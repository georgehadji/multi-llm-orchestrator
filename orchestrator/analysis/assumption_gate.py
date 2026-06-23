"""
Assumption Gate — surface hidden assumptions before generation.
=================================================================

Karpathy Principle 1: "Think Before Coding"
- State assumptions explicitly. If uncertain, ASK.
- If multiple interpretations exist, present ALL of them.

This gate runs BEFORE task decomposition to catch ambiguity early.
Cost: ~$0.001 per call (cheapest model). Pre-checks skip the LLM
call for clearly unambiguous task descriptions.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # FIXED: from .api_clients import UnifiedClient
    from ...api_clients import UnifiedClient

# FIXED: from .models import Model
from ...models import Model

# Cheapest available model for the lightweight assumption-surfacing call.
# UnifiedClient has no get_cheapest_model() method; pin directly to the
# lowest-cost non-free model so the gate stays at ~$0.001 per call.
_ASSUMPTION_MODEL = Model.ZHIPU_GLM_5_2

logger = logging.getLogger("orchestrator.assumption_gate")


@dataclass
class AssumptionReport:
    """Report of hidden assumptions detected in a task description."""

    has_ambiguity: bool = False
    assumptions: list[dict] = field(default_factory=list)
    interpretations: list[str] = field(default_factory=list)
    clarification_questions: list[str] = field(default_factory=list)
    confidence: float = 1.0

    def to_prompt_context(self) -> str:
        """Render as context for the user to review."""
        if not self.has_ambiguity and not self.assumptions:
            return ""

        lines = ["## Assumptions & Clarifications"]
        if self.assumptions:
            lines.append("\n### Assumptions Made")
            for a in self.assumptions:
                confidence = a.get("confidence", "medium")
                lines.append(f"- {a['statement']} (confidence: {confidence})")

        if self.interpretations:
            lines.append("\n### Multiple Interpretations")
            for i, interp in enumerate(self.interpretations, 1):
                lines.append(f"{i}. {interp}")

        if self.clarification_questions:
            lines.append("\n### Questions to Clarify")
            for q in self.clarification_questions:
                lines.append(f"- {q}")

        return "\n".join(lines)


async def surface_assumptions(
    task_description: str,
    client: UnifiedClient,
    threshold: float = 0.7,
) -> AssumptionReport:
    """Ask the LLM to surface hidden assumptions before implementation.

    Only triggers the LLM call if the task description contains
    ambiguous language or appears to warrant clarification.

    Args:
        task_description: The task description to analyze.
        client: API client for LLM calls.
        threshold: Confidence threshold below which ambiguity is flagged.

    Returns:
        AssumptionReport with surfaced assumptions, or empty report
        if description is clearly unambiguous.
    """
    # Quick pre-check: skip LLM call for clearly unambiguous descriptions
    if _is_unambiguous(task_description):
        return AssumptionReport(has_ambiguity=False)

    prompt = (
        "Analyze this task description for hidden assumptions and ambiguity.\n\n"
        f"TASK: {task_description}\n\n"
        "Return JSON:\n"
        "{\n"
        '  "has_ambiguity": true/false,\n'
        '  "assumptions": [{"statement": "...", "confidence": "high|medium|low"}],\n'
        '  "interpretations": ["interpretation 1", "interpretation 2"],\n'
        '  "clarification_questions": ["question 1", "question 2"],\n'
        '  "confidence": 0.0-1.0\n'
        "}\n\n"
        "Only flag actual ambiguity. Do not fabricate issues for clear descriptions."
    )

    try:
        response = await client.call(
            model=_ASSUMPTION_MODEL,
            prompt=prompt,
            max_tokens=300,
            temperature=0.1,
        )

        data = json.loads(response.text)
        return AssumptionReport(
            has_ambiguity=data.get("has_ambiguity", False),
            assumptions=data.get("assumptions", []),
            interpretations=data.get("interpretations", []),
            clarification_questions=data.get("clarification_questions", []),
            confidence=data.get("confidence", 1.0),
        )
    except Exception:
        logger.warning("Assumption surfacing failed, proceeding without check")
        return AssumptionReport(has_ambiguity=False)


def _is_unambiguous(description: str) -> bool:
    """Quick heuristic check for clearly unambiguous descriptions.

    Unambiguous signals:
    - Contains specific file paths: "in src/auth.py"
    - Contains exact values: "set timeout to 30s"
    - Contains test-first language: "write a test that..."

    Ambiguous signals (triggers the LLM check):
    - Contains vague verbs: "make it better", "fix it", "improve"
    - Contains ambiguous nouns: "the system", "the thing"
    - Contains no specific targets
    """
    # Ambiguous patterns — if any match, need LLM check
    vague_patterns = [
        r"\bmake it\b",
        r"\bfix it\b",
        r"\bimprove\b",
        r"\bthe system\b",
        r"\bthe thing\b",
        r"\bthe app\b",
    ]
    for pattern in vague_patterns:
        if re.search(pattern, description, re.IGNORECASE):
            return False

    # Specific patterns — if any match, description is clear
    specific_patterns = [
        r"\bin \w+\.\w+\b",  # "in auth.py"
        r"\bset \w+ to \w+\b",  # "set timeout to 30s"
        r"\bwrite a test\b",  # test-first
        r"\badd a \w+\.\w+\b",  # "add a Button.tsx"
    ]
    return any(re.search(pattern, description, re.IGNORECASE) for pattern in specific_patterns)
