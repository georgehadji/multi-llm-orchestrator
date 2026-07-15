"""
MechanismResearcher — LLM-Driven Mechanism Generation (Level 2)
================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Implements the paper's 4-round dialogue to generate new search mechanisms
from project execution traces:

1. **Explore** — analyses trace metrics and surveys adjacent search domains
2. **Critique** — reasons over candidate mechanisms and identifies weaknesses
3. **Specify** — produces a structured interface spec and patch description
4. **Generate Code** — emits runnable Python code for the mechanism

Each round uses the orchestrator's ``UnifiedClient`` via the configured
``ROUTING_TABLE`` for ``TaskType.MECHANISM_RESEARCH``.

Model cascading:
- Rounds 1-3 use the primary model (DeepSeek V4 Pro by default).
- Round 4 uses the primary model first; if validation fails, retries with
  the fallback model (Claude Opus 4.8).

Usage:
    researcher = MechanismResearcher(client=unified_client, selector=model_selector)
    result = await researcher.research(trace)
    # result = {"name": "tabu_search", "code": "class TabuSearchManager: ...", ...}
"""

from __future__ import annotations

import json
import logging
import textwrap
from dataclasses import dataclass, field
from typing import Any

from ..models import ROUTING_TABLE, TaskType
from .trace_analyzer import SearchTrace

logger = logging.getLogger("orchestrator.bilevel.mechanism_researcher")


# ── Data structures ───────────────────────────────────────────────────────


@dataclass
class MechanismSpec:
    """Output of the mechanism research pipeline."""

    name: str = ""
    description: str = ""
    code: str = ""
    version: str = "1.0"
    model_used: str = ""
    rounds_completed: int = 0
    validation_log: list[str] = field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        return bool(self.code) and self.rounds_completed >= 3

    @property
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "code": self.code,
            "version": self.version,
            "model_used": self.model_used,
            "rounds_completed": self.rounds_completed,
        }


# ── Prompts ───────────────────────────────────────────────────────────────


EXPLORE_PROMPT = """\
You are a search-mechanism researcher for an autonomous coding system.
Analyse the trace below and propose 1-2 novel search mechanisms that could
improve the system's performance.

Current metrics:
- Repetition rate: {repetition_rate:.0%}
- Dominant model share: {dominant_model_share:.0%}
- Stagnation score: {stagnation_score:.0%}
- Average cost per score point: ${avg_cost_per_point:.4f}

Respond with a JSON list of mechanism proposals, each with:
  name: short kebab-case name
  description: one-line summary
  rationale: why this would help given the trace metrics
"""

CRITIQUE_PROMPT = """\
Critique the following mechanism proposals for a search system.
For each, identify potential failure modes, edge cases, and improvements.

Proposals:
{proposals}

Respond with a JSON list of critiques, each with:
  name: matching proposal name
  strengths: list of strengths
  weaknesses: list of potential issues
  improvements: list of concrete suggestions
"""

SPECIFY_PROMPT = """\
Given the following mechanism proposals and critiques, produce a detailed
specification for the most promising mechanism.

Critiques:
{critiques}

Respond with a JSON object containing:
  name: mechanism name
  description: detailed description
  interface: Python class/method signatures
  patch_description: what the mechanism changes in the pipeline
  success_criteria: how to measure success
"""

GENERATE_PROMPT = """\
Generate a complete, runnable Python module for the following mechanism spec.
The module must be self-contained (no external imports beyond the standard
library and orchestrator's existing module paths).

Spec:
{spec}

Requirements:
- Define a single public class matching the interface spec
- Include type annotations
- Include a module-level docstring
- Use only the standard library + references to orchestrator.*
- The class must be importable: `from module import ClassName`

Respond with a JSON object:
  name: mechanism name
  code: the complete Python source code as a single string
"""


# ── Researcher ────────────────────────────────────────────────────────────


class MechanismResearcher:
    """Generate new search mechanisms via LLM dialogue.

    Args:
        client: ``UnifiedClient`` or compatible ``call_model()`` interface.
        selector: Model selector for routing to the right model.
        primary_model: Override for the primary research model.
        fallback_model: Override for the code-generation fallback model.
    """

    def __init__(
        self,
        client: Any = None,
        selector: Any = None,
        primary_model: Any = None,
        fallback_model: Any = None,
    ) -> None:
        self._client = client
        self._selector = selector
        self._primary_model = primary_model
        self._fallback_model = fallback_model

    async def research(
        self,
        trace: SearchTrace,
    ) -> MechanismSpec:
        """Run the full 4-round research dialogue.

        Args:
            trace: ``SearchTrace`` from the ``TraceAnalyzer``.

        Returns:
            A ``MechanismSpec`` with the generated mechanism.
        """
        spec = MechanismSpec()

        # Round 1: Explore
        explore_result = await self._call_round(
            EXPLORE_PROMPT.format(
                repetition_rate=trace.repetition_rate,
                dominant_model_share=trace.dominant_model_share,
                stagnation_score=trace.stagnation_score,
                avg_cost_per_point=trace.avg_cost_per_point,
            ),
            round_name="explore",
            spec=spec,
        )
        if not explore_result:
            spec.validation_log.append("Explore round failed")
            return spec

        # Round 2: Critique
        critique_result = await self._call_round(
            CRITIQUE_PROMPT.format(proposals=json.dumps(explore_result, indent=2)),
            round_name="critique",
            spec=spec,
        )
        if not critique_result:
            spec.validation_log.append("Critique round failed")
            return spec

        # Round 3: Specify
        specify_result = await self._call_round(
            SPECIFY_PROMPT.format(critiques=json.dumps(critique_result, indent=2)),
            round_name="specify",
            spec=spec,
        )
        if not specify_result:
            spec.validation_log.append("Specify round failed")
            return spec

        # Round 4: Generate Code (with fallback)
        code_result = await self._generate_with_fallback(
            GENERATE_PROMPT.format(spec=json.dumps(specify_result, indent=2)),
            spec=spec,
        )
        if not code_result:
            spec.validation_log.append("Generate round failed")
            return spec

        spec.rounds_completed = 4
        return spec

    # ── Internal ──────────────────────────────────────────────────────

    async def _call_round(
        self,
        prompt: str,
        round_name: str,
        spec: MechanismSpec,
        model_override: Any = None,
    ) -> Any:
        """Call the LLM for one research round and parse the JSON response."""
        if self._client is None:
            logger.warning("No client available for %s round", round_name)
            return None

        model = model_override or self._get_model_for_round(round_name)
        spec.model_used = str(model) if model else "unknown"
        spec.validation_log.append(f"{round_name}: using {model}")

        try:
            response = await self._client.call_model(
                model=model,
                system=(
                    "You are a search-mechanism researcher. "
                    "Respond with valid JSON only, no markdown fences."
                ),
                prompt=prompt,
            )
            text = response.text if hasattr(response, "text") else str(response)

            # Parse JSON from response
            import re

            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            # Try array
            json_match = re.search(r"\[.*\]", text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())

            spec.validation_log.append(f"{round_name}: no JSON found in response")
            return None

        except Exception as exc:
            logger.warning("MechanismResearch %s round failed: %s", round_name, exc)
            spec.validation_log.append(f"{round_name}: {exc}")
            return None

    async def _generate_with_fallback(
        self,
        prompt: str,
        spec: MechanismSpec,
    ) -> Any:
        """Generate code, retrying with fallback model on validation failure."""
        # Primary attempt
        primary_model = self._get_model_for_round("generate")
        result = await self._call_round(prompt, "generate_code", spec, model_override=primary_model)

        if result and self._validate_code(result):
            return result

        # Fallback attempt with Claude Opus
        fallback = self._fallback_model or self._get_model_for_round("generate_fallback")
        spec.validation_log.append(f"Primary code validation failed, retrying with {fallback}")
        logger.info("MechanismResearcher: primary code failed, falling back to %s", fallback)

        result = await self._call_round(
            prompt, "generate_code_fallback", spec, model_override=fallback
        )
        if result and self._validate_code(result):
            spec.validation_log.append("Fallback generation succeeded")
            return result

        spec.validation_log.append("Both primary and fallback code generation failed")
        return None

    def _validate_code(self, result: dict[str, Any]) -> bool:
        """Validate generated code: must parse as Python and contain a class."""
        code = result.get("code", "")
        if not code:
            return False
        try:
            import ast

            tree = ast.parse(code)
            # Must contain at least one class definition
            return any(isinstance(node, ast.ClassDef) for node in ast.walk(tree))
        except SyntaxError:
            return False

    def _get_model_for_round(self, round_name: str) -> Any:
        """Get the model for a given research round.

        Uses the first entry in the ROUTING_TABLE for MECHANISM_RESEARCH
        as the primary model. Overrides can be set in constructor.
        """
        if round_name == "generate_fallback":
            # Claude Opus for fallback code generation
            from ..models import Model

            return Model.CLAUDE_OPUS_4_8

        if self._primary_model:
            return self._primary_model

        routing = ROUTING_TABLE.get(TaskType.MECHANISM_RESEARCH, [])
        if routing:
            return routing[0]
        return None
