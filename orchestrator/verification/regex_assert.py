"""
Regex Assertion Verifier — checks generated output against
deterministic string / regex contracts.

Useful for structured outputs where expected patterns are known:
- Must contain ``import`` statements (code tasks)
- Must contain specific keywords or phrases
- Must NOT contain forbidden patterns (placeholders, error messages)
- Must match expected format (URLs, email addresses, etc.)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import ClassVar

from orchestrator.models import TaskType, Verdict

logger = logging.getLogger(__name__)


@dataclass
class AssertionRule:
    """A single regex assertion to check against the response."""

    pattern: str  # regex pattern (compiled internally)
    name: str  # signal label for telemetry
    kind: str = "must_match"  # "must_match" or "must_not_match"
    description: str = ""

    def __post_init__(self) -> None:
        self._compiled: re.Pattern[str] | None = None

    def matches(self, text: str) -> bool:
        """Check whether the pattern appears in *text*."""
        if self._compiled is None:
            self._compiled = re.compile(self.pattern)
        return bool(self._compiled.search(text))

    def __hash__(self) -> int:
        return hash((self.pattern, self.name, self.kind))


# ─────────────────────────────────────────────
# Default rules by task type
# ─────────────────────────────────────────────

DEFAULT_RULES: dict[str, list[AssertionRule]] = {
    "code_generation": [
        AssertionRule(r"def |class ", "has_func_or_class", kind="must_match"),
        AssertionRule(r"TODO|FIXME|XXX", "no_todo", kind="must_not_match"),
    ],
    "code_review": [
        AssertionRule(r"```", "has_code_block", kind="must_match"),
        AssertionRule(r"risk|vulnerability|issue", "has_review_terms", kind="must_match"),
    ],
    "data_extraction": [
        AssertionRule(r"[:{\[]", "has_structured_data", kind="must_match"),
    ],
    "creative_writing": [
        AssertionRule(r"\n\n", "has_paragraph_break", kind="must_match"),
        AssertionRule(r"^\S", "starts_with_content", kind="must_match"),
    ],
    "summarization": [
        AssertionRule(r"\b\d{1,3}\b", "has_numeric_stats", kind="must_match"),
    ],
}

# Catch-all for any unlisted task_type
FALLBACK_RULES: list[AssertionRule] = [
    AssertionRule(r".", "non_empty", kind="must_match"),
]


def _get_rules(task_type: TaskType) -> list[AssertionRule]:
    """Look up rules for *task_type*; fallback if absent."""
    return DEFAULT_RULES.get(task_type.value, FALLBACK_RULES)


# ─────────────────────────────────────────────
# Verifier
# ─────────────────────────────────────────────


class RegexAssertVerifier:
    """Verify response against regex-based assertion rules.

    Built-in defaults exist per task type; custom rules can be
    provided at construction time.

    Usage:
        verifier = RegexAssertVerifier()

        # Or with custom rules:
        from orchestrator.verification.regex_assert import AssertionRule
        custom_rules = {
            "code_generation": [
                AssertionRule(r"async def", "has_async", kind="must_match"),
            ],
        }
        verifier = RegexAssertVerifier(rules=custom_rules)
    """

    RELEVANT_TASK_TYPES: ClassVar[frozenset[str]] = frozenset(
        {
            "code_generation",
            "code_review",
            "data_extraction",
            "creative_writing",
            "summarization",
        }
    )

    def __init__(
        self,
        rules: dict[str, list[AssertionRule]] | None = None,
    ) -> None:
        """Initialize verifier.

        Args:
            rules: Per-task-type rule overrides. If ``None``, uses
                built-in ``DEFAULT_RULES``.  Missing keys fall back
                to defaults.
        """
        self._rules: dict[str, list[AssertionRule]] = {}
        if rules is not None:
            self._rules.update(rules)
        # Fill in defaults for missing task types
        for key, default_list in DEFAULT_RULES.items():
            if key not in self._rules:
                self._rules[key] = list(default_list)

    async def verify(
        self,
        *,
        prompt: str,
        response: str,
        task_type: TaskType,
    ) -> Verdict:
        """Run regex assertions against *response*."""
        if task_type.value not in self.RELEVANT_TASK_TYPES:
            return Verdict(
                passed=True,
                score=0.5,
                signals=("not_applicable",),
                detail=f"RegexAssertVerifier skipped for {task_type.value}",
            )

        rules = self._rules.get(task_type.value, _get_rules(task_type))
        signals: list[str] = []
        failures: list[str] = []

        for rule in rules:
            matched = rule.matches(response)
            if rule.kind == "must_match":
                if matched:
                    signals.append(rule.name)
                else:
                    failures.append(f"Expected '{rule.name}' ({rule.description or rule.pattern})")
            elif rule.kind == "must_not_match":
                if matched:
                    failures.append(f"Forbidden '{rule.name}' ({rule.description or rule.pattern})")
                else:
                    signals.append(f"no_{rule.name}")

        passed = len(failures) == 0
        # Score: start at 0.7, -0.2 per failure, floor 0.0
        score = max(0.0, 0.7 - 0.2 * len(failures))

        return Verdict(
            passed=passed,
            score=score,
            signals=tuple(signals),
            detail="; ".join(failures) if failures else "All regex assertions passed",
        )
