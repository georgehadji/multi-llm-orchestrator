"""
CritiqueReport — typed feedback between evaluation and generation
==================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Carries structured critique output between pipeline phases so that
the generator receives typed, categorised feedback instead of raw
LLM text. Every edge in the graph becomes EXTRACTED with confidence 1.0.

Design:
  - Immutable dataclasses — no shared mutable state
  - CritiqueItem maps to a specific severity + category
  - CritiqueReport aggregates items and renders to prompt context
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class CritiqueSeverity(Enum):
    """Severity levels for critique items.

    BLOCKER — must fix before proceeding
    MAJOR — should fix
    MINOR — nice to fix
    SUGGESTION — optional improvement
    """

    BLOCKER = "blocker"
    MAJOR = "major"
    MINOR = "minor"
    SUGGESTION = "suggestion"


@dataclass(frozen=True)
class CritiqueItem:
    """A single critique finding.

    Attributes:
        severity: How urgently this must be addressed
        category: Domain of the issue
            (security, architecture, style, correctness, performance, completeness)
        description: What's wrong
        location: Optional file/line reference
        suggestion: Optional concrete fix suggestion
    """

    severity: CritiqueSeverity
    category: str
    description: str
    location: Optional[str] = None
    suggestion: Optional[str] = None


@dataclass(frozen=True)
class CritiqueReport:
    """Structured critique result from the evaluator.

    Attributes:
        task_id: The task that was evaluated
        score: Overall quality score (0.0-10.0)
        items: Individual critique findings
        passed_validators: Whether deterministic validators passed
        model_used: Model that produced the evaluation
        tokens_used: Tokens consumed by evaluation

    Properties:
        has_blockers — True if any BLOCKER items exist
        has_major — True if any MAJOR items exist
    """

    task_id: str
    score: float = 5.0
    items: list[CritiqueItem] = field(default_factory=list)
    passed_validators: bool = False
    model_used: Optional[str] = None
    tokens_used: int = 0

    @property
    def has_blockers(self) -> bool:
        return any(i.severity == CritiqueSeverity.BLOCKER for i in self.items)

    @property
    def has_major(self) -> bool:
        return any(i.severity == CritiqueSeverity.MAJOR for i in self.items)

    def to_prompt_context(self, max_items: int = 15) -> str:
        """Render this critique as a structured prompt for the next generation pass.

        Args:
            max_items: Maximum number of items to include (prevents prompt bloat)

        Returns:
            Formatted string suitable for insertion into a system prompt.
        """
        lines = [
            "## CRITIQUE FEEDBACK",
            f"Overall score: {self.score:.1f}/10",
        ]

        if self.has_blockers:
            lines.append(
                f"BLOCKERS: {sum(1 for i in self.items if i.severity == CritiqueSeverity.BLOCKER)}"
            )
        if self.has_major:
            lines.append(
                f"MAJOR issues: {sum(1 for i in self.items if i.severity == CritiqueSeverity.MAJOR)}"
            )

        lines.append("")

        # Sort by severity: BLOCKER first, then MAJOR, MINOR, SUGGESTION
        severity_order = {
            CritiqueSeverity.BLOCKER: 0,
            CritiqueSeverity.MAJOR: 1,
            CritiqueSeverity.MINOR: 2,
            CritiqueSeverity.SUGGESTION: 3,
        }
        sorted_items = sorted(self.items, key=lambda i: severity_order.get(i.severity, 99))

        for item in sorted_items[:max_items]:
            tag = f"[{item.severity.value.upper()}]"
            location = f" at {item.location}" if item.location else ""
            suggestion = f"\n  Fix: {item.suggestion}" if item.suggestion else ""
            lines.append(f"{tag} [{item.category}]{location}: {item.description}{suggestion}")

        omitted = len(self.items) - max_items
        if omitted > 0:
            lines.append(f"\n... and {omitted} more item(s)")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize to dictionary for persistence."""
        return {
            "task_id": self.task_id,
            "score": self.score,
            "items": [
                {
                    "severity": i.severity.value,
                    "category": i.category,
                    "description": i.description,
                    "location": i.location,
                    "suggestion": i.suggestion,
                }
                for i in self.items
            ],
            "passed_validators": self.passed_validators,
            "model_used": self.model_used,
            "tokens_used": self.tokens_used,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CritiqueReport":
        """Deserialize from dictionary."""
        return cls(
            task_id=data["task_id"],
            score=data.get("score", 5.0),
            items=[
                CritiqueItem(
                    severity=CritiqueSeverity(item["severity"]),
                    category=item["category"],
                    description=item["description"],
                    location=item.get("location"),
                    suggestion=item.get("suggestion"),
                )
                for item in data.get("items", [])
            ],
            passed_validators=data.get("passed_validators", False),
            model_used=data.get("model_used"),
            tokens_used=data.get("tokens_used", 0),
        )
