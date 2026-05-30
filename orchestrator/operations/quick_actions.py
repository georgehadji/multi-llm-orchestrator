"""
QuickActionResolver — Contextual actions after Plan Mode.
==========================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Part of Category 1 (Wave 3: W3 Quick Action Buttons).
After the orchestrator produces a plan (task list), this module suggests
contextual next actions: implement, refine, estimate cost, show alternatives,
generate documentation, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class QuickActionKind(str, Enum):
    """Kinds of quick actions available after plan creation."""

    IMPLEMENT = "implement"  # Execute the plan
    REFINE = "refine"  # Refine the plan with feedback
    ESTIMATE = "estimate"  # Show cost/time estimates
    ALTERNATIVE = "alternative"  # Generate alternative approach
    DOCUMENT = "document"  # Generate project documentation
    REVIEW = "review"  # Review plan for issues
    EXPORT = "export"  # Export to file
    CANCEL = "cancel"  # Discard plan and start fresh


@dataclass
class QuickAction:
    """A single suggested quick action."""

    kind: QuickActionKind
    label: str
    description: str
    hotkey: str | None = None
    auto_trigger: bool = False  # Trigger automatically if conditions met

    def to_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind.value,
            "label": self.label,
            "description": self.description,
            "hotkey": self.hotkey or "",
        }


class QuickActionResolver:
    """Resolves which quick actions to suggest based on plan state.

    Usage:
        resolver = QuickActionResolver()
        actions = resolver.suggest(tasks_count=5, estimated_cost=2.50)
        for action in actions:
            print(f"[{action.hotkey}] {action.label} — {action.description}")
    """

    def __init__(self) -> None:
        self._all_actions: dict[QuickActionKind, QuickAction] = {
            QuickActionKind.IMPLEMENT: QuickAction(
                kind=QuickActionKind.IMPLEMENT,
                label="Implement Plan",
                description="Execute all tasks with the selected autonomy level",
                hotkey="Enter",
                auto_trigger=True,
            ),
            QuickActionKind.REFINE: QuickAction(
                kind=QuickActionKind.REFINE,
                label="Refine Plan",
                description="Adjust the plan with feedback before execution",
                hotkey="r",
            ),
            QuickActionKind.ESTIMATE: QuickAction(
                kind=QuickActionKind.ESTIMATE,
                label="Show Estimate",
                description="Display cost and time breakdown per task",
                hotkey="e",
            ),
            QuickActionKind.ALTERNATIVE: QuickAction(
                kind=QuickActionKind.ALTERNATIVE,
                label="Alternative Approach",
                description="Generate a different task breakdown",
                hotkey="a",
            ),
            QuickActionKind.DOCUMENT: QuickAction(
                kind=QuickActionKind.DOCUMENT,
                label="Generate Docs",
                description="Create README and architecture documentation",
                hotkey="d",
            ),
            QuickActionKind.REVIEW: QuickAction(
                kind=QuickActionKind.REVIEW,
                label="Review Plan",
                description="Run a security/cost/feasibility review on this plan",
                hotkey="v",
            ),
            QuickActionKind.EXPORT: QuickAction(
                kind=QuickActionKind.EXPORT,
                label="Export Plan",
                description="Save the plan as JSON/YAML for later use",
                hotkey="x",
            ),
            QuickActionKind.CANCEL: QuickAction(
                kind=QuickActionKind.CANCEL,
                label="Cancel",
                description="Discard this plan and start fresh",
                hotkey="Esc",
            ),
        }

    def suggest(
        self,
        tasks_count: int = 0,
        estimated_cost: float = 0.0,
        estimated_time_minutes: int = 0,
        has_dependencies: bool = False,
        is_retry: bool = False,
    ) -> list[QuickAction]:
        """Suggest contextual quick actions based on plan state.

        Args:
            tasks_count: Number of tasks in the plan
            estimated_cost: Estimated USD cost
            estimated_time_minutes: Estimated execution time
            has_dependencies: Whether tasks have inter-dependencies
            is_retry: Whether this is a retry of a previous plan

        Returns:
            Ordered list of suggested actions
        """
        suggestions: list[QuickAction] = []

        # Always suggest implement and cancel
        suggestions.append(self._all_actions[QuickActionKind.IMPLEMENT])

        # Refine if the plan is large or costly
        if tasks_count > 5 or estimated_cost > 5.0:
            suggestions.append(self._all_actions[QuickActionKind.REFINE])

        # Estimate if there's a notable cost
        if estimated_cost > 2.0 or estimated_time_minutes > 30:
            suggestions.append(self._all_actions[QuickActionKind.ESTIMATE])

        # Alternative if this is a retry
        if is_retry:
            suggestions.append(self._all_actions[QuickActionKind.ALTERNATIVE])

        # Review for complex plans
        if tasks_count > 10 or has_dependencies:
            suggestions.append(self._all_actions[QuickActionKind.REVIEW])

        # Doc for larger projects
        if tasks_count > 3:
            suggestions.append(self._all_actions[QuickActionKind.DOCUMENT])

        # Alternative and export always available
        if self._all_actions[QuickActionKind.ALTERNATIVE] not in suggestions:
            suggestions.append(self._all_actions[QuickActionKind.ALTERNATIVE])
        suggestions.append(self._all_actions[QuickActionKind.EXPORT])
        suggestions.append(self._all_actions[QuickActionKind.CANCEL])

        return suggestions

    @staticmethod
    def present(actions: list[QuickAction]) -> str:
        """Format quick actions for display."""
        lines = ["\nAvailable Actions:", "-" * 40]
        for action in actions:
            hotkey = f"[{action.hotkey}] " if action.hotkey else ""
            lines.append(f"  {hotkey}{action.label:<20} — {action.description}")
        return "\n".join(lines)

    def resolve(self, hotkey: str) -> QuickAction | None:
        """Resolve a hotkey to a QuickAction."""
        for action in self._all_actions.values():
            if action.hotkey and action.hotkey.lower() == hotkey.lower():
                return action
        return None