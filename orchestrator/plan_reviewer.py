"""
PlanBuilder - Interactive plan refinement before execution.
============================================================
Author: Georgios-Chrysovalantis Chatzivantsidis
Part of Category 10, Phase 2 (Replit-inspired).
"""

from dataclasses import dataclass, field
from typing import Any, Callable
import logging

logger = logging.getLogger(__name__)


@dataclass
class PlanReviewResult:
    approved: bool = False
    feedback: list = field(default_factory=list)
    modifications: dict = field(default_factory=dict)


class PlanReviewer:
    """Reviews decomposed task plans before execution."""

    def __init__(self):
        self._approvers = []

    def add_approver(self, fn):
        self._approvers.append(fn)

    async def review(self, tasks, context=None):
        result = PlanReviewResult(approved=True)
        for approver in self._approvers:
            r = approver(tasks)
            result.feedback.extend(r.feedback)
            result.modifications.update(r.modifications)
            if not r.approved:
                result.approved = False
        return result

    @staticmethod
    def estimate_resources(tasks):
        total = len(tasks)
        return {
            "task_count": total,
            "estimated_cost_usd": total * 0.15,
            "estimated_time_minutes": total * 5,
            "has_dependencies": any(t.get("dependencies") for t in tasks),
            "parallelizable": total - sum(1 for t in tasks if t.get("dependencies")),
        }

    @staticmethod
    def validate_plan(tasks):
        issues = []
        task_ids = {t.get("id", "") for t in tasks}
        for task in tasks:
            tid = task.get("id", "?")
            if not task.get("description"):
                issues.append(f"Task {tid}: missing description")
            if not task.get("type"):
                issues.append(f"Task {tid}: missing task type")
        for task in tasks:
            for dep in task.get("dependencies", []):
                if dep and dep not in task_ids:
                    issues.append(f"Task {task.get('id', '?')}: depends on unknown task '{dep}'")
        return issues
