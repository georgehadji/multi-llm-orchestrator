"""
PlanReviewData - Task approve/reject/refine data for plan review panel.
========================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis
Part of Category 7, Phase U1 (UI): Plan review panel backend.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class ReviewableTask:
    task_id: str
    description: str
    type: str = ""
    dependencies: list = field(default_factory=list)
    estimated_cost: float = 0.0
    status: str = "pending"  # pending, approved, rejected, modified
    feedback: str = ""
    modifications: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "task_id": self.task_id,
            "description": self.description,
            "type": self.type,
            "dependencies": self.dependencies,
            "estimated_cost": self.estimated_cost,
            "status": self.status,
            "feedback": self.feedback,
        }


@dataclass
class PlanReview:
    plan_id: str
    description: str = ""
    criteria: str = ""
    tasks: list[ReviewableTask] = field(default_factory=list)
    estimated_total_cost: float = 0.0
    estimated_time_minutes: int = 0
    status: str = "pending"  # pending, approved, rejected, in_progress

    def to_dict(self):
        return {
            "plan_id": self.plan_id,
            "description": self.description,
            "criteria": self.criteria,
            "tasks": [t.to_dict() for t in self.tasks],
            "estimated_total_cost": self.estimated_total_cost,
            "estimated_time_minutes": self.estimated_time_minutes,
            "status": self.status,
        }


class PlanReviewData:
    """Provides data for the interactive plan review panel."""

    def __init__(self):
        self._reviews: dict[str, PlanReview] = {}

    def create_review(self, plan_id, tasks, description="", criteria=""):
        """Create a new plan review from a task list."""
        reviewable = [
            ReviewableTask(
                task_id=t.get("id", f"task_{i}"),
                description=t.get("description", ""),
                type=t.get("type", ""),
                dependencies=t.get("dependencies", []),
                estimated_cost=t.get("estimated_cost", 0.15),
            )
            for i, t in enumerate(tasks)
        ]
        review = PlanReview(
            plan_id=plan_id,
            description=description,
            criteria=criteria,
            tasks=reviewable,
            estimated_total_cost=sum(r.estimated_cost for r in reviewable),
            estimated_time_minutes=len(reviewable) * 5,
        )
        self._reviews[plan_id] = review
        return review

    def approve_task(self, plan_id, task_id, feedback=""):
        review = self._reviews.get(plan_id)
        if review:
            for t in review.tasks:
                if t.task_id == task_id:
                    t.status = "approved"
                    t.feedback = feedback
            self._update_plan_status(review)

    def reject_task(self, plan_id, task_id, feedback=""):
        review = self._reviews.get(plan_id)
        if review:
            for t in review.tasks:
                if t.task_id == task_id:
                    t.status = "rejected"
                    t.feedback = feedback
            self._update_plan_status(review)

    def modify_task(self, plan_id, task_id, modifications):
        review = self._reviews.get(plan_id)
        if review:
            for t in review.tasks:
                if t.task_id == task_id:
                    t.status = "modified"
                    t.modifications = modifications
            self._update_plan_status(review)

    def approve_all(self, plan_id):
        review = self._reviews.get(plan_id)
        if review:
            for t in review.tasks:
                if t.status == "pending":
                    t.status = "approved"
            review.status = "approved"

    def _update_plan_status(self, review):
        statuses = {t.status for t in review.tasks}
        if "rejected" in statuses:
            review.status = "rejected"
        elif all(s == "approved" for s in statuses):
            review.status = "approved"
        else:
            review.status = "in_progress"

    def get_review(self, plan_id):
        review = self._reviews.get(plan_id)
        return review.to_dict() if review else None

    def get_pending_tasks(self, plan_id):
        review = self._reviews.get(plan_id)
        return [t.to_dict() for t in review.tasks if t.status == "pending"] if review else []
