"""
Goal, SubGoal, Plan — Core data types for recursive planning.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Goal:
    """A high-level goal that can be decomposed."""
    description: str
    context: str = ""


class SubGoal:
    """A single sub-goal within a plan."""
    def __init__(self, id: str, description: str, is_atomic: bool = False,
                 agent_role: str = "", depends_on: list[str] = None):
        self.id = id
        self.description = description
        self.is_atomic = is_atomic
        self.agent_role = agent_role
        self.depends_on = depends_on or []


@dataclass
class Plan:
    """A full plan with ordered tasks."""
    tasks: list[SubGoal] = field(default_factory=list)

    @classmethod
    def merge(cls, plans: list["Plan"]) -> "Plan":
        all_tasks = []
        seen_ids = set()
        for plan in plans:
            for task in plan.tasks:
                if task.id not in seen_ids:
                    seen_ids.add(task.id)
                    all_tasks.append(task)
        return Plan(tasks=all_tasks)
