"""
GoalDecomposer — Recursive HTN planning for breaking goals into sub-goals
===========================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Capability 3 of the Agentic System Implementation Plan.
Breaks high-level goals into sub-goals recursively until each leaf
is an atomic action suitable for a single agent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .goal import Goal, SubGoal, Plan

logger = logging.getLogger("orchestrator.planning.decomposer")

MAX_DEPTH = 4


@dataclass
class GoalDecomposer:
    """Recursive HTN planner that breaks goals into atomic actions.

    Uses task type to determine decomposition strategy:
    - CODE_GEN tasks are leaf actions
    - REASONING tasks may decompose further
    - Goals with "and", "multiple", etc. are split
    """

    async def decompose(self, goal: str, context: str = "", depth: int = 0) -> Plan:
        """Decompose a goal into sub-goals.

        Args:
            goal: The goal to decompose.
            context: Optional context.
            depth: Current recursion depth.

        Returns:
            Plan with ordered sub-goals.
        """
        if depth >= MAX_DEPTH:
            return Plan(
                tasks=[SubGoal(id=f"sg_{hash(goal) % 10000}", description=goal, is_atomic=True)]
            )

        # Check if this is a compound goal
        sub_goals = self._split_goal(goal)
        if len(sub_goals) <= 1:
            return Plan(
                tasks=[SubGoal(id=f"sg_{hash(goal) % 10000}", description=goal, is_atomic=True)]
            )

        # Recurse on each sub-goal
        sub_plans = []
        for sg in sub_goals:
            sub_plan = await self.decompose(sg, context, depth + 1)
            sub_plans.append(sub_plan)

        return Plan.merge(sub_plans)

    def _split_goal(self, goal: str) -> list[str]:
        """Split a compound goal into sub-goals.

        Uses heuristics: "and", bullet points, numbered lists.
        """
        goal = goal.strip()
        results: list[str] = []

        # Check for bullet points or numbered lists
        lines = goal.split("\n")
        items = [
            l.strip()
            for l in lines
            if l.strip().startswith(("- ", "* ", "1. ", "2. ", "3. ", "4. ", "5. "))
        ]
        if len(items) >= 2:
            return items

        # Split on " and " (top-level)
        parts = goal.split(" and ")
        if len(parts) >= 2:
            for p in parts:
                p = p.strip().strip(".,").strip()
                if p:
                    results.append(p)
            return results

        return [goal]


class SubGoal:
    """A single sub-goal within a plan."""

    def __init__(
        self,
        id: str,
        description: str,
        is_atomic: bool = False,
        agent_role: str = "",
        depends_on: list[str] = None,
    ):
        self.id = id
        self.description = description
        self.is_atomic = is_atomic
        self.agent_role = agent_role
        self.depends_on = depends_on or []


class Goal:
    """A high-level goal that can be decomposed."""

    def __init__(self, description: str, context: str = ""):
        self.description = description
        self.context = context
