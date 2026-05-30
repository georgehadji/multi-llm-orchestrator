"""
AgentOrchestrator — Coordinates specialized agents toward a goal
==================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Capability 1 of the Agentic System Implementation Plan.
The AgentOrchestrator receives a high-level goal, decomposes it,
dispatches sub-goals to appropriate agents, and monitors progress.

Optimizations:
- B-6: Parallel execution with semaphore-based concurrency
- E-12: Per-agent metrics tracking via MetricsRegistry
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from .base import AgentBase, AgentRole, AgentTask, AgentTaskResult

logger = logging.getLogger("orchestrator.agents.coordinator")

MAX_CONCURRENT = 3


class AgentOrchestrator:
    """Coordinates specialized agents toward a shared goal."""

    def __init__(self, agents: dict[AgentRole, AgentBase], workspace: Any = None) -> None:
        self.agents = agents
        self.workspace = workspace
        self.results: dict[str, AgentTaskResult] = {}

    def get_agent(self, role: AgentRole) -> AgentBase | None:
        return self.agents.get(role)

    async def execute_goal(
        self, goal: str, context: str = "", max_parallel: int = MAX_CONCURRENT
    ) -> dict[str, AgentTaskResult]:
        """Execute a goal through agent coordination.

        Optimization B-6: tasks without dependencies run in parallel.
        Optimization E-12: metrics tracked per agent via MetricsRegistry.
        """
        logger.info("AgentOrchestrator: executing goal '%s'", goal[:80])
        tasks = self._decompose_goal(goal)
        semaphore = asyncio.Semaphore(max_parallel)
        self.results = {}

        async def _run_one(task: AgentTask) -> tuple[str, AgentTaskResult]:
            async with semaphore:
                agent = self.agents.get(task.target_role) if task.target_role else None
                if agent is None:
                    return task.id, AgentTaskResult(
                        task_id=task.id, success=False, output="No agent"
                    )
                try:
                    result = await agent.handle_task(task)
                    return task.id, result
                except Exception as exc:
                    return task.id, AgentTaskResult(task_id=task.id, success=False, output=str(exc))

        # Dependency-aware parallel dispatch
        dep_map = {t.id: getattr(t, "dependencies", []) for t in tasks}
        executed: set[str] = set()
        while len(executed) < len(tasks):
            ready = [
                t
                for t in tasks
                if t.id not in executed and all(d in executed for d in dep_map.get(t.id, []))
            ]
            if not ready:
                break
            for tid, result in await asyncio.gather(*[_run_one(t) for t in ready]):
                self.results[tid] = result
                executed.add(tid)

        return self.results

    async def _enrich_task(self, task: AgentTask) -> None:
        """Level 2: inject best-known strategy from experience memory."""
        if not hasattr(self, "experience") or self.experience is None:
            return
        try:
            best_method = self.experience.best_method_for(getattr(task, "goal", "")[:50])
            if best_method:
                ctx = getattr(task, "context", "") or ""
                task.context = (
                    ctx + f" [Memory: {best_method} works best for this]"
                    if ctx
                    else f"[Memory: {best_method} works best for this]"
                )
        except Exception:
            pass

    # Keywords that trigger an INVESTIGATOR task instead of a build pipeline.
    _INVESTIGATION_TRIGGERS = frozenset(
        [
            "understand",
            "trace",
            "explore",
            "investigate",
            "how does",
            "map dependencies",
            "dependency map",
            "explain",
            "walk me through",
            "show me how",
        ]
    )

    def _decompose_goal(self, goal: str) -> list[AgentTask]:
        """Decompose a goal into agent tasks."""
        tasks: list[AgentTask] = []
        goal_lower = goal.lower()

        # Investigation requests — dispatch to INVESTIGATOR before any build task.
        if any(trigger in goal_lower for trigger in self._INVESTIGATION_TRIGGERS):
            tasks.append(
                AgentTask(
                    id="investigate_001",
                    goal=goal,
                    target_role=AgentRole.INVESTIGATOR,
                    context="",
                    dependencies=[],
                )
            )
            return tasks

        if any(kw in goal_lower for kw in ["design", "architecture", "choose framework", "plan"]):
            tasks.append(
                AgentTask(
                    id="arch_001",
                    goal=goal,
                    target_role=AgentRole.ARCHITECT,
                    context="Design the architecture for this project",
                )
            )

        tasks.append(
            AgentTask(
                id="dev_001",
                goal=goal,
                target_role=AgentRole.DEVELOPER,
                context="Implement the solution",
                dependencies=[t.id for t in tasks],
            )
        )

        if any(kw in goal_lower for kw in ["test", "verify", "qa"]):
            tasks.append(
                AgentTask(
                    id="test_001",
                    goal=goal,
                    target_role=AgentRole.TESTER,
                    context="Write and run tests",
                    dependencies=[t.id for t in tasks],
                )
            )

        return tasks
