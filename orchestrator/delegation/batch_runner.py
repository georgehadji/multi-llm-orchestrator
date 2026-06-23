"""
BatchRunner — Parallel Task Execution Within Dependency Levels
================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Spawns SubAgents for independent tasks within a dependency level,
executing them concurrently. Each subagent gets a proportional budget
slice.

Key design decisions:
- Default max_concurrent_children = 3 (matching Hermes Agent)
- Default max_spawn_depth = 2 (orchestrator → leaf, one level deep)
- Budget is split evenly among concurrent tasks (80% of parent budget)
- Failed tasks are returned with FAILED status — does NOT abort the
  entire batch (other tasks continue)

Integration: Called from engine.py.run_project() when the dependency
resolver identifies multiple ready tasks at the same topological level.
Opt-in via ORCH_BATCH_PARALLELISM=true env var.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from .subagent import SubAgent, SubAgentConfig

if TYPE_CHECKING:
    from ..budget import Budget
    from ..models import Task, TaskResult

logger = logging.getLogger("orchestrator.delegation.batch")


# ─────────────────────────────────────────────────────────────────────────────
# BatchRunner
# ─────────────────────────────────────────────────────────────────────────────


class BatchRunner:
    """Spawns multiple SubAgents for independent tasks, gathers results.

    Usage:
        runner = BatchRunner(max_concurrent=3, max_depth=2)
        results = await runner.run_batch(
            tasks=[task1, task2, task3],
            dependency_contexts={"t1": "...", "t2": "..."},
            parent_budget=budget,
            execute_fn=orchestrator._execute_task,
        )
        for task_id, result in results.items():
            print(f"{task_id}: {result.status.value}")
    """

    def __init__(
        self,
        max_concurrent: int = 3,
        max_depth: int = 2,
    ) -> None:
        """Initialize batch runner.

        Args:
            max_concurrent: Maximum number of tasks to run simultaneously.
                Default 3. Must be ≥ 1.
            max_depth: Maximum spawn depth. 1 = only leaf tasks;
                2 = orchestrator can spawn one level of children.
                Default 2.
        """
        self.max_concurrent = max(1, max_concurrent)
        self.max_depth = max(1, max_depth)
        self._current_depth: int = 0

    # ── Public API ──────────────────────────────────────────────────────────

    async def run_batch(
        self,
        tasks: list["Task"],
        dependency_contexts: dict[str, str] | None = None,
        parent_budget: "Budget | None" = None,
        execute_fn: Any = None,
    ) -> dict[str, "TaskResult"]:
        """Execute multiple independent tasks concurrently.

        Each task gets its own SubAgent with a proportional budget slice.
        Tasks are limited by ``max_concurrent`` — additional tasks wait
        until a slot opens.

        Args:
            tasks: List of Task objects to execute. Must have unique IDs.
            dependency_contexts: Optional dict of ``{task_id: context_str}``
                for dependency injection.
            parent_budget: Optional parent Budget. When provided, each
                task gets an equal slice (80% of max_usd / len(tasks)).
                When None, tasks use no budget limit.
            execute_fn: Async callable ``(Task) -> TaskResult``. Defers to
                SubAgent when None (which raises if no execute_fn set).

        Returns:
            Dict of ``{task_id: TaskResult}`` for all tasks. Failed tasks
            have ``status == FAILED``.
        """
        if not tasks:
            return {}

        # Guard: enforce depth limit
        if self._current_depth >= self.max_depth:
            logger.warning(
                "BatchRunner depth limit reached (%d >= %d). " "Tasks will run sequentially.",
                self._current_depth,
                self.max_depth,
            )
            return await self._run_sequential(tasks, dependency_contexts, execute_fn)

        # Calculate budget slice per task
        budget_per_task = None
        if parent_budget is not None and len(tasks) > 0:
            slice_amount = (parent_budget.max_usd * 0.8) / len(tasks)
            budget_per_task = parent_budget.__class__(max_usd=slice_amount)

        # Create subagents
        config = SubAgentConfig(
            role="leaf",
            max_iterations=90,
        )
        agents = [
            SubAgent(
                config=config,
                budget_slice=budget_per_task,
                execute_fn=execute_fn,
            )
            for _ in tasks
        ]

        # Build context map
        contexts = dependency_contexts or {}

        # Execute with concurrency limit
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def _run_one(agent: SubAgent, task: "Task") -> tuple[str, "TaskResult"]:
            async with semaphore:
                ctx = contexts.get(task.id, "")
                result = await agent.execute(task, dependency_context=ctx)
                return task.id, result

        logger.info(
            "BatchRunner: executing %d tasks (concurrent=%d, depth=%d)",
            len(tasks),
            self.max_concurrent,
            self._current_depth,
        )

        # Gather all results
        coros = [_run_one(agent, task) for agent, task in zip(agents, tasks)]
        outcomes = await asyncio.gather(*coros, return_exceptions=True)

        # Build result dict
        results: dict[str, "TaskResult"] = {}
        for outcome in outcomes:
            if isinstance(outcome, Exception):
                logger.warning("BatchRunner task failed with exception: %s", outcome)
                continue
            if isinstance(outcome, tuple):
                task_id, result = outcome
                results[task_id] = result

        return results

    # ── Internal ────────────────────────────────────────────────────────────

    async def _run_sequential(
        self,
        tasks: list["Task"],
        dependency_contexts: dict[str, str] | None,
        execute_fn: Any,
    ) -> dict[str, "TaskResult"]:
        """Fallback: run tasks one at a time when depth limit is reached."""
        results: dict[str, "TaskResult"] = {}
        agent = SubAgent(
            config=SubAgentConfig(role="leaf"),
            budget_slice=None,
            execute_fn=execute_fn,
        )
        contexts = dependency_contexts or {}
        for task in tasks:
            ctx = contexts.get(task.id, "")
            result = await agent.execute(task, dependency_context=ctx)
            results[task.id] = result
        return results
