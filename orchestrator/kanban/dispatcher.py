"""
Kanban Dispatcher — Background Loop for Worker Assignment
=============================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Long-lived background loop that monitors the KanbanBoard and spawns
workers to claim + execute available tasks.

Designed to run alongside the gateway (as a daemon task) or as a
standalone process via `orchestrator kanban start`.

Integration: KanbanDispatcher.start() runs an infinite loop that polls
the board at a configurable interval, claims tasks, and spawns workers.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .board import KanbanBoard

logger = logging.getLogger("orchestrator.kanban.dispatcher")


# ─────────────────────────────────────────────────────────────────────────────
# KanbanDispatcher
# ─────────────────────────────────────────────────────────────────────────────


class KanbanDispatcher:
    """Background loop that monitors the board and spawns workers.

    Usage:
        dispatcher = KanbanDispatcher(board)
        await dispatcher.start()  # runs forever
        # ... or ...
        await dispatcher.run_once()  # single poll cycle
    """

    def __init__(
        self,
        board: "KanbanBoard",
        poll_interval: float = 30.0,
        max_workers: int = 2,
        worker_name: str = "orch-worker",
    ) -> None:
        """Initialize dispatcher.

        Args:
            board: KanbanBoard instance.
            poll_interval: Seconds between board polls (default 30).
            max_workers: Maximum concurrent workers (default 2).
            worker_name: Assignee name for claimed tasks.
        """
        self._board = board
        self._poll_interval = poll_interval
        self._max_workers = max_workers
        self._worker_name = worker_name
        self._running = False

    # ── Public API ──────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the infinite dispatch loop.

        Runs until ``shutdown()`` is called. Polls the board every
        ``poll_interval`` seconds and spawns workers for available tasks.
        """
        self._running = True
        logger.info(
            "KanbanDispatcher: started (poll=%ds, workers=%d)",
            self._poll_interval,
            self._max_workers,
        )

        while self._running:
            try:
                await self.run_once()
            except Exception as exc:
                logger.error("KanbanDispatcher: poll cycle error: %s", exc)

            # Sleep between polls (check for shutdown every second)
            for _ in range(int(self._poll_interval)):
                if not self._running:
                    break
                await asyncio.sleep(1)

        logger.info("KanbanDispatcher: stopped")

    async def shutdown(self) -> None:
        """Stop the dispatch loop."""
        self._running = False
        logger.info("KanbanDispatcher: shut down")

    async def run_once(self) -> int:
        """Execute a single poll cycle.

        Claims up to ``max_workers`` tasks and spawns workers.

        Returns:
            Number of tasks dispatched in this cycle.
        """
        dispatched = 0
        for _ in range(self._max_workers):
            task = await self._board.claim_next(self._worker_name)
            if task is None:
                break  # no more tasks

            # Fire-and-forget: spawn worker without blocking
            asyncio.create_task(self._execute_task(task.task_id, task.project_spec))
            dispatched += 1

        if dispatched:
            logger.info("KanbanDispatcher: dispatched %d task(s)", dispatched)

        return dispatched

    # ── Internal ────────────────────────────────────────────────────────────

    async def _execute_task(self, task_id: str, project_spec: str) -> None:
        """Execute a single kanban task.

        Runs the orchestrator on the project spec and marks the task
        as completed or failed.

        Args:
            task_id: Kanban task ID.
            project_spec: JSON string with project description/criteria.
        """
        from ..engine import Orchestrator
        from ..budget import Budget

        import json

        try:
            spec = json.loads(project_spec) if isinstance(project_spec, str) else {}
        except json.JSONDecodeError:
            spec = {}

        description = spec.get("description", project_spec[:100])
        budget = Budget(max_usd=3.0, max_time_seconds=900)
        orch = Orchestrator(budget=budget)

        try:
            async with orch:
                state = await orch.run_project(
                    project_description=description,
                    success_criteria=spec.get("criteria", ""),
                )

            await self._board.complete(
                task_id,
                {
                    "status": state.status.value,
                    "tasks_passed": sum(
                        1 for r in state.results.values() if r.status.value == "completed"
                    ),
                    "total_cost": state.budget.spent_usd,
                },
            )

            logger.info("Kanban: completed %s", task_id)

        except Exception as exc:
            logger.error("Kanban: task %s failed: %s", task_id, exc)
            await self._board.fail(task_id, str(exc))

    @property
    def is_running(self) -> bool:
        return self._running
