"""
Resume from checkpoint — restores budget and executes remaining tasks.

P3-3 of REFACTORING_PLAN_V7.md — extracted from engine._resume_project.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

from ..models import ProjectState, TaskStatus

logger = logging.getLogger(__name__)


class ResumptionService:
    """Resumes a partially-completed project from a saved :class:`ProjectState`.

    Dependencies are injected so the service remains independently testable:

    - ``budget``:  the live :class:`~orchestrator.budget.Budget` instance;
      mutated in-place to restore persisted spend.
    - ``results``:  the live results dict; populated from the checkpoint
      and updated as remaining tasks complete.
    - ``execute_task_fn``:  ``async (task, policy) -> TaskResult`` — usually
      ``Orchestrator._execute_task``.
    - ``determine_final_status_fn``:  ``(state) -> ProjectStatus`` — usually
      ``Orchestrator._determine_final_status``.
    """

    def __init__(
        self,
        budget: Any,
        results: dict,  # type: ignore[type-arg]
        execute_task_fn: Callable,  # type: ignore[type-arg]
        determine_final_status_fn: Callable,  # type: ignore[type-arg]
    ) -> None:
        self._budget = budget
        self._results = results
        self._execute_task = execute_task_fn
        self._determine_final_status = determine_final_status_fn

    async def resume(self, state: ProjectState) -> ProjectState:
        """Resume execution from *state*, return the updated state."""
        self._restore_budget(state)
        self._results.update(state.results)

        remaining = [
            tid
            for tid in state.execution_order
            if tid not in self._results
            or self._results[tid].status in (TaskStatus.PENDING, TaskStatus.FAILED)
        ]

        if remaining:
            logger.info("Resuming: %d tasks remaining", len(remaining))
            for task_id in remaining:
                if task_id in state.tasks:
                    task = state.tasks[task_id]
                    # Import directly from the module to bypass the operations
                    # package __init__.py (which has fragile wildcard imports).
                    try:
                        from ..operations.resilience import RetryTemplate

                        policy = RetryTemplate.for_task_type(task.type)
                    except Exception:
                        policy = None
                    result = await self._execute_task(task, policy=policy)
                    self._results[task_id] = result
                    # M7: build a new state instead of mutating in place
                    import dataclasses as _dc

                    state = _dc.replace(state, results={**state.results, task_id: result})

        # M7: return a new state with updated status
        import dataclasses as _dc

        final_status = self._determine_final_status(state)
        state = _dc.replace(state, status=final_status)
        return state

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _restore_budget(self, state: ProjectState) -> None:
        """Restore persisted budget fields; reset start_time for new session."""
        if state.budget is None:
            return
        self._budget.spent_usd = state.budget.spent_usd
        self._budget.phase_spent = dict(state.budget.phase_spent)
        # Preserve original_start_time for elapsed-time calculation when resuming
        self._budget.original_start_time = state.budget.original_start_time
        # Reset start_time so the new session gets fresh wall-clock tracking
        self._budget.start_time = time.time()
        logger.info(
            "Restored budget: $%.4f already spent, $%.4f remaining",
            self._budget.spent_usd,
            self._budget.max_usd - self._budget.spent_usd,
        )
