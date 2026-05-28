"""
StateCoordinator — Project state lifecycle management
=======================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Handles ProjectState creation, final status determination, and summary logging.
Extracted from engine.py to dismantle the God Object.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ..models import ProjectState, ProjectStatus, TaskStatus

if TYPE_CHECKING:
    from ..models import Task, TaskResult, Model
    from ..budget import Budget

logger = logging.getLogger("orchestrator.engine_core.state_coordinator")


class StateCoordinator:
    """Manages the creation and evaluation of ProjectState objects."""

    def determine_final_status(self, state: ProjectState) -> ProjectStatus:
        """Determine terminal project status based on results and budget."""
        budget_exhausted = state.budget.remaining_usd <= 0
        time_ok = state.budget.time_remaining()

        if budget_exhausted:
            return ProjectStatus.BUDGET_EXHAUSTED
        if not time_ok:
            return ProjectStatus.TIMEOUT

        if not state.results:
            return ProjectStatus.SYSTEM_FAILURE

        # COMPLETED or DEGRADED both count as "passed" for final status
        all_passed = all(
            r.status in (TaskStatus.COMPLETED, TaskStatus.DEGRADED) 
            for r in state.results.values()
        )

        degraded_heavy = any(
            r.degraded_fallback_count > r.iterations * 0.5
            for r in state.results.values()
            if r.iterations > 0
        )

        det_ok = all(r.deterministic_check_passed for r in state.results.values())
        all_tasks_executed = len(state.results) == len(state.tasks)

        if all_tasks_executed and all_passed and det_ok and not degraded_heavy:
            return ProjectStatus.SUCCESS
        elif all_tasks_executed and all_passed and not det_ok:
            return ProjectStatus.COMPLETED_DEGRADED
        else:
            return ProjectStatus.PARTIAL_SUCCESS

    def make_state(
        self,
        project_desc: str,
        criteria: str,
        budget: Budget,
        tasks: dict[str, Task],
        results: dict[str, TaskResult],
        api_health: dict[Model, bool],
        status: ProjectStatus = ProjectStatus.PARTIAL_SUCCESS,
        execution_order: list[str] | None = None,
    ) -> ProjectState:
        """Factory for ProjectState DTO."""
        return ProjectState(
            project_description=project_desc,
            success_criteria=criteria,
            budget=budget,
            tasks=tasks,
            results=dict(results),
            api_health=dict(api_health),
            status=status,
            execution_order=execution_order if execution_order is not None else list(tasks.keys()),
        )

    def log_summary(self, state: ProjectState) -> None:
        """Log a comprehensive summary of the project outcome."""
        logger.info("=" * 60)
        logger.info(f"PROJECT STATUS: {state.status.value}")
        logger.info(f"Budget: ${state.budget.spent_usd:.4f} / ${state.budget.max_usd}")
        logger.info(f"Time: {state.budget.elapsed_seconds:.1f}s / {state.budget.max_time_seconds}s")
        for tid, result in state.results.items():
            logger.info(
                f"  {tid}: score={result.score:.3f} status={result.status.value} "
                f"model={result.model_used.value} iters={result.iterations} "
                f"cost=${result.cost_usd:.4f}"
            )
        logger.info("=" * 60)
