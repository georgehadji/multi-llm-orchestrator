"""
PipelineRunner — Multi-task execution orchestrator
===================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Handles level-based parallel execution of tasks, progress reporting,
and dashboard notifications. Orchestrates the TaskPipeline for each task.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from ..models import Task, ProjectState
    from .pipeline import TaskPipeline
    from .project_planner import ProjectPlanner

logger = logging.getLogger("orchestrator.engine_core.pipeline_runner")


def _build_failure_result(task: Any, reason: str) -> Any:
    """Build a FAILED TaskResult for a task whose execute_task_fn raised.

    Mirrors application/task_executor.py::_build_failure_result — same
    placeholder-model convention (the task never got far enough to select a
    real one).
    """
    from ..models import Model, TaskResult, TaskStatus

    return TaskResult(
        task_id=task.id,
        output="",
        score=0.0,
        model_used=Model.GPT_4O_MINI,
        reviewer_model=None,
        tokens_used={"input": 0, "output": 0},
        iterations=0,
        cost_usd=0.0,
        status=TaskStatus.FAILED,
        critique=reason,
        deterministic_check_passed=False,
        degraded_fallback_count=0,
        attempt_history=[],
        task_type=task.type.value,
    )


class PipelineRunner:
    """Orchestrates parallel execution of tasks across project levels."""

    def __init__(
        self,
        pipeline: TaskPipeline,
        planner: ProjectPlanner,
        max_parallel_tasks: int = 3,
        event_bus: Any = None,
        dashboard: Any = None,
    ):
        self._pipeline = pipeline
        self._planner = planner
        self._max_parallel_tasks = max_parallel_tasks
        self._event_bus = event_bus
        self._dashboard = dashboard

    async def execute_all(
        self,
        tasks: dict[str, Task],
        execution_order: list[str],
        project_desc: str,
        success_criteria: str,
        output_dir: Path | None = None,
        execute_task_fn: Callable[[Task], Any] | None = None,
        make_state_fn: Callable[..., ProjectState] | None = None,
    ) -> ProjectState:
        """Execute all tasks respecting dependencies, with level-based parallelism."""
        levels = self._planner.get_execution_levels(tasks)
        semaphore = asyncio.Semaphore(self._max_parallel_tasks)
        results: dict[str, Any] = {}
        results_lock = asyncio.Lock()

        # Progress tracking setup
        progress_writer = None
        if output_dir and make_state_fn:
            try:
                from ..progress_writer import ProgressWriter

                partial_state = make_state_fn(
                    project_desc, success_criteria, tasks, execution_order
                )
                partial_state.results = results
                progress_writer = ProgressWriter(output_dir, partial_state)
            except ImportError:
                pass

        for level_idx, level in enumerate(levels):
            logger.info("Executing level %d (%d tasks)", level_idx, len(level))

            async def run_one(tid: str) -> Any:
                task = tasks[tid]
                async with semaphore:
                    if execute_task_fn:
                        try:
                            res = await execute_task_fn(task)
                        except Exception as exc:
                            # T4-C1: previously left `results[tid]` unset on
                            # failure (asyncio.gather's return_exceptions=True
                            # only logged it below) — the failure was invisible
                            # to anything reading ProjectState.results later.
                            logger.error(
                                "Task %s failed in level %d: %s",
                                tid,
                                level_idx,
                                exc,
                                exc_info=exc,
                            )
                            res = _build_failure_result(task, str(exc))
                        async with results_lock:
                            results[tid] = res
                            if progress_writer:
                                await progress_writer.write_update()
                        return res
                return None

            level_results = await asyncio.gather(
                *(run_one(tid) for tid in level),
                return_exceptions=True,
            )

            # Surface task failures immediately — don't silently swallow them
            for tid, outcome in zip(level, level_results):
                if isinstance(outcome, BaseException):
                    logger.error(
                        "Task %s failed in level %d: %s",
                        tid,
                        level_idx,
                        outcome,
                        exc_info=outcome,
                    )

        if make_state_fn:
            return make_state_fn(
                project_desc, success_criteria, tasks, execution_order, results=results
            )
        return results  # type: ignore
