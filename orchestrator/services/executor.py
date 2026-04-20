"""
ExecutorService — stable task execution interface.
===================================================
Provides a clean, injectable boundary around the generate→critique→revise→evaluate
loop. The actual loop implementation lives in engine.py._execute_task() and is
injected via ``execute_fn``; this service adds:

  - Execution timing (wall-clock ms per task)
  - Structured error normalization (bare exceptions → TaskError)
  - Task-level metrics accumulation (total tasks, failures, retries)
  - A single injection point for future cross-cutting concerns
    (rate shaping, concurrency budgets, HITL checkpoints)

Phase 1 (current):  Interface established; implementation delegated via callback.
Phase 2 (planned):  Migrate engine._execute_task() body here in chunks.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from ..concurrency_controller import TaskConcurrencyGuard
from ..exceptions import TaskError, TaskTimeoutError
from ..models import Task, TaskResult, TaskStatus

logger = logging.getLogger("orchestrator.services.executor")


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ExecutorResult:
    """
    Wrapper around TaskResult that adds execution-level metadata.

    Attributes:
        task_result:        The inner TaskResult from the generation loop.
        wall_time_ms:       Wall-clock execution time in milliseconds.
        executor_retries:   How many times ExecutorService itself retried (≥0).
        error:              Set when execution raised and could not recover.
    """

    task_result: TaskResult
    wall_time_ms: float
    executor_retries: int = 0
    error: Exception | None = None

    @property
    def succeeded(self) -> bool:
        return self.task_result.status in (TaskStatus.COMPLETED, TaskStatus.DEGRADED)


@dataclass
class ExecutorMetrics:
    """Monotonic counters — never decremented. Thread/coroutine-safe reads."""

    total_submitted: int = 0
    total_completed: int = 0
    total_failed: int = 0
    total_degraded: int = 0
    total_executor_retries: int = 0
    cumulative_wall_ms: float = 0.0

    def record(self, result: ExecutorResult) -> None:
        self.total_submitted += 1
        self.cumulative_wall_ms += result.wall_time_ms
        self.total_executor_retries += result.executor_retries
        if result.error is not None:
            self.total_failed += 1
        elif result.task_result.status == TaskStatus.COMPLETED:
            self.total_completed += 1
        elif result.task_result.status == TaskStatus.DEGRADED:
            self.total_degraded += 1
        else:
            self.total_failed += 1

    def to_dict(self) -> dict[str, Any]:
        avg_ms = (
            self.cumulative_wall_ms / self.total_submitted
            if self.total_submitted
            else 0.0
        )
        return {
            "total_submitted": self.total_submitted,
            "total_completed": self.total_completed,
            "total_failed": self.total_failed,
            "total_degraded": self.total_degraded,
            "total_executor_retries": self.total_executor_retries,
            "avg_wall_ms": round(avg_ms, 1),
        }


# Type alias for the injected implementation callback.
ExecuteFn = Callable[[Task], Awaitable[TaskResult]]


# ─────────────────────────────────────────────────────────────────────────────
# Service
# ─────────────────────────────────────────────────────────────────────────────


class ExecutorService:
    """
    Application-layer service for single-task execution.

    Usage (in engine.py):

        self._executor = ExecutorService(execute_fn=self._execute_task)
        ...
        result = await self._executor.execute(task)

    The service is intentionally stateless regarding task content; it only tracks
    aggregate metrics and enforces the execution contract.

    Args:
        execute_fn:     Async callable ``(Task) -> TaskResult``.  Injected from
                        engine.py; will migrate into this class in Phase 2.
        task_timeout:   Hard wall-clock timeout (seconds) for a single task,
                        applied on top of whatever the inner ``execute_fn`` does.
                        None = no extra timeout (inner fn manages its own).
    """

    def __init__(
        self,
        execute_fn: ExecuteFn,
        task_timeout: float | None = None,
        guard: TaskConcurrencyGuard | None = None,
    ) -> None:
        self._execute_fn = execute_fn
        self._task_timeout = task_timeout
        self._guard = guard
        self.metrics = ExecutorMetrics()
        self._lock = asyncio.Lock()  # guards metrics update

    # ── Public interface ──────────────────────────────────────────────────────

    async def execute(self, task: Task) -> ExecutorResult:
        """
        Execute ``task`` and return a structured ``ExecutorResult``.

        Guarantees:
          - Always returns an ``ExecutorResult`` (never raises).
          - On unrecoverable error, ``ExecutorResult.error`` is set and
            ``task_result.status == TaskStatus.FAILED``.
          - Wall time is always measured and included.

        Note: This method does NOT raise. Callers that need exception semantics
        should check ``result.error``.
        """
        t0 = time.monotonic()
        task_result, error = await self._run_with_guard(task)
        wall_ms = (time.monotonic() - t0) * 1000

        result = ExecutorResult(
            task_result=task_result,
            wall_time_ms=wall_ms,
            error=error,
        )

        async with self._lock:
            self.metrics.record(result)

        if error:
            logger.warning(
                "task=%s FAILED in %.0fms: %s",
                task.id,
                wall_ms,
                error,
            )
        else:
            logger.debug(
                "task=%s status=%s in %.0fms score=%.3f",
                task.id,
                task_result.status.value,
                wall_ms,
                task_result.score,
            )

        return result

    def metrics_snapshot(self) -> dict[str, Any]:
        """Return a copy of current aggregate metrics."""
        return self.metrics.to_dict()

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _run_with_guard(
        self, task: Task
    ) -> tuple[TaskResult, Exception | None]:
        """
        Call ``_execute_fn`` with optional hard timeout.
        Normalises all exceptions into ``(failed_TaskResult, exc)`` pairs so
        ``execute()`` always has a TaskResult to work with.
        """
        try:
            if self._guard is not None:
                async with self._guard:
                    if self._task_timeout is not None:
                        raw = await asyncio.wait_for(
                            self._execute_fn(task),
                            timeout=self._task_timeout,
                        )
                    else:
                        raw = await self._execute_fn(task)
            elif self._task_timeout is not None:
                raw = await asyncio.wait_for(
                    self._execute_fn(task),
                    timeout=self._task_timeout,
                )
            else:
                raw = await self._execute_fn(task)
            return raw, None

        except asyncio.TimeoutError as exc:
            te = TaskTimeoutError(
                task_id=task.id,
                timeout_seconds=self._task_timeout or 0,
                cause=exc,
            )
            return self._failed_result(task, str(te)), te

        except TaskError as exc:
            return self._failed_result(task, str(exc)), exc

        except Exception as exc:
            wrapped = TaskError(
                f"Unexpected error executing task '{task.id}': {exc}",
                cause=exc,
            )
            return self._failed_result(task, str(wrapped)), wrapped

    @staticmethod
    def _failed_result(task: Task, message: str) -> TaskResult:
        """Minimal failed TaskResult for error paths."""
        from ..models import Model

        return TaskResult(
            task_id=task.id,
            output="",
            score=0.0,
            model_used=Model.GPT_4O_MINI,
            status=TaskStatus.FAILED,
            task_type=task.type.value,
            critique=message,
            iterations=0,
            cost_usd=0.0,
            tokens_used={"input": 0, "output": 0},
        )
