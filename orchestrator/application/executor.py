"""
ExecutorService — stable task execution interface.
===================================================
Provides a clean, injectable boundary around the generate->critique->revise->evaluate
loop. The actual loop implementation lives in engine.py._execute_task() and is
injected via execute_fn.

Part of Application Layer (Phase 4) — Canonical location.
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
from ..telemetry import TelemetryCollector
from ..tracing import Tracer
from ..resilience import ResiliencePolicy as _ResiliencePolicy

logger = logging.getLogger("orchestrator.services.executor")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class ExecutorResult:
    """Wrapper around TaskResult that adds execution-level metadata."""

    task_result: TaskResult
    wall_time_ms: float
    executor_retries: int = 0
    error: Exception | None = None

    @property
    def succeeded(self) -> bool:
        return self.task_result.status in (TaskStatus.COMPLETED, TaskStatus.DEGRADED)


@dataclass
class ExecutorMetrics:
    """Monotonic counters — never decremented."""

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
        avg_ms = self.cumulative_wall_ms / self.total_submitted if self.total_submitted else 0.0
        return {
            "total_submitted": self.total_submitted,
            "total_completed": self.total_completed,
            "total_failed": self.total_failed,
            "total_degraded": self.total_degraded,
            "total_executor_retries": self.total_executor_retries,
            "avg_wall_ms": round(avg_ms, 1),
        }


ExecuteFn = Callable[[Task], Awaitable[TaskResult]]


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class ExecutorService:
    """
    Application-layer service for single-task execution.

    Usage:
        self._executor = ExecutorService(execute_fn=self._execute_task)
        result = await self._executor.execute(task)
    """

    def __init__(
        self,
        execute_fn: ExecuteFn,
        task_timeout: float | None = None,
        guard: TaskConcurrencyGuard | None = None,
        tracer: Tracer | None = None,
        telemetry: TelemetryCollector | None = None,
    ) -> None:
        self._execute_fn = execute_fn
        self._task_timeout = task_timeout
        self._guard = guard
        self._tracer = tracer
        self._telemetry = telemetry
        self.metrics = ExecutorMetrics()
        self._lock = asyncio.Lock()

    async def execute(self, task: Task, policy: _ResiliencePolicy | None = None) -> ExecutorResult:
        """Execute task and return structured ExecutorResult."""
        t0 = time.monotonic()

        if self._tracer is not None:
            with self._tracer.trace(
                "executor.task",
                {"task_id": task.id, "task_type": task.type.value},
            ) as span:
                task_result, error = await self._run_with_guard(task, policy)
                if error:
                    span.set_status("ERROR")
                    span.add_event("exception", {"exception.message": str(error)})
        else:
            task_result, error = await self._run_with_guard(task, policy)

        wall_ms = (time.monotonic() - t0) * 1000

        result = ExecutorResult(
            task_result=task_result,
            wall_time_ms=wall_ms,
            error=error,
        )

        async with self._lock:
            self.metrics.record(result)

        if self._telemetry is not None and result.task_result.model_used is not None:
            try:
                from ..models import Model

                model = (
                    result.task_result.model_used
                    if isinstance(result.task_result.model_used, Model)
                    else Model(result.task_result.model_used)
                )
                self._telemetry.record_call(
                    model=model,
                    latency_ms=wall_ms,
                    cost_usd=result.task_result.cost_usd or 0.0,
                    success=result.succeeded,
                    quality_score=result.task_result.score if result.succeeded else None,
                )
            except Exception:
                logger.debug("Telemetry recording failed for task %s", task.id, exc_info=True)

        if error:
            logger.warning("task=%s FAILED in %.0fms: %s", task.id, wall_ms, error)
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
        return self.metrics.to_dict()

    async def _run_with_guard(
        self, task: Task, policy: _ResiliencePolicy | None = None
    ) -> tuple[TaskResult, Exception | None]:
        try:
            if self._guard is not None:
                async with self._guard:
                    if self._task_timeout is not None:
                        raw = await asyncio.wait_for(
                            self._execute_fn(task, policy=policy),  # type: ignore[call-arg]
                            timeout=self._task_timeout,
                        )
                    else:
                        raw = await self._execute_fn(task, policy=policy)  # type: ignore[call-arg]
            elif self._task_timeout is not None:
                raw = await asyncio.wait_for(
                    self._execute_fn(task, policy=policy),  # type: ignore[call-arg]
                    timeout=self._task_timeout,
                )
            else:
                raw = await self._execute_fn(task, policy=policy)  # type: ignore[call-arg]
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
