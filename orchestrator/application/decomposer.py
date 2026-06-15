"""
DecomposerService — stable project decomposition interface.
===========================================================
Wraps engine.py._decompose() via callback injection. Establishes the
decompose(project, criteria) -> GeneratorResult boundary.

Part of Application Layer (Phase 4) — Canonical location.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from ..exceptions import OrchestratorError, TaskError
from ..models import Task
from ..tracing import Tracer
from ..resilience import ResiliencePolicy as _ResiliencePolicy
from ..project_context import ProjectContext as _ProjectContext

logger = logging.getLogger("orchestrator.services.generator")

DecomposeFn = Callable[..., Awaitable[dict[str, Task]]]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class DecomposerResult:
    """Outcome of a decomposition call."""

    tasks: dict[str, Task]
    wall_time_ms: float
    error: Exception | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None and bool(self.tasks)

    @property
    def task_count(self) -> int:
        return len(self.tasks)


@dataclass
class DecomposerMetrics:
    """Monotonic counters for decomposition calls."""

    total_calls: int = 0
    total_succeeded: int = 0
    total_failed: int = 0
    total_tasks_generated: int = 0
    cumulative_wall_ms: float = 0.0

    def record(self, result: DecomposerResult) -> None:
        self.total_calls += 1
        self.cumulative_wall_ms += result.wall_time_ms
        if result.succeeded:
            self.total_succeeded += 1
            self.total_tasks_generated += result.task_count
        else:
            self.total_failed += 1

    def to_dict(self) -> dict[str, Any]:
        avg_ms = self.cumulative_wall_ms / self.total_calls if self.total_calls else 0.0
        return {
            "total_calls": self.total_calls,
            "total_succeeded": self.total_succeeded,
            "total_failed": self.total_failed,
            "total_tasks_generated": self.total_tasks_generated,
            "avg_wall_ms": round(avg_ms, 1),
        }


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class DecomposerService:
    """
    Application-layer service for project decomposition.

    Usage:
        self._decomposer = DecomposerService(decompose_fn=self._decompose)
        result = await self._decomposer.decompose(project, criteria)
        if not result.succeeded:
            raise OrchestratorError(f"Decomposition failed: {result.error}")
        tasks = result.tasks
    """

    def __init__(
        self,
        decompose_fn: DecomposeFn,
        decompose_timeout: float | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        self._decompose_fn = decompose_fn
        self._decompose_timeout = decompose_timeout
        self._tracer = tracer
        self.metrics = DecomposerMetrics()
        self._lock = asyncio.Lock()

    @property
    def decompose_fn(self):
        """Late-bound decompose_fn — allows container.wire_executor() to set."""
        return self._decompose_fn

    @decompose_fn.setter
    def decompose_fn(self, fn):
        self._decompose_fn = fn

    async def decompose(
        self,
        project: str,
        criteria: str,
        policy: _ResiliencePolicy | None = None,
        project_context: "ProjectContext | None" = None,  # type: ignore[name-defined]
        **kwargs: Any,
    ) -> DecomposerResult:
        """Decompose project into an ordered task dict. Never raises."""
        t0 = time.monotonic()

        if self._tracer is not None:
            with self._tracer.trace(
                "generator.decompose",
                {"project": project[:50], "criteria": criteria[:50]},
            ) as span:
                tasks, error = await self._run_with_guard(
                    project, criteria, policy, project_context=project_context, **kwargs
                )
                if error:
                    span.set_status("ERROR")
                    span.add_event("exception", {"exception.message": str(error)})
        else:
            tasks, error = await self._run_with_guard(
                project, criteria, policy, project_context=project_context, **kwargs
            )

        wall_ms = (time.monotonic() - t0) * 1000

        result = DecomposerResult(tasks=tasks or {}, wall_time_ms=wall_ms, error=error)

        async with self._lock:
            self.metrics.record(result)

        if error:
            logger.warning("decompose FAILED in %.0fms: %s", wall_ms, error)
        else:
            logger.debug("decompose succeeded in %.0fms — %d tasks", wall_ms, result.task_count)

        return result

    def metrics_snapshot(self) -> dict[str, Any]:
        return self.metrics.to_dict()

    async def _run_with_guard(
        self,
        project: str,
        criteria: str,
        policy: _ResiliencePolicy | None = None,
        project_context: _ProjectContext | None = None,
        **kwargs: Any,
    ) -> tuple[dict[str, Task] | None, Exception | None]:
        try:
            decompose_kwargs = dict(kwargs)
            decompose_kwargs["policy"] = policy
            if project_context is not None:
                decompose_kwargs["project_context"] = project_context

            if self._decompose_timeout is not None:
                raw = await asyncio.wait_for(
                    self._decompose_fn(project, criteria, **decompose_kwargs),
                    timeout=self._decompose_timeout,
                )
            else:
                raw = await self._decompose_fn(project, criteria, **decompose_kwargs)
            return raw, None

        except asyncio.TimeoutError as exc:
            wrapped = OrchestratorError(
                f"Decomposition timed out after {self._decompose_timeout}s",
                cause=exc,
            )
            return None, wrapped

        except (OrchestratorError, TaskError) as exc:
            return None, exc

        except Exception as exc:
            wrapped = OrchestratorError(f"Unexpected decomposition error: {exc}", cause=exc)
            return None, wrapped
