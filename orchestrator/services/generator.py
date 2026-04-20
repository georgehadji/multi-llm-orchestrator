"""
GeneratorService — stable project decomposition interface.
==========================================================
Wraps engine.py._decompose() via callback injection (same pattern as
ExecutorService). Establishes the ``decompose(project, criteria) → GeneratorResult``
boundary so callers can depend on a stable contract while the complex
decomposition logic is migrated here incrementally.

Phase 1 (current):  Interface + timing + error normalization; implementation
                    delegated via ``decompose_fn`` callback.
Phase 3 (planned):  Migrate engine._decompose() body into this class.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from ..exceptions import OrchestratorError, TaskError
from ..models import Task

logger = logging.getLogger("orchestrator.services.generator")

# Type alias for the injected decompose implementation.
DecomposeFn = Callable[..., Awaitable[dict[str, Task]]]


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class GeneratorResult:
    """
    Outcome of a decomposition call.

    Attributes:
        tasks:          Ordered dict of task_id → Task (empty on failure).
        wall_time_ms:   Wall-clock time for the full decomposition in ms.
        error:          Set on unrecoverable failure; None on success.
    """

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
class GeneratorMetrics:
    """Monotonic counters for decomposition calls."""

    total_calls: int = 0
    total_succeeded: int = 0
    total_failed: int = 0
    total_tasks_generated: int = 0
    cumulative_wall_ms: float = 0.0

    def record(self, result: GeneratorResult) -> None:
        self.total_calls += 1
        self.cumulative_wall_ms += result.wall_time_ms
        if result.succeeded:
            self.total_succeeded += 1
            self.total_tasks_generated += result.task_count
        else:
            self.total_failed += 1

    def to_dict(self) -> dict[str, Any]:
        avg_ms = (
            self.cumulative_wall_ms / self.total_calls if self.total_calls else 0.0
        )
        return {
            "total_calls": self.total_calls,
            "total_succeeded": self.total_succeeded,
            "total_failed": self.total_failed,
            "total_tasks_generated": self.total_tasks_generated,
            "avg_wall_ms": round(avg_ms, 1),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Service
# ─────────────────────────────────────────────────────────────────────────────


class GeneratorService:
    """
    Application-layer service for project decomposition.

    Usage (in engine.py):

        self._generator = GeneratorService(decompose_fn=self._decompose)
        ...
        result = await self._generator.decompose(project, criteria)
        if not result.succeeded:
            raise OrchestratorError(f"Decomposition failed: {result.error}")
        tasks = result.tasks

    Args:
        decompose_fn:   Async callable ``(project, criteria, **kwargs) → dict[str, Task]``.
                        Injected from engine.py; will migrate into this class in Phase 3.
        decompose_timeout:
                        Hard wall-clock timeout (seconds) for a single decomposition call.
                        None = no extra timeout.
    """

    def __init__(
        self,
        decompose_fn: DecomposeFn,
        decompose_timeout: float | None = None,
    ) -> None:
        self._decompose_fn = decompose_fn
        self._decompose_timeout = decompose_timeout
        self.metrics = GeneratorMetrics()
        self._lock = asyncio.Lock()

    # ── Public interface ──────────────────────────────────────────────────────

    async def decompose(
        self,
        project: str,
        criteria: str,
        **kwargs: Any,
    ) -> GeneratorResult:
        """
        Decompose ``project`` into an ordered task dict.

        Guarantees:
          - Always returns a ``GeneratorResult`` (never raises).
          - On failure, ``result.error`` is set and ``result.tasks`` is empty.
          - Wall time is always measured.

        Extra keyword args (e.g. ``app_profile``) are forwarded to ``decompose_fn``.
        """
        t0 = time.monotonic()
        tasks, error = await self._run_with_guard(project, criteria, **kwargs)
        wall_ms = (time.monotonic() - t0) * 1000

        result = GeneratorResult(tasks=tasks or {}, wall_time_ms=wall_ms, error=error)

        async with self._lock:
            self.metrics.record(result)

        if error:
            logger.warning(
                "decompose FAILED in %.0fms: %s", wall_ms, error
            )
        else:
            logger.debug(
                "decompose succeeded in %.0fms — %d tasks", wall_ms, result.task_count
            )

        return result

    def metrics_snapshot(self) -> dict[str, Any]:
        return self.metrics.to_dict()

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _run_with_guard(
        self, project: str, criteria: str, **kwargs: Any
    ) -> tuple[dict[str, Task] | None, Exception | None]:
        try:
            if self._decompose_timeout is not None:
                raw = await asyncio.wait_for(
                    self._decompose_fn(project, criteria, **kwargs),
                    timeout=self._decompose_timeout,
                )
            else:
                raw = await self._decompose_fn(project, criteria, **kwargs)
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
            wrapped = OrchestratorError(
                f"Unexpected decomposition error: {exc}", cause=exc
            )
            return None, wrapped
