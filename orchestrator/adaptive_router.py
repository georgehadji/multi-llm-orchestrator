"""
Adaptive Model Router — circuit breaker v2.

Per-model states:
  HEALTHY   — routing proceeds normally
  DEGRADED  — too many recent timeouts; skip during cooldown_seconds
  DISABLED  — permanent failure (auth error); never route here

Also tracks EMA latency to prefer faster healthy models.

DESIGN: All public methods are async and use asyncio.Lock for thread safety.
        This prevents event loop blocking and protects shared state from
        concurrent access in async contexts.

FIX-BUG-002: Replaced threading.Lock with asyncio.Lock to prevent potential
        event loop blocking and follow async best practices.
"""

from __future__ import annotations

import asyncio
import time
from enum import Enum

from .models import Model, TaskType


class ModelState(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DISABLED = "disabled"


class AdaptiveRouter:
    """
    Async-safe adaptive model router with circuit breaker functionality.

    All public methods are async and protected by asyncio.Lock to prevent
    race conditions when multiple concurrent tasks update model health state.

    FIX-BUG-002: Uses asyncio.Lock instead of threading.Lock.
    """

    def __init__(
        self,
        timeout_threshold: int = 3,
        cooldown_seconds: float = 300.0,
    ) -> None:
        self._timeout_counts: dict[Model, int] = dict.fromkeys(Model, 0)
        self._degraded_since: dict[Model, float | None] = dict.fromkeys(Model)
        self._disabled: set[Model] = set()
        self._latencies: dict[Model, float] = {}
        self._lock = asyncio.Lock()  # Async-safe lock for all shared state
        self.timeout_threshold = timeout_threshold
        self.cooldown_seconds = cooldown_seconds

    def _get_state_unsafe(self, model: Model) -> ModelState:
        """
        Get model state - caller must hold _lock.

        Internal method for use when lock is already held.
        """
        if model in self._disabled:
            return ModelState.DISABLED
        since = self._degraded_since.get(model)
        if since is not None:
            if time.monotonic() - since < self.cooldown_seconds:
                return ModelState.DEGRADED
            # cooldown elapsed — recover
            self._degraded_since[model] = None
            self._timeout_counts[model] = 0
        return ModelState.HEALTHY

    async def get_state(self, model: Model) -> ModelState:
        """Get model state (async-safe)."""
        async with self._lock:
            return self._get_state_unsafe(model)

    def is_available(self, model: Model) -> bool:
        """
        Check if model is available for routing (fast, non-blocking).

        Reads mutable state without acquiring the lock — this is safe in
        CPython asyncio because dict.get() is a single C call that does not
        release the GIL, and this method contains no ``await`` points that
        could yield control mid-read.  When the lock IS held by a concurrent
        writer, the method returns True (optimistic), avoiding blocking in
        list comprehensions.
        """
        # Optimistic: if lock is held, a writer is in progress — assume available
        try:
            if self._lock.locked():
                return True
        except AttributeError:
            pass
        if model in self._disabled:
            return False
        since = self._degraded_since.get(model)
        if since is None:
            return True
        return (time.monotonic() - since) >= self.cooldown_seconds

    async def record_timeout(self, model: Model) -> None:
        """
        Record a timeout for a model (async-safe).

        After timeout_threshold consecutive timeouts, model enters DEGRADED state
        for cooldown_seconds duration.
        """
        async with self._lock:
            if model in self._disabled:
                return
            self._timeout_counts[model] += 1
            count = self._timeout_counts[model]
            since = self._degraded_since.get(model)

            # Check and update inside lock to avoid reentrancy issues
            if count >= self.timeout_threshold and since is None:
                self._degraded_since[model] = time.monotonic()

    async def record_success(self, model: Model) -> None:
        """
        Record a successful call (async-safe).

        Resets timeout counter and clears DEGRADED state.
        """
        async with self._lock:
            self._timeout_counts[model] = 0
            self._degraded_since[model] = None

    async def reset(self) -> None:
        """
        Reset all router health state (async-safe).

        Clears timeout counts, degradation timers, disabled set, and latency
        EMAs. Used by FallbackHandler.reset() to restore a clean slate.
        """
        async with self._lock:
            self._timeout_counts = dict.fromkeys(Model, 0)
            self._degraded_since = dict.fromkeys(Model)
            self._disabled.clear()
            self._latencies.clear()

    async def record_auth_failure(self, model: Model) -> None:
        """
        Record an authentication failure (async-safe).

        Permanently disables the model until process restart.
        """
        async with self._lock:
            self._disabled.add(model)

    async def record_latency(self, model: Model, latency_ms: float, alpha: float = 0.1) -> None:
        """
        Record latency observation for EMA calculation (async-safe).

        Args:
            model: The model to record latency for
            latency_ms: Observed latency in milliseconds
            alpha: Smoothing factor for EMA (default 0.1)
        """
        async with self._lock:
            if model in self._latencies:
                self._latencies[model] = alpha * latency_ms + (1 - alpha) * self._latencies[model]
            else:
                self._latencies[model] = latency_ms

    async def preferred_model(
        self,
        candidates: list[Model],
        task_type: TaskType | None = None,
    ) -> Model | None:
        """
        Return healthy candidate with lowest observed EMA latency (async-safe).

        Args:
            candidates: List of candidate models to choose from
            task_type: Optional task type for future task-aware routing

        Returns:
            Best healthy model, or None if no healthy candidates
        """
        async with self._lock:
            healthy = [m for m in candidates if self._get_state_unsafe(m) == ModelState.HEALTHY]
            if not healthy:
                return None
            # Copy latencies while holding lock
            latencies_copy = {m: self._latencies.get(m, float("inf")) for m in healthy}

        # Sort outside lock
        healthy.sort(key=lambda m: latencies_copy[m])
        return healthy[0]
