"""
Adaptive Model Router — circuit breaker v2.

Per-model states:
  HEALTHY   — routing proceeds normally
  DEGRADED  — too many recent timeouts; skip during cooldown_seconds
  DISABLED  — permanent failure (auth error); never route here

Also tracks EMA latency to prefer faster healthy models.

DESIGN: All public methods are thread-safe using threading.Lock.
        This protects shared state from concurrent access in async contexts
        where multiple tasks may update router state simultaneously.
"""
from __future__ import annotations
import time
from threading import Lock
from enum import Enum
from typing import Optional
from .models import Model, TaskType


class ModelState(str, Enum):
    HEALTHY  = "healthy"
    DEGRADED = "degraded"
    DISABLED = "disabled"


class AdaptiveRouter:
    """
    Thread-safe adaptive model router with circuit breaker functionality.
    
    All public methods are protected by a lock to prevent race conditions
    when multiple concurrent tasks update model health state.
    """

    def __init__(
        self,
        timeout_threshold: int = 3,
        cooldown_seconds: float = 300.0,
    ) -> None:
        self._timeout_counts: dict[Model, int] = {m: 0 for m in Model}
        self._degraded_since: dict[Model, Optional[float]] = {m: None for m in Model}
        self._disabled: set[Model] = set()
        self._latencies: dict[Model, float] = {}
        self._lock = Lock()  # Thread-safe lock for all shared state
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

    def get_state(self, model: Model) -> ModelState:
        """Get model state (thread-safe)."""
        with self._lock:
            return self._get_state_unsafe(model)

    def is_available(self, model: Model) -> bool:
        """Check if model is available for routing (thread-safe)."""
        with self._lock:
            return self._get_state_unsafe(model) == ModelState.HEALTHY

    def record_timeout(self, model: Model) -> None:
        """
        Record a timeout for a model (thread-safe).
        
        After timeout_threshold consecutive timeouts, model enters DEGRADED state
        for cooldown_seconds duration.
        """
        with self._lock:
            if model in self._disabled:
                return
            self._timeout_counts[model] += 1
            count = self._timeout_counts[model]
            since = self._degraded_since.get(model)
        
        # Check outside lock to minimize hold time
        if count >= self.timeout_threshold and since is None:
            with self._lock:
                # Double-check after reacquiring lock
                if self._degraded_since.get(model) is None:
                    self._degraded_since[model] = time.monotonic()

    def record_success(self, model: Model) -> None:
        """
        Record a successful call (thread-safe).
        
        Resets timeout counter and clears DEGRADED state.
        """
        with self._lock:
            self._timeout_counts[model] = 0
            self._degraded_since[model] = None

    def record_auth_failure(self, model: Model) -> None:
        """
        Record an authentication failure (thread-safe).
        
        Permanently disables the model until process restart.
        """
        with self._lock:
            self._disabled.add(model)

    def record_latency(self, model: Model, latency_ms: float,
                       alpha: float = 0.1) -> None:
        """
        Record latency observation for EMA calculation (thread-safe).
        
        Args:
            model: The model to record latency for
            latency_ms: Observed latency in milliseconds
            alpha: Smoothing factor for EMA (default 0.1)
        """
        with self._lock:
            if model in self._latencies:
                self._latencies[model] = (
                    alpha * latency_ms + (1 - alpha) * self._latencies[model]
                )
            else:
                self._latencies[model] = latency_ms

    def preferred_model(
        self,
        candidates: list[Model],
        task_type: Optional[TaskType] = None,
    ) -> Optional[Model]:
        """
        Return healthy candidate with lowest observed EMA latency (thread-safe).
        
        Args:
            candidates: List of candidate models to choose from
            task_type: Optional task type for future task-aware routing
            
        Returns:
            Best healthy model, or None if no healthy candidates
        """
        with self._lock:
            healthy = [m for m in candidates if self._get_state_unsafe(m) == ModelState.HEALTHY]
            if not healthy:
                return None
            # Copy latencies while holding lock
            latencies_copy = {m: self._latencies.get(m, float("inf")) for m in healthy}
        
        # Sort outside lock
        healthy.sort(key=lambda m: latencies_copy[m])
        return healthy[0]
