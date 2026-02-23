"""
Adaptive Model Router — circuit breaker v2.

Per-model states:
  HEALTHY   — routing proceeds normally
  DEGRADED  — too many recent timeouts; skip during cooldown_seconds
  DISABLED  — permanent failure (auth error); never route here

Also tracks EMA latency to prefer faster healthy models.
"""
from __future__ import annotations
import time
from enum import Enum
from typing import Optional
from .models import Model, TaskType


class ModelState(str, Enum):
    HEALTHY  = "healthy"
    DEGRADED = "degraded"
    DISABLED = "disabled"


class AdaptiveRouter:

    def __init__(
        self,
        timeout_threshold: int = 3,
        cooldown_seconds: float = 300.0,
    ) -> None:
        self._timeout_counts: dict[Model, int] = {m: 0 for m in Model}
        self._degraded_since: dict[Model, Optional[float]] = {m: None for m in Model}
        self._disabled: set[Model] = set()
        self._latencies: dict[Model, float] = {}
        self.timeout_threshold = timeout_threshold
        self.cooldown_seconds = cooldown_seconds

    def get_state(self, model: Model) -> ModelState:
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

    def is_available(self, model: Model) -> bool:
        return self.get_state(model) == ModelState.HEALTHY

    def record_timeout(self, model: Model) -> None:
        if model in self._disabled:
            return
        self._timeout_counts[model] += 1
        if self._timeout_counts[model] >= self.timeout_threshold:
            if self._degraded_since.get(model) is None:
                self._degraded_since[model] = time.monotonic()

    def record_success(self, model: Model) -> None:
        self._timeout_counts[model] = 0
        self._degraded_since[model] = None

    def record_auth_failure(self, model: Model) -> None:
        self._disabled.add(model)

    def record_latency(self, model: Model, latency_ms: float,
                       alpha: float = 0.1) -> None:
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
        """Return healthy candidate with lowest observed EMA latency."""
        healthy = [m for m in candidates if self.is_available(m)]
        if not healthy:
            return None
        healthy.sort(key=lambda m: self._latencies.get(m, float("inf")))
        return healthy[0]
