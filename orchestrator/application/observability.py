"""
ObservabilityService — per-model latency, cost, and error-rate tracking.
========================================================================
Collects lightweight metrics for every LLM call.

Part of Application Layer (Phase 4) — Canonical location.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("orchestrator.services.observability")


# ---------------------------------------------------------------------------
# Per-model bucket
# ---------------------------------------------------------------------------


@dataclass
class _ModelBucket:
    """Mutable accumulator for one model."""

    model_id: str
    calls: int = 0
    errors: int = 0
    fallback_triggers: int = 0
    total_latency_ms: float = 0.0
    total_cost_usd: float = 0.0
    _window: list[bool] = field(default_factory=list)  # True = success
    _window_size: int = 20
    _last_error: str | None = None
    _last_call_ts: float = field(default_factory=time.monotonic)

    def record(
        self,
        latency_ms: float,
        cost_usd: float,
        success: bool,
        error: str | None = None,
    ) -> None:
        self.calls += 1
        self.total_latency_ms += latency_ms
        self.total_cost_usd += cost_usd
        self._last_call_ts = time.monotonic()
        if not success:
            self.errors += 1
            self._last_error = error
        self._window.append(success)
        if len(self._window) > self._window_size:
            self._window.pop(0)

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.calls if self.calls else 0.0

    @property
    def error_rate(self) -> float:
        if not self._window:
            return 0.0
        failures = self._window.count(False)
        return failures / len(self._window)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "calls": self.calls,
            "errors": self.errors,
            "fallback_triggers": self.fallback_triggers,
            "error_rate_window": round(self.error_rate, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "total_cost_usd": round(self.total_cost_usd, 6),
            "last_error": self._last_error,
        }


# ---------------------------------------------------------------------------
# Public service
# ---------------------------------------------------------------------------


@dataclass
class ModelSummary:
    """Immutable snapshot of metrics for one model."""

    model_id: str
    calls: int
    errors: int
    fallback_triggers: int
    error_rate: float
    avg_latency_ms: float
    total_cost_usd: float
    last_error: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "calls": self.calls,
            "errors": self.errors,
            "fallback_triggers": self.fallback_triggers,
            "error_rate": self.error_rate,
            "avg_latency_ms": self.avg_latency_ms,
            "total_cost_usd": self.total_cost_usd,
            "last_error": self.last_error,
        }


class ObservabilityService:
    """
    Per-model latency, cost, and error-rate tracker.
    Thread-safe: all mutations serialised through a single asyncio.Lock.
    """

    def __init__(self, error_rate_threshold: float = 0.5) -> None:
        self.error_rate_threshold = error_rate_threshold
        self._buckets: dict[str, _ModelBucket] = {}
        self._lock = asyncio.Lock()

    async def record_call(
        self,
        model_id: str,
        latency_ms: float,
        cost_usd: float,
        success: bool,
        error: str | None = None,
    ) -> None:
        async with self._lock:
            bucket = self._get_or_create(model_id)
            bucket.record(latency_ms=latency_ms, cost_usd=cost_usd, success=success, error=error)
        if not success:
            logger.debug(
                "obs: model=%s FAILED latency=%.0fms error=%s", model_id, latency_ms, error
            )

    async def record_fallback(self, primary_model_id: str) -> None:
        async with self._lock:
            bucket = self._get_or_create(primary_model_id)
            bucket.fallback_triggers += 1
        logger.debug("obs: fallback triggered for model=%s", primary_model_id)

    def model_summary(self, model_id: str) -> ModelSummary | None:
        bucket = self._buckets.get(model_id)
        if bucket is None:
            return None
        return self._bucket_to_summary(bucket)

    def snapshot(self) -> list[ModelSummary]:
        return [self._bucket_to_summary(b) for b in self._buckets.values()]

    def ranked_by_error_rate(self) -> list[ModelSummary]:
        return sorted(self.snapshot(), key=lambda s: s.error_rate, reverse=True)

    def is_degraded(self, model_id: str) -> bool:
        bucket = self._buckets.get(model_id)
        if bucket is None or bucket.calls < 3:
            return False
        return bucket.error_rate >= self.error_rate_threshold

    def total_cost_usd(self) -> float:
        return sum(b.total_cost_usd for b in self._buckets.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_cost_usd": round(self.total_cost_usd(), 6),
            "models": [s.to_dict() for s in self.snapshot()],
        }

    def _get_or_create(self, model_id: str) -> _ModelBucket:
        if model_id not in self._buckets:
            self._buckets[model_id] = _ModelBucket(model_id=model_id)
        return self._buckets[model_id]

    @staticmethod
    def _bucket_to_summary(b: _ModelBucket) -> ModelSummary:
        return ModelSummary(
            model_id=b.model_id,
            calls=b.calls,
            errors=b.errors,
            fallback_triggers=b.fallback_triggers,
            error_rate=round(b.error_rate, 4),
            avg_latency_ms=round(b.avg_latency_ms, 1),
            total_cost_usd=round(b.total_cost_usd, 6),
            last_error=b._last_error,
        )
