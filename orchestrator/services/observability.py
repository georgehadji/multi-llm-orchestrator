"""
ObservabilityService — per-model latency, cost, and error-rate tracking.
========================================================================
Collects lightweight metrics for every LLM call so that the engine can:
  - Identify slow or degraded models before spending budget on them
  - Surface cost drift in dashboards / telemetry
  - Feed circuit-breaker and cascade-fallback decisions with real data

Design:
  - All mutations go through a single asyncio.Lock so the service is
    safe to call from concurrent tasks.
  - Metrics are stored in memory only; persistence is out-of-scope here
    (the existing telemetry_store handles that).
  - snapshot() returns an immutable copy so callers cannot corrupt state.

Usage:
    obs = ObservabilityService()

    # Record a successful call:
    obs.record_call(
        model_id="openai/gpt-4o",
        latency_ms=430.0,
        cost_usd=0.0012,
        success=True,
    )

    # Record a failed call:
    obs.record_call(model_id="openai/gpt-4o", latency_ms=15000.0,
                    cost_usd=0.0, success=False, error="TimeoutError")

    # Query per-model summary:
    summary = obs.model_summary("openai/gpt-4o")
    # {"calls": 10, "errors": 1, "error_rate": 0.1, "avg_latency_ms": 450.0, ...}

    # Get all models sorted by error rate (descending):
    ranked = obs.ranked_by_error_rate()
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("orchestrator.services.observability")


# ─────────────────────────────────────────────────────────────────────────────
# Per-model bucket
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class _ModelBucket:
    """Mutable accumulator for one model. Not exposed directly — always copy."""

    model_id: str
    calls: int = 0
    errors: int = 0
    fallback_triggers: int = 0
    total_latency_ms: float = 0.0
    total_cost_usd: float = 0.0
    # Sliding window for error rate (last 20 outcomes)
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
        """Sliding-window error rate over last _window_size calls."""
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


# ─────────────────────────────────────────────────────────────────────────────
# Public service
# ─────────────────────────────────────────────────────────────────────────────


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
    snapshot() / model_summary() return copies — no shared mutable state leaks.
    """

    def __init__(self, error_rate_threshold: float = 0.5) -> None:
        """
        Args:
            error_rate_threshold: Error rate (0–1) above which a model is
                                  considered degraded. Used by is_degraded().
        """
        self.error_rate_threshold = error_rate_threshold
        self._buckets: dict[str, _ModelBucket] = {}
        self._lock = asyncio.Lock()

    # ── Write path ────────────────────────────────────────────────────────────

    async def record_call(
        self,
        model_id: str,
        latency_ms: float,
        cost_usd: float,
        success: bool,
        error: str | None = None,
    ) -> None:
        """Record one completed LLM call (success or failure)."""
        async with self._lock:
            bucket = self._get_or_create(model_id)
            bucket.record(latency_ms=latency_ms, cost_usd=cost_usd,
                          success=success, error=error)
        if not success:
            logger.debug(
                "obs: model=%s FAILED latency=%.0fms error=%s",
                model_id, latency_ms, error,
            )

    async def record_fallback(self, primary_model_id: str) -> None:
        """Increment fallback_triggers counter for *primary_model_id*."""
        async with self._lock:
            bucket = self._get_or_create(primary_model_id)
            bucket.fallback_triggers += 1
        logger.debug("obs: fallback triggered for model=%s", primary_model_id)

    # ── Read path (returns copies) ────────────────────────────────────────────

    def model_summary(self, model_id: str) -> ModelSummary | None:
        """Return an immutable summary for *model_id*, or None if unknown."""
        bucket = self._buckets.get(model_id)
        if bucket is None:
            return None
        return self._bucket_to_summary(bucket)

    def snapshot(self) -> list[ModelSummary]:
        """Return immutable summaries for all observed models."""
        return [self._bucket_to_summary(b) for b in self._buckets.values()]

    def ranked_by_error_rate(self) -> list[ModelSummary]:
        """Return all model summaries sorted by error_rate descending."""
        return sorted(self.snapshot(), key=lambda s: s.error_rate, reverse=True)

    def is_degraded(self, model_id: str) -> bool:
        """True if the model's sliding-window error rate exceeds the threshold."""
        bucket = self._buckets.get(model_id)
        if bucket is None or bucket.calls < 3:
            return False
        return bucket.error_rate >= self.error_rate_threshold

    def total_cost_usd(self) -> float:
        """Total cost across all models."""
        return sum(b.total_cost_usd for b in self._buckets.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_cost_usd": round(self.total_cost_usd(), 6),
            "models": [s.to_dict() for s in self.snapshot()],
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_or_create(self, model_id: str) -> _ModelBucket:
        """Must be called under self._lock."""
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
