"""
TelemetryCollector — updates ModelProfile stats after each API call.
=====================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis
Uses Exponential Moving Average (EMA, alpha=0.1) for latency and quality,
a fixed-size rolling window (last 10 calls) for success_rate, a sorted
latency buffer (last 50 samples) for real p95 computation, and cost EMA.

OPTIMIZATION: p95 calculation now uses numpy if available for O(n) performance,
otherwise falls back to partial sort for O(n) average case.

Adaptive re-planning rationale:
  These updated profiles feed directly into ConstraintPlanner._score(), so
  routing decisions automatically improve over time — models with consistently
  high evaluation scores and low failure rates become preferred, while models
  that repeatedly fail validation or violate policies are progressively de-ranked.

No I/O: all state lives in the in-memory ModelProfile objects passed at init.
Thread-safety: NOT thread-safe. The asyncio event loop serialises all updates.
"""

from __future__ import annotations

import collections
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import Model
    from .policy import ModelProfile

logger = logging.getLogger("orchestrator.telemetry")

# ── EMA configuration ──────────────────────────────────────────────────────────
# OPTIMIZATION: Increased alpha from 0.1 to 0.2 for faster regression detection
# This makes the system respond to quality changes in ~5 calls instead of ~10
_EMA_ALPHA: float = 0.2  # learning rate for latency, quality, and cost EMAs

# ── Success rate rolling window ────────────────────────────────────────────────
_SUCCESS_WINDOW: int = 10  # number of recent calls to track

# ── Trust factor dynamics ──────────────────────────────────────────────────────
_TRUST_DEGRADE: float = 0.95  # multiplier on each failure / policy violation
_TRUST_RECOVER: float = 1.001  # multiplier on each success
_TRUST_CAP: float = 1.0  # ceiling (trust can never exceed 1.0)

# ── Latency p95 buffer ─────────────────────────────────────────────────────────
_LATENCY_BUFFER_SIZE: int = 50  # circular buffer for real p95 computation

# Try to import numpy for faster percentile calculation
try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


class TelemetryCollector:
    """
    Records call outcomes and updates the corresponding ModelProfile in-place.

    Usage (by engine.py after each client.call()):
        self._telemetry.record_call(model, response.latency_ms,
                                    response.cost_usd, success=True,
                                    quality_score=eval_score)
    """

    def __init__(self, profiles: dict[Model, ModelProfile]):
        self._profiles = profiles
        # Rolling window per model: deque of bool (True=success, False=failure)
        self._success_windows: dict[Model, collections.deque] = {
            m: collections.deque(maxlen=_SUCCESS_WINDOW) for m in profiles
        }
        # Latency buffer per model using deque for O(1) append
        self._latency_buffers: dict[Model, collections.deque] = {
            m: collections.deque(maxlen=_LATENCY_BUFFER_SIZE) for m in profiles
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def record_call(
        self,
        model: Model,
        latency_ms: float,
        cost_usd: float,
        success: bool,
        quality_score: float | None = None,
    ) -> None:
        """
        Update ModelProfile after a single API call.

        OPTIMIZATION: p95 calculation uses numpy.percentile if available (O(n)),
        otherwise uses statistics.quantiles with partial sort.
        """
        profile = self._profiles.get(model)
        if profile is None:
            logger.warning(f"TelemetryCollector: unknown model {model!r}")
            return

        profile.call_count += 1

        if not success:
            profile.failure_count += 1

        # ── Latency EMA + p95 (skip if latency_ms <= 0, e.g. cache hits) ─
        if latency_ms > 0:
            profile.avg_latency_ms = (
                _EMA_ALPHA * latency_ms + (1 - _EMA_ALPHA) * profile.avg_latency_ms
            )

            # Update latency buffer
            buf = self._latency_buffers.setdefault(
                model, collections.deque(maxlen=_LATENCY_BUFFER_SIZE)
            )
            buf.append(latency_ms)

            # OPTIMIZED p95 calculation
            profile.latency_p95_ms = self._calculate_p95(list(buf))

        # ── Cost EMA (skip if cost_usd <= 0 to avoid dragging EMA to zero) ───
        if cost_usd > 0:
            profile.avg_cost_usd = _EMA_ALPHA * cost_usd + (1 - _EMA_ALPHA) * profile.avg_cost_usd

        # ── Success rate: rolling window ──────────────────────────────────────
        win = self._success_windows.setdefault(model, collections.deque(maxlen=_SUCCESS_WINDOW))
        win.append(success)
        profile.success_rate = sum(win) / len(win)

        # ── Quality EMA (only when evaluator score available) ─────────────────
        if quality_score is not None:
            profile.quality_score = (
                _EMA_ALPHA * quality_score + (1 - _EMA_ALPHA) * profile.quality_score
            )

        # ── Trust factor: degradation / recovery ──────────────────────────────
        if success:
            profile.trust_factor = min(_TRUST_CAP, profile.trust_factor * _TRUST_RECOVER)
        else:
            profile.trust_factor *= _TRUST_DEGRADE

    def record_policy_violation(self, model: Model) -> None:
        """Record a policy violation for the given model."""
        profile = self._profiles.get(model)
        if profile is None:
            return
        profile.trust_factor *= _TRUST_DEGRADE
        profile.violation_count += 1

    def record_validator_failure(self, model: Model) -> None:
        """Record a validator failure for the given model."""
        # TODO: Implement proper tracking
        # For now, just degrade trust factor slightly
        profile = self._profiles.get(model)
        if profile is None:
            return
        profile.trust_factor *= _TRUST_DEGRADE

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _calculate_p95(self, values: list[float]) -> float:
        """
        Calculate 95th percentile efficiently.

        Uses numpy if available for vectorized operation,
        otherwise uses statistics.quantiles for partial sort.
        """
        if not values:
            return 0.0

        if len(values) == 1:
            return values[0]

        # Use numpy for O(n) performance
        if _HAS_NUMPY:
            return float(np.percentile(values, 95, interpolation="linear"))

        # Fallback to statistics.quantiles (uses partial sort, O(n) average)
        try:
            import statistics

            # quantiles returns a list, we want the 95th (index 18 for 20 quantiles)
            return statistics.quantiles(values, n=20)[18]
        except Exception:
            # Last resort: full sort
            return sorted(values)[int(0.95 * len(values))]

    def get_profile_stats(self, model: Model) -> dict:
        """Get current statistics for a model (for debugging)."""
        profile = self._profiles.get(model)
        if profile is None:
            return {}

        buf = self._latency_buffers.get(model, collections.deque())
        win = self._success_windows.get(model, collections.deque())

        return {
            "model": model.value,
            "call_count": profile.call_count,
            "failure_count": profile.failure_count,
            "success_rate": profile.success_rate,
            "avg_latency_ms": profile.avg_latency_ms,
            "latency_p95_ms": profile.latency_p95_ms,
            "trust_factor": profile.trust_factor,
            "quality_score": profile.quality_score,
            "buffer_size": len(buf),
            "window_size": len(win),
        }
