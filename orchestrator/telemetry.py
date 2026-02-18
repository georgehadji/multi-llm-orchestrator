"""
TelemetryCollector — updates ModelProfile stats after each API call.
=====================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis
Uses Exponential Moving Average (EMA, alpha=0.1) for latency and quality,
a fixed-size rolling window (last 10 calls) for success_rate, a sorted
latency buffer (last 50 samples) for real p95 computation, and cost EMA.

Adaptive re-planning rationale:
  These updated profiles feed directly into ConstraintPlanner._score(), so
  routing decisions automatically improve over time — models with consistently
  high evaluation scores and low failure rates become preferred, while models
  that repeatedly fail validation or violate policies are progressively de-ranked.

No I/O: all state lives in the in-memory ModelProfile objects passed at init.
Thread-safety: NOT thread-safe. The asyncio event loop serialises all updates.
"""
from __future__ import annotations

import bisect
import collections
import logging
from typing import Optional

from .models import Model
from .policy import ModelProfile

logger = logging.getLogger("orchestrator.telemetry")

# ── EMA configuration ──────────────────────────────────────────────────────────
_EMA_ALPHA: float = 0.1        # learning rate for latency, quality, and cost EMAs

# ── Success rate rolling window ────────────────────────────────────────────────
_SUCCESS_WINDOW: int = 10      # number of recent calls to track

# ── Trust factor dynamics ──────────────────────────────────────────────────────
_TRUST_DEGRADE: float = 0.95   # multiplier on each failure / policy violation
_TRUST_RECOVER: float = 1.001  # multiplier on each success
_TRUST_CAP: float = 1.0        # ceiling (trust can never exceed 1.0)

# ── Latency p95 buffer ─────────────────────────────────────────────────────────
_LATENCY_BUFFER_SIZE: int = 50  # sorted circular buffer for real p95 computation


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
            m: collections.deque(maxlen=_SUCCESS_WINDOW)
            for m in profiles
        }
        # Sorted latency buffer per model (ascending order, capped at _LATENCY_BUFFER_SIZE)
        self._latency_buffers: dict[Model, list[float]] = {
            m: [] for m in profiles
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
        quality_score: Optional[float] = None,
    ) -> None:
        """
        Update ModelProfile after a single API call.

        Parameters
        ----------
        model : Model
            The model that was called.
        latency_ms : float
            Round-trip latency in milliseconds. Pass 0.0 for cache hits or
            quality-feedback-only calls (these are excluded from latency tracking).
        cost_usd : float
            Actual cost of the call in USD. Pass 0.0 for cache hits or failed
            calls (these are excluded from cost EMA to avoid dragging it to zero).
        success : bool
            True if the call succeeded (no exception, no policy violation).
        quality_score : Optional[float]
            LLM evaluator score [0.0–1.0] for the output of this call.
            Pass None when no evaluator score is available (e.g. reviewer calls).
        """
        profile = self._profiles.get(model)
        if profile is None:
            logger.warning(f"TelemetryCollector: unknown model {model!r}")
            return

        profile.call_count += 1

        if not success:
            profile.failure_count += 1

        # ── Latency EMA + real p95 (skip if latency_ms <= 0, e.g. cache hits) ─
        if latency_ms > 0:
            profile.avg_latency_ms = (
                _EMA_ALPHA * latency_ms
                + (1 - _EMA_ALPHA) * profile.avg_latency_ms
            )

            # Real p95 via sorted buffer (bisect maintains ascending order)
            buf = self._latency_buffers.setdefault(model, [])
            bisect.insort(buf, latency_ms)
            if len(buf) > _LATENCY_BUFFER_SIZE:
                buf.pop(0)   # discard the oldest/smallest sample
            profile.latency_samples = list(buf)
            idx = int(0.95 * len(buf))
            profile.latency_p95_ms = buf[min(idx, len(buf) - 1)]

        # ── Cost EMA (skip if cost_usd <= 0 to avoid dragging EMA to zero) ───
        if cost_usd > 0:
            profile.avg_cost_usd = (
                _EMA_ALPHA * cost_usd
                + (1 - _EMA_ALPHA) * profile.avg_cost_usd
            )

        # ── Success rate: rolling window ──────────────────────────────────────
        win = self._success_windows.setdefault(
            model, collections.deque(maxlen=_SUCCESS_WINDOW)
        )
        win.append(success)
        profile.success_rate = sum(win) / len(win)

        # ── Quality EMA (only when evaluator score available) ─────────────────
        if quality_score is not None:
            profile.quality_score = (
                _EMA_ALPHA * quality_score
                + (1 - _EMA_ALPHA) * profile.quality_score
            )

        # ── Trust factor: degradation / recovery ──────────────────────────────
        if success:
            profile.trust_factor = min(
                _TRUST_CAP,
                profile.trust_factor * _TRUST_RECOVER,
            )
        else:
            profile.trust_factor *= _TRUST_DEGRADE

        logger.debug(
            "Telemetry %s: trust=%.4f sr=%.3f lat=%.0fms q=%.3f cost=%.6f",
            model.value,
            profile.trust_factor,
            profile.success_rate,
            profile.avg_latency_ms,
            profile.quality_score,
            profile.avg_cost_usd,
        )

    def record_validator_failure(self, model: Model) -> None:
        """
        Increment the validator_fail_count for the model.

        Called by engine.py when deterministic validators (pytest, ruff, etc.)
        fail for a task that used this model. The count feeds into MetricsExporter
        for observability and can be used to down-rank models with high fail rates.
        """
        profile = self._profiles.get(model)
        if profile is None:
            logger.warning(
                "TelemetryCollector: record_validator_failure called for unknown model %r",
                model,
            )
            return
        profile.validator_fail_count += 1
        logger.warning(
            "Validator failure recorded for %s — total=%d",
            model.value,
            profile.validator_fail_count,
        )

    def record_policy_violation(self, model: Model) -> None:
        """
        Called when a model is used but found to violate a policy.
        Degrades trust_factor identically to a failed call to signal
        the planner to prefer this model less in future decisions.
        """
        profile = self._profiles.get(model)
        if profile is None:
            return
        profile.trust_factor *= _TRUST_DEGRADE
        logger.warning(
            "Policy violation recorded for %s — trust_factor now %.4f",
            model.value,
            profile.trust_factor,
        )

    def error_rate(self, model: Model) -> float:
        """
        Return the error rate for the model: failure_count / call_count.

        Returns 0.0 if the model is unknown or has never been called.
        """
        profile = self._profiles.get(model)
        if profile is None or profile.call_count == 0:
            return 0.0
        return profile.failure_count / profile.call_count

    def get_profiles(self) -> dict[Model, ModelProfile]:
        """
        Return the live profile dictionary.
        Callers receive a reference (not a copy) — do not mutate directly.
        """
        return self._profiles
