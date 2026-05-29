"""
Model health tracking — circuit breaker and telemetry recording.

P3-2 of REFACTORING_PLAN_V7.md — extracted from engine._record_success/_record_failure.
"""

from __future__ import annotations

import logging
from typing import Any

from ..models import Model

logger = logging.getLogger(__name__)


class ModelHealthTracker:
    """Tracks per-model health via a circuit breaker and telemetry recording.

    Owns its own copies of ``consecutive_failures`` and ``api_health``.
    Callers read back the current state via the read-only properties below.

    M6 change: dicts are no longer shared by reference with the engine.
    Engine reads back through ``tracker.api_health`` / ``tracker.consecutive_failures``.
    """

    def __init__(
        self,
        telemetry: Any,
        dashboard: Any,
        adaptive_router: Any,
        state_mgr: Any,
        circuit_breaker_threshold: int = 3,
        initial_consecutive_failures: dict[Model, int] | None = None,
        initial_api_health: dict[Model, bool] | None = None,
    ) -> None:
        self._telemetry = telemetry
        self._consecutive_failures: dict[Model, int] = dict(
            initial_consecutive_failures or {}
        )
        self._api_health: dict[Model, bool] = dict(initial_api_health or {})
        self._dashboard = dashboard
        self._adaptive_router = adaptive_router
        self._state_mgr = state_mgr
        self._threshold = circuit_breaker_threshold

    # ------------------------------------------------------------------ #
    # Read-only properties (defensive copies so callers can't mutate)
    # ------------------------------------------------------------------ #

    @property
    def consecutive_failures(self) -> dict[Model, int]:
        """Snapshot of current consecutive-failure counts."""
        return dict(self._consecutive_failures)

    @property
    def api_health(self) -> dict[Model, bool]:
        """Snapshot of current model health flags."""
        return dict(self._api_health)

    def update_from_persisted_state(
        self,
        consecutive_failures: dict[Model, int],
        api_health: dict[Model, bool],
    ) -> None:
        """Merge persisted circuit-breaker state into the tracker's own dicts.

        Called by the engine after ``_load_circuit_breaker_state()`` so the
        tracker reflects crash-recovery values without needing a shared reference.
        """
        self._consecutive_failures.update(consecutive_failures)
        self._api_health.update(api_health)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def record_success(self, model: Model, response: Any) -> None:
        """Record a successful API call; reset the circuit breaker counter."""
        self._consecutive_failures[model] = 0

        if self._adaptive_router is not None:
            await self._adaptive_router.record_success(model)
            await self._adaptive_router.record_latency(model, response.latency_ms)

        if self._dashboard is not None:
            try:
                self._dashboard.on_model_success(model)
            except Exception as exc:
                logger.debug("Dashboard notification failed: %s", exc)

        self._telemetry.record_call(
            model,
            latency_ms=response.latency_ms,
            cost_usd=response.cost_usd,
            success=True,
        )

    async def record_failure(self, model: Model, error: Exception | None = None) -> None:
        """Record a failed API call.

        - 401 / 404 errors mark the model unhealthy immediately (permanent failures).
        - Transient errors increment the consecutive-failure counter; when the
          counter reaches the threshold the model is also marked unhealthy.
        """
        if self._dashboard is not None:
            try:
                self._dashboard.on_model_failure(model)
            except Exception as exc:
                logger.debug("Dashboard notification failed: %s", exc)

        error_str = str(error) if error else ""
        is_permanent = self._is_permanent_error(error_str)

        if is_permanent:
            if self._api_health.get(model, True):
                self._api_health[model] = False
                logger.warning(
                    "Model %s marked unhealthy immediately: %s",
                    model.value,
                    self._permanent_reason(error_str),
                )
            # Permanently disable in the adaptive router for auth failures
            if self._adaptive_router is not None and (
                "401" in error_str or "invalid_authentication" in error_str.lower()
            ):
                await self._adaptive_router.record_auth_failure(model)
            self._telemetry.record_call(model, latency_ms=0.0, cost_usd=0.0, success=False)
            return

        # Transient failure path
        self._consecutive_failures[model] = self._consecutive_failures.get(model, 0) + 1
        try:
            await self._state_mgr.save_circuit_breaker_state(
                model.value, self._consecutive_failures[model]
            )
        except Exception as cb_err:
            logger.debug(
                "Could not persist circuit breaker state for %s: %s", model.value, cb_err
            )

        if self._adaptive_router is not None and self._is_timeout(error_str, error):
            await self._adaptive_router.record_timeout(model)

        self._telemetry.record_call(model, latency_ms=0.0, cost_usd=0.0, success=False)

        if self._consecutive_failures[model] >= self._threshold:
            if self._api_health.get(model, True):
                self._api_health[model] = False
                logger.warning(
                    "Circuit breaker tripped for %s after %d consecutive failures",
                    model.value,
                    self._consecutive_failures[model],
                )

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _is_permanent_error(error_str: str) -> bool:
        lower = error_str.lower()
        return (
            "401" in error_str
            or "invalid_authentication" in lower
            or "404" in error_str
            or "not found" in lower
            or ("400" in error_str and "invalid_request_error" in lower)
        )

    @staticmethod
    def _permanent_reason(error_str: str) -> str:
        if "401" in error_str:
            return "auth error (401) — check your API key"
        if "404" in error_str:
            return "model not found (404) — check model name"
        return f"invalid request (400)"

    @staticmethod
    def _is_timeout(error_str: str, error: Exception | None) -> bool:
        lower = error_str.lower()
        return (
            "timeout" in lower
            or "timed out" in lower
            or "asyncio.timeouterror" in lower
            or "TimeoutError" in (type(error).__name__ if error else "")
        )
