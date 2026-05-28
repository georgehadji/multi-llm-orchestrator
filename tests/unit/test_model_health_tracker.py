"""
Unit tests for orchestrator.application.model_health_tracker.ModelHealthTracker

P3-2 of REFACTORING_PLAN_V7.md — extracted from engine._record_success/_record_failure.

Write tests FIRST (TDD RED phase), then implement the class.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.models import Model, TaskStatus


pytestmark = pytest.mark.asyncio


# ─────────────────────────────────────────────────────────────────────────────
# Helpers / fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _make_response(latency_ms: float = 200.0, cost_usd: float = 0.001,
                   input_tokens: int = 100, output_tokens: int = 50):
    r = MagicMock()
    r.latency_ms = latency_ms
    r.cost_usd = cost_usd
    r.input_tokens = input_tokens
    r.output_tokens = output_tokens
    return r


def _make_tracker(threshold: int = 3, initial_failures: dict | None = None):
    """Build a ModelHealthTracker with test doubles."""
    from orchestrator.application.model_health_tracker import ModelHealthTracker

    telemetry = MagicMock()
    telemetry.record_call = MagicMock()

    adaptive_router = MagicMock()
    adaptive_router.record_success = AsyncMock()
    adaptive_router.record_latency = AsyncMock()
    adaptive_router.record_failure = AsyncMock()
    adaptive_router.record_timeout = AsyncMock()
    adaptive_router.record_auth_failure = AsyncMock()

    state_mgr = MagicMock()
    state_mgr.save_circuit_breaker_state = AsyncMock()

    consecutive_failures = dict.fromkeys(Model, 0)
    if initial_failures:
        consecutive_failures.update(initial_failures)

    api_health = dict.fromkeys(Model, True)

    tracker = ModelHealthTracker(
        telemetry=telemetry,
        consecutive_failures=consecutive_failures,
        api_health=api_health,
        dashboard=None,
        adaptive_router=adaptive_router,
        state_mgr=state_mgr,
        circuit_breaker_threshold=threshold,
    )
    return tracker, consecutive_failures, api_health, telemetry, adaptive_router, state_mgr


# ─────────────────────────────────────────────────────────────────────────────
# record_success tests
# ─────────────────────────────────────────────────────────────────────────────


async def test_record_success_resets_failure_counter():
    tracker, failures, _, _, _, _ = _make_tracker()
    model = Model.GPT_4O_MINI
    failures[model] = 2  # Pre-existing failures

    resp = _make_response()
    await tracker.record_success(model, resp)

    assert failures[model] == 0


async def test_record_success_calls_telemetry():
    tracker, _, _, telemetry, _, _ = _make_tracker()
    model = Model.GPT_4O_MINI
    resp = _make_response(latency_ms=123.0, cost_usd=0.005)

    await tracker.record_success(model, resp)

    telemetry.record_call.assert_called_once_with(
        model,
        latency_ms=123.0,
        cost_usd=0.005,
        success=True,
    )


async def test_record_success_calls_adaptive_router():
    tracker, _, _, _, adaptive_router, _ = _make_tracker()
    model = Model.GPT_4O_MINI
    resp = _make_response(latency_ms=50.0)

    await tracker.record_success(model, resp)

    adaptive_router.record_success.assert_awaited_once_with(model)
    adaptive_router.record_latency.assert_awaited_once_with(model, 50.0)


# ─────────────────────────────────────────────────────────────────────────────
# record_failure tests
# ─────────────────────────────────────────────────────────────────────────────


async def test_record_failure_increments_counter():
    tracker, failures, _, _, _, _ = _make_tracker()
    model = Model.GPT_4O_MINI

    await tracker.record_failure(model, error=Exception("timeout"))

    assert failures[model] == 1


async def test_record_failure_does_not_trip_below_threshold():
    tracker, _, api_health, _, _, _ = _make_tracker(threshold=3)
    model = Model.GPT_4O_MINI

    await tracker.record_failure(model, error=Exception("err"))
    await tracker.record_failure(model, error=Exception("err"))  # 2 failures

    assert api_health[model] is True  # Not yet at threshold


async def test_record_failure_trips_circuit_breaker_at_threshold():
    tracker, _, api_health, _, _, _ = _make_tracker(threshold=3)
    model = Model.GPT_4O_MINI

    for _ in range(3):
        await tracker.record_failure(model, error=Exception("transient"))

    assert api_health[model] is False


async def test_record_failure_401_marks_unhealthy_immediately():
    tracker, failures, api_health, _, _, _ = _make_tracker(threshold=3)
    model = Model.GPT_4O_MINI

    await tracker.record_failure(model, error=Exception("401 unauthorized"))

    # Should mark unhealthy immediately without incrementing counter
    assert api_health[model] is False
    assert failures[model] == 0  # Not incremented for permanent errors


async def test_record_failure_persists_circuit_breaker_state():
    tracker, _, _, _, _, state_mgr = _make_tracker(threshold=3)
    model = Model.GPT_4O_MINI

    await tracker.record_failure(model, error=Exception("transient"))

    state_mgr.save_circuit_breaker_state.assert_awaited_once_with(model.value, 1)


async def test_record_failure_404_marks_unhealthy_immediately():
    tracker, _, api_health, _, _, _ = _make_tracker(threshold=3)
    model = Model.GPT_4O_MINI

    await tracker.record_failure(model, error=Exception("404 model not found"))

    assert api_health[model] is False
