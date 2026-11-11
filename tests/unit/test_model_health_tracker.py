"""
Unit tests for orchestrator.application.model_health_tracker.ModelHealthTracker

M6 update: constructor no longer accepts consecutive_failures/api_health by
reference.  Tests now read state back through the tracker's own properties.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

pytestmark = pytest.mark.unit

from orchestrator.models import Model

pytestmark = pytest.mark.asyncio


# ─────────────────────────────────────────────────────────────────────────────
# Helpers / fixtures
# ─────────────────────────────────────────────────────────────────────────────


def _make_response(
    latency_ms: float = 200.0,
    cost_usd: float = 0.001,
    input_tokens: int = 100,
    output_tokens: int = 50,
):
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

    tracker = ModelHealthTracker(
        telemetry=telemetry,
        dashboard=None,
        adaptive_router=adaptive_router,
        state_mgr=state_mgr,
        circuit_breaker_threshold=threshold,
        initial_consecutive_failures=initial_failures,
    )
    return tracker, telemetry, adaptive_router, state_mgr


# ─────────────────────────────────────────────────────────────────────────────
# record_success tests
# ─────────────────────────────────────────────────────────────────────────────


async def test_record_success_resets_failure_counter():
    tracker, _, _, _ = _make_tracker(initial_failures={Model.GPT_4O_MINI: 2})
    model = Model.GPT_4O_MINI

    resp = _make_response()
    await tracker.record_success(model, resp)

    assert tracker.consecutive_failures.get(model, 0) == 0


async def test_record_success_calls_telemetry():
    tracker, telemetry, _, _ = _make_tracker()
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
    tracker, _, adaptive_router, _ = _make_tracker()
    model = Model.GPT_4O_MINI
    resp = _make_response(latency_ms=50.0)

    await tracker.record_success(model, resp)

    adaptive_router.record_success.assert_awaited_once_with(model)
    adaptive_router.record_latency.assert_awaited_once_with(model, 50.0)


async def test_record_success_clears_the_persisted_count():
    """The saved count must go back to 0, or the next start re-opens the breaker."""
    tracker, _, _, state_mgr = _make_tracker(threshold=3)
    model = Model.GPT_4O_MINI
    await tracker.record_failure(model, error=Exception("transient"))
    state_mgr.save_circuit_breaker_state.reset_mock()

    await tracker.record_success(model, _make_response())

    state_mgr.save_circuit_breaker_state.assert_awaited_once_with(model.value, 0)


async def test_record_success_does_not_write_when_nothing_to_clear():
    tracker, _, _, state_mgr = _make_tracker()

    await tracker.record_success(Model.GPT_4O_MINI, _make_response())

    state_mgr.save_circuit_breaker_state.assert_not_awaited()


async def test_record_success_closes_a_tripped_breaker():
    tracker, _, _, _ = _make_tracker(threshold=3)
    model = Model.GPT_4O_MINI
    for _ in range(3):
        await tracker.record_failure(model, error=Exception("transient"))
    assert tracker.api_health[model] is False

    await tracker.record_success(model, _make_response())

    assert tracker.api_health[model] is True


async def test_failures_split_by_a_success_do_not_trip():
    """'Consecutive' means consecutive: fail, fail, ok, fail, fail stays closed."""
    tracker, _, _, _ = _make_tracker(threshold=3)
    model = Model.GPT_4O_MINI
    for _ in range(2):
        await tracker.record_failure(model, error=Exception("transient"))
    await tracker.record_success(model, _make_response())
    for _ in range(2):
        await tracker.record_failure(model, error=Exception("transient"))

    assert tracker.api_health.get(model, True) is True


async def test_load_state_only_asks_for_recent_trips():
    tracker, _, _, state_mgr = _make_tracker(threshold=3)
    state_mgr.load_circuit_breaker_state = AsyncMock(return_value={})

    await tracker.load_state()

    state_mgr.load_circuit_breaker_state.assert_awaited_once_with(
        max_age_seconds=tracker.TRIP_TTL_SECONDS
    )


async def test_state_manager_drops_stale_circuit_breaker_rows(tmp_path, monkeypatch):
    """A trip from a long-finished outage must not exclude the model forever."""
    from orchestrator.infrastructure import state as state_module
    from orchestrator.infrastructure.state import StateManager

    sm = StateManager(db_path=tmp_path / "s.db")
    try:
        await sm.save_circuit_breaker_state("old-model", 5)
        saved_at = state_module.time.time()
        monkeypatch.setattr(state_module.time, "time", lambda: saved_at + 7200)
        await sm.save_circuit_breaker_state("new-model", 4)

        assert await sm.load_circuit_breaker_state() == {"old-model": 5, "new-model": 4}
        assert await sm.load_circuit_breaker_state(max_age_seconds=1800) == {"new-model": 4}
    finally:
        await sm.close()


# ─────────────────────────────────────────────────────────────────────────────
# record_failure tests
# ─────────────────────────────────────────────────────────────────────────────


async def test_record_failure_increments_counter():
    tracker, _, _, _ = _make_tracker()
    model = Model.GPT_4O_MINI

    await tracker.record_failure(model, error=Exception("timeout"))

    assert tracker.consecutive_failures.get(model, 0) == 1


async def test_record_failure_does_not_trip_below_threshold():
    tracker, _, _, _ = _make_tracker(threshold=3)
    model = Model.GPT_4O_MINI

    await tracker.record_failure(model, error=Exception("err"))
    await tracker.record_failure(model, error=Exception("err"))  # 2 failures

    assert tracker.api_health.get(model, True) is True  # Not yet at threshold


async def test_record_failure_trips_circuit_breaker_at_threshold():
    tracker, _, _, _ = _make_tracker(threshold=3)
    model = Model.GPT_4O_MINI

    for _ in range(3):
        await tracker.record_failure(model, error=Exception("transient"))

    assert tracker.api_health.get(model, True) is False


async def test_record_failure_401_marks_unhealthy_immediately():
    tracker, _, _, _ = _make_tracker(threshold=3)
    model = Model.GPT_4O_MINI

    await tracker.record_failure(model, error=Exception("401 unauthorized"))

    # Should mark unhealthy immediately without incrementing counter
    assert tracker.api_health.get(model, True) is False
    assert tracker.consecutive_failures.get(model, 0) == 0  # Not incremented for permanent


async def test_record_failure_persists_circuit_breaker_state():
    tracker, _, _, state_mgr = _make_tracker(threshold=3)
    model = Model.GPT_4O_MINI

    await tracker.record_failure(model, error=Exception("transient"))

    state_mgr.save_circuit_breaker_state.assert_awaited_once_with(model.value, 1)


async def test_record_failure_404_marks_unhealthy_immediately():
    tracker, _, _, _ = _make_tracker(threshold=3)
    model = Model.GPT_4O_MINI

    await tracker.record_failure(model, error=Exception("404 model not found"))

    assert tracker.api_health.get(model, True) is False


# ─────────────────────────────────────────────────────────────────────────────
# M6: state ownership
# ─────────────────────────────────────────────────────────────────────────────


async def test_tracker_owns_its_dicts():
    """M6: the tracker must not share its dicts with the caller."""
    from orchestrator.application.model_health_tracker import ModelHealthTracker

    init_failures = {Model.GPT_4O_MINI: 0}
    tracker = ModelHealthTracker(
        telemetry=MagicMock(),
        dashboard=None,
        adaptive_router=None,
        state_mgr=MagicMock(save_circuit_breaker_state=AsyncMock()),
        initial_consecutive_failures=init_failures,
    )
    # Mutating the original dict must NOT affect the tracker
    init_failures[Model.GPT_4O_MINI] = 99
    assert tracker.consecutive_failures.get(Model.GPT_4O_MINI, 0) == 0


async def test_update_from_persisted_state():
    """M6: update_from_persisted_state merges loaded state correctly."""
    tracker, _, _, _ = _make_tracker(threshold=3)
    model = Model.GPT_4O_MINI

    tracker.update_from_persisted_state(
        consecutive_failures={model: 2},
        api_health={model: True},
    )

    assert tracker.consecutive_failures[model] == 2
    assert tracker.api_health[model] is True
