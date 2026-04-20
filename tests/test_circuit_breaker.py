"""Unit tests for orchestrator.circuit_breaker."""

from __future__ import annotations

import asyncio

import pytest

from orchestrator.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpen,
    CircuitState,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


async def _fail(cb: CircuitBreaker, n: int) -> None:
    """Record n consecutive failures."""
    for _ in range(n):
        await cb.record_failure(RuntimeError("simulated failure"))


async def _succeed(cb: CircuitBreaker, n: int) -> None:
    """Record n consecutive successes."""
    for _ in range(n):
        await cb.record_success()


# ─────────────────────────────────────────────────────────────────────────────
# CLOSED state
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_starts_closed():
    cb = CircuitBreaker(name="test", failure_threshold=3)
    assert cb.state == CircuitState.CLOSED
    assert cb.is_closed
    assert not cb.is_open


@pytest.mark.asyncio
async def test_check_passes_when_closed():
    cb = CircuitBreaker(name="test", failure_threshold=3)
    await cb.check()  # should not raise


@pytest.mark.asyncio
async def test_failures_below_threshold_stay_closed():
    cb = CircuitBreaker(name="test", failure_threshold=5)
    await _fail(cb, 4)
    assert cb.state == CircuitState.CLOSED


# ─────────────────────────────────────────────────────────────────────────────
# OPEN state
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_trips_open_at_threshold():
    cb = CircuitBreaker(name="test", failure_threshold=3)
    await _fail(cb, 3)
    assert cb.state == CircuitState.OPEN
    assert cb.is_open
    assert cb.trip_count == 1


@pytest.mark.asyncio
async def test_check_raises_when_open():
    cb = CircuitBreaker(name="test", failure_threshold=2)
    await _fail(cb, 2)
    with pytest.raises(CircuitBreakerOpen) as exc_info:
        await cb.check()
    assert "OPEN" in str(exc_info.value)
    assert exc_info.value.code == "CIRCUIT_BREAKER_OPEN"


@pytest.mark.asyncio
async def test_context_raises_when_open():
    cb = CircuitBreaker(name="test", failure_threshold=2)
    await _fail(cb, 2)
    with pytest.raises(CircuitBreakerOpen):
        async with cb.context():
            pass  # should not reach here


@pytest.mark.asyncio
async def test_time_until_reset_positive_when_open():
    cb = CircuitBreaker(name="test", failure_threshold=2, reset_timeout=60.0)
    await _fail(cb, 2)
    remaining = cb.time_until_reset()
    assert 0 < remaining <= 60.0


@pytest.mark.asyncio
async def test_time_until_reset_zero_when_closed():
    cb = CircuitBreaker(name="test", failure_threshold=5)
    assert cb.time_until_reset() == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# HALF_OPEN / recovery
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_transitions_to_half_open_after_timeout(monkeypatch):
    cb = CircuitBreaker(name="test", failure_threshold=2, reset_timeout=0.01)
    await _fail(cb, 2)
    assert cb.state == CircuitState.OPEN
    await asyncio.sleep(0.05)  # wait for reset_timeout to expire
    await cb.check()  # should not raise; transitions to HALF_OPEN
    assert cb.state == CircuitState.HALF_OPEN


@pytest.mark.asyncio
async def test_closes_after_successes_in_half_open():
    cb = CircuitBreaker(
        name="test",
        failure_threshold=2,
        reset_timeout=0.01,
        success_threshold=2,
    )
    await _fail(cb, 2)
    await asyncio.sleep(0.05)
    await cb.check()  # HALF_OPEN
    await _succeed(cb, 2)
    assert cb.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_reopens_on_failure_in_half_open():
    cb = CircuitBreaker(
        name="test",
        failure_threshold=2,
        reset_timeout=0.01,
        success_threshold=3,
    )
    await _fail(cb, 2)
    await asyncio.sleep(0.05)
    await cb.check()  # HALF_OPEN
    # One failure re-opens
    await cb.record_failure(RuntimeError("still broken"))
    assert cb.state == CircuitState.OPEN
    assert cb.trip_count == 2


# ─────────────────────────────────────────────────────────────────────────────
# context() manager
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_context_records_success_on_clean_exit():
    cb = CircuitBreaker(name="test", failure_threshold=5)
    async with cb.context():
        pass
    assert cb.total_successes == 1
    assert cb.total_failures == 0


@pytest.mark.asyncio
async def test_context_records_failure_on_exception():
    cb = CircuitBreaker(name="test", failure_threshold=5)
    with pytest.raises(ValueError):
        async with cb.context():
            raise ValueError("oops")
    assert cb.total_failures == 1
    assert cb.total_successes == 0


@pytest.mark.asyncio
async def test_context_trips_after_threshold_failures():
    cb = CircuitBreaker(name="test", failure_threshold=3)
    for _ in range(3):
        with pytest.raises(RuntimeError):
            async with cb.context():
                raise RuntimeError("fail")
    assert cb.state == CircuitState.OPEN
    with pytest.raises(CircuitBreakerOpen):
        async with cb.context():
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stats_structure():
    cb = CircuitBreaker(name="my-cb", failure_threshold=5)
    s = cb.stats()
    assert s["name"] == "my-cb"
    assert s["state"] == "closed"
    assert s["trip_count"] == 0
    assert "total_calls" in s
    assert "time_until_reset_s" in s


@pytest.mark.asyncio
async def test_total_counters_accumulate():
    cb = CircuitBreaker(name="test", failure_threshold=10)
    async with cb.context():
        pass
    with pytest.raises(ValueError):
        async with cb.context():
            raise ValueError("err")
    assert cb.total_calls == 2
    assert cb.total_successes == 1
    assert cb.total_failures == 1
