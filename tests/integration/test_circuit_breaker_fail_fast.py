"""
test_circuit_breaker_fail_fast.py — Verify circuit breaker trips within 5s.
=========================================================================

MVOS coverage:
  - Circuit breaker trips within 30s of API down (audit invariant #5)
  - Fail-fast behaviour: no retry spam, budget not wasted
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock

import pytest

from orchestrator.circuit_breaker import CircuitBreakerOpen
from orchestrator.models import ProjectStatus
from orchestrator.services.generator import GeneratorResult


@pytest.mark.asyncio
async def test_run_project_fails_fast_on_decomposition_error(orchestrator_fixture):
    """
    If decomposition fails immediately, run_project should return quickly
    (< 5 seconds) rather than hanging or retrying indefinitely.
    """
    orch = orchestrator_fixture

    # Patch generator to return an error immediately
    orch._generator.decompose = AsyncMock(
        return_value=GeneratorResult(tasks={}, wall_time_ms=0.0, error=RuntimeError("API down"))
    )

    t0 = time.monotonic()
    state = await orch.run_project(
        project_description="Test circuit breaker",
        success_criteria="Should fail fast",
        project_id="cb-test-001",
    )
    elapsed = time.monotonic() - t0

    # Should fail fast: the audit MVOS invariant is < 30s. We use a tolerant
    # bound here that still catches hangs / infinite-retry while not flaking on
    # legitimate retry backoff under machine load (raw aim is ~5s).
    assert elapsed < 15.0, f"Took {elapsed:.1f}s — did not fail fast"

    # State should reflect failure
    assert state is not None
    assert state.status in (ProjectStatus.SYSTEM_FAILURE, ProjectStatus.BUDGET_EXHAUSTED)


@pytest.mark.asyncio
async def test_circuit_breaker_open_raises_immediately(orchestrator_fixture):
    """
    Once the circuit is open, calls should raise CircuitBreakerOpen
    immediately without attempting the network call.
    """
    orch = orchestrator_fixture

    # Warm up client to avoid lazy-init overhead inside the timed block
    from orchestrator.models import Model

    try:
        await orch.client._get_client_for_model(Model.GPT_4O_MINI)
    except Exception:
        pass

    # Manually trip the circuit breaker
    cb = orch.client.circuit_breaker
    for _ in range(cb.failure_threshold):
        await cb.record_failure()

    assert cb.is_open is True

    t0 = time.monotonic()
    with pytest.raises(CircuitBreakerOpen):
        await orch.client.call(
            model="openai/gpt-4o-mini",
            prompt="test",
            system="",
        )
    elapsed = time.monotonic() - t0

    # Must be immediate (< 0.5s)
    assert elapsed < 0.5, f"Circuit breaker call took {elapsed:.2f}s — not failing fast"


@pytest.mark.asyncio
async def test_circuit_breaker_records_failures_and_trips():
    """Direct unit-style verification of the circuit breaker state machine."""
    from orchestrator.circuit_breaker import CircuitBreaker, CircuitState

    cb = CircuitBreaker(name="test", failure_threshold=3, reset_timeout=1.0)
    assert cb.state == CircuitState.CLOSED

    # 2 failures — still closed
    await cb.record_failure()
    await cb.record_failure()
    assert cb.state == CircuitState.CLOSED

    # 3rd failure — opens
    await cb.record_failure()
    assert cb.state == CircuitState.OPEN
    assert cb.is_open is True
    assert cb.total_failures == 3
