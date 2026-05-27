"""Regression tests for bugs BUG-001 through BUG-004.

Each test reproduces the original failure (fails without patch)
and verifies the fix (passes with patch).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.models import Budget, Task, TaskType
from orchestrator.circuit_breaker import CircuitBreaker, CircuitState

# ═══════════════════════════════════════════════════════════════════════════════
# BUG-001: Budget._get_lock() TOCTOU race
# ═══════════════════════════════════════════════════════════════════════════════


class TestBug001BudgetLockTOCTOU:

    @pytest.mark.asyncio
    async def test_lock_is_shared_by_concurrent_tasks(self):
        """Two concurrent tasks charging budget must share the same lock."""
        budget = Budget(max_usd=10.0)

        async def charge_five():
            for _ in range(5):
                await budget.charge(1.0, "generation")
                await asyncio.sleep(0)

        t1 = asyncio.create_task(charge_five())
        t2 = asyncio.create_task(charge_five())

        await asyncio.gather(t1, t2)

        # Bug scenario: if two different Lock objects were used, spent_usd
        # could be less than 10.0 due to unprotected concurrent writes.
        assert (
            budget.spent_usd == 10.0
        ), f"Expected spent_usd=10.0 (5+5 charges of $1), got {budget.spent_usd}"

    @pytest.mark.asyncio
    async def test_reserve_is_atomic(self):
        """Concurrent reserves must not exceed total budget."""
        budget = Budget(max_usd=5.0)

        async def try_reserve(amount: float) -> bool:
            return await budget.reserve(amount)

        results = await asyncio.gather(
            try_reserve(3.0),
            try_reserve(3.0),  # Should fail — only $5 total
        )

        # Exactly one should succeed (3.0 < 5.0, but 3.0+3.0 > 5.0)
        successes = sum(results)
        assert successes == 1, f"Expected exactly 1 successful reserve, got {successes}"
        assert budget._reserved_usd == 3.0, f"Expected 3.0 reserved, got {budget._reserved_usd}"

    def test_lock_is_eagerly_initialized(self):
        """Budget should have its lock ready immediately after construction."""
        budget = Budget(max_usd=5.0)
        assert budget._lock is not None, "Budget._lock should not be None"
        assert isinstance(budget._lock, asyncio.Lock), "Budget._lock should be asyncio.Lock"


# ═══════════════════════════════════════════════════════════════════════════════
# BUG-002: CircuitBreaker HALF_OPEN probe_in_flight race
# ═══════════════════════════════════════════════════════════════════════════════


class TestBug002CircuitBreakerProbeRace:

    @pytest.mark.asyncio
    async def test_half_open_blocks_second_probe_before_threshold(self):
        """With success_threshold=2, a first probe success must still block
        subsequent callers until threshold is met or the probe fails."""
        cb = CircuitBreaker(
            name="test-probe",
            failure_threshold=1,
            reset_timeout=0.02,
            success_threshold=2,  # Need 2 successes to close
        )

        # Trip the breaker (exception is expected and caught)
        with pytest.raises(ConnectionError):
            async with cb.context():
                raise ConnectionError("simulated failure")

        assert cb.state == CircuitState.OPEN

        # Wait for reset timeout
        await asyncio.sleep(0.03)

        # First probe call — transitions to HALF_OPEN
        await cb.check()
        assert cb.state == CircuitState.HALF_OPEN

        # Record success (successes=1 < threshold=2) — with BUG-002,
        # this would clear probe_in_flight, allowing a second probe.
        await cb.record_success()

        # Second check — must be blocked (probe still in flight)
        from orchestrator.circuit_breaker import CircuitBreakerOpen

        with pytest.raises(CircuitBreakerOpen):
            await cb.check()

    @pytest.mark.asyncio
    async def test_half_open_recovers_after_full_threshold(self):
        """After success_threshold successes, the breaker should close."""
        cb = CircuitBreaker(
            name="test-recover",
            failure_threshold=1,
            reset_timeout=0.02,
            success_threshold=2,
        )

        with pytest.raises(ConnectionError):
            async with cb.context():
                raise ConnectionError("trip")

        await asyncio.sleep(0.03)

        # First probe + success (successes=1)
        await cb.check()
        await cb.record_success()
        assert cb.state == CircuitState.HALF_OPEN

        # Second probe — probe_in_flight was still True (BUG-002 fix),
        # so check() should block additional callers:
        from orchestrator.circuit_breaker import CircuitBreakerOpen

        with pytest.raises(CircuitBreakerOpen):
            await cb.check()

        # Simulate probe failure — this clears probe_in_flight
        # and re-opens the circuit; wait for reset timeout
        await cb.record_failure(TimeoutError("probe timeout"))
        assert cb._state.probe_in_flight is False

        # Wait for reset timeout so check() can probe again
        await asyncio.sleep(0.03)
        await cb.check()
        assert cb.state == CircuitState.HALF_OPEN
        await cb.record_success()
        await cb.record_success()

        assert cb.state == CircuitState.CLOSED


# ═══════════════════════════════════════════════════════════════════════════════
# BUG-003: Evaluator hardcoded latency_ms=0.0
# ═══════════════════════════════════════════════════════════════════════════════


class TestBug003EvaluatorLatency:

    @pytest.mark.asyncio
    async def test_evaluator_telemetry_has_nonzero_latency(self):
        """Evaluator telemetry must include actual wall-clock latency."""
        from orchestrator.application.evaluator import EvaluatorService

        budget = Budget(max_usd=10.0)

        # Mock client that returns a response with non-trivial delay
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.text = '{"score": 0.85, "issues": []}'
        mock_response.cost_usd = 0.01
        mock_response.input_tokens = 50
        mock_response.output_tokens = 30

        async def delayed_call(*args, **kwargs):
            await asyncio.sleep(0.01)  # 10ms minimum delay
            return mock_response

        mock_client.call = delayed_call

        telemetry = MagicMock()
        get_models = MagicMock(return_value=[MagicMock(value="openai/gpt-4o-mini")])

        evaluator = EvaluatorService(
            client=mock_client,
            budget=budget,
            get_models_fn=get_models,
            consistency_runs=1,
            telemetry=telemetry,
        )

        task = Task(id="test_001", type=TaskType.CODE_GEN, prompt="write code")

        await evaluator.evaluate(task, "def foo(): pass")

        # Verify telemetry.record_call was called with non-zero latency
        telemetry.record_call.assert_called_once()
        call_kwargs = telemetry.record_call.call_args[1]
        recorded_latency = call_kwargs.get("latency_ms", 0.0)
        assert recorded_latency > 0.0, f"Expected non-zero latency, got {recorded_latency}"


# ═══════════════════════════════════════════════════════════════════════════════
# BUG-004: __aexit__ null dereference on _telemetry_store
# ═══════════════════════════════════════════════════════════════════════════════


class TestBug004NullTelemetryStore:

    @pytest.mark.asyncio
    async def test_aexit_handles_none_telemetry_store(self):
        """__aexit__ must not raise AttributeError when _telemetry_store is None."""
        from orchestrator.engine import Orchestrator

        orch = Orchestrator()

        # Simulate TelemetryStore import failure
        orch._telemetry_store = None

        # This must not raise AttributeError (the bug)
        try:
            await orch.__aexit__(None, None, None)
        except AttributeError as e:
            pytest.fail(f"__aexit__ raised AttributeError: {e}")
        except Exception:
            # Other exceptions during shutdown are acceptable
            pass

    @pytest.mark.asyncio
    async def test_aexit_logs_correctly_when_telemetry_missing(self):
        """When telemetry_store is None, log should not falsely claim failure."""
        from orchestrator.engine import Orchestrator

        orch = Orchestrator()
        orch._telemetry_store = None

        with patch("orchestrator.engine.logger") as mock_logger:
            try:
                await orch.__aexit__(None, None, None)
            except Exception:
                pass

            # Verify no "Failed to flush telemetry store" warning was emitted
            # when telemetry was never initialized
            warning_messages = [call.args[0] for call in mock_logger.warning.call_args_list]
            assert not any(
                "Failed to flush telemetry store" in msg for msg in warning_messages
            ), "Should not log 'failed to flush' when telemetry was never initialized"

    @pytest.mark.asyncio
    async def test_aexit_still_flushes_when_telemetry_present(self):
        """When _telemetry_store is valid, it should still be flushed."""
        from orchestrator.engine import Orchestrator

        orch = Orchestrator()
        mock_store = MagicMock()
        orch._telemetry_store = mock_store

        try:
            await orch.__aexit__(None, None, None)
        except Exception:
            pass

        # Verify the flush was attempted
        mock_store.flush.assert_called_once()
