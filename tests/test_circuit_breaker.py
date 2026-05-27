"""
Tests for orchestrator/circuit_breaker.py — CircuitBreaker state machine.
"""

from __future__ import annotations

import asyncio
import pytest

from orchestrator.circuit_breaker import (
    CircuitBreaker, CircuitBreakerOpen, CircuitState, CircuitBreakerRegistry,
)


class TestCircuitBreaker:
    """Unit tests for CircuitBreaker."""

    # ── Initial State ──
    def test_initial_state_closed(self, circuit_breaker):
        """New breaker must be CLOSED."""
        assert circuit_breaker.state == CircuitState.CLOSED

    def test_check_in_closed_allows_call(self, circuit_breaker):
        """check() in CLOSED must not raise."""
        asyncio.run(circuit_breaker.check())

    # ── Transition to OPEN ──
    @pytest.mark.asyncio
    async def test_failures_trip_to_open(self, circuit_breaker):
        """Sufficient failures must transition to OPEN."""
        with pytest.raises(ConnectionError):
            async with circuit_breaker.context():
                raise ConnectionError("fail")
        with pytest.raises(ConnectionError):
            async with circuit_breaker.context():
                raise ConnectionError("fail")
        assert circuit_breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_context_manager_raises(self, circuit_breaker):
        """Check that context manager reraises errors."""
        with pytest.raises(ConnectionError):
            async with circuit_breaker.context():
                raise ConnectionError("fail")
        await asyncio.sleep(0.01)
        with pytest.raises(CircuitBreakerOpen):
            await circuit_breaker.check()

    # ── HALF_OPEN Probe ──
    @pytest.mark.asyncio
    async def test_half_open_blocks_second_probe(self):
        """BUG-002 fix: Single probe in HALF_OPEN."""
        cb = CircuitBreaker(name="test", failure_threshold=1, reset_timeout=0.02, success_threshold=2)
        with pytest.raises(ConnectionError):
            async with cb.context():
                raise ConnectionError("trip")
        await asyncio.sleep(0.03)
        await cb.check()
        await cb.record_success()
        with pytest.raises(CircuitBreakerOpen):
            await cb.check()

    # ── Edge Cases ──
    def test_circuit_breaker_without_name(self):
        """Must accept empty name."""
        cb = CircuitBreaker(name="")
        assert cb.name == ""

    @pytest.mark.parametrize("threshold", [1, 3, 10])
    def test_custom_failure_threshold(self, threshold):
        """Must accept various failure thresholds."""
        cb = CircuitBreaker(name="test", failure_threshold=threshold)
        assert cb.failure_threshold == threshold

    # ── Registry ──
    def test_registry_get_or_create(self):
        """Registry must create breakers on demand."""
        registry = CircuitBreakerRegistry()
        import asyncio; cb = asyncio.run(registry.get("model-key"))
        assert "model-key" in cb.name


class TestCircuitBreakerEdgeCases:
    """Edge case tests."""

    @pytest.mark.asyncio
    async def test_record_failure_in_open_is_noop(self):
        """BUG-001: record_failure in OPEN must be no-op."""
        cb = CircuitBreaker(name="test", failure_threshold=1, reset_timeout=0.02)
        with pytest.raises(ConnectionError):
            async with cb.context():
                raise ConnectionError("trip")
        assert cb.state == CircuitState.OPEN
        prev_failures = cb._state.failures
        await cb.record_failure(RuntimeError("extra"))
        assert cb._state.failures == prev_failures

    @pytest.mark.asyncio
    async def test_success_in_closed_resets_failures(self):
        """Success must reset failure counter in CLOSED."""
        cb = CircuitBreaker(name="test", failure_threshold=3)
        await cb.record_failure(RuntimeError("e1"))
        assert cb._state.failures == 1
        await cb.record_success()
        assert cb._state.failures == 0
