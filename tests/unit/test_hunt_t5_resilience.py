"""
T5 (resilience state machines) proof-of-defect and no-regression tests.

One VERIFIED DEFECT from docs/hunts/t5-resilience/inventory.md:

C1 — orchestrator/operations/circuit_breaker.py had silently diverged from
     the canonical orchestrator/circuit_breaker.py (the one every live
     construction site actually uses: engine_core/container.py,
     infrastructure/llm_client.py, engine_slimming.py). The operations/
     copy was missing:
       (a) the already-OPEN no-op guard in record_failure(), and
       (b) critically, `self._state.probe_in_flight = True` in check()'s
           HALF_OPEN branch — the fix (BUG-001/BUG-002, per the root
           file's own comments) that ensures only ONE probe call is let
           through per HALF_OPEN window, not every concurrent caller.
     Because orchestrator/operations/resilience.py imports
     CircuitBreakerOpen from this module (`from .circuit_breaker import
     CircuitBreakerOpen, CircuitBreakerRegistry`), and every live breaker
     is actually an instance of the ROOT module's classes, a real
     CircuitBreakerOpen raised by a live breaker was a *different,
     unrelated exception class* than the one run_with_resilience's
     `except CircuitBreakerOpen` clause checked for — it would not have
     been caught, defeating the "skip immediately, no retries" fail-fast
     behavior run_with_resilience explicitly documents.
"""

from __future__ import annotations

import asyncio

import pytest

pytestmark = pytest.mark.unit


# --- C1 -----------------------------------------------------------------


@pytest.mark.unit
def test_c1_operations_circuit_breaker_is_canonical():
    from orchestrator.circuit_breaker import CircuitBreaker as canonical
    from orchestrator.circuit_breaker import CircuitBreakerOpen as canonical_open
    from orchestrator.operations.circuit_breaker import CircuitBreaker as via_ops
    from orchestrator.operations.circuit_breaker import CircuitBreakerOpen as via_ops_open

    assert via_ops is canonical
    assert via_ops_open is canonical_open


@pytest.mark.unit
def test_c1_resilience_module_catches_the_real_circuit_breaker_open():
    """The core defect: a CircuitBreakerOpen raised by the canonical
    CircuitBreaker (what every live breaker actually is) must be an
    instance of the exception class orchestrator.operations.resilience
    catches."""
    from orchestrator.circuit_breaker import CircuitBreakerOpen as RealOpen
    from orchestrator.operations.resilience import CircuitBreakerOpen as CaughtBy

    exc = RealOpen("test-breaker", reset_in=5.0)
    assert isinstance(exc, CaughtBy)


@pytest.mark.asyncio
async def test_c1_half_open_allows_only_one_concurrent_probe():
    """Real trigger, not simulated: two concurrent callers hit check() while
    the breaker is already HALF_OPEN with no probe currently in flight —
    exactly one must be let through, the other rejected with
    CircuitBreakerOpen. This is the exact property probe_in_flight exists
    to guarantee (root circuit_breaker.py:187-189); the pre-fix
    operations/circuit_breaker.py never set the flag in this branch, so
    every concurrent caller was admitted.

    Arranges the HALF_OPEN-with-no-probe-in-flight state directly (rather
    than via record_failure(), whose threshold/counter interactions would
    otherwise re-trip the breaker back to OPEN) — this isolates exactly
    the check() behavior under test.
    """
    from orchestrator.operations.circuit_breaker import (
        CircuitBreaker,
        CircuitBreakerOpen,
        CircuitState,
    )

    cb = CircuitBreaker(name="t5-c1")
    cb._state.state = CircuitState.HALF_OPEN
    cb._state.probe_in_flight = False

    results = await asyncio.gather(cb.check(), cb.check(), return_exceptions=True)

    allowed = [r for r in results if r is None]
    rejected = [r for r in results if isinstance(r, CircuitBreakerOpen)]
    assert len(allowed) == 1, f"expected exactly one probe admitted, got {len(allowed)}"
    assert len(rejected) == 1, f"expected exactly one caller rejected, got {len(rejected)}"
