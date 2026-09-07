"""Guard: a circuit breaker trip actually publishes `circuit_breaker.tripped`.

Why this exists
----------------
`circuit_breaker._open()` published trip events via `from .events import
get_event_bus` — a name that does not exist in `orchestrator.events` (that
package holds hooks/triggers/ab-testing, not the unified event bus). The
import raised `ImportError` on every single trip, silently swallowed by a
bare `except Exception: pass`. Circuit-breaker trip events had never reached
a subscriber. See R3 in test_suite_remediation_plan.md.
"""

from __future__ import annotations

import asyncio

import pytest

from orchestrator.circuit_breaker import CircuitBreaker
from orchestrator.unified_events.core import EventType, UnifiedEventBus

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_event_bus_singleton():
    UnifiedEventBus._instance = None
    yield
    UnifiedEventBus._instance = None


@pytest.mark.asyncio
async def test_trip_publishes_circuit_breaker_open_event():
    received: list = []
    seen = asyncio.Event()

    async def _capture(event):
        received.append(event)
        seen.set()

    bus = await UnifiedEventBus.get_instance()
    bus.subscribe(_capture)
    await bus.start()
    try:
        breaker = CircuitBreaker(name="test-provider", failure_threshold=1)
        await breaker.record_failure(RuntimeError("boom"))

        # The trip-event publish runs as a scheduled task and is then
        # processed asynchronously by the bus's own queue loop.
        await asyncio.wait_for(seen.wait(), timeout=2.0)
    finally:
        await bus.stop()

    assert received, "no event was published when the breaker tripped"
    event = received[-1]
    assert event.event_type == EventType.CIRCUIT_BREAKER_OPEN
    assert event.metadata["name"] == "test-provider"
    assert event.metadata["failures"] == 1
