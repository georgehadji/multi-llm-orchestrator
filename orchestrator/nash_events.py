"""
Nash Stability Event System — Backward-compatibility shim
============================================================
Canonical event bus lives in orchestrator/unified_events/core.py.
This module is a re-export shim for callers still importing from .nash_events.
"""

from .unified_events.core import (
    DomainEvent,
    EventType,
    UnifiedEventBus,
    get_event_bus,
)

# Legacy aliases for backward compatibility
NashEventBus = UnifiedEventBus
NashEvent = DomainEvent

__all__ = [
    "DomainEvent",
    "EventType",
    "NashEventBus",
    "NashEvent",
    "UnifiedEventBus",
    "get_event_bus",
]
