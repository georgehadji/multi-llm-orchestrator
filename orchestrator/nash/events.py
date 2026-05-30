"""
Nash Event Bus — Re-export shim
================================
Unified into unified_events/core.py.
This file kept as backward-compat shim.
"""

from ..unified_events.core import (  # noqa: F401
    EventType,
    UnifiedEventBus as NashEventBus,
    get_event_bus,
)

# Nash Specific Event classes re-mapped to DomainEvent base if needed
# For now, we allow them to import the base classes from unified_events
from ..unified_events.core import (  # noqa: F401
    DriftDetectedEvent,
    KnowledgeGraphUpdatedEvent,
    StabilityScoreUpdatedEvent,
    TemplateSelectedEvent,
)
