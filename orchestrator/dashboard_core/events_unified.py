# Compatibility shim — dashboard_core/core.py imports from here.
# Canonical source is orchestrator.unified_events.core.
from ..unified_events.core import DomainEvent, get_event_bus

__all__ = ["DomainEvent", "get_event_bus"]
