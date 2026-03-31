"""
Unified Events System
=====================
Single event bus for all orchestrator events.
"""

from .core import (
    BudgetWarningEvent,
    CapabilityCompletedEvent,
    CapabilityUsedEvent,
    DomainEvent,
    EventType,
    FallbackTriggeredEvent,
    ModelSelectedEvent,
    ProjectCompletedEvent,
    ProjectStartedEvent,
    TaskCompletedEvent,
    TaskFailedEvent,
    TaskProgressEvent,
    TaskStartedEvent,
    UnifiedEventBus,
    get_current_project,
    get_event_bus,
    log_capability_use,
    set_current_project,
)

__all__ = [
    "UnifiedEventBus",
    "get_event_bus",
    "DomainEvent",
    "EventType",
    "ProjectStartedEvent",
    "ProjectCompletedEvent",
    "TaskStartedEvent",
    "TaskCompletedEvent",
    "TaskFailedEvent",
    "TaskProgressEvent",
    "CapabilityUsedEvent",
    "CapabilityCompletedEvent",
    "BudgetWarningEvent",
    "ModelSelectedEvent",
    "FallbackTriggeredEvent",
    "log_capability_use",
    "set_current_project",
    "get_current_project",
]
