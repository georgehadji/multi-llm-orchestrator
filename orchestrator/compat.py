"""
Backward Compatibility Layer (v6.0)
=====================================

Provides backward compatibility for v5.x code migrating to v6.0.
"""

from __future__ import annotations

# Import compatibility aliases
try:
    from .streaming import ProjectEventBus
except ImportError:
    ProjectEventBus = None

try:
    from .streaming import PipelineEvent as StreamEvent
except ImportError:
    StreamEvent = None

# Event aliases — consolidated into unified_events.core
try:
    from .unified_events.core import (
        ProjectCompletedEvent as ProjectCompleted,
        ProjectStartedEvent as ProjectStarted,
        TaskCompletedEvent as TaskCompleted,
        TaskStartedEvent as TaskStarted,
    )
except ImportError:
    ProjectStarted = None
    TaskStarted = None
    TaskCompleted = None
    ProjectCompleted = None

# Dashboard aliases — all consolidated into dashboard_core/mission_control.py
from .dashboard_core.mission_control import (
    MissionControlView as DashboardView,
    create_view as run_dashboard,
)


def print_migration_guide():
    """Print migration guide for v5.x to v6.0."""
    print("""
    Migration Guide: v5.x → v6.0
    ============================

    1. Event System
       Old: from orchestrator.events import DomainEvent
       New: from orchestrator.unified_events.core import DomainEvent

    2. Dashboard
       Old: from orchestrator.dashboard_live import run_live_dashboard
       New: from orchestrator.dashboard_core.mission_control import create_view

    See: MIGRATION_GUIDE_v6.md for details.
    """)


__all__ = [
    "ProjectEventBus",
    "StreamEvent",
    "ProjectStarted",
    "TaskStarted",
    "TaskCompleted",
    "ProjectCompleted",
    "DashboardView",
    "run_dashboard",
    "print_migration_guide",
]
