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

# Event aliases
try:
    from .events import (
        ProjectCompletedEvent as ProjectCompleted,
    )
    from .events import (
        ProjectStartedEvent as ProjectStarted,
    )
    from .events import (
        TaskCompletedEvent as TaskCompleted,
    )
    from .events import (
        TaskStartedEvent as TaskStarted,
    )
except ImportError:
    ProjectStarted = None
    TaskStarted = None
    TaskCompleted = None
    ProjectCompleted = None

# Dashboard aliases
try:
    from .dashboard_live import run_live_dashboard
except ImportError:
    run_live_dashboard = None

try:
    from .dashboard_mission_control import run_mission_control
except ImportError:
    run_mission_control = None

try:
    from .dashboard_antd import run_ant_design_dashboard
except ImportError:
    run_ant_design_dashboard = None


def print_migration_guide():
    """Print migration guide for v5.x to v6.0."""
    print("""
    Migration Guide: v5.x → v6.0
    ============================

    1. Event System
       Old: from orchestrator.streaming import ProjectEventBus
       New: from orchestrator import get_event_bus

    2. Dashboard
       Old: from orchestrator import run_live_dashboard
       New: from orchestrator import run_dashboard

    3. Events
       Old: from orchestrator.streaming import ProjectStarted
       New: from orchestrator.events import ProjectStartedEvent

    See: MIGRATION_GUIDE_v6.md for details.
    """)


__all__ = [
    "ProjectEventBus",
    "StreamEvent",
    "ProjectStarted",
    "TaskStarted",
    "TaskCompleted",
    "ProjectCompleted",
    "run_live_dashboard",
    "run_mission_control",
    "run_ant_design_dashboard",
    "print_migration_guide",
]
