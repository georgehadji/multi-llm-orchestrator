"""
Backward Compatibility Layer
============================
Save relevant parts to orchestrator/__init__.py or orchestrator/compat.py

This module provides backward compatibility for code using the old API.
It translates old calls to the new unified system.
"""

from __future__ import annotations

import warnings
import functools
from typing import Any, Optional

# ═══════════════════════════════════════════════════════════════════════════════
# Deprecation Warnings
# ═══════════════════════════════════════════════════════════════════════════════


def deprecated(old_name: str, new_name: str, removal_version: str = "7.0"):
    """Decorator to mark functions as deprecated."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"{old_name} is deprecated and will be removed in v{removal_version}. "
                f"Use {new_name} instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        return wrapper

    return decorator


# ═══════════════════════════════════════════════════════════════════════════════
# Dashboard Compatibility
# ═══════════════════════════════════════════════════════════════════════════════


@deprecated("dashboard_live.run_live_dashboard()", "dashboard_core.run_dashboard(view='live')")
def run_live_dashboard(*args, **kwargs):
    """Backward compat for live dashboard."""
    from .dashboard_core_core import run_dashboard

    return run_dashboard(view="mission-control", *args, **kwargs)


@deprecated(
    "dashboard_mission_control.run_mission_control()",
    "dashboard_core.run_dashboard(view='mission-control')",
)
def run_mission_control(*args, **kwargs):
    """Backward compat for mission control dashboard."""
    from .dashboard_core_core import run_dashboard

    return run_dashboard(view="mission-control", *args, **kwargs)


@deprecated(
    "dashboard_antd.run_ant_design_dashboard()", "dashboard_core.run_dashboard(view='ant-design')"
)
def run_ant_design_dashboard(*args, **kwargs):
    """Backward compat for Ant Design dashboard."""
    from .dashboard_core_core import run_dashboard

    return run_dashboard(view="ant-design", *args, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════════
# Events Compatibility
# ═══════════════════════════════════════════════════════════════════════════════


# Old imports that now point to unified system
@deprecated("streaming.ProjectEventBus", "unified_events.UnifiedEventBus")
class ProjectEventBus:
    """Backward compat for streaming.ProjectEventBus."""

    def __init__(self):
        self._bus = None

    async def _get_bus(self):
        if self._bus is None:
            from .unified_events_core import UnifiedEventBus

            self._bus = await UnifiedEventBus.get_instance()
        return self._bus

    async def publish(self, event):
        bus = await self._get_bus()
        await bus.publish(event)

    def subscribe(self):
        """Return async iterator for events."""
        import asyncio

        bus = asyncio.get_event_loop().run_until_complete(self._get_bus())
        return bus.subscribe_iter()


@deprecated("hooks.HookRegistry", "unified_events.UnifiedEventBus.subscribe()")
class HookRegistry:
    """Backward compat for hooks.HookRegistry."""

    def __init__(self):
        self._callbacks = {}

    def add(self, event: str, callback):
        """Add event handler."""
        import asyncio
        from .unified_events_core import EventType

        async def subscribe():
            bus = await self._get_bus()

            async def handler(evt):
                if evt.event_type.name == event:
                    await callback(**evt.metadata)

            bus.subscribe(handler)

        asyncio.create_task(subscribe())

    def fire(self, event: str, **kwargs):
        """Fire event."""
        import asyncio
        from .unified_events_core import EventType, DomainEvent

        async def publish():
            bus = await self._get_bus()
            evt_type = getattr(EventType, event, EventType.INFO)
            event_obj = DomainEvent(
                event_type=evt_type,
                aggregate_id=kwargs.get("project_id", ""),
                metadata=kwargs,
            )
            await bus.publish(event_obj)

        asyncio.create_task(publish())

    async def _get_bus(self):
        from .unified_events_core import UnifiedEventBus

        return await UnifiedEventBus.get_instance()


# Event type mapping
class EventTypeCompat:
    """Backward compat for EventType enum."""

    PROJECT_STARTED = "PROJECT_STARTED"
    PROJECT_COMPLETED = "PROJECT_COMPLETED"
    TASK_STARTED = "TASK_STARTED"
    TASK_COMPLETED = "TASK_COMPLETED"
    TASK_FAILED = "TASK_FAILED"
    BUDGET_WARNING = "BUDGET_WARNING"
    MODEL_SELECTED = "MODEL_SELECTED"


# ═══════════════════════════════════════════════════════════════════════════════
# Capability Logger Compatibility
# ═══════════════════════════════════════════════════════════════════════════════


@deprecated("capability_logger.log_capability()", "unified_events.UnifiedEventBus.log_capability()")
async def log_capability(capability: str, project_id: str = "", **kwargs):
    """Backward compat for capability logging."""
    from .unified_events_core import get_event_bus

    bus = await get_event_bus()
    await bus.log_capability(capability, project_id, kwargs)


@deprecated("capability_logger.CapabilityLogger", "unified_events.UnifiedEventBus")
class CapabilityLogger:
    """Backward compat for CapabilityLogger."""

    async def log(self, capability: str, project_id: str = "", **kwargs):
        await log_capability(capability, project_id, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════════
# Stream Events Compatibility
# ═══════════════════════════════════════════════════════════════════════════════


@deprecated("streaming events (ProjectStarted, TaskStarted, etc.)", "unified_events domain events")
class StreamEvent:
    """Base class for stream events."""

    pass


class ProjectStarted(StreamEvent):
    """Backward compat."""

    def __init__(self, project_id: str, total_tasks: int, budget_usd: float):
        self.project_id = project_id
        self.total_tasks = total_tasks
        self.budget_usd = budget_usd


class TaskStarted(StreamEvent):
    """Backward compat."""

    def __init__(self, task_id: str, task_type: str, iteration: int = 1):
        self.task_id = task_id
        self.task_type = task_type
        self.iteration = iteration


class TaskCompleted(StreamEvent):
    """Backward compat."""

    def __init__(self, task_id: str, score: float, cost_usd: float):
        self.task_id = task_id
        self.score = score
        self.cost_usd = cost_usd


class ProjectCompleted(StreamEvent):
    """Backward compat."""

    def __init__(
        self,
        project_id: str,
        status: str,
        total_cost_usd: float,
        elapsed_seconds: float,
        tasks_completed: int,
        tasks_failed: int,
    ):
        self.project_id = project_id
        self.status = status
        self.total_cost_usd = total_cost_usd
        self.elapsed_seconds = elapsed_seconds
        self.tasks_completed = tasks_completed
        self.tasks_failed = tasks_failed


# ═══════════════════════════════════════════════════════════════════════════════
# Migration Helper
# ═══════════════════════════════════════════════════════════════════════════════


def print_migration_guide():
    """Print migration guide for developers."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     ORCHESTRATOR MIGRATION GUIDE v6.0                        ║
╠══════════════════════════════════════════════════════════════════════════════╣

DASHBOARDS (Consolidated)
─────────────────────────
OLD: from orchestrator.dashboard_live import run_live_dashboard
NEW: from orchestrator.dashboard_core import run_dashboard
     run_dashboard(view='mission-control')

OLD: from orchestrator.dashboard_mission_control import run_mission_control  
NEW: from orchestrator.dashboard_core import run_dashboard
     run_dashboard(view='mission-control')

OLD: from orchestrator.dashboard_antd import run_ant_design_dashboard
NEW: from orchestrator.dashboard_core import run_dashboard
     run_dashboard(view='ant-design')

EVENTS (Unified)
────────────────
OLD: from orchestrator.streaming import ProjectEventBus, ProjectStarted
NEW: from orchestrator.unified_events import (
         UnifiedEventBus, ProjectStartedEvent, get_event_bus
     )
     bus = await get_event_bus()
     await bus.publish(ProjectStartedEvent(...))

OLD: from orchestrator.hooks import HookRegistry, EventType
NEW: from orchestrator.unified_events import UnifiedEventBus, EventType
     bus = await get_event_bus()
     bus.subscribe(callback)

OLD: from orchestrator.capability_logger import log_capability
NEW: from orchestrator.unified_events import get_event_bus
     bus = await get_event_bus()
     await bus.log_capability('name', project_id)

PLUGINS (Optional)
──────────────────
OLD: (built-in validators)
NEW: pip install orchestrator-plugins-validators
     from orchestrator_plugins.validators import PythonTypeCheckerValidator

OLD: slack_integration.py
NEW: pip install orchestrator-plugins-integrations
     from orchestrator_plugins.integrations import SlackIntegration

╚══════════════════════════════════════════════════════════════════════════════╝
""")


# Run migration guide if called directly
if __name__ == "__main__":
    print_migration_guide()
