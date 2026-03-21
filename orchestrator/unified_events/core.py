"""
Unified Events System — Event Sourcing Lite
===========================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Consolidates:
- streaming.py (streaming pipeline events)
- events.py (domain events)
- hooks.py (hook registry)
- capability_logger.py (capability usage logging)

Into a single event-driven architecture with:
- Typed domain events
- Automatic projections (read models)
- Event persistence
- Capability tracking
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from abc import ABC
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Set, Awaitable
from collections import defaultdict
from contextvars import ContextVar

logger = logging.getLogger("orchestrator.unified_events")


# ═══════════════════════════════════════════════════════════════════════════════
# Event Types
# ═══════════════════════════════════════════════════════════════════════════════

class EventType(Enum):
    """All event types in the unified system."""
    # Project lifecycle
    PROJECT_STARTED = auto()
    PROJECT_COMPLETED = auto()
    PROJECT_FAILED = auto()
    
    # Task lifecycle
    TASK_CREATED = auto()
    TASK_STARTED = auto()
    TASK_PROGRESS = auto()
    TASK_COMPLETED = auto()
    TASK_FAILED = auto()
    TASK_RETRY = auto()
    
    # Model/Routing
    MODEL_SELECTED = auto()
    MODEL_UNAVAILABLE = auto()
    FALLBACK_TRIGGERED = auto()
    CIRCUIT_BREAKER_OPEN = auto()
    
    # Quality & Validation
    VALIDATION_PASSED = auto()
    VALIDATION_FAILED = auto()
    QUALITY_GATE_PASSED = auto()
    QUALITY_GATE_FAILED = auto()
    
    # Budget & Cost
    BUDGET_WARNING = auto()
    BUDGET_EXHAUSTED = auto()
    COST_RECORDED = auto()
    
    # System
    ERROR = auto()
    WARNING = auto()
    INFO = auto()
    METRIC = auto()
    
    # Capability usage (from capability_logger.py)
    CAPABILITY_USED = auto()
    CAPABILITY_COMPLETED = auto()
    CAPABILITY_FAILED = auto()


# ═══════════════════════════════════════════════════════════════════════════════
# Domain Events
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class DomainEvent:
    """
    Base class for all domain events.
    Immutable, serializable, timestamped.
    """
    event_type: EventType
    aggregate_id: str  # e.g., project_id or task_id
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "event_type": self.event_type.name,
            "aggregate_id": self.aggregate_id,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DomainEvent:
        """Deserialize from dictionary."""
        return cls(
            event_type=EventType[data["event_type"]],
            aggregate_id=data["aggregate_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {}),
        )


# Project events
@dataclass(frozen=True)
class ProjectStartedEvent(DomainEvent):
    project_description: str = ""
    budget: float = 0.0
    
    def __init__(self, aggregate_id: str, project_id: str, description: str, budget: float):
        super().__init__(
            event_type=EventType.PROJECT_STARTED,
            aggregate_id=aggregate_id,
            metadata={
                "project_id": project_id,
                "description": description,
                "budget": budget,
            }
        )


@dataclass(frozen=True)
class ProjectCompletedEvent(DomainEvent):
    status: str = ""
    total_cost: float = 0.0
    tasks_completed: int = 0
    tasks_failed: int = 0
    
    def __init__(self, aggregate_id: str, project_id: str, status: str, total_cost: float,
                 tasks_completed: int = 0, tasks_failed: int = 0):
        super().__init__(
            event_type=EventType.PROJECT_COMPLETED,
            aggregate_id=aggregate_id,
            metadata={
                "project_id": project_id,
                "status": status,
                "total_cost": total_cost,
                "tasks_completed": tasks_completed,
                "tasks_failed": tasks_failed,
            }
        )


# Task events
@dataclass(frozen=True)
class TaskStartedEvent(DomainEvent):
    task_type: str = ""
    model: str = ""
    
    def __init__(self, aggregate_id: str, task_id: str, task_type: str, model: str = ""):
        super().__init__(
            event_type=EventType.TASK_STARTED,
            aggregate_id=aggregate_id,
            metadata={
                "task_id": task_id,
                "task_type": task_type,
                "model": model,
            }
        )


@dataclass(frozen=True)
class TaskCompletedEvent(DomainEvent):
    score: float = 0.0
    cost: float = 0.0
    duration_ms: int = 0
    
    def __init__(self, aggregate_id: str, task_id: str, score: float = 0.0, 
                 cost: float = 0.0, duration_ms: int = 0):
        super().__init__(
            event_type=EventType.TASK_COMPLETED,
            aggregate_id=aggregate_id,
            metadata={
                "task_id": task_id,
                "score": score,
                "cost": cost,
                "duration_ms": duration_ms,
            }
        )


@dataclass(frozen=True)
class TaskFailedEvent(DomainEvent):
    error: str = ""
    will_retry: bool = False
    
    def __init__(self, aggregate_id: str, task_id: str, error: str, will_retry: bool = False):
        super().__init__(
            event_type=EventType.TASK_FAILED,
            aggregate_id=aggregate_id,
            metadata={
                "task_id": task_id,
                "error": error,
                "will_retry": will_retry,
            }
        )


@dataclass(frozen=True)
class TaskProgressEvent(DomainEvent):
    iteration: int = 0
    score: float = 0.0
    message: str = ""
    
    def __init__(self, aggregate_id: str, task_id: str, iteration: int, 
                 score: float, message: str = ""):
        super().__init__(
            event_type=EventType.TASK_PROGRESS,
            aggregate_id=aggregate_id,
            metadata={
                "task_id": task_id,
                "iteration": iteration,
                "score": score,
                "message": message,
            }
        )


# Model/Routing events
@dataclass(frozen=True)
class ModelSelectedEvent(DomainEvent):
    model: str = ""
    reason: str = ""
    
    def __init__(self, aggregate_id: str, task_id: str, model: str, reason: str = ""):
        super().__init__(
            event_type=EventType.MODEL_SELECTED,
            aggregate_id=aggregate_id,
            metadata={
                "task_id": task_id,
                "model": model,
                "reason": reason,
            }
        )


@dataclass(frozen=True)
class FallbackTriggeredEvent(DomainEvent):
    original_model: str = ""
    fallback_model: str = ""
    reason: str = ""
    
    def __init__(self, aggregate_id: str, original_model: str, fallback_model: str, reason: str):
        super().__init__(
            event_type=EventType.FALLBACK_TRIGGERED,
            aggregate_id=aggregate_id,
            metadata={
                "original_model": original_model,
                "fallback_model": fallback_model,
                "reason": reason,
            }
        )


# Capability events (replaces capability_logger.py)
@dataclass(frozen=True)
class CapabilityUsedEvent(DomainEvent):
    capability: str = ""
    project_id: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def __init__(self, aggregate_id: str, capability: str, project_id: str = "", 
                 parameters: Optional[Dict[str, Any]] = None):
        super().__init__(
            event_type=EventType.CAPABILITY_USED,
            aggregate_id=aggregate_id,
            metadata={
                "capability": capability,
                "project_id": project_id,
                "parameters": parameters or {},
            }
        )


@dataclass(frozen=True)
class CapabilityCompletedEvent(DomainEvent):
    capability: str = ""
    duration_ms: int = 0
    result_summary: str = ""
    
    def __init__(self, aggregate_id: str, capability: str, duration_ms: int = 0,
                 result_summary: str = ""):
        super().__init__(
            event_type=EventType.CAPABILITY_COMPLETED,
            aggregate_id=aggregate_id,
            metadata={
                "capability": capability,
                "duration_ms": duration_ms,
                "result_summary": result_summary,
            }
        )


# Budget events
@dataclass(frozen=True)
class BudgetWarningEvent(DomainEvent):
    phase: str = ""
    spent: float = 0.0
    cap: float = 0.0
    ratio: float = 0.0
    
    def __init__(self, aggregate_id: str, phase: str, spent: float, cap: float, ratio: float):
        super().__init__(
            event_type=EventType.BUDGET_WARNING,
            aggregate_id=aggregate_id,
            metadata={
                "phase": phase,
                "spent": spent,
                "cap": cap,
                "ratio": ratio,
            }
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Event Store
# ═══════════════════════════════════════════════════════════════════════════════

class EventStore:
    """
    SQLite-based event store for persistence.
    Enables replay and audit trail.
    """
    
    def __init__(self, db_path: str = ".orchestrator_events.db"):
        self.db_path = Path(db_path)
        self._local = threading.local()
        self._init_db()
    
    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local connection."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(str(self.db_path))
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn
    
    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                aggregate_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_aggregate ON events(aggregate_id)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_type ON events(event_type)
        """)
        conn.commit()
        conn.close()
    
    def append(self, event: DomainEvent) -> None:
        """Persist an event."""
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO events (event_type, aggregate_id, timestamp, data) VALUES (?, ?, ?, ?)",
            (event.event_type.name, event.aggregate_id, 
             event.timestamp.isoformat(), json.dumps(event.to_dict()))
        )
        conn.commit()
    
    def get_events(self, aggregate_id: Optional[str] = None,
                   event_type: Optional[EventType] = None,
                   since: Optional[datetime] = None) -> List[DomainEvent]:
        """Query events with filters."""
        conn = self._get_conn()
        query = "SELECT data FROM events WHERE 1=1"
        params = []
        
        if aggregate_id:
            query += " AND aggregate_id = ?"
            params.append(aggregate_id)
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type.name)
        if since:
            query += " AND timestamp > ?"
            params.append(since.isoformat())
        
        query += " ORDER BY timestamp ASC"
        
        cursor = conn.execute(query, params)
        events = []
        for row in cursor:
            data = json.loads(row["data"])
            events.append(DomainEvent.from_dict(data))
        return events
    
    def get_aggregate(self, aggregate_id: str) -> List[DomainEvent]:
        """Get all events for an aggregate (for replay)."""
        return self.get_events(aggregate_id=aggregate_id)


import threading


# ═══════════════════════════════════════════════════════════════════════════════
# Projections (Read Models)
# ═══════════════════════════════════════════════════════════════════════════════

class Projection(ABC):
    """Base class for read model projections."""
    
    def apply(self, event: DomainEvent) -> None:
        """Apply event to update projection state."""
        handler = getattr(self, f"on_{event.event_type.name.lower()}", None)
        if handler:
            handler(event)


class ProjectStateProjection(Projection):
    """
    Projection that maintains current project state.
    Replaces manual state tracking.
    """
    
    def __init__(self):
        self.projects: Dict[str, Dict[str, Any]] = {}
        self.tasks: Dict[str, Dict[str, Any]] = {}
    
    def on_project_started(self, event: ProjectStartedEvent) -> None:
        self.projects[event.aggregate_id] = {
            "id": event.aggregate_id,
            "status": "running",
            "description": event.metadata.get("description", ""),
            "budget": event.metadata.get("budget", 0),
            "started_at": event.timestamp,
            "tasks": [],
        }
    
    def on_project_completed(self, event: ProjectCompletedEvent) -> None:
        if event.aggregate_id in self.projects:
            self.projects[event.aggregate_id]["status"] = event.metadata.get("status", "completed")
            self.projects[event.aggregate_id]["completed_at"] = event.timestamp
            self.projects[event.aggregate_id]["total_cost"] = event.metadata.get("total_cost", 0)
    
    def on_task_started(self, event: TaskStartedEvent) -> None:
        task_id = event.metadata.get("task_id", "")
        self.tasks[task_id] = {
            "id": task_id,
            "project_id": event.aggregate_id,
            "type": event.metadata.get("task_type", ""),
            "model": event.metadata.get("model", ""),
            "status": "running",
            "started_at": event.timestamp,
        }
        if event.aggregate_id in self.projects:
            self.projects[event.aggregate_id]["tasks"].append(task_id)
    
    def on_task_completed(self, event: TaskCompletedEvent) -> None:
        task_id = event.metadata.get("task_id", "")
        if task_id in self.tasks:
            self.tasks[task_id]["status"] = "completed"
            self.tasks[task_id]["score"] = event.metadata.get("score", 0)
            self.tasks[task_id]["cost"] = event.metadata.get("cost", 0)
            self.tasks[task_id]["completed_at"] = event.timestamp
    
    def on_task_failed(self, event: TaskFailedEvent) -> None:
        task_id = event.metadata.get("task_id", "")
        if task_id in self.tasks:
            self.tasks[task_id]["status"] = "failed"
            self.tasks[task_id]["error"] = event.metadata.get("error", "")
    
    def on_task_progress(self, event: TaskProgressEvent) -> None:
        task_id = event.metadata.get("task_id", "")
        if task_id in self.tasks:
            self.tasks[task_id]["iteration"] = event.metadata.get("iteration", 0)
            self.tasks[task_id]["score"] = event.metadata.get("score", 0)
    
    def get_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        return self.projects.get(project_id)
    
    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        return self.tasks.get(task_id)
    
    def get_active_tasks(self, project_id: str) -> List[Dict[str, Any]]:
        return [
            self.tasks[tid] for tid in self.projects.get(project_id, {}).get("tasks", [])
            if self.tasks.get(tid, {}).get("status") == "running"
        ]


class MetricsProjection(Projection):
    """Projection for real-time metrics."""
    
    def __init__(self):
        self.metrics = defaultdict(lambda: defaultdict(float))
        self.counters = defaultdict(int)
    
    def on_task_completed(self, event: TaskCompletedEvent) -> None:
        self.counters["tasks_completed"] += 1
        score = event.metadata.get("score", 0)
        self.metrics["quality"]["total_score"] += score
        self.metrics["quality"]["avg_score"] = (
            self.metrics["quality"]["total_score"] / self.counters["tasks_completed"]
        )
        self.metrics["cost"]["total"] += event.metadata.get("cost", 0)
    
    def on_task_failed(self, event: TaskFailedEvent) -> None:
        self.counters["tasks_failed"] += 1
    
    def on_fallback_triggered(self, event: FallbackTriggeredEvent) -> None:
        self.counters["fallbacks"] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        return {
            "counters": dict(self.counters),
            "metrics": {k: dict(v) for k, v in self.metrics.items()},
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Unified Event Bus
# ═══════════════════════════════════════════════════════════════════════════════

class UnifiedEventBus:
    """
    Single event bus for all orchestrator events.
    Replaces: streaming.py, events.py, hooks.py, capability_logger.py
    """
    
    _instance: Optional[UnifiedEventBus] = None
    _lock = asyncio.Lock()
    
    def __init__(self, store: Optional[EventStore] = None):
        self.store = store
        self.subscribers: List[Callable[[DomainEvent], Awaitable[None]]] = []
        self.projections: List[Projection] = []
        self._event_queue: asyncio.Queue[DomainEvent] = asyncio.Queue()
        self._running = False
        self._process_task: Optional[asyncio.Task] = None
        
        # Default projections
        self.projection_state = ProjectStateProjection()
        self.projection_metrics = MetricsProjection()
        self.add_projection(self.projection_state)
        self.add_projection(self.projection_metrics)
    
    @classmethod
    async def get_instance(cls) -> UnifiedEventBus:
        """Get or create singleton instance."""
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def add_projection(self, projection: Projection) -> None:
        """Add a read model projection."""
        self.projections.append(projection)
    
    def subscribe(self, callback: Callable[[DomainEvent], Awaitable[None]]) -> None:
        """Subscribe to events (replaces hooks.py)."""
        self.subscribers.append(callback)
    
    def unsubscribe(self, callback: Callable[[DomainEvent], Awaitable[None]]) -> None:
        """Unsubscribe from events."""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
    
    async def publish(self, event: DomainEvent) -> None:
        """Publish event to all subscribers and projections."""
        await self._event_queue.put(event)
    
    async def start(self) -> None:
        """Start event processing loop."""
        if self._running:
            return
        self._running = True
        self._process_task = asyncio.create_task(self._process_loop())
    
    async def stop(self) -> None:
        """Stop event processing loop."""
        self._running = False
        if self._process_task:
            await self._event_queue.put(None)  # Sentinel
            self._process_task.cancel()
            try:
                await self._process_task
            except asyncio.CancelledError:
                pass
    
    async def _process_loop(self) -> None:
        """Process events from queue."""
        while self._running:
            try:
                event = await self._event_queue.get()
                if event is None:  # Sentinel
                    break
                await self._handle_event(event)
            except asyncio.CancelledError:
                # Properly handle cancellation
                logger.debug("Event processing loop was cancelled")
                break
            except Exception as e:
                logger.exception("Error processing event")
    
    async def _handle_event(self, event: DomainEvent) -> None:
        """Handle a single event."""
        # Persist
        if self.store:
            try:
                self.store.append(event)
            except Exception as e:
                logger.warning(f"Failed to persist event: {e}")
        
        # Update projections
        for projection in self.projections:
            try:
                projection.apply(event)
            except Exception as e:
                logger.warning(f"Projection failed: {e}")
        
        # Notify subscribers
        for callback in self.subscribers:
            try:
                await callback(event)
            except Exception as e:
                logger.warning(f"Subscriber failed: {e}")
    
    async def subscribe_iter(self) -> AsyncIterator[DomainEvent]:
        """Async iterator for events (replaces streaming.py)."""
        queue: asyncio.Queue[DomainEvent] = asyncio.Queue()
        
        async def handler(event: DomainEvent) -> None:
            await queue.put(event)
        
        self.subscribe(handler)
        try:
            while True:
                event = await queue.get()
                yield event
        finally:
            self.unsubscribe(handler)
    
    # Convenience methods for common events
    
    async def log_capability(self, capability: str, project_id: str = "",
                            parameters: Optional[Dict[str, Any]] = None) -> None:
        """Log capability usage (replaces capability_logger.py)."""
        event = CapabilityUsedEvent(
            aggregate_id=f"capability:{capability}:{datetime.utcnow().isoformat()}",
            capability=capability,
            project_id=project_id,
            parameters=parameters,
        )
        await self.publish(event)
    
    async def log_task_progress(self, task_id: str, iteration: int, 
                                score: float, message: str = "") -> None:
        """Log task progress update."""
        event = TaskProgressEvent(
            aggregate_id=task_id,
            task_id=task_id,
            iteration=iteration,
            score=score,
            message=message,
        )
        await self.publish(event)
    
    # Query methods (using projections)
    
    def get_project_state(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Get current project state."""
        return self.projection_state.get_project(project_id)
    
    def get_task_state(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get current task state."""
        return self.projection_state.get_task(task_id)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return self.projection_metrics.get_metrics()


# ═══════════════════════════════════════════════════════════════════════════════
# Global Instance & Convenience Functions
# ═══════════════════════════════════════════════════════════════════════════════

async def get_event_bus() -> UnifiedEventBus:
    """Get the global event bus instance."""
    return await UnifiedEventBus.get_instance()


# Context variable for automatic event tracking
_current_project: ContextVar[Optional[str]] = ContextVar('current_project', default=None)


def set_current_project(project_id: str) -> None:
    """Set current project context for automatic event attribution."""
    _current_project.set(project_id)


def get_current_project() -> Optional[str]:
    """Get current project from context."""
    return _current_project.get()


# Decorator for automatic capability logging
def log_capability_use(capability_name: str):
    """Decorator to automatically log capability usage."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            bus = await get_event_bus()
            project_id = get_current_project() or ""
            
            # Log start
            await bus.log_capability(capability_name, project_id, {"status": "started"})
            
            start_time = datetime.utcnow()
            try:
                result = await func(*args, **kwargs)
                
                # Log completion
                duration = (datetime.utcnow() - start_time).total_seconds() * 1000
                await bus.publish(CapabilityCompletedEvent(
                    aggregate_id=f"capability:{capability_name}:{start_time.isoformat()}",
                    capability=capability_name,
                    duration_ms=int(duration),
                    result_summary="success",
                ))
                
                return result
            except Exception as e:
                # Log failure
                await bus.publish(DomainEvent(
                    event_type=EventType.CAPABILITY_FAILED,
                    aggregate_id=f"capability:{capability_name}:{start_time.isoformat()}",
                    metadata={"capability": capability_name, "error": str(e)},
                ))
                raise
        return wrapper
    return decorator
