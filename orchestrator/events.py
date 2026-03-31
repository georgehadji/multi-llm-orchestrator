"""
Event-Driven Architecture for Multi-LLM Orchestrator
=====================================================

Replaces hooks.py with a robust event system supporting:
- Multiple backends (memory, sqlite)
- Persistent events
- Async handlers
- Event replay
- CQRS projections

Usage:
    from orchestrator.events import EventBus, TaskCompletedEvent

    bus = EventBus.create("sqlite")  # or "memory"

    @bus.subscribe("task.completed")
    async def on_complete(event: TaskCompletedEvent):
        print(f"Task {event.task_id} done!")

    await bus.publish(TaskCompletedEvent(task_id="123", score=0.95))
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = logging.getLogger("orchestrator.events")

T = TypeVar("T", bound="DomainEvent")


# ═══════════════════════════════════════════════════════════════════════════════
# Domain Events
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class DomainEvent:
    """Base class for all domain events. Immutable facts about past occurrences."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = field(default="domain_event")
    aggregate_id: str = field(default="")
    timestamp: datetime = field(default_factory=datetime.utcnow)
    payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.event_type == "domain_event":
            object.__setattr__(
                self,
                "event_type",
                self.__class__.__name__.replace("Event", "").lower().replace("_", "."),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "aggregate_id": self.aggregate_id,
            "timestamp": self.timestamp.isoformat(),
            "payload": self.payload,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls: type[T], data: dict[str, Any]) -> T:
        # Handle different event types
        event_type = data.get("event_type", "")
        event_class = EVENT_REGISTRY.get(event_type, DomainEvent)

        return event_class(
            event_id=data["event_id"],
            event_type=data["event_type"],
            aggregate_id=data.get("aggregate_id", ""),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            payload=data.get("payload", {}),
            metadata=data.get("metadata", {}),
        )


# Event registry for deserialization
EVENT_REGISTRY: dict[str, type[DomainEvent]] = {}


def register_event(cls: type[T]) -> type[T]:
    """Decorator to register event classes for deserialization."""
    EVENT_REGISTRY[cls.__name__.lower().replace("_", ".")] = cls
    return cls


@register_event
@dataclass(frozen=True)
class TaskStartedEvent(DomainEvent):
    """Emitted when a task starts execution."""

    task_id: str = ""
    task_type: str = ""
    model: str = ""
    project_id: str = ""

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(
            self,
            "payload",
            {
                "task_id": self.task_id,
                "task_type": self.task_type,
                "model": self.model,
                "project_id": self.project_id,
            },
        )


@register_event
@dataclass(frozen=True)
class TaskCompletedEvent(DomainEvent):
    """Emitted when a task completes successfully."""

    task_id: str = ""
    model: str = ""
    score: float = 0.0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    tokens_input: int = 0
    tokens_output: int = 0
    iterations: int = 1
    status: str = "completed"  # completed, degraded, partial

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(
            self,
            "payload",
            {
                "task_id": self.task_id,
                "model": self.model,
                "score": self.score,
                "cost_usd": self.cost_usd,
                "latency_ms": self.latency_ms,
                "tokens_input": self.tokens_input,
                "tokens_output": self.tokens_output,
                "iterations": self.iterations,
                "status": self.status,
            },
        )


@register_event
@dataclass(frozen=True)
class TaskProgressEvent(DomainEvent):
    """Emitted during task execution to report progress."""

    task_id: str = ""
    iteration: int = 0
    score: float = 0.0
    best_score: float = 0.0
    model: str = ""

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(
            self,
            "payload",
            {
                "task_id": self.task_id,
                "iteration": self.iteration,
                "score": self.score,
                "best_score": self.best_score,
                "model": self.model,
            },
        )


@register_event
@dataclass(frozen=True)
class TaskFailedEvent(DomainEvent):
    """Emitted when a task fails."""

    task_id: str = ""
    model: str = ""
    error: str = ""
    reason: str = ""  # Alias for error (compatibility)
    attempt: int = 0
    will_retry: bool = False

    def __post_init__(self):
        super().__post_init__()
        # Use error as reason if reason not provided
        reason_val = self.reason if self.reason else self.error
        object.__setattr__(
            self,
            "payload",
            {
                "task_id": self.task_id,
                "model": self.model,
                "error": self.error,
                "reason": reason_val,
                "attempt": self.attempt,
                "will_retry": self.will_retry,
            },
        )


@register_event
@dataclass(frozen=True)
class ModelSelectedEvent(DomainEvent):
    """Emitted when a model is selected for a task."""

    task_id: str = ""
    model: str = ""
    strategy: str = ""
    reason: str = ""
    confidence: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(
            self,
            "payload",
            {
                "task_id": self.task_id,
                "model": self.model,
                "strategy": self.strategy,
                "reason": self.reason,
                "confidence": self.confidence,
            },
        )


@register_event
@dataclass(frozen=True)
class ValidationFailedEvent(DomainEvent):
    """Emitted when deterministic validators fail."""

    task_id: str = ""
    model: str = ""
    validators: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(
            self,
            "payload",
            {
                "task_id": self.task_id,
                "model": self.model,
                "validators": self.validators,
                "errors": self.errors,
            },
        )


@register_event
@dataclass(frozen=True)
class BudgetWarningEvent(DomainEvent):
    """Emitted when budget thresholds are exceeded."""

    phase: str = ""
    spent: float = 0.0
    spent_usd: float = 0.0  # Alias for spent
    budget: float = 0.0
    cap_usd: float = 0.0  # Alias for budget
    ratio: float = 0.0
    project_id: str = ""

    def __post_init__(self):
        super().__post_init__()
        # Use aliases if provided, otherwise fall back to main fields
        spent_val = self.spent_usd if self.spent_usd else self.spent
        object.__setattr__(
            self,
            "payload",
            {
                "phase": self.phase,
                "spent": spent_val,
                "budget": self.budget,
                "ratio": self.ratio,
                "project_id": self.project_id,
            },
        )


@register_event
@dataclass(frozen=True)
class ProjectStartedEvent(DomainEvent):
    """Emitted when a project starts."""

    project_id: str = ""
    description: str = ""
    budget: float = 0.0
    budget_usd: float = 0.0  # Alias for budget
    total_tasks: int = 0

    def __post_init__(self):
        super().__post_init__()
        budget_val = self.budget_usd if self.budget_usd else self.budget
        object.__setattr__(
            self,
            "payload",
            {
                "project_id": self.project_id,
                "description": self.description,
                "budget": budget_val,
                "budget_usd": budget_val,
                "total_tasks": self.total_tasks,
            },
        )


@register_event
@dataclass(frozen=True)
class ProjectCompletedEvent(DomainEvent):
    """Emitted when a project completes."""

    project_id: str = ""
    status: str = ""  # success, partial, failed
    total_cost: float = 0.0
    total_cost_usd: float = 0.0  # Alias for total_cost
    duration_seconds: float = 0.0
    elapsed_seconds: float = 0.0  # Alias for duration_seconds
    tasks_completed: int = 0
    tasks_failed: int = 0

    def __post_init__(self):
        super().__post_init__()
        cost_val = self.total_cost_usd if self.total_cost_usd else self.total_cost
        elapsed_val = self.elapsed_seconds if self.elapsed_seconds else self.duration_seconds
        object.__setattr__(
            self,
            "payload",
            {
                "project_id": self.project_id,
                "status": self.status,
                "total_cost": cost_val,
                "total_cost_usd": cost_val,
                "duration_seconds": elapsed_val,
                "elapsed_seconds": elapsed_val,
                "tasks_completed": self.tasks_completed,
                "tasks_failed": self.tasks_failed,
            },
        )


@register_event
@dataclass(frozen=True)
class CircuitBreakerTrippedEvent(DomainEvent):
    """Emitted when circuit breaker opens for a model."""

    model: str = ""
    failure_count: int = 0
    threshold: int = 0

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(
            self,
            "payload",
            {
                "model": self.model,
                "failure_count": self.failure_count,
                "threshold": self.threshold,
            },
        )


@register_event
@dataclass(frozen=True)
class ProductionOutcomeRecordedEvent(DomainEvent):
    """Emitted when production feedback is recorded."""

    project_id: str = ""
    deployment_id: str = ""
    model: str = ""
    status: str = ""  # success, partial, failure
    score: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(
            self,
            "payload",
            {
                "project_id": self.project_id,
                "deployment_id": self.deployment_id,
                "model": self.model,
                "status": self.status,
                "score": self.score,
            },
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Event Store
# ═══════════════════════════════════════════════════════════════════════════════


class EventStore(ABC):
    """Abstract event store for persisting domain events."""

    @abstractmethod
    async def append(self, event: DomainEvent) -> None:
        """Persist an event."""
        pass

    @abstractmethod
    async def get_events(
        self,
        aggregate_id: str | None = None,
        event_types: list[str] | None = None,
        since: datetime | None = None,
        limit: int | None = None,
    ) -> list[DomainEvent]:
        """Retrieve events matching criteria."""
        pass

    @abstractmethod
    async def replay(
        self,
        handler: Callable[[DomainEvent], Awaitable[None]],
        event_types: list[str] | None = None,
        since: datetime | None = None,
    ) -> None:
        """Replay all events through a handler."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Cleanup resources."""
        pass


class InMemoryEventStore(EventStore):
    """In-memory event store for development/testing."""

    def __init__(self):
        self._events: list[DomainEvent] = []

    async def append(self, event: DomainEvent) -> None:
        self._events.append(event)

    async def get_events(
        self,
        aggregate_id: str | None = None,
        event_types: list[str] | None = None,
        since: datetime | None = None,
        limit: int | None = None,
    ) -> list[DomainEvent]:
        events = list(self._events)

        if aggregate_id:
            events = [e for e in events if e.aggregate_id == aggregate_id]

        if event_types:
            events = [e for e in events if e.event_type in event_types]

        if since:
            events = [e for e in events if e.timestamp >= since]

        if limit:
            events = events[-limit:]

        return events

    async def replay(
        self,
        handler: Callable[[DomainEvent], Awaitable[None]],
        event_types: list[str] | None = None,
        since: datetime | None = None,
    ) -> None:
        events = await self.get_events(event_types=event_types, since=since)
        for event in events:
            await handler(event)

    async def close(self) -> None:
        self._events.clear()


class SQLiteEventStore(EventStore):
    """SQLite-backed event store for production."""

    def __init__(self, db_path: str = ".events/event_store.db"):
        self.db_path = Path(db_path)
        self._initialized = False
        self._lock = asyncio.Lock()

    async def _init(self):
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return

            # Ensure directory exists
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            # Use run_in_executor for sync sqlite operations
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._create_tables)

            self._initialized = True

    def _create_tables(self):
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    aggregate_id TEXT,
                    timestamp TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    metadata TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_aggregate
                ON events(aggregate_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_type
                ON events(event_type)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON events(timestamp)
            """)
            conn.commit()

    async def append(self, event: DomainEvent) -> None:
        await self._init()

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._insert_event, event)

    def _insert_event(self, event: DomainEvent):
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                """
                INSERT INTO events (event_id, event_type, aggregate_id, timestamp, payload, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.event_type,
                    event.aggregate_id,
                    event.timestamp.isoformat(),
                    json.dumps(event.payload),
                    json.dumps(event.metadata),
                ),
            )
            conn.commit()

    async def get_events(
        self,
        aggregate_id: str | None = None,
        event_types: list[str] | None = None,
        since: datetime | None = None,
        limit: int | None = None,
    ) -> list[DomainEvent]:
        await self._init()

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._fetch_events,
            aggregate_id,
            event_types,
            since.isoformat() if since else None,
            limit,
        )

    def _fetch_events(
        self,
        aggregate_id: str | None,
        event_types: list[str] | None,
        since: str | None,
        limit: int | None,
    ) -> list[DomainEvent]:
        with sqlite3.connect(str(self.db_path)) as conn:
            query = "SELECT event_id, event_type, aggregate_id, timestamp, payload, metadata FROM events WHERE 1=1"
            params = []

            if aggregate_id:
                query += " AND aggregate_id = ?"
                params.append(aggregate_id)

            if event_types:
                placeholders = ",".join("?" * len(event_types))
                query += f" AND event_type IN ({placeholders})"
                params.extend(event_types)

            if since:
                query += " AND timestamp >= ?"
                params.append(since)

            query += " ORDER BY timestamp ASC"

            if limit:
                query += f" LIMIT {limit}"

            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

            events = []
            for row in rows:
                event_data = {
                    "event_id": row[0],
                    "event_type": row[1],
                    "aggregate_id": row[2],
                    "timestamp": row[3],
                    "payload": json.loads(row[4]),
                    "metadata": json.loads(row[5]) if row[5] else {},
                }
                events.append(DomainEvent.from_dict(event_data))

            return events

    async def replay(
        self,
        handler: Callable[[DomainEvent], Awaitable[None]],
        event_types: list[str] | None = None,
        since: datetime | None = None,
    ) -> None:
        events = await self.get_events(event_types=event_types, since=since)
        for event in events:
            await handler(event)

    async def close(self) -> None:
        pass  # SQLite connections are per-operation


# ═══════════════════════════════════════════════════════════════════════════════
# Event Bus
# ═══════════════════════════════════════════════════════════════════════════════


class EventBus:
    """
    Central event bus for publishing and subscribing to domain events.

    Features:
    - Persistent event store
    - Async handlers with timeout
    - Error isolation
    - Event replay
    """

    def __init__(self, store: EventStore, handler_timeout: float = 30.0):
        self.store = store
        self.handler_timeout = handler_timeout
        self._handlers: dict[str, list[Callable[[DomainEvent], Awaitable[None]]]] = defaultdict(
            list
        )
        self._metrics = {
            "published": 0,
            "handled": 0,
            "errors": 0,
        }

    @classmethod
    def create(cls, backend: str = "memory", **kwargs) -> EventBus:
        """Factory method to create event bus with different backends."""
        if backend == "memory":
            return cls(InMemoryEventStore(), **kwargs)
        elif backend == "sqlite":
            return cls(SQLiteEventStore(kwargs.get("db_path", ".events/event_store.db")), **kwargs)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def subscribe(
        self,
        event_type: str,
        handler: Callable[[DomainEvent], Awaitable[None]],
    ) -> Callable[[], None]:
        """
        Subscribe to events of a specific type.

        Returns an unsubscribe function.
        """
        self._handlers[event_type].append(handler)

        def unsubscribe():
            if handler in self._handlers[event_type]:
                self._handlers[event_type].remove(handler)

        return unsubscribe

    async def publish(self, event: DomainEvent) -> None:
        """
        Publish an event to all subscribers.

        Flow:
        1. Persist event to store
        2. Dispatch to all handlers concurrently
        3. Log errors but never fail
        """
        # 1. Persist
        try:
            await self.store.append(event)
            self._metrics["published"] += 1
        except Exception as e:
            logger.error(f"Failed to persist event {event.event_id}: {e}")
            raise  # Can't proceed without persistence

        # 2. Dispatch
        handlers = self._handlers.get(event.event_type, [])

        if not handlers:
            logger.debug(f"No handlers for event type: {event.event_type}")
            return

        # Run handlers concurrently with error isolation
        results = await asyncio.gather(
            *[self._run_handler(handler, event) for handler in handlers], return_exceptions=True
        )

        # 3. Log results
        for handler, result in zip(handlers, results, strict=False):
            if isinstance(result, Exception):
                self._metrics["errors"] += 1
                logger.error(
                    f"Handler {handler.__name__} failed for {event.event_type}: {result}",
                    exc_info=result,
                )
            else:
                self._metrics["handled"] += 1

    async def _run_handler(
        self,
        handler: Callable[[DomainEvent], Awaitable[None]],
        event: DomainEvent,
    ) -> None:
        """Run a handler with timeout and error handling."""
        try:
            await asyncio.wait_for(handler(event), timeout=self.handler_timeout)
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"Handler {handler.__name__} timed out after {self.handler_timeout}s"
            )

    async def replay(
        self,
        event_types: list[str] | None = None,
        since: datetime | None = None,
        handler_filter: Callable[[str], bool] | None = None,
    ) -> dict[str, int]:
        """
        Replay events from the store.

        Returns statistics about replayed events.
        """
        events = await self.store.get_events(event_types=event_types, since=since)

        stats = {"replayed": 0, "errors": 0}

        for event in events:
            handlers = self._handlers.get(event.event_type, [])

            if handler_filter:
                handlers = [h for h in handlers if handler_filter(h.__name__)]

            for handler in handlers:
                try:
                    await handler(event)
                    stats["replayed"] += 1
                except Exception as e:
                    stats["errors"] += 1
                    logger.error(f"Replay handler {handler.__name__} failed: {e}")

        return stats

    def get_metrics(self) -> dict[str, int]:
        """Get event bus metrics."""
        return dict(self._metrics)

    async def close(self) -> None:
        """Cleanup resources."""
        await self.store.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Global Event Bus Instance
# ═══════════════════════════════════════════════════════════════════════════════

_event_bus: EventBus | None = None


def get_event_bus(backend: str = "sqlite", **kwargs) -> EventBus:
    """Get or create global event bus instance."""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus.create(backend, **kwargs)
    return _event_bus


def reset_event_bus() -> None:
    """Reset global event bus (for testing)."""
    global _event_bus
    _event_bus = None


@asynccontextmanager
async def event_bus_context(backend: str = "sqlite", **kwargs):
    """Context manager for event bus lifecycle."""
    bus = get_event_bus(backend, **kwargs)
    try:
        yield bus
    finally:
        await bus.close()
        reset_event_bus()
