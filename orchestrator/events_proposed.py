"""
Proposed Event-Driven Architecture Implementation
=================================================

This module shows the target architecture for the event system.
It would replace hooks.py and provide the foundation for CQRS.

Usage:
    from orchestrator.events import EventBus, DomainEvent

    bus = EventBus.create("redis")  # or "memory", "kafka"

    @bus.subscribe("task.completed")
    async def on_task_completed(event: TaskCompletedEvent):
        print(f"Task {event.task_id} completed!")

    await bus.publish(TaskCompletedEvent(task_id="123", score=0.95))
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = logging.getLogger("orchestrator.events")

T = TypeVar('T', bound='DomainEvent')


# ═══════════════════════════════════════════════════════════════════════════════
# Domain Events
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class DomainEvent:
    """
    Base class for all domain events.

    Events are immutable facts that happened in the past.
    They are the source of truth for the system state.
    """
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = field(default="domain_event")
    aggregate_id: str = field(default="")
    timestamp: datetime = field(default_factory=datetime.utcnow)
    payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Ensure event_type is set from class name if not provided
        if self.event_type == "domain_event":
            object.__setattr__(
                self,
                'event_type',
                self.__class__.__name__.replace('Event', '').lower()
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
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
        """Deserialize from dictionary."""
        return cls(
            event_id=data["event_id"],
            event_type=data["event_type"],
            aggregate_id=data["aggregate_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            payload=data["payload"],
            metadata=data["metadata"],
        )


@dataclass(frozen=True)
class TaskStartedEvent(DomainEvent):
    """Emitted when a task starts execution."""
    task_id: str = field(default="")
    task_type: str = field(default="")
    model: str = field(default="")

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, 'event_type', 'task.started')


@dataclass(frozen=True)
class TaskCompletedEvent(DomainEvent):
    """Emitted when a task completes successfully."""
    task_id: str = field(default="")
    model: str = field(default="")
    score: float = field(default=0.0)
    cost_usd: float = field(default=0.0)
    latency_ms: float = field(default=0.0)

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, 'event_type', 'task.completed')


@dataclass(frozen=True)
class TaskFailedEvent(DomainEvent):
    """Emitted when a task fails."""
    task_id: str = field(default="")
    model: str = field(default="")
    error: str = field(default="")
    attempt: int = field(default=0)

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, 'event_type', 'task.failed')


@dataclass(frozen=True)
class ModelSelectedEvent(DomainEvent):
    """Emitted when a model is selected for a task."""
    task_id: str = field(default="")
    model: str = field(default="")
    strategy: str = field(default="")
    reason: str = field(default="")

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, 'event_type', 'model.selected')


@dataclass(frozen=True)
class ProductionOutcomeRecordedEvent(DomainEvent):
    """Emitted when production feedback is recorded."""
    project_id: str = field(default="")
    deployment_id: str = field(default="")
    model: str = field(default="")
    status: str = field(default="")  # success, partial, failure
    score: float = field(default=0.0)

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, 'event_type', 'production.outcome_recorded')


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
    ) -> list[DomainEvent]:
        """Retrieve events matching criteria."""
        pass

    @abstractmethod
    async def replay(
        self,
        handler: Callable[[DomainEvent], Awaitable[None]],
        event_types: list[str] | None = None,
    ) -> None:
        """Replay all events through a handler."""
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
    ) -> list[DomainEvent]:
        events = self._events

        if aggregate_id:
            events = [e for e in events if e.aggregate_id == aggregate_id]

        if event_types:
            events = [e for e in events if e.event_type in event_types]

        if since:
            events = [e for e in events if e.timestamp >= since]

        return events

    async def replay(
        self,
        handler: Callable[[DomainEvent], Awaitable[None]],
        event_types: list[str] | None = None,
    ) -> None:
        events = await self.get_events(event_types=event_types)
        for event in events:
            await handler(event)


class SQLiteEventStore(EventStore):
    """SQLite-backed event store for single-node production."""

    def __init__(self, db_path: str = "events.db"):
        self.db_path = db_path
        self._initialized = False

    async def _init(self):
        if self._initialized:
            return

        import aiosqlite
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    aggregate_id TEXT,
                    timestamp TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    metadata TEXT
                )
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_aggregate
                ON events(aggregate_id)
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_type
                ON events(event_type)
            """)
            await db.commit()

        self._initialized = True

    async def append(self, event: DomainEvent) -> None:
        await self._init()

        import aiosqlite
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
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
                )
            )
            await db.commit()

    async def get_events(
        self,
        aggregate_id: str | None = None,
        event_types: list[str] | None = None,
        since: datetime | None = None,
    ) -> list[DomainEvent]:
        await self._init()

        import aiosqlite
        async with aiosqlite.connect(self.db_path) as db:
            query = "SELECT * FROM events WHERE 1=1"
            params = []

            if aggregate_id:
                query += " AND aggregate_id = ?"
                params.append(aggregate_id)

            if event_types:
                placeholders = ','.join('?' * len(event_types))
                query += f" AND event_type IN ({placeholders})"
                params.extend(event_types)

            if since:
                query += " AND timestamp >= ?"
                params.append(since.isoformat())

            query += " ORDER BY timestamp ASC"

            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()

                events = []
                for row in rows:
                    events.append(DomainEvent(
                        event_id=row[0],
                        event_type=row[1],
                        aggregate_id=row[2],
                        timestamp=datetime.fromisoformat(row[3]),
                        payload=json.loads(row[4]),
                        metadata=json.loads(row[5]) if row[5] else {},
                    ))
                return events

    async def replay(self, handler, event_types=None):
        events = await self.get_events(event_types=event_types)
        for event in events:
            await handler(event)


# ═══════════════════════════════════════════════════════════════════════════════
# Event Bus
# ═══════════════════════════════════════════════════════════════════════════════

class EventBus:
    """
    Central event bus for publishing and subscribing to domain events.

    Features:
    - Multiple subscribers per event type
    - Async event handlers
    - Persistent event store
    - Event replay capability
    """

    def __init__(self, store: EventStore):
        self.store = store
        self._handlers: dict[str, list[Callable[[DomainEvent], Awaitable[None]]]] = defaultdict(list)
        self._running = False

    @classmethod
    def create(cls, backend: str = "memory", **kwargs) -> EventBus:
        """Factory method to create event bus with different backends."""
        if backend == "memory":
            return cls(InMemoryEventStore())
        elif backend == "sqlite":
            return cls(SQLiteEventStore(kwargs.get("db_path", "events.db")))
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
            self._handlers[event_type].remove(handler)

        return unsubscribe

    async def publish(self, event: DomainEvent) -> None:
        """
        Publish an event to all subscribers.

        Events are first persisted, then dispatched to handlers.
        """
        # 1. Persist event
        await self.store.append(event)

        # 2. Dispatch to handlers
        handlers = self._handlers.get(event.event_type, [])

        if not handlers:
            return

        # Run handlers concurrently
        results = await asyncio.gather(*[
            self._run_handler(handler, event)
            for handler in handlers
        ], return_exceptions=True)

        # Log errors but don't fail
        for handler, result in zip(handlers, results, strict=False):
            if isinstance(result, Exception):
                logger.error(
                    f"Event handler {handler.__name__} failed for {event.event_type}: {result}"
                )

    async def _run_handler(
        self,
        handler: Callable[[DomainEvent], Awaitable[None]],
        event: DomainEvent,
    ) -> None:
        """Run a handler with timeout."""
        try:
            await asyncio.wait_for(handler(event), timeout=30.0)
        except asyncio.TimeoutError:
            logger.warning(f"Handler {handler.__name__} timed out")
            raise

    async def replay(
        self,
        event_types: list[str] | None = None,
        since: datetime | None = None,
    ) -> None:
        """Replay events from the store."""
        events = await self.store.get_events(event_types=event_types, since=since)

        for event in events:
            handlers = self._handlers.get(event.event_type, [])
            for handler in handlers:
                try:
                    await handler(event)
                except Exception as e:
                    logger.error(f"Replay handler failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# Projection (CQRS Read Model)
# ═══════════════════════════════════════════════════════════════════════════════

class Projection(ABC):
    """
    Base class for CQRS projections (read models).

    Projections subscribe to events and build read-optimized views.
    """

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self._subscriptions: list[Callable] = []

    def subscribe_to(self, event_type: str) -> None:
        """Subscribe to an event type."""
        handler = getattr(self, f'on_{event_type.replace(".", "_")}', None)
        if handler:
            unsub = self.event_bus.subscribe(event_type, handler)
            self._subscriptions.append(unsub)

    def unsubscribe_all(self) -> None:
        """Unsubscribe from all events."""
        for unsub in self._subscriptions:
            unsub()
        self._subscriptions.clear()

    @abstractmethod
    async def rebuild(self) -> None:
        """Rebuild the projection from event history."""
        pass


class ModelPerformanceProjection(Projection):
    """
    Projection that maintains model performance statistics.

    This is a CQRS read model optimized for fast queries.
    """

    def __init__(self, event_bus: EventBus):
        super().__init__(event_bus)
        self._scores: dict[str, dict[str, float]] = {}  # model -> task_type -> score
        self._subscribe_to_events()

    def _subscribe_to_events(self):
        self.subscribe_to("task.completed")
        self.subscribe_to("task.failed")
        self.subscribe_to("production.outcome_recorded")

    async def on_task_completed(self, event: TaskCompletedEvent) -> None:
        """Update scores when a task completes."""

        # Simple EMA update
        current = self._scores.get(event.model, {}).get("score", 0.5)
        new_score = 0.9 * current + 0.1 * event.score

        if event.model not in self._scores:
            self._scores[event.model] = {}

        self._scores[event.model]["score"] = new_score
        self._scores[event.model]["last_success"] = event.timestamp.isoformat()

    async def on_task_failed(self, event: TaskFailedEvent) -> None:
        """Update scores when a task fails."""
        current = self._scores.get(event.model, {}).get("score", 0.5)
        new_score = 0.95 * current  # Penalize failure

        if event.model not in self._scores:
            self._scores[event.model] = {}

        self._scores[event.model]["score"] = new_score

    async def on_production_outcome_recorded(self, event: ProductionOutcomeRecordedEvent) -> None:
        """Incorporate production feedback."""
        # Weight production data more heavily
        current = self._scores.get(event.model, {}).get("production_score", 0.5)
        new_score = 0.8 * current + 0.2 * event.score

        if event.model not in self._scores:
            self._scores[event.model] = {}

        self._scores[event.model]["production_score"] = new_score

    async def rebuild(self) -> None:
        """Rebuild from event history."""
        self._scores.clear()
        await self.event_bus.replay(event_types=[
            "task.completed",
            "task.failed",
            "production.outcome_recorded",
        ])

    def get_score(self, model: str) -> float:
        """Get current score for a model."""
        scores = self._scores.get(model, {})

        # Blend different score sources
        base = scores.get("score", 0.5)
        production = scores.get("production_score", 0.5)

        return 0.7 * base + 0.3 * production


# ═══════════════════════════════════════════════════════════════════════════════
# Usage Example
# ═══════════════════════════════════════════════════════════════════════════════

async def example_usage():
    """Example of using the event system."""

    # Create event bus
    bus = EventBus.create("sqlite", db_path="events.db")

    # Create projection
    projection = ModelPerformanceProjection(bus)

    # Subscribe to events
    @bus.subscribe("task.completed")
    async def notify_slack(event: TaskCompletedEvent):
        print(f"🎉 Task {event.task_id} completed with score {event.score}")

    @bus.subscribe("task.failed")
    async def alert_on_failure(event: TaskFailedEvent):
        print(f"🚨 Task {event.task_id} failed: {event.error}")

    # Publish events
    await bus.publish(TaskStartedEvent(
        aggregate_id="task-123",
        task_id="task-123",
        task_type="code_gen",
        model="gpt-4o",
    ))

    await bus.publish(TaskCompletedEvent(
        aggregate_id="task-123",
        task_id="task-123",
        model="gpt-4o",
        score=0.95,
        cost_usd=0.02,
        latency_ms=1200,
    ))

    # Query projection
    score = projection.get_score("gpt-4o")
    print(f"GPT-4o score: {score}")

    # Cleanup
    projection.unsubscribe_all()


if __name__ == "__main__":
    asyncio.run(example_usage())
