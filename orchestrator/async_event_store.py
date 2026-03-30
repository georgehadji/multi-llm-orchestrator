"""
Async Event Store — Non-blocking SQLite event persistence
=========================================================
Author: Senior Distributed Systems Architect

CRITICAL FIX: Replaces synchronous EventStore in unified_events/core.py
This module provides async event persistence that never blocks the event loop.

INVARIANTS:
1. All operations are async (never blocks event loop)
2. Connection is lazily initialized
3. WAL mode for concurrent reads
4. Single writer pattern via lock
5. Graceful shutdown with proper cleanup

USAGE:
    from orchestrator.async_event_store import AsyncEventStore

    store = AsyncEventStore()
    await store.append(event)
    events = await store.get_events(aggregate_id="project_123")
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import aiosqlite

if TYPE_CHECKING:
    from datetime import datetime

logger = logging.getLogger("orchestrator.async_event_store")

# Import EventType and DomainEvent from unified_events if available
try:
    from .unified_events.core import DomainEvent, EventType
    HAS_UNIFIED_EVENTS = True
except ImportError:
    HAS_UNIFIED_EVENTS = False
    EventType = None
    DomainEvent = None


class AsyncEventStore:
    """
    Async SQLite event store for persistence.

    Enables replay and audit trail without blocking the event loop.

    FEATURES:
    - Async I/O throughout (no blocking)
    - WAL mode for concurrent reads
    - Single writer pattern for consistency
    - Connection pooling with lazy init
    - Graceful shutdown

    MIGRATION FROM SYNC:
        # Before (blocks event loop):
        from orchestrator.unified_events.core import EventStore
        store = EventStore()
        store.append(event)  # BLOCKS!

        # After (non-blocking):
        from orchestrator.async_event_store import AsyncEventStore
        store = AsyncEventStore()
        await store.append(event)  # NON-BLOCKING
    """

    def __init__(self, db_path: str = ".orchestrator_events.db"):
        self.db_path = Path(db_path)
        self._conn: aiosqlite.Connection | None = None
        self._lock: asyncio.Lock | None = None
        self._write_lock: asyncio.Lock | None = None
        self._initialized = False

    async def _get_conn(self) -> aiosqlite.Connection:
        """
        Get or create async connection.

        Lazy initialization - connection created on first use.
        Thread-safe via lock.
        """
        if self._lock is None:
            self._lock = asyncio.Lock()

        if self._conn is None:
            async with self._lock:
                if self._conn is None:
                    # Ensure parent directory exists
                    self.db_path.parent.mkdir(parents=True, exist_ok=True)

                    # Create connection
                    self._conn = await aiosqlite.connect(str(self.db_path))
                    self._conn.row_factory = aiosqlite.Row

                    # Enable WAL mode for concurrent reads
                    await self._conn.execute("PRAGMA journal_mode=WAL")

                    # Initialize schema if needed
                    if not self._initialized:
                        await self._init_schema()
                        self._initialized = True

                    logger.debug(f"AsyncEventStore connected to {self.db_path}")

        return self._conn

    async def _init_schema(self) -> None:
        """Initialize database schema."""
        await self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                aggregate_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_events_aggregate ON events(aggregate_id);
            CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
            CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
        """)
        await self._conn.commit()

    async def append(self, event: DomainEvent) -> int:
        """
        Persist an event (async, non-blocking).

        Args:
            event: DomainEvent to persist

        Returns:
            Event ID (rowid)
        """
        if self._write_lock is None:
            self._write_lock = asyncio.Lock()

        conn = await self._get_conn()

        async with self._write_lock:  # Single writer pattern
            cursor = await conn.execute(
                "INSERT INTO events (event_type, aggregate_id, timestamp, data) VALUES (?, ?, ?, ?)",
                (
                    event.event_type.name,
                    event.aggregate_id,
                    event.timestamp.isoformat(),
                    json.dumps(event.to_dict())
                )
            )
            await conn.commit()
            return cursor.lastrowid

    async def append_batch(self, events: list[DomainEvent]) -> list[int]:
        """
        Persist multiple events in a single transaction.

        Args:
            events: List of DomainEvent to persist

        Returns:
            List of event IDs
        """
        if not events:
            return []

        if self._write_lock is None:
            self._write_lock = asyncio.Lock()

        conn = await self._get_conn()
        ids = []

        async with self._write_lock:
            for event in events:
                cursor = await conn.execute(
                    "INSERT INTO events (event_type, aggregate_id, timestamp, data) VALUES (?, ?, ?, ?)",
                    (
                        event.event_type.name,
                        event.aggregate_id,
                        event.timestamp.isoformat(),
                        json.dumps(event.to_dict())
                    )
                )
                ids.append(cursor.lastrowid)

            await conn.commit()

        return ids

    async def get_events(
        self,
        aggregate_id: str | None = None,
        event_type: EventType | None = None,
        since: datetime | None = None,
        limit: int = 1000
    ) -> list[DomainEvent]:
        """
        Query events with filters (async, non-blocking).

        Args:
            aggregate_id: Filter by aggregate ID
            event_type: Filter by event type
            since: Filter events after this timestamp
            limit: Maximum number of events to return

        Returns:
            List of DomainEvent objects
        """
        if not HAS_UNIFIED_EVENTS:
            logger.warning("unified_events not available, returning raw data")
            return []

        conn = await self._get_conn()

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

        query += " ORDER BY timestamp ASC LIMIT ?"
        params.append(limit)

        events = []
        async with conn.execute(query, params) as cursor:
            async for row in cursor:
                try:
                    data = json.loads(row["data"])
                    events.append(DomainEvent.from_dict(data))
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Failed to parse event: {e}")

        return events

    async def get_aggregate(self, aggregate_id: str) -> list[DomainEvent]:
        """
        Get all events for an aggregate (for replay).

        Args:
            aggregate_id: Aggregate ID to query

        Returns:
            List of DomainEvent objects in chronological order
        """
        return await self.get_events(aggregate_id=aggregate_id, limit=10000)

    async def get_event_count(
        self,
        aggregate_id: str | None = None,
        event_type: EventType | None = None,
        since: datetime | None = None
    ) -> int:
        """
        Count events matching filters.

        Args:
            aggregate_id: Filter by aggregate ID
            event_type: Filter by event type
            since: Filter events after this timestamp

        Returns:
            Number of matching events
        """
        conn = await self._get_conn()

        query = "SELECT COUNT(*) FROM events WHERE 1=1"
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

        async with conn.execute(query, params) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else 0

    async def delete_events(
        self,
        aggregate_id: str | None = None,
        before: datetime | None = None
    ) -> int:
        """
        Delete events (for cleanup/retention).

        Args:
            aggregate_id: Delete events for this aggregate
            before: Delete events before this timestamp

        Returns:
            Number of deleted events
        """
        if self._write_lock is None:
            self._write_lock = asyncio.Lock()

        conn = await self._get_conn()

        query = "DELETE FROM events WHERE 1=1"
        params = []

        if aggregate_id:
            query += " AND aggregate_id = ?"
            params.append(aggregate_id)
        if before:
            query += " AND timestamp < ?"
            params.append(before.isoformat())

        async with self._write_lock:
            cursor = await conn.execute(query, params)
            await conn.commit()
            return cursor.rowcount

    async def get_stats(self) -> dict:
        """
        Get event store statistics.

        Returns:
            Dict with event counts and size info
        """
        conn = await self._get_conn()

        stats = {}

        # Total events
        async with conn.execute("SELECT COUNT(*) FROM events") as cursor:
            row = await cursor.fetchone()
            stats["total_events"] = row[0] if row else 0

        # Events by type
        stats["by_type"] = {}
        async with conn.execute(
            "SELECT event_type, COUNT(*) as cnt FROM events GROUP BY event_type"
        ) as cursor:
            async for row in cursor:
                stats["by_type"][row[0]] = row[1]

        # Database size
        if self.db_path.exists():
            stats["db_size_bytes"] = self.db_path.stat().st_size

        return stats

    async def checkpoint(self) -> None:
        """
        Force WAL checkpoint to main database.

        Call this periodically to keep WAL file small.
        """
        conn = await self._get_conn()
        await conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        logger.debug("EventStore WAL checkpointed")

    async def close(self) -> None:
        """
        Close connection gracefully.

        IMPORTANT: Must be called before event loop closes
        to allow aiosqlite background thread to finish.
        """
        if self._conn is not None:
            try:
                await self._conn.close()
                # Yield control so aiosqlite background thread can finish
                await asyncio.sleep(0)
                logger.debug("AsyncEventStore connection closed")
            except Exception as e:
                logger.warning(f"Error closing AsyncEventStore: {e}")
            finally:
                self._conn = None

    async def __aenter__(self) -> AsyncEventStore:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()


# Singleton instance for convenience
_instance: AsyncEventStore | None = None


def get_async_event_store() -> AsyncEventStore:
    """Get singleton AsyncEventStore instance."""
    global _instance
    if _instance is None:
        _instance = AsyncEventStore()
    return _instance


def reset_async_event_store() -> None:
    """Reset singleton (for testing)."""
    global _instance
    _instance = None
