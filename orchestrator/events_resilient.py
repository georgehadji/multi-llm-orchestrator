"""
Resilient Event Store with Corruption Resistance
================================================

Implements the minimax regret improvement for Black Swan Scenario 1:
Event Store Corruption/Loss

Features:
- Write-Ahead Logging (WAL) for durability
- Synchronous writes to primary + async to secondary
- Checksum validation on read
- Automatic failover to replica
- Corruption detection and repair

Usage:
    from orchestrator.events_resilient import ResilientEventStore

    store = ResilientEventStore(
        primary_path=".events/primary.db",
        replica_paths=[".events/replica1.db", ".events/replica2.db"],
    )
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .events import DomainEvent, EventStore
from .log_config import get_logger

if TYPE_CHECKING:
    from datetime import datetime

logger = get_logger(__name__)


class ResilientEventStore(EventStore):
    """
    Event store with corruption resistance and automatic recovery.

    Implements defense in depth against data loss:
    1. WAL mode for crash safety
    2. Checksum validation
    3. Async replication
    4. Automatic failover
    """

    def __init__(
        self,
        primary_path: str,
        replica_paths: list[str],
        sync_mode: str = "WAL",
    ):
        self.primary_path = Path(primary_path)
        self.primary_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize primary
        self.primary = self._create_store(primary_path)
        self._enable_wal(primary_path)

        # Initialize replicas
        self.replicas = []
        for path in replica_paths:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self.replicas.append(self._create_store(path))

        self.checksums: dict[str, str] = {}
        self.sync_mode = sync_mode

        logger.info(f"Initialized resilient event store with {len(replicas)} replicas")

    def _create_store(self, path: str) -> SQLiteEventStore:
        """Create a SQLite event store."""
        from .events import SQLiteEventStore
        return SQLiteEventStore(path)

    def _enable_wal(self, db_path: str) -> None:
        """Enable Write-Ahead Logging for crash safety."""
        with sqlite3.connect(db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA wal_autocheckpoint=1000")
        logger.debug(f"Enabled WAL mode for {db_path}")

    async def append(self, event: DomainEvent) -> None:
        """
        Append event with checksum and replication.

        Flow:
        1. Calculate checksum
        2. Write to primary with retry
        3. Async replicate to secondaries
        """
        # 1. Calculate checksum
        event_data = event.to_dict()
        checksum = hashlib.sha256(
            json.dumps(event_data, sort_keys=True, default=str).encode()
        ).hexdigest()

        # 2. Write to primary with retry
        for attempt in range(3):
            try:
                await self.primary.append(event)
                break
            except sqlite3.DatabaseError as e:
                logger.warning(f"Primary write failed (attempt {attempt + 1}): {e}")
                if attempt == 2:
                    raise  # Failed after retries
                await asyncio.sleep(0.1 * (2 ** attempt))

        # 3. Store checksum
        self.checksums[event.event_id] = checksum

        # 4. Async replicate to secondaries
        asyncio.create_task(self._replicate_with_retry(event, checksum))

    async def _replicate_with_retry(
        self,
        event: DomainEvent,
        checksum: str,
    ) -> None:
        """Replicate to secondary stores with exponential backoff."""
        for replica in self.replicas:
            for attempt in range(5):
                try:
                    await replica.append(event)
                    break
                except Exception as e:
                    logger.warning(f"Replication failed (attempt {attempt + 1}): {e}")
                    if attempt < 4:
                        await asyncio.sleep(0.5 * (2 ** attempt))
                    else:
                        logger.error(f"Failed to replicate to {replica.db_path}")

    async def get_events(
        self,
        aggregate_id: str | None = None,
        event_types: list[str] | None = None,
        since: datetime | None = None,
        limit: int | None = None,
    ) -> list[DomainEvent]:
        """
        Read events with corruption detection.
        """
        try:
            events = await self.primary.get_events(
                aggregate_id=aggregate_id,
                event_types=event_types,
                since=since,
                limit=limit,
            )

            # Validate checksums
            corrupted = []
            for event in events:
                if event.event_id in self.checksums:
                    expected = self.checksums[event.event_id]
                    actual = self._calculate_checksum(event)
                    if expected != actual:
                        logger.error(f"Checksum mismatch for event {event.event_id}")
                        corrupted.append(event)

            if corrupted:
                logger.error(f"Detected {len(corrupted)} corrupted events")
                # Attempt recovery
                recovered = await self._recover_events(corrupted)

                # Replace corrupted with recovered
                for i, event in enumerate(events):
                    if event in corrupted:
                        recovered_event = next(
                            (r for r in recovered if r.event_id == event.event_id),
                            None
                        )
                        if recovered_event:
                            events[i] = recovered_event

            return events

        except sqlite3.DatabaseError as e:
            logger.critical(f"Primary database error: {e}, failing over")
            return await self._failover_to_replica(
                aggregate_id=aggregate_id,
                event_types=event_types,
                since=since,
                limit=limit,
            )

    def _calculate_checksum(self, event: DomainEvent) -> str:
        """Calculate SHA-256 checksum of event."""
        data = event.to_dict()
        return hashlib.sha256(
            json.dumps(data, sort_keys=True, default=str).encode()
        ).hexdigest()

    async def _recover_events(
        self,
        corrupted: list[DomainEvent],
    ) -> list[DomainEvent]:
        """Attempt to recover corrupted events from replicas."""
        recovered = []

        for event in corrupted:
            for replica in self.replicas:
                try:
                    # Try to find event in replica
                    replica_events = await replica.get_events(
                        aggregate_id=event.aggregate_id,
                    )
                    for re in replica_events:
                        if re.event_id == event.event_id:
                            # Verify checksum
                            if self._calculate_checksum(re) == self.checksums.get(event.event_id):
                                recovered.append(re)
                                logger.info(f"Recovered event {event.event_id} from replica")
                                break
                except Exception as e:
                    logger.warning(f"Failed to recover from replica: {e}")

        return recovered

    async def _failover_to_replica(
        self,
        aggregate_id: str | None = None,
        event_types: list[str] | None = None,
        since: datetime | None = None,
        limit: int | None = None,
    ) -> list[DomainEvent]:
        """Failover to first healthy replica."""
        for replica in self.replicas:
            try:
                events = await replica.get_events(
                    aggregate_id=aggregate_id,
                    event_types=event_types,
                    since=since,
                    limit=limit,
                )
                logger.info(f"Failover successful to {replica.db_path}")
                return events
            except Exception as e:
                logger.error(f"Replica {replica.db_path} also failed: {e}")

        raise Exception("All event stores failed - catastrophic data loss")

    async def replay(
        self,
        handler,
        event_types: list[str] | None = None,
        since: datetime | None = None,
    ) -> None:
        """Replay events from primary or failover."""
        try:
            await self.primary.replay(handler, event_types, since)
        except Exception:
            # Try replicas
            for replica in self.replicas:
                try:
                    await replica.replay(handler, event_types, since)
                    return
                except Exception:
                    continue
            raise

    async def close(self) -> None:
        """Close all stores."""
        await self.primary.close()
        for replica in self.replicas:
            await replica.close()

    async def verify_integrity(self) -> dict[str, Any]:
        """
        Verify integrity of all stores.

        Returns report of any discrepancies.
        """
        report = {
            "primary_healthy": True,
            "replica_health": [],
            "corrupted_events": [],
            "missing_in_replicas": [],
        }

        # Check primary
        try:
            primary_events = await self.primary.get_events()
            for event in primary_events:
                if event.event_id in self.checksums:
                    if self._calculate_checksum(event) != self.checksums[event.event_id]:
                        report["corrupted_events"].append(event.event_id)
        except Exception as e:
            report["primary_healthy"] = False
            logger.error(f"Primary integrity check failed: {e}")

        # Check replicas
        for i, replica in enumerate(self.replicas):
            try:
                replica_events = await replica.get_events()
                replica_ids = {e.event_id for e in replica_events}
                primary_ids = {e.event_id for e in primary_events}

                missing = primary_ids - replica_ids
                if missing:
                    report["missing_in_replicas"].append({
                        "replica_index": i,
                        "missing_count": len(missing),
                    })

                report["replica_health"].append({"index": i, "healthy": True})
            except Exception as e:
                report["replica_health"].append({"index": i, "healthy": False, "error": str(e)})

        return report
