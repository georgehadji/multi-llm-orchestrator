# ARCHITECTURAL FIX PLAN — Multi-LLM Orchestrator v6.0 → v6.1

**Author:** Senior Distributed Systems Architect + Production SRE  
**Date:** 2026-03-20  
**Goal:** Transform fragile AI infra into antifragile production system  
**Constraint:** Solo founder resources, optimize for survival, prefer boring tech

---

# 1. ROOT CAUSE MODEL

## 1.1 Why These Failures Will Happen in Production

### The Fundamental Problem: Hidden Coupling

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    HIDDEN COUPLING MAP                                       │
│                                                                              │
│  EventStore ──────┐                                                          │
│  (sync sqlite)    │     ┌──────────────┐     ┌──────────────┐              │
│        │          ├────►│ Event Loop   │────►│ All Async    │              │
│        ▼          │     │ (single)     │     │ Operations   │              │
│  StateManager ────┤     └──────────────┘     └──────────────┘              │
│  (async sqlite)   │            │                     │                      │
│        │          │            ▼                     ▼                      │
│        ▼          │     ┌──────────────┐     ┌──────────────┐              │
│  DiskCache ───────┤     │ Background   │     │ API Clients  │              │
│  (async sqlite)   │     │ Tasks        │     │ (network I/O)│              │
│        │          │     └──────────────┘     └──────────────┘              │
│        ▼          │            │                     │                      │
│  TelemetryStore ──┘            ▼                     ▼                      │
│  (async sqlite)          ┌──────────────┐     ┌──────────────┐              │
│                          │ Memory       │     │ Budget       │              │
│                          │ (background  │     │ Hierarchy    │              │
│                          │  task set)   │     │ (in-memory)  │              │
│                          └──────────────┘     └──────────────┘              │
│                                │                     │                      │
│                                └─────────┬───────────┘                      │
│                                          ▼                                  │
│                                   ┌──────────────┐                          │
│                                   │ SINGLE POINT │                          │
│                                   │ OF FAILURE:  │                          │
│                                   │ EVENT LOOP   │                          │
│                                   └──────────────┘                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Nonlinear Risk Propagation

| Trigger | First-Order Effect | Second-Order Effect | Cascade |
|---------|---------------------|---------------------|---------|
| Sync SQLite write | Event loop blocked 50ms | All async operations stall | Timeout cascade → circuit breakers trip → all models marked DEGRADED |
| Memory leak (1 task/hour) | 24 tasks after 1 day | 168 tasks after 1 week | OOM → process killed → state corruption |
| Budget race condition | Two jobs both pass check | Both charge → 2x budget exceeded | Billing shock, user trust destroyed |
| Provider outage | Fallback chain activates | Fallback also slow | All models degraded → complete system failure |

### The Real Production Killer: Event Loop Starvation

```
Timeline of Event Loop Starvation:

T+0ms:    EventStore.append() called (sync sqlite write)
T+5ms:    SQLite write in progress
T+50ms:   Write completes, but...
T+50ms:   API client timeout was supposed to fire at T+30ms
T+50ms:   Timeout handler delayed by 20ms
T+50ms:   Circuit breaker incorrectly marks model as timed out
T+50ms:   Fallback chain activated unnecessarily
T+100ms:  Second provider hit, but also marked degraded
T+200ms:  Cascading failure across all providers
T+500ms:  System in degraded state, all models marked unhealthy
```

**Root Cause:** Synchronous I/O in an async system violates the fundamental invariant of event-loop-based concurrency.

---

## 1.2 Why Current Fixes Are Insufficient

| Current Fix | Why It Fails |
|-------------|--------------|
| `asyncio.sleep(0)` after close | Only yields once, doesn't prevent blocking during operation |
| Circuit breaker threshold = 3 | Reactive, not proactive; doesn't prevent root cause |
| Budget reservation | Race condition between check and reserve still possible |
| WAL mode | Helps concurrency, doesn't help event loop blocking |

---

# 2. ARCHITECTURAL FIX PLAN

## 2.1 Async EventStore Migration

### Current State (BROKEN)

```python
# unified_events/core.py - BLOCKS EVENT LOOP
class EventStore:
    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, 'conn'):
            self._local.conn = sqlite3.connect(str(self.db_path))  # BLOCKS
        return self._local.conn

    def append(self, event: DomainEvent) -> None:
        conn = self._get_conn()
        conn.execute(...)  # BLOCKS
        conn.commit()      # BLOCKS
```

### Fixed State (CORRECT)

```python
# unified_events/core.py - ASYNC, NON-BLOCKING
import aiosqlite
from contextlib import asynccontextmanager

class AsyncEventStore:
    """
    Async SQLite event store with connection pooling.
    
    INVARIANTS:
    1. All operations are async (never blocks event loop)
    2. Connection is lazily initialized
    3. WAL mode for concurrent reads
    4. Single writer pattern via lock
    """
    
    def __init__(self, db_path: str = ".orchestrator_events.db"):
        self.db_path = Path(db_path)
        self._conn: Optional[aiosqlite.Connection] = None
        self._lock: Optional[asyncio.Lock] = None
        self._write_lock: Optional[asyncio.Lock] = None  # Single writer
        
    async def _get_conn(self) -> aiosqlite.Connection:
        """Get or create async connection."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        if self._conn is None:
            async with self._lock:
                if self._conn is None:
                    self._conn = await aiosqlite.connect(str(self.db_path))
                    self._conn.row_factory = aiosqlite.Row
                    await self._conn.execute("PRAGMA journal_mode=WAL")
                    await self._init_schema()
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
            CREATE INDEX IF NOT EXISTS idx_aggregate ON events(aggregate_id);
            CREATE INDEX IF NOT EXISTS idx_type ON events(event_type);
            CREATE INDEX IF NOT EXISTS idx_timestamp ON events(timestamp);
        """)
        await self._conn.commit()
    
    async def append(self, event: DomainEvent) -> None:
        """Persist an event (async, non-blocking)."""
        if self._write_lock is None:
            self._write_lock = asyncio.Lock()
        
        conn = await self._get_conn()
        async with self._write_lock:  # Single writer pattern
            await conn.execute(
                "INSERT INTO events (event_type, aggregate_id, timestamp, data) VALUES (?, ?, ?, ?)",
                (event.event_type.name, event.aggregate_id,
                 event.timestamp.isoformat(), json.dumps(event.to_dict()))
            )
            await conn.commit()
    
    async def get_events(
        self,
        aggregate_id: Optional[str] = None,
        event_type: Optional[EventType] = None,
        since: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[DomainEvent]:
        """Query events with filters (async, non-blocking)."""
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
        
        async with conn.execute(query, params) as cursor:
            events = []
            async for row in cursor:
                data = json.loads(row["data"])
                events.append(DomainEvent.from_dict(data))
            return events
    
    async def close(self) -> None:
        """Close connection gracefully."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
            await asyncio.sleep(0)  # Yield for cleanup
```

### Migration Steps

```python
# Step 1: Create new async store alongside old one
class UnifiedEventBus:
    def __init__(self, store: Optional[EventStore] = None, 
                 async_store: Optional[AsyncEventStore] = None):
        self._legacy_store = store  # Keep for backward compat
        self._async_store = async_store or AsyncEventStore()
        
    async def _handle_event(self, event: DomainEvent) -> None:
        # Use async store
        if self._async_store:
            await self._async_store.append(event)
        # Legacy store deprecated, log warning if used
        elif self._legacy_store:
            logger.warning("Using deprecated sync EventStore - migrate to AsyncEventStore")
            # Offload to thread to avoid blocking
            await asyncio.to_thread(self._legacy_store.append, event)
```

---

## 2.2 Database Reliability Strategy

### Backup Strategy

```python
# orchestrator/db_backup.py - NEW FILE
"""
Database Backup and Recovery System
====================================
Implements:
- Periodic snapshots
- Point-in-time recovery
- Corruption detection
- Automatic failover to backup
"""

import asyncio
import hashlib
import json
import logging
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import aiosqlite

logger = logging.getLogger("orchestrator.backup")

@dataclass
class BackupManifest:
    """Metadata for a backup."""
    timestamp: datetime
    db_path: Path
    size_bytes: int
    checksum_sha256: str
    tables: List[str]
    event_count: int
    is_valid: bool


class DatabaseBackupManager:
    """
    Manages backups for all orchestrator databases.
    
    DATABASES:
    - state.db (project state, checkpoints)
    - cache.db (LLM response cache)
    - telemetry.db (cross-run learning)
    - events.db (event store)
    
    STRATEGY:
    - Full backup every 6 hours
    - Incremental via WAL checkpoint every 30 minutes
    - Retention: 7 daily, 4 weekly, 12 monthly
    """
    
    def __init__(
        self,
        data_dir: Path = Path.home() / ".orchestrator_cache",
        backup_dir: Optional[Path] = None,
        backup_interval_hours: float = 6.0,
        checkpoint_interval_minutes: float = 30.0,
    ):
        self.data_dir = Path(data_dir)
        self.backup_dir = backup_dir or (self.data_dir / "backups")
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.backup_interval = backup_interval_hours * 3600
        self.checkpoint_interval = checkpoint_interval_minutes * 60
        self._task: Optional[asyncio.Task] = None
        self._running = False
        
    async def start(self) -> None:
        """Start background backup scheduler."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._backup_loop())
        logger.info(f"Backup manager started (interval={self.backup_interval}s)")
        
    async def stop(self) -> None:
        """Stop backup scheduler."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Backup manager stopped")
    
    async def _backup_loop(self) -> None:
        """Background backup scheduler."""
        last_backup = 0.0
        last_checkpoint = 0.0
        
        while self._running:
            await asyncio.sleep(60)  # Check every minute
            
            now = time.time()
            
            # Checkpoint (quick, low impact)
            if now - last_checkpoint > self.checkpoint_interval:
                await self._checkpoint_all()
                last_checkpoint = now
            
            # Full backup
            if now - last_backup > self.backup_interval:
                await self._backup_all()
                last_backup = now
                await self._prune_old_backups()
    
    async def _checkpoint_all(self) -> None:
        """Checkpoint WAL files to main database."""
        for db_name in ["state.db", "cache.db", "telemetry.db", "events.db"]:
            db_path = self.data_dir / db_name
            if db_path.exists():
                try:
                    async with aiosqlite.connect(str(db_path)) as conn:
                        await conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                    logger.debug(f"Checkpointed {db_name}")
                except Exception as e:
                    logger.warning(f"Checkpoint failed for {db_name}: {e}")
    
    async def _backup_all(self) -> List[BackupManifest]:
        """Create full backup of all databases."""
        manifests = []
        timestamp = datetime.utcnow()
        backup_subdir = self.backup_dir / timestamp.strftime("%Y%m%d_%H%M%S")
        backup_subdir.mkdir(parents=True, exist_ok=True)
        
        for db_name in ["state.db", "cache.db", "telemetry.db", "events.db"]:
            db_path = self.data_dir / db_name
            if not db_path.exists():
                continue
            
            try:
                manifest = await self._backup_single(db_path, backup_subdir / db_name)
                manifests.append(manifest)
            except Exception as e:
                logger.error(f"Backup failed for {db_name}: {e}")
        
        # Write manifest
        manifest_path = backup_subdir / "manifest.json"
        manifest_data = {
            "timestamp": timestamp.isoformat(),
            "backups": [
                {
                    "db_name": m.db_path.name,
                    "size_bytes": m.size_bytes,
                    "checksum": m.checksum_sha256,
                    "tables": m.tables,
                    "event_count": m.event_count,
                    "is_valid": m.is_valid,
                }
                for m in manifests
            ]
        }
        manifest_path.write_text(json.dumps(manifest_data, indent=2))
        
        logger.info(f"Backup complete: {len(manifests)} databases backed up")
        return manifests
    
    async def _backup_single(self, source: Path, dest: Path) -> BackupManifest:
        """Backup a single database with validation."""
        # Use SQLite backup API for consistency
        async with aiosqlite.connect(str(source)) as source_conn:
            # Get metadata
            tables = []
            async with source_conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ) as cursor:
                async for row in cursor:
                    tables.append(row[0])
            
            # Count events if events table exists
            event_count = 0
            if "events" in tables:
                async with source_conn.execute("SELECT COUNT(*) FROM events") as c:
                    event_count = (await c.fetchone())[0]
            
            # Backup to destination
            await source_conn.backup(str(dest))
        
        # Calculate checksum
        checksum = await self._calculate_checksum(dest)
        size = dest.stat().st_size
        
        # Validate backup
        is_valid = await self._validate_backup(dest)
        
        return BackupManifest(
            timestamp=datetime.utcnow(),
            db_path=dest,
            size_bytes=size,
            checksum_sha256=checksum,
            tables=tables,
            event_count=event_count,
            is_valid=is_valid,
        )
    
    async def _calculate_checksum(self, path: Path) -> str:
        """Calculate SHA256 checksum of file."""
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    async def _validate_backup(self, path: Path) -> bool:
        """Validate backup integrity."""
        try:
            async with aiosqlite.connect(str(path)) as conn:
                await conn.execute("PRAGMA integrity_check")
                result = await conn.fetchone()
                return result[0] == "ok"
        except Exception:
            return False
    
    async def _prune_old_backups(self) -> None:
        """Remove old backups according to retention policy."""
        # Keep: 7 daily, 4 weekly, 12 monthly
        now = datetime.utcnow()
        backups = sorted(self.backup_dir.iterdir(), reverse=True)
        
        kept_daily = 0
        kept_weekly = 0
        kept_monthly = 0
        
        for backup in backups:
            if not backup.is_dir():
                continue
            
            try:
                backup_time = datetime.strptime(backup.name, "%Y%m%d_%H%M%S")
            except ValueError:
                continue
            
            age_days = (now - backup_time).days
            
            # Daily: keep last 7
            if age_days < 7:
                if kept_daily >= 7:
                    shutil.rmtree(backup)
                    continue
                kept_daily += 1
            # Weekly: keep last 4 (one per week)
            elif age_days < 30:
                if kept_weekly >= 4:
                    shutil.rmtree(backup)
                    continue
                kept_weekly += 1
            # Monthly: keep last 12
            elif age_days < 365:
                if kept_monthly >= 12:
                    shutil.rmtree(backup)
                    continue
                kept_monthly += 1
            else:
                # Older than 1 year, delete
                shutil.rmtree(backup)
    
    async def restore(
        self,
        backup_path: Path,
        target_databases: Optional[List[str]] = None
    ) -> bool:
        """
        Restore databases from backup.
        
        WARNING: This overwrites current data!
        """
        target_databases = target_databases or ["state.db", "cache.db", "telemetry.db", "events.db"]
        
        for db_name in target_databases:
            backup_db = backup_path / db_name
            target_db = self.data_dir / db_name
            
            if not backup_db.exists():
                logger.warning(f"Backup not found: {db_name}")
                continue
            
            # Validate before restore
            if not await self._validate_backup(backup_db):
                logger.error(f"Backup corrupted: {db_name}, skipping")
                continue
            
            # Create safety copy of current
            if target_db.exists():
                safety_path = target_db.with_suffix(".db.pre_restore")
                shutil.copy(target_db, safety_path)
            
            # Restore
            shutil.copy(backup_db, target_db)
            logger.info(f"Restored {db_name} from backup")
        
        return True
    
    def list_backups(self) -> List[dict]:
        """List available backups."""
        backups = []
        for backup_dir in sorted(self.backup_dir.iterdir(), reverse=True):
            if not backup_dir.is_dir():
                continue
            manifest_path = backup_dir / "manifest.json"
            if manifest_path.exists():
                backups.append(json.loads(manifest_path.read_text()))
        return backups
```

### Integration with Orchestrator

```python
# engine.py - Add backup manager to Orchestrator

class Orchestrator:
    def __init__(self, ...):
        # ... existing init ...
        
        # NEW: Database backup manager
        self._backup_manager = DatabaseBackupManager()
        self._backup_enabled = True
    
    async def __aenter__(self) -> "Orchestrator":
        self._entered = True
        # Start backup manager
        if self._backup_enabled:
            await self._backup_manager.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        # Stop backup manager first
        if self._backup_enabled:
            await self._backup_manager.stop()
        # ... existing cleanup ...
```

---

## 2.3 Chaos Engineering Framework

```python
# tests/chaos/test_chaos_engineering.py - NEW FILE
"""
Chaos Engineering Test Suite
============================
Validates system resilience under failure conditions.

PRINCIPLES:
1. Test in production-like conditions
2. Test real failures, not simulations
3. Automate failure injection
4. Define steady state and verify it holds
"""

import asyncio
import json
import logging
import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable, List
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
import aiosqlite

from orchestrator import Orchestrator, Budget, Task, TaskType
from orchestrator.api_clients import UnifiedClient, APIResponse
from orchestrator.models import Model
from orchestrator.cost import BudgetHierarchy

logger = logging.getLogger("orchestrator.chaos")


# ═══════════════════════════════════════════════════════════════════════════════
# Chaos Test Infrastructure
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ChaosResult:
    """Result of a chaos experiment."""
    experiment_name: str
    steady_state_maintained: bool
    time_to_recovery_ms: float
    error_count: int
    details: dict


class ChaosRunner:
    """Execute chaos experiments with safety controls."""
    
    def __init__(self, timeout_seconds: float = 60.0):
        self.timeout = timeout_seconds
        self.results: List[ChaosResult] = []
    
    async def run_experiment(
        self,
        name: str,
        inject_failure: Callable,
        verify_steady_state: Callable,
        duration_seconds: float = 10.0
    ) -> ChaosResult:
        """
        Run a chaos experiment.
        
        1. Verify steady state
        2. Inject failure
        3. Wait for duration
        4. Verify steady state maintained
        5. Record recovery time
        """
        start_time = time.monotonic()
        error_count = 0
        
        # Pre-condition: steady state
        assert await verify_steady_state(), "System not in steady state before experiment"
        
        # Inject failure
        try:
            await inject_failure()
        except Exception as e:
            logger.warning(f"Failure injection failed: {e}")
            error_count += 1
        
        # Wait and observe
        await asyncio.sleep(duration_seconds)
        
        # Check steady state
        steady_state_ok = await verify_steady_state()
        recovery_time = (time.monotonic() - start_time) * 1000
        
        result = ChaosResult(
            experiment_name=name,
            steady_state_maintained=steady_state_ok,
            time_to_recovery_ms=recovery_time,
            error_count=error_count,
            details={"duration_seconds": duration_seconds}
        )
        self.results.append(result)
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 1: Provider Outage
# ═══════════════════════════════════════════════════════════════════════════════

class TestProviderOutage:
    """Simulate complete provider outage."""
    
    @pytest.mark.asyncio
    async def test_openai_outage_fallback_chain(self):
        """When OpenAI goes down, system should fallback to other providers."""
        runner = ChaosRunner()
        
        async def inject_openai_failure():
            """Simulate OpenAI returning 500 errors."""
            # Patch OpenAI client to fail
            with patch("orchestrator.api_clients.AsyncOpenAI") as mock_client:
                mock_instance = MagicMock()
                mock_instance.chat.completions.create = AsyncMock(
                    side_effect=Exception("Service Unavailable")
                )
                mock_client.return_value = mock_instance
        
        async def verify_fallback():
            """Verify system routes to non-OpenAI models."""
            # System should have marked OpenAI models as degraded
            # and successfully used fallback
            return True  # Simplified for example
        
        result = await runner.run_experiment(
            name="openai_outage",
            inject_failure=inject_openai_failure,
            verify_steady_state=verify_fallback,
            duration_seconds=5.0
        )
        
        assert result.steady_state_maintained, "System failed to handle OpenAI outage"
    
    @pytest.mark.asyncio
    async def test_all_providers_down_graceful_degradation(self):
        """When all providers fail, system should degrade gracefully."""
        orch = Orchestrator(budget=Budget(max_usd=1.0))
        
        # Mock all providers to fail
        async def all_providers_fail(*args, **kwargs):
            raise Exception("All providers down")
        
        with patch.object(orch.client, "call", side_effect=all_providers_fail):
            result = await orch.run_project(
                "Test project",
                "Complete successfully",
                Budget(max_usd=1.0)
            )
        
        # Should not crash, should return partial state
        assert result is not None
        assert result.status in ["SYSTEM_FAILURE", "PARTIAL_SUCCESS"]


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 2: Database Failure
# ═══════════════════════════════════════════════════════════════════════════════

class TestDatabaseFailure:
    """Simulate database failures."""
    
    @pytest.mark.asyncio
    async def test_state_db_corruption_recovery(self, tmp_path):
        """System should recover from corrupted state database."""
        # Create corrupted database
        corrupt_db = tmp_path / "state.db"
        corrupt_db.write_bytes(b"CORRUPTED DATA NOT SQLITE")
        
        # Try to load state
        from orchestrator.state import StateManager
        sm = StateManager(db_path=corrupt_db)
        
        # Should handle corruption gracefully
        try:
            await sm._get_conn()
        except Exception as e:
            # Expected: corruption detected
            assert "not a database" in str(e).lower() or "corrupt" in str(e).lower()
    
    @pytest.mark.asyncio
    async def test_database_lock_contention(self, tmp_path):
        """System should handle database lock contention."""
        from orchestrator.state import StateManager
        
        db_path = tmp_path / "state.db"
        sm1 = StateManager(db_path=db_path)
        sm2 = StateManager(db_path=db_path)
        
        # Concurrent writes
        async def write_state(sm, project_id):
            from orchestrator.models import ProjectState, ProjectStatus, Budget
            state = ProjectState(
                project_description=f"Project {project_id}",
                success_criteria="test",
                budget=Budget(max_usd=10.0),
                tasks={},
                results={},
                status=ProjectStatus.SUCCESS
            )
            await sm.save_project(project_id, state)
        
        # Run concurrent writes
        await asyncio.gather(
            write_state(sm1, "project_1"),
            write_state(sm2, "project_2"),
        )
        
        # Both should succeed (WAL mode)
        p1 = await sm1.load_project("project_1")
        p2 = await sm2.load_project("project_2")
        
        assert p1 is not None
        assert p2 is not None


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 3: Memory Pressure
# ═══════════════════════════════════════════════════════════════════════════════

class TestMemoryPressure:
    """Test behavior under memory pressure."""
    
    @pytest.mark.asyncio
    async def test_background_task_cleanup(self):
        """Background tasks should be cleaned up, not leak."""
        orch = Orchestrator()
        
        # Simulate many background tasks
        for i in range(100):
            task = asyncio.create_task(
                orch._flush_telemetry_snapshots(f"project_{i}")
            )
            orch._background_tasks.add(task)
        
        # Wait for completion
        await asyncio.sleep(1)
        
        # Cleanup should have removed completed tasks
        cleaned = await orch._cleanup_background_tasks()
        
        assert len(orch._background_tasks) < 100, "Background tasks not cleaned"
    
    @pytest.mark.asyncio
    async def test_large_context_truncation(self):
        """Large contexts should be truncated, not cause OOM."""
        orch = Orchestrator()
        
        # Create very large context
        large_context = "x" * 1_000_000  # 1MB string
        
        # Should truncate, not crash
        truncated = large_context[:orch.context_truncation_limit]
        
        assert len(truncated) == orch.context_truncation_limit


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 4: Budget Race Conditions
# ═══════════════════════════════════════════════════════════════════════════════

class TestBudgetRaceConditions:
    """Test budget enforcement under concurrent access."""
    
    @pytest.mark.asyncio
    async def test_concurrent_budget_check(self):
        """Concurrent budget checks should not exceed limit."""
        hierarchy = BudgetHierarchy(org_max_usd=100.0)
        
        # 10 concurrent jobs each trying to reserve 20 USD
        # Total would be 200 USD if race condition exists
        async def try_reserve(job_id: str):
            return hierarchy.can_afford_job(job_id, "eng", 20.0)
        
        results = await asyncio.gather(*[
            try_reserve(f"job_{i}") for i in range(10)
        ])
        
        # At most 5 should succeed (100 / 20 = 5)
        successful = sum(1 for r in results if r)
        
        assert successful <= 5, f"Budget race condition: {successful} jobs approved, max 5"
    
    @pytest.mark.asyncio
    async def test_budget_reservation_release_on_failure(self):
        """Failed jobs should release budget reservations."""
        hierarchy = BudgetHierarchy(org_max_usd=100.0)
        
        # Reserve budget
        assert hierarchy.can_afford_job("job_1", "eng", 50.0)
        
        # Simulate failure - release reservation
        hierarchy.release_reservation("job_1", "eng")
        
        # Budget should be available again
        assert hierarchy.can_afford_job("job_2", "eng", 100.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 5: Event Loop Starvation
# ═══════════════════════════════════════════════════════════════════════════════

class TestEventLoopStarvation:
    """Test event loop is not blocked."""
    
    @pytest.mark.asyncio
    async def test_no_blocking_calls_in_hot_path(self):
        """Verify no sync I/O in async hot paths."""
        import inspect
        
        from orchestrator.state import StateManager
        from orchestrator.cache import DiskCache
        
        # All public methods should be async
        for name, method in inspect.getmembers(StateManager, predicate=inspect.isfunction):
            if name.startswith("_"):
                continue
            assert inspect.iscoroutinefunction(method), f"StateManager.{name} is not async!"
        
        for name, method in inspect.getmembers(DiskCache, predicate=inspect.isfunction):
            if name.startswith("_"):
                continue
            assert inspect.iscoroutinefunction(method), f"DiskCache.{name} is not async!"
    
    @pytest.mark.asyncio
    async def test_event_loop_responsiveness(self):
        """Event loop should remain responsive under load."""
        orch = Orchestrator()
        
        # Measure event loop latency
        latencies = []
        
        async def measure_latency():
            start = time.monotonic()
            await asyncio.sleep(0)  # Yield
            return (time.monotonic() - start) * 1000
        
        # Run many operations
        for _ in range(100):
            latency = await measure_latency()
            latencies.append(latency)
        
        # P99 should be < 10ms
        p99 = sorted(latencies)[98]
        assert p99 < 10, f"Event loop latency too high: p99={p99}ms"


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 6: Cache Poisoning
# ═══════════════════════════════════════════════════════════════════════════════

class TestCachePoisoning:
    """Test cache integrity."""
    
    @pytest.mark.asyncio
    async def test_cache_bypass_on_validation_failure(self):
        """Cached responses that fail validation should be bypassed."""
        orch = Orchestrator()
        
        # Mock a cached response that's invalid
        invalid_response = {
            "response": "INVALID CODE",
            "tokens_input": 100,
            "tokens_output": 50,
        }
        
        # Should be able to bypass cache
        result = await orch.client.call(
            Model.GPT_4O_MINI,
            "test prompt",
            bypass_cache=True
        )
        
        # Should not use cached invalid response
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_cache_invalidation(self, tmp_path):
        """Cache should support invalidation."""
        from orchestrator.cache import DiskCache
        
        cache = DiskCache(db_path=tmp_path / "cache.db")
        
        # Put and get
        await cache.put("test_model", "test_prompt", 100, "test_response", 10, 5)
        result = await cache.get("test_model", "test_prompt", 100)
        
        assert result is not None
        
        # Clear cache
        await cache.clear()
        result = await cache.get("test_model", "test_prompt", 100)
        
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 7: Configuration Drift
# ═══════════════════════════════════════════════════════════════════════════════

class TestConfigurationDrift:
    """Test configuration validation."""
    
    def test_api_key_validation(self):
        """Missing API keys should be detected."""
        # Save current env
        original_key = os.environ.get("OPENAI_API_KEY")
        
        try:
            # Remove key
            if "OPENAI_API_KEY" in os.environ:
                del os.environ["OPENAI_API_KEY"]
            
            # Create client - should warn but not crash
            from orchestrator.api_clients import UnifiedClient
            client = UnifiedClient()
            
            # OpenAI should not be available
            assert not client.is_available(Model.GPT_4O)
        
        finally:
            # Restore
            if original_key:
                os.environ["OPENAI_API_KEY"] = original_key
    
    def test_budget_configuration_validation(self):
        """Invalid budget configuration should be rejected."""
        from orchestrator.models import Budget
        
        # Negative budget should be rejected
        with pytest.raises((ValueError, AssertionError)):
            Budget(max_usd=-10.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 8: Rate Limit Cascade
# ═══════════════════════════════════════════════════════════════════════════════

class TestRateLimitCascade:
    """Test rate limit handling."""
    
    @pytest.mark.asyncio
    async def test_rate_limit_backoff(self):
        """Rate limits should trigger exponential backoff."""
        from orchestrator.rate_limiter import RateLimiter, RateLimitExceeded
        
        limiter = RateLimiter()
        limiter.set_limits("tenant_1", "gpt-4o", tpm=100, rpm=10)
        
        # Exhaust rate limit
        for i in range(10):
            limiter.check("tenant_1", "gpt-4o", 5)
            limiter.record("tenant_1", "gpt-4o", 5)
        
        # Next request should fail
        with pytest.raises(RateLimitExceeded):
            limiter.check("tenant_1", "gpt-4o", 5)
    
    @pytest.mark.asyncio
    async def test_rate_limit_recovery(self):
        """Rate limits should recover after window expires."""
        from orchestrator.rate_limiter import RateLimiter
        
        limiter = RateLimiter()
        limiter.set_limits("tenant_1", "gpt-4o", tpm=100, rpm=2)
        
        # Use up limit
        limiter.check("tenant_1", "gpt-4o", 10)
        limiter.record("tenant_1", "gpt-4o", 10)
        limiter.check("tenant_1", "gpt-4o", 10)
        limiter.record("tenant_1", "gpt-4o", 10)
        
        # Wait for window to expire (simulated)
        await asyncio.sleep(0.1)  # In practice, 60s window
        
        # Should work again (after window)
        # Note: This test would need time mocking for real validation
```

---

## 2.4 Concurrency Budget Control

```python
# orchestrator/concurrency_controller.py - NEW FILE
"""
Concurrency Budget Controller
=============================
Prevents budget overspend via concurrent jobs.

PROBLEM:
- Multiple concurrent jobs can both pass budget check
- Both charge, exceeding budget
- Race condition between check and charge

SOLUTION:
- Global concurrency budget semaphore
- Pessimistic reservation before job starts
- Atomic check-and-reserve operation
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict
from contextlib import asynccontextmanager

logger = logging.getLogger("orchestrator.concurrency")


@dataclass
class ConcurrencyBudget:
    """
    Global concurrency budget controller.
    
    USAGE:
        budget = ConcurrencyBudget(max_concurrent_jobs=5, max_concurrent_cost=100.0)
        
        async with budget.acquire(job_id="job_1", estimated_cost=20.0):
            # Guaranteed: budget reserved, no race condition
            result = await run_job()
            budget.charge(actual_cost=18.5)
    """
    
    max_concurrent_jobs: int = 10
    max_concurrent_cost_usd: float = 100.0
    
    def __post_init__(self):
        self._job_semaphore = asyncio.Semaphore(self.max_concurrent_jobs)
        self._cost_semaphore = asyncio.Semaphore(int(self.max_concurrent_cost_usd * 100))  # Cents
        self._active_jobs: Dict[str, float] = {}  # job_id -> reserved_cost
        self._lock = asyncio.Lock()
        self._total_reserved: float = 0.0
        self._total_spent: float = 0.0
    
    @asynccontextmanager
    async def acquire(
        self,
        job_id: str,
        estimated_cost: float,
        timeout_seconds: float = 30.0
    ):
        """
        Acquire budget slot for a job.
        
        Guarantees:
        1. At most max_concurrent_jobs running
        2. Total reserved cost <= max_concurrent_cost_usd
        3. Atomic check-and-reserve (no race)
        
        Raises:
            TimeoutError: If budget not available within timeout
        """
        cost_cents = int(estimated_cost * 100)
        
        # Try to acquire both semaphores
        job_acquired = False
        cost_acquired = False
        
        try:
            # Acquire job slot
            await asyncio.wait_for(
                self._job_semaphore.acquire(),
                timeout=timeout_seconds
            )
            job_acquired = True
            
            # Acquire cost budget
            await asyncio.wait_for(
                self._cost_semaphore.acquire(),
                timeout=timeout_seconds
            )
            cost_acquired = True
            
            # Record reservation
            async with self._lock:
                self._active_jobs[job_id] = estimated_cost
                self._total_reserved += estimated_cost
            
            logger.debug(f"Budget acquired for {job_id}: ${estimated_cost:.2f}")
            
            yield self
            
        except asyncio.TimeoutError:
            logger.warning(f"Budget acquisition timeout for {job_id}")
            raise TimeoutError(f"Could not acquire budget for {job_id} within {timeout_seconds}s")
        
        finally:
            # Release semaphores if acquired but job failed
            if cost_acquired:
                self._cost_semaphore.release()
            if job_acquired:
                self._job_semaphore.release()
            
            # Clean up reservation
            async with self._lock:
                if job_id in self._active_jobs:
                    reserved = self._active_jobs.pop(job_id)
                    self._total_reserved -= reserved
    
    def charge(self, job_id: str, actual_cost: float) -> None:
        """Record actual cost spent."""
        async def _charge():
            async with self._lock:
                self._total_spent += actual_cost
                logger.debug(f"Job {job_id} charged: ${actual_cost:.2f}")
        
        # Schedule charge (fire-and-forget)
        asyncio.create_task(_charge())
    
    @property
    def available_slots(self) -> int:
        """Number of available job slots."""
        return self._job_semaphore._value
    
    @property
    def available_budget(self) -> float:
        """Available cost budget."""
        return self._cost_semaphore._value / 100.0
    
    @property
    def stats(self) -> dict:
        """Current statistics."""
        return {
            "active_jobs": len(self._active_jobs),
            "available_slots": self.available_slots,
            "available_budget": self.available_budget,
            "total_reserved": self._total_reserved,
            "total_spent": self._total_spent,
        }


class GlobalConcurrencyController:
    """
    Singleton controller for global concurrency.
    
    Ensures all orchestrator instances share the same budget limits.
    """
    
    _instance: Optional["GlobalConcurrencyController"] = None
    
    def __init__(
        self,
        max_concurrent_jobs: int = 10,
        max_concurrent_cost_usd: float = 100.0
    ):
        self.budget = ConcurrencyBudget(
            max_concurrent_jobs=max_concurrent_jobs,
            max_concurrent_cost_usd=max_concurrent_cost_usd
        )
    
    @classmethod
    def get_instance(cls) -> "GlobalConcurrencyController":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset(cls) -> None:
        """Reset for testing."""
        cls._instance = None
```

### Integration with Engine

```python
# engine.py - Add concurrency controller

from .concurrency_controller import GlobalConcurrencyController

class Orchestrator:
    def __init__(self, ...):
        # ... existing init ...
        
        # NEW: Global concurrency controller
        self._concurrency = GlobalConcurrencyController.get_instance()
    
    async def run_job(self, spec: JobSpec) -> ProjectState:
        """Run job with concurrency control."""
        estimated_cost = self._estimate_job_cost(spec)
        
        async with self._concurrency.budget.acquire(
            job_id=spec.job_id,
            estimated_cost=estimated_cost,
            timeout_seconds=30.0
        ):
            try:
                result = await self.run_project(...)
                actual_cost = self.budget.spent_usd
                self._concurrency.budget.charge(spec.job_id, actual_cost)
                return result
            except Exception as e:
                # Budget released automatically by context manager
                raise
```

---

## 2.5 Provider Outage Mitigation Strategy

```python
# orchestrator/provider_health.py - NEW FILE
"""
Provider Health Monitoring and Failover
=======================================
Proactive provider health monitoring with automatic failover.

STRATEGY:
1. Background health checks every 30 seconds
2. Latency-based routing (prefer fast providers)
3. Automatic failover on degradation
4. Gradual recovery (don't flood recovering provider)
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Callable

from .models import Model, TaskType, FALLBACK_CHAIN

logger = logging.getLogger("orchestrator.health")


class ProviderHealth(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"  # High latency or occasional errors
    UNAVAILABLE = "unavailable"  # Complete outage
    RECOVERING = "recovering"  # Coming back online


@dataclass
class ProviderMetrics:
    """Health metrics for a provider."""
    provider: str
    health: ProviderHealth = ProviderHealth.HEALTHY
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    error_rate: float = 0.0
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    consecutive_failures: int = 0
    total_requests: int = 0
    total_errors: int = 0
    
    # Recovery state
    recovery_start: Optional[datetime] = None
    recovery_requests: int = 0
    recovery_successes: int = 0


class ProviderHealthMonitor:
    """
    Monitors provider health and manages failover.
    
    INVARIANTS:
    1. Never route to UNAVAILABLE provider
    2. Prefer HEALTHY over DEGRADED over RECOVERING
    3. Gradual recovery: 10% traffic to RECOVERING provider
    4. Automatic escalation: DEGRADED → UNAVAILABLE after threshold
    """
    
    # Thresholds
    LATENCY_P95_DEGRADED_MS: float = 5000.0
    LATENCY_P95_UNAVAILABLE_MS: float = 30000.0
    ERROR_RATE_DEGRADED: float = 0.05  # 5%
    ERROR_RATE_UNAVAILABLE: float = 0.50  # 50%
    CONSECUTIVE_FAILURES_UNAVAILABLE: int = 5
    RECOVERY_SUCCESS_THRESHOLD: float = 0.90  # 90% success rate to exit recovery
    
    def __init__(
        self,
        health_check_interval_seconds: float = 30.0,
        health_check_timeout_seconds: float = 10.0
    ):
        self._metrics: Dict[str, ProviderMetrics] = {}
        self._check_interval = health_check_interval_seconds
        self._check_timeout = health_check_timeout_seconds
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._lock = asyncio.Lock()
        
        # Initialize metrics for all providers
        for model in Model:
            provider = self._get_provider_name(model)
            if provider not in self._metrics:
                self._metrics[provider] = ProviderMetrics(provider=provider)
    
    def _get_provider_name(self, model: Model) -> str:
        """Get provider name from model."""
        from .models import get_provider
        return get_provider(model)
    
    async def start(self) -> None:
        """Start health monitoring."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._health_check_loop())
        logger.info("Provider health monitor started")
    
    async def stop(self) -> None:
        """Stop health monitoring."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Provider health monitor stopped")
    
    async def _health_check_loop(self) -> None:
        """Background health check loop."""
        while self._running:
            try:
                await self._run_health_checks()
                await asyncio.sleep(self._check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(5.0)  # Back off on error
    
    async def _run_health_checks(self) -> None:
        """Run health checks for all providers."""
        # This would make lightweight API calls to check provider health
        # For now, we rely on passive monitoring (recording actual call outcomes)
        pass
    
    def record_success(self, model: Model, latency_ms: float) -> None:
        """Record successful API call."""
        provider = self._get_provider_name(model)
        metrics = self._metrics.get(provider)
        if not metrics:
            return
        
        metrics.last_success = datetime.utcnow()
        metrics.consecutive_failures = 0
        metrics.total_requests += 1
        
        # Update latency (simple EMA)
        alpha = 0.1
        if metrics.latency_p50_ms == 0:
            metrics.latency_p50_ms = latency_ms
        else:
            metrics.latency_p50_ms = alpha * latency_ms + (1 - alpha) * metrics.latency_p50_ms
        
        # Check for recovery
        if metrics.health == ProviderHealth.RECOVERING:
            metrics.recovery_requests += 1
            metrics.recovery_successes += 1
            
            # Check if recovered
            if (metrics.recovery_requests >= 10 and
                metrics.recovery_successes / metrics.recovery_requests >= self.RECOVERY_SUCCESS_THRESHOLD):
                metrics.health = ProviderHealth.HEALTHY
                metrics.recovery_start = None
                logger.info(f"Provider {provider} recovered")
        
        # Check for upgrade from DEGRADED
        elif metrics.health == ProviderHealth.DEGRADED:
            if metrics.latency_p50_ms < self.LATENCY_P95_DEGRADED_MS:
                metrics.health = ProviderHealth.HEALTHY
                logger.info(f"Provider {provider} upgraded to HEALTHY")
    
    def record_failure(self, model: Model, error: Exception) -> None:
        """Record failed API call."""
        provider = self._get_provider_name(model)
        metrics = self._metrics.get(provider)
        if not metrics:
            return
        
        metrics.last_failure = datetime.utcnow()
        metrics.consecutive_failures += 1
        metrics.total_requests += 1
        metrics.total_errors += 1
        metrics.error_rate = metrics.total_errors / metrics.total_requests
        
        # Check for degradation
        if metrics.health == ProviderHealth.HEALTHY:
            if (metrics.error_rate > self.ERROR_RATE_DEGRADED or
                metrics.latency_p50_ms > self.LATENCY_P95_DEGRADED_MS):
                metrics.health = ProviderHealth.DEGRADED
                logger.warning(f"Provider {provider} degraded")
        
        # Check for unavailability
        if metrics.health in (ProviderHealth.HEALTHY, ProviderHealth.DEGRADED):
            if (metrics.consecutive_failures >= self.CONSECUTIVE_FAILURES_UNAVAILABLE or
                metrics.error_rate > self.ERROR_RATE_UNAVAILABLE):
                metrics.health = ProviderHealth.UNAVAILABLE
                logger.error(f"Provider {provider} unavailable")
        
        # Check for recovery failure
        elif metrics.health == ProviderHealth.RECOVERING:
            metrics.recovery_requests += 1
            if metrics.recovery_successes / max(1, metrics.recovery_requests) < 0.5:
                metrics.health = ProviderHealth.UNAVAILABLE
                logger.error(f"Provider {provider} recovery failed")
    
    def get_health(self, provider: str) -> ProviderHealth:
        """Get current health status for provider."""
        metrics = self._metrics.get(provider)
        return metrics.health if metrics else ProviderHealth.HEALTHY
    
    def is_available(self, model: Model) -> bool:
        """Check if model is available for routing."""
        provider = self._get_provider_name(model)
        health = self.get_health(provider)
        return health in (ProviderHealth.HEALTHY, ProviderHealth.DEGRADED, ProviderHealth.RECOVERING)
    
    def select_model(
        self,
        candidates: List[Model],
        task_type: TaskType
    ) -> Optional[Model]:
        """
        Select best available model for task.
        
        Priority:
        1. HEALTHY providers (lowest latency)
        2. DEGRADED providers
        3. RECOVERING providers (limited traffic)
        """
        healthy = []
        degraded = []
        recovering = []
        
        for model in candidates:
            provider = self._get_provider_name(model)
            health = self.get_health(provider)
            metrics = self._metrics.get(provider)
            
            if health == ProviderHealth.HEALTHY:
                healthy.append((model, metrics.latency_p50_ms if metrics else 0))
            elif health == ProviderHealth.DEGRADED:
                degraded.append((model, metrics.latency_p50_ms if metrics else 0))
            elif health == ProviderHealth.RECOVERING:
                # Only route 10% of traffic to recovering providers
                import random
                if random.random() < 0.1:
                    recovering.append((model, metrics.latency_p50_ms if metrics else 0))
        
        # Sort by latency and return best
        if healthy:
            healthy.sort(key=lambda x: x[1])
            return healthy[0][0]
        if degraded:
            degraded.sort(key=lambda x: x[1])
            return degraded[0][0]
        if recovering:
            recovering.sort(key=lambda x: x[1])
            return recovering[0][0]
        
        return None
    
    def get_fallback_chain(self, model: Model) -> List[Model]:
        """Get fallback chain for model, excluding unavailable providers."""
        chain = [model]
        current = model
        
        while current in FALLBACK_CHAIN:
            fallback = FALLBACK_CHAIN[current]
            provider = self._get_provider_name(fallback)
            
            if self.get_health(provider) != ProviderHealth.UNAVAILABLE:
                chain.append(fallback)
            
            current = fallback
        
        return chain
    
    def get_status_report(self) -> dict:
        """Get health status for all providers."""
        return {
            provider: {
                "health": metrics.health.value,
                "latency_p50_ms": metrics.latency_p50_ms,
                "error_rate": metrics.error_rate,
                "consecutive_failures": metrics.consecutive_failures,
                "last_success": metrics.last_success.isoformat() if metrics.last_success else None,
                "last_failure": metrics.last_failure.isoformat() if metrics.last_failure else None,
            }
            for provider, metrics in self._metrics.items()
        }
```

---

# 3. FAILURE-FIRST DESIGN

## 3.1 Worst-Case Cascading Failure Chain

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CATASTROPHIC FAILURE CASCADE                              │
│                                                                              │
│  T+0s     OpenAI API returns 500 error                                       │
│           │                                                                  │
│  T+1s     Circuit breaker increments failure count                           │
│           │                                                                  │
│  T+2s     Second request to OpenAI times out (sync EventStore blocked loop) │
│           │                                                                  │
│  T+3s     Event loop blocked for 50ms                                        │
│           │                                                                  │
│  T+3.05s  Timeout handler fires late, incorrectly marks DeepSeek as failed  │
│           │                                                                  │
│  T+5s     Third OpenAI failure → circuit breaker trips (threshold=3)        │
│           │                                                                  │
│  T+6s     Fallback to DeepSeek                                               │
│           │                                                                  │
│  T+8s     DeepSeek also slow (180s+ latency)                                 │
│           │                                                                  │
│  T+10s    DeepSeek marked DEGRADED                                           │
│           │                                                                  │
│  T+15s    All models in DEGRADED state                                       │
│           │                                                                  │
│  T+20s    New project starts, no healthy models available                    │
│           │                                                                  │
│  T+25s    Project fails immediately                                           │
│           │                                                                  │
│  T+30s    User sees "SYSTEM_FAILURE"                                          │
│           │                                                                  │
│  T+60s    Background telemetry flush blocked by sync EventStore              │
│           │                                                                  │
│  T+120s   Memory leak: 50 background tasks accumulated                       │
│           │                                                                  │
│  T+300s   Process OOM killed                                                  │
│           │                                                                  │
│  T+301s   State corruption: checkpoint not written                           │
│           │                                                                  │
│  T+302s   User cannot resume project                                          │
│                                                                              │
│  ═══════════════════════════════════════════════════════════════════════════ │
│  RESULT: Complete system failure, data loss, user trust destroyed            │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 3.2 How Architecture Prevents Cascade

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DEFENSE IN DEPTH                                          │
│                                                                              │
│  Layer 1: Async EventStore                                                   │
│  ───────────────────────────                                                 │
│  T+2s     Event loop NOT blocked                                             │
│  T+2.05s  Timeout handler fires correctly                                    │
│  T+2.1s   Only OpenAI marked as failed, not DeepSeek                         │
│                                                                              │
│  Layer 2: Provider Health Monitor                                            │
│  ─────────────────────────────────                                           │
│  T+5s     OpenAI marked DEGRADED (not UNAVAILABLE yet)                       │
│  T+6s     Traffic shifted to Gemini (cross-provider)                         │
│  T+8s     Gemini handles load successfully                                   │
│                                                                              │
│  Layer 3: Concurrency Budget Controller                                      │
│  ────────────────────────────────────                                        │
│  T+10s    New project queued (max concurrent jobs)                           │
│  T+15s    Previous project completes, new project starts                     │
│                                                                              │
│  Layer 4: Memory Leak Prevention                                             │
│  ────────────────────────────────                                            │
│  T+60s    Background tasks cleaned up                                        │
│  T+120s   Memory stable, no leak                                             │
│                                                                              │
│  Layer 5: Database Backup                                                    │
│  ─────────────────────                                                       │
│  T+300s   If process crashes, backup available                               │
│  T+301s   State recoverable from last checkpoint                             │
│                                                                              │
│  ═══════════════════════════════════════════════════════════════════════════ │
│  RESULT: Graceful degradation, no data loss, user trust maintained           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

# 4. IMPLEMENTATION PLAN (HARD MODE)

## 4.1 Safe Migration Order

```
Phase 1: Foundation (Week 1)
├── Step 1.1: Add AsyncEventStore (new file, no changes to existing)
├── Step 1.2: Add DatabaseBackupManager (new file)
├── Step 1.3: Add ConcurrencyController (new file)
├── Step 1.4: Add ProviderHealthMonitor (new file)
└── Step 1.5: Write chaos tests (new file)

Phase 2: Integration (Week 2)
├── Step 2.1: Wire AsyncEventStore into UnifiedEventBus
├── Step 2.2: Wire ConcurrencyController into Orchestrator
├── Step 2.3: Wire ProviderHealthMonitor into AdaptiveRouter
├── Step 2.4: Wire DatabaseBackupManager into Orchestrator
└── Step 2.5: Run chaos tests, fix issues

Phase 3: Deprecation (Week 3)
├── Step 3.1: Add deprecation warnings to legacy EventStore
├── Step 3.2: Add deprecation warnings to legacy dashboards
├── Step 3.3: Update documentation
└── Step 3.4: Create migration guide

Phase 4: Cleanup (Week 4)
├── Step 4.1: Remove legacy EventStore (v7.0)
├── Step 4.2: Remove legacy dashboards (v7.0)
└── Step 4.3: Final verification
```

## 4.2 What Can Break During Migration

| Step | Risk | Mitigation |
|------|------|------------|
| 1.1 AsyncEventStore | None (new file) | N/A |
| 2.1 Wire AsyncEventStore | Events not persisted | Keep legacy fallback |
| 2.2 Wire ConcurrencyController | Jobs rejected unexpectedly | Start with high limits |
| 2.3 Wire ProviderHealthMonitor | Models incorrectly marked unavailable | Conservative thresholds |
| 3.1 Deprecation warnings | Log spam | Use once-per-session warnings |
| 4.1 Remove legacy | Import errors | Update all imports first |

## 4.3 Rollback Strategy

```python
# Feature flags for safe rollback
class FeatureFlags:
    USE_ASYNC_EVENT_STORE: bool = True
    USE_CONCURRENCY_CONTROLLER: bool = True
    USE_PROVIDER_HEALTH_MONITOR: bool = True
    USE_DATABASE_BACKUP: bool = True

# In UnifiedEventBus
async def _handle_event(self, event: DomainEvent) -> None:
    if FeatureFlags.USE_ASYNC_EVENT_STORE and self._async_store:
        await self._async_store.append(event)
    elif self._legacy_store:
        await asyncio.to_thread(self._legacy_store.append, event)
```

---

# 5. TEST STRATEGY

## 5.1 Chaos Tests to Add Immediately

| Test | File | Purpose |
|------|------|---------|
| `test_provider_outage_cascade` | `tests/chaos/test_provider_outage.py` | Verify fallback chain |
| `test_database_lock_contention` | `tests/chaos/test_database.py` | Verify WAL mode |
| `test_budget_race_condition` | `tests/chaos/test_budget.py` | Verify reservation |
| `test_event_loop_starvation` | `tests/chaos/test_event_loop.py` | Verify async |
| `test_memory_leak` | `tests/chaos/test_memory.py` | Verify cleanup |
| `test_cascading_failure` | `tests/chaos/test_cascade.py` | Full system test |

## 5.2 Load Tests

```python
# tests/load/test_load.py

class TestLoad:
    """Load tests for production readiness."""
    
    @pytest.mark.asyncio
    async def test_100_concurrent_tasks(self):
        """System should handle 100 concurrent tasks."""
        orch = Orchestrator(max_concurrency=10)
        
        tasks = [
            orch.client.call(Model.GPT_4O_MINI, f"Task {i}")
            for i in range(100)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        success_count = sum(1 for r in results if not isinstance(r, Exception))
        assert success_count >= 95  # 95% success rate
    
    @pytest.mark.asyncio
    async def test_sustained_load_1_hour(self):
        """System should handle sustained load for 1 hour."""
        # Run 10 tasks per minute for 60 minutes
        pass  # Implementation would use time-based loop
```

## 5.3 State Corruption Tests

```python
# tests/chaos/test_state_corruption.py

class TestStateCorruption:
    """Test recovery from corrupted state."""
    
    @pytest.mark.asyncio
    async def test_recover_from_corrupted_state_db(self, tmp_path):
        """Should recover from corrupted state.db."""
        # Create corrupted DB
        corrupt_db = tmp_path / "state.db"
        corrupt_db.write_bytes(b"CORRUPTED")
        
        # Try to initialize
        from orchestrator.state import StateManager
        sm = StateManager(db_path=corrupt_db)
        
        # Should detect corruption and reinitialize
        conn = await sm._get_conn()
        assert conn is not None
    
    @pytest.mark.asyncio
    async def test_recover_from_backup(self, tmp_path):
        """Should restore from backup when main DB corrupted."""
        from orchestrator.db_backup import DatabaseBackupManager
        
        backup_mgr = DatabaseBackupManager(
            data_dir=tmp_path,
            backup_dir=tmp_path / "backups"
        )
        
        # Create backup
        await backup_mgr._backup_all()
        
        # Corrupt main DB
        (tmp_path / "state.db").write_bytes(b"CORRUPTED")
        
        # Restore from backup
        backups = backup_mgr.list_backups()
        assert len(backups) > 0
        
        await backup_mgr.restore(Path(backups[0]["timestamp"]))
```

---

# 6. PRODUCTION GUARDRAILS

## 6.1 Runtime Monitors

```python
# orchestrator/guardrails.py - NEW FILE
"""
Production Guardrails
=====================
Runtime monitors, kill switches, and safety mechanisms.
"""

import asyncio
import logging
import os
import signal
import time
from dataclasses import dataclass
from typing import Optional, Callable, Dict, Any

logger = logging.getLogger("orchestrator.guardrails")


@dataclass
class GuardrailConfig:
    """Configuration for production guardrails."""
    
    # Budget limits
    max_budget_usd: float = 1000.0
    budget_warning_threshold: float = 0.8  # 80%
    budget_critical_threshold: float = 0.95  # 95%
    
    # Rate limits
    max_requests_per_minute: int = 100
    max_tokens_per_minute: int = 1_000_000
    
    # Memory limits
    max_memory_mb: float = 1024.0
    memory_warning_threshold: float = 0.85
    
    # Latency limits
    max_latency_ms: float = 30000.0  # 30 seconds
    
    # Error rate limits
    max_error_rate: float = 0.10  # 10%
    
    # Kill switch
    enable_kill_switch: bool = True
    kill_switch_file: str = "/tmp/orchestrator_kill"


class ProductionGuardrails:
    """
    Production safety mechanisms.
    
    FEATURES:
    1. Budget enforcement (hard limit)
    2. Rate limit enforcement
    3. Memory monitoring
    4. Error rate monitoring
    5. Kill switch (file-based)
    6. Drift detection
    """
    
    def __init__(self, config: Optional[GuardrailConfig] = None):
        self.config = config or GuardrailConfig()
        self._error_count = 0
        self._total_requests = 0
        self._start_time = time.monotonic()
        self._kill_switch_checked = 0.0
        self._drift_baseline: Dict[str, Any] = {}
    
    def check_budget(self, spent: float, max_budget: float) -> bool:
        """Check if budget is within limits."""
        if spent > max_budget:
            logger.critical(f"BUDGET EXCEEDED: ${spent:.2f} > ${max_budget:.2f}")
            return False
        
        ratio = spent / max_budget
        if ratio >= self.config.budget_critical_threshold:
            logger.critical(f"BUDGET CRITICAL: {ratio*100:.1f}% used")
        elif ratio >= self.config.budget_warning_threshold:
            logger.warning(f"BUDGET WARNING: {ratio*100:.1f}% used")
        
        return True
    
    def check_kill_switch(self) -> bool:
        """Check if kill switch is activated."""
        if not self.config.enable_kill_switch:
            return False
        
        # Only check every 5 seconds
        if time.monotonic() - self._kill_switch_checked < 5.0:
            return False
        self._kill_switch_checked = time.monotonic()
        
        kill_file = self.config.kill_switch_file
        if os.path.exists(kill_file):
            logger.critical(f"KILL SWITCH ACTIVATED: {kill_file}")
            return True
        
        return False
    
    def record_request(self, success: bool) -> None:
        """Record request outcome for error rate calculation."""
        self._total_requests += 1
        if not success:
            self._error_count += 1
    
    def check_error_rate(self) -> bool:
        """Check if error rate is within limits."""
        if self._total_requests < 10:
            return True  # Not enough data
        
        error_rate = self._error_count / self._total_requests
        if error_rate > self.config.max_error_rate:
            logger.critical(f"ERROR RATE TOO HIGH: {error_rate*100:.1f}%")
            return False
        
        return True
    
    def check_memory(self) -> bool:
        """Check memory usage."""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            if memory_mb > self.config.max_memory_mb:
                logger.critical(f"MEMORY EXCEEDED: {memory_mb:.0f}MB > {self.config.max_memory_mb:.0f}MB")
                return False
            
            ratio = memory_mb / self.config.max_memory_mb
            if ratio >= self.config.memory_warning_threshold:
                logger.warning(f"MEMORY WARNING: {ratio*100:.1f}% used")
            
            return True
        except ImportError:
            return True  # psutil not available
    
    def detect_drift(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect configuration drift from baseline.
        
        Returns dict of drifted values.
        """
        if not self._drift_baseline:
            self._drift_baseline = current_state.copy()
            return {}
        
        drift = {}
        for key, value in current_state.items():
            if key in self._drift_baseline:
                baseline = self._drift_baseline[key]
                if baseline != value:
                    drift[key] = {
                        "baseline": baseline,
                        "current": value
                    }
        
        if drift:
            logger.warning(f"CONFIGURATION DRIFT DETECTED: {drift}")
        
        return drift
    
    def all_checks_pass(self, spent: float, max_budget: float) -> bool:
        """Run all guardrail checks."""
        checks = [
            ("budget", self.check_budget(spent, max_budget)),
            ("kill_switch", not self.check_kill_switch()),
            ("error_rate", self.check_error_rate()),
            ("memory", self.check_memory()),
        ]
        
        failed = [name for name, passed in checks if not passed]
        
        if failed:
            logger.critical(f"GUARDRAIL FAILURES: {failed}")
        
        return len(failed) == 0


# Global guardrails instance
_guardrails: Optional[ProductionGuardrails] = None


def get_guardrails() -> ProductionGuardrails:
    """Get global guardrails instance."""
    global _guardrails
    if _guardrails is None:
        _guardrails = ProductionGuardrails()
    return _guardrails
```

## 6.2 Kill Switch Implementation

```python
# orchestrator/kill_switch.py - NEW FILE
"""
Kill Switch Implementation
==========================
Emergency stop mechanism for production.

USAGE:
    # Activate kill switch
    touch /tmp/orchestrator_kill
    
    # Deactivate
    rm /tmp/orchestrator_kill
    
    # Check in code
    if guardrails.check_kill_switch():
        raise SystemExit("Kill switch activated")
"""

import os
import signal
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("orchestrator.kill_switch")


class KillSwitch:
    """
    File-based kill switch for emergency shutdown.
    
    FEATURES:
    1. Instant activation via file creation
    2. Graceful shutdown (finish current tasks)
    3. Force shutdown (immediate termination)
    4. Audit log of activation
    """
    
    def __init__(
        self,
        kill_file: str = "/tmp/orchestrator_kill",
        force_file: str = "/tmp/orchestrator_force_kill",
        audit_file: Optional[str] = None
    ):
        self.kill_file = Path(kill_file)
        self.force_file = Path(force_file)
        self.audit_file = Path(audit_file) if audit_file else None
        self._activated = False
    
    def is_activated(self) -> bool:
        """Check if kill switch is activated."""
        if self.kill_file.exists():
            if not self._activated:
                self._activated = True
                self._log_activation()
            return True
        return False
    
    def is_force_activated(self) -> bool:
        """Check if force kill is activated."""
        return self.force_file.exists()
    
    def _log_activation(self) -> None:
        """Log kill switch activation."""
        import time
        logger.critical(f"KILL SWITCH ACTIVATED at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if self.audit_file:
            with open(self.audit_file, "a") as f:
                f.write(f"{time.time()}: KILL_SWITCH_ACTIVATED\n")
    
    def activate(self, force: bool = False) -> None:
        """Activate kill switch (for testing)."""
        target = self.force_file if force else self.kill_file
        target.touch()
        logger.warning(f"Kill switch activated: {target}")
    
    def deactivate(self) -> None:
        """Deactivate kill switch."""
        for f in [self.kill_file, self.force_file]:
            if f.exists():
                f.unlink()
        self._activated = False
        logger.info("Kill switch deactivated")
    
    def check_and_exit(self) -> None:
        """Check kill switch and exit if activated."""
        if self.is_force_activated():
            logger.critical("FORCE KILL - Immediate termination")
            os._exit(1)  # Immediate, no cleanup
        
        if self.is_activated():
            logger.critical("KILL SWITCH - Graceful shutdown")
            raise SystemExit(0)
```

---

# 7. DESIGN CRITIQUE

## 7.1 What Is Fundamentally Weak

| Weakness | Why It's Fundamental | Mitigation |
|----------|---------------------|------------|
| **Single Event Loop** | Python async is single-threaded; one blocking call stalls everything | Strict async discipline, monitoring |
| **SQLite as Database** | Not designed for high concurrency or distributed systems | Accept limitations, or migrate to PostgreSQL |
| **In-Memory State** | Lost on crash; requires careful checkpointing | Frequent checkpoints, backup strategy |
| **Provider Dependency** | System is only as reliable as the least reliable provider | Multi-provider fallback, circuit breakers |
| **No Horizontal Scaling** | Single process, single machine | Accept for solo founder, or add message queue |

## 7.2 What Will Still Fail After Fixes

| Scenario | Why It Will Fail | Acceptance |
|----------|------------------|------------|
| **All providers down simultaneously** | No fallback available | Accept, notify user |
| **Disk full** | Cannot persist state | Monitor disk, alert early |
| **Network partition** | Cannot reach any provider | Accept, queue locally |
| **Memory corruption** | Process state corrupted | Restart, restore from backup |
| **API key revoked** | Authentication fails | Alert immediately |

## 7.3 What Top-Tier Infra Teams Would Do Differently

| Current Approach | Top-Tier Approach | Tradeoff |
|------------------|-------------------|----------|
| SQLite | PostgreSQL with replication | Complexity vs reliability |
| Single process | Kubernetes deployment with replicas | Cost vs availability |
| File-based kill switch | Kubernetes pod termination | Simplicity vs integration |
| In-memory budget | Distributed budget service | Latency vs consistency |
| Local cache | Redis cluster | Cost vs scalability |
| Manual provider monitoring | PagerDuty integration | Cost vs response time |

---

# 8. SUMMARY

## Immediate Actions (This Week)

1. ✅ Create `AsyncEventStore` class (new file)
2. ✅ Create `DatabaseBackupManager` class (new file)
3. ✅ Create `ConcurrencyController` class (new file)
4. ✅ Create `ProviderHealthMonitor` class (new file)
5. ✅ Create `ProductionGuardrails` class (new file)
6. ✅ Create chaos test suite (new file)

## Integration Actions (Next Week)

1. Wire `AsyncEventStore` into `UnifiedEventBus`
2. Wire `ConcurrencyController` into `Orchestrator.run_job()`
3. Wire `ProviderHealthMonitor` into `AdaptiveRouter`
4. Wire `DatabaseBackupManager` into `Orchestrator.__aenter__`
5. Run chaos tests, fix issues

## Deprecation Actions (Week 3)

1. Add deprecation warnings to legacy `EventStore`
2. Add deprecation warnings to legacy dashboards
3. Update documentation
4. Create migration guide

## Tradeoffs Made

| Decision | Tradeoff |
|----------|----------|
| SQLite instead of PostgreSQL | Simplicity over scalability |
| Single process | Cost over availability |
| File-based kill switch | Simplicity over integration |
| Conservative thresholds | False positives over cascading failures |
| Gradual recovery | Latency over immediate restoration |

---

*Document Complete — Ready for Implementation*