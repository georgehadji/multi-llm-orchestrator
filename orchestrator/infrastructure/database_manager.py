"""
DatabaseManager — Centralized SQLite schema management
=======================================================
Owns schema creation, migration, and connection pooling for all 9
SQLite databases used by the orchestrator.  Modules that previously
called ``sqlite3.connect()`` or ``aiosqlite.connect()`` directly
should receive a ``DatabaseManager`` instance and call ``get_db(name)``.

Usage:
    from orchestrator.infrastructure.database_manager import DatabaseManager
    from orchestrator.infrastructure.path_provider import CachePathProvider

    paths = CachePathProvider()
    dbm = DatabaseManager(paths)
    await dbm.initialize_all()

    conn = await dbm.get_async("state")
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("orchestrator.database_manager")


class DatabaseManager:
    """Centralized owner of all SQLite schema and migrations.

    Args:
        paths: CachePathProvider instance for path resolution.
        wal_mode: Enable WAL journal mode (default True).
        synchronous: PRAGMA synchronous setting (\"FULL\" for durability).
    """

    # Every database known to the orchestrator, with its schema DDL.
    _SCHEMAS: dict[str, str] = {
        "state": """
            CREATE TABLE IF NOT EXISTS projects (
                project_id TEXT PRIMARY KEY,
                state      TEXT NOT NULL,
                status     TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS checkpoints (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id TEXT NOT NULL,
                task_id    TEXT NOT NULL,
                state      TEXT NOT NULL,
                created_at REAL NOT NULL,
                FOREIGN KEY (project_id) REFERENCES projects(project_id)
            );
            CREATE TABLE IF NOT EXISTS circuit_breaker_state (
                model_name    TEXT PRIMARY KEY,
                failure_count INTEGER NOT NULL DEFAULT 0,
                updated_at    REAL NOT NULL
            );
        """,
        "cache": """
            CREATE TABLE IF NOT EXISTS cache (
                key        TEXT PRIMARY KEY,
                value      BLOB NOT NULL,
                size_bytes INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                expires_at REAL
            );
        """,
        "kanban": """
            CREATE TABLE IF NOT EXISTS kanban_tasks (
                task_id       TEXT PRIMARY KEY,
                project_spec  TEXT NOT NULL,
                status        TEXT NOT NULL DEFAULT 'todo',
                priority      INTEGER NOT NULL DEFAULT 0,
                assignee      TEXT NOT NULL DEFAULT '',
                created_at    REAL NOT NULL,
                claimed_at    REAL DEFAULT 0.0,
                completed_at  REAL DEFAULT 0.0,
                failure_count INTEGER NOT NULL DEFAULT 0,
                error_log     TEXT NOT NULL DEFAULT ''
            );
        """,
        "patterns": """
            CREATE TABLE IF NOT EXISTS patterns (
                pattern_id          TEXT PRIMARY KEY,
                task_type           TEXT NOT NULL,
                prompt_text         TEXT NOT NULL,
                generated_code      TEXT NOT NULL DEFAULT '',
                quality_score       REAL NOT NULL DEFAULT 0.0,
                reuse_count         INTEGER NOT NULL DEFAULT 0,
                avg_score_on_reuse  REAL NOT NULL DEFAULT 0.0,
                status              TEXT NOT NULL DEFAULT 'active',
                created_at          REAL NOT NULL,
                archived_at         REAL
            );
        """,
        "telemetry": """
            CREATE TABLE IF NOT EXISTS telemetry_records (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                model      TEXT NOT NULL,
                task_type  TEXT NOT NULL,
                latency_ms INTEGER NOT NULL,
                cost_usd   REAL NOT NULL DEFAULT 0.0,
                success    INTEGER NOT NULL DEFAULT 1,
                error      TEXT,
                timestamp  REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_telemetry_model
                ON telemetry_records(model, timestamp);
        """,
        "trajectories": """
            CREATE TABLE IF NOT EXISTS trajectories (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                task_type  TEXT NOT NULL,
                model      TEXT NOT NULL,
                score      REAL NOT NULL DEFAULT 0.0,
                critique   TEXT,
                created_at REAL NOT NULL
            );
        """,
        "skills": """
            CREATE TABLE IF NOT EXISTS skills (
                task_type TEXT PRIMARY KEY,
                content   TEXT NOT NULL,
                version   INTEGER NOT NULL DEFAULT 1,
                score     REAL NOT NULL DEFAULT 0.0,
                updated_at REAL NOT NULL
            );
        """,
        "events": """
            CREATE TABLE IF NOT EXISTS events (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type   TEXT NOT NULL,
                aggregate_id TEXT NOT NULL,
                payload      TEXT NOT NULL,
                timestamp    REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_events_type
                ON events(event_type, timestamp);
        """,
        "budget": """
            CREATE TABLE IF NOT EXISTS budget_hierarchy (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
        """,
    }

    def __init__(
        self,
        paths: Any = None,
        wal_mode: bool = True,
        synchronous: str = "FULL",
    ) -> None:
        from .path_provider import CachePathProvider

        self._paths = paths if paths is not None else CachePathProvider()
        self._wal_mode = wal_mode
        self._synchronous = synchronous
        self._lock = asyncio.Lock()
        self._initialized: set[str] = set()

    # ── public API ──────────────────────────────────────────────────────────

    async def initialize_all(self) -> None:
        """Create schemas for all known databases (idempotent)."""
        import aiosqlite

        for name, schema in self._SCHEMAS.items():
            db_path = self._paths.db(f"{name}.db")
            async with self._lock:
                if name in self._initialized:
                    continue
                try:
                    conn = await aiosqlite.connect(str(db_path))
                    try:
                        await conn.execute(
                            "PRAGMA journal_mode=WAL"
                            if self._wal_mode
                            else "PRAGMA journal_mode=DELETE"
                        )
                        await conn.execute(f"PRAGMA synchronous={self._synchronous}")
                        await conn.executescript(schema)
                        await conn.commit()
                        self._initialized.add(name)
                        logger.debug("DatabaseManager: initialized %s (%s)", name, db_path)
                    finally:
                        await conn.close()
                except aiosqlite.Error as exc:
                    logger.error("DatabaseManager: failed to init %s: %s", name, exc)
                    raise

    async def get_async(self, name: str) -> Any:
        """Return an aiosqlite connection for *name*, initialising on first use."""
        import aiosqlite

        if name not in self._initialized:
            await self._ensure_initialized(name)

        db_path = self._paths.db(f"{name}.db")
        return await aiosqlite.connect(str(db_path))

    async def _ensure_initialized(self, name: str) -> None:
        schema = self._SCHEMAS.get(name)
        if schema is None:
            raise ValueError(f"Unknown database {name!r}. Known: {list(self._SCHEMAS)}")
        async with self._lock:
            if name in self._initialized:
                return
            import aiosqlite

            db_path = self._paths.db(f"{name}.db")
            conn = await aiosqlite.connect(str(db_path))
            try:
                await conn.execute(
                    "PRAGMA journal_mode=WAL" if self._wal_mode else "PRAGMA journal_mode=DELETE"
                )
                await conn.execute(f"PRAGMA synchronous={self._synchronous}")
                await conn.executescript(schema)
                await conn.commit()
                self._initialized.add(name)
            finally:
                await conn.close()

    def db_path(self, name: str) -> Path:
        """Return the file-system path for database *name*."""
        return self._paths.db(f"{name}.db")

    @property
    def known_databases(self) -> list[str]:
        return sorted(self._SCHEMAS)
