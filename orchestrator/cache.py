"""
Disk Cache — prompt hash → response persistence
================================================
Author: Georgios-Chrysovalantis Chatzivantsidis
SQLite-backed cache for deduplicating identical API calls.
Uses aiosqlite for non-blocking async I/O and WAL mode for
concurrent read/write safety under multi-task execution.

FIX #4: Persistent connection with one-time schema init
        instead of per-operation connect + _ensure_schema.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

import aiosqlite

from .models import prompt_hash

logger = logging.getLogger("orchestrator.cache")

DEFAULT_CACHE_PATH = Path.home() / ".orchestrator_cache" / "cache.db"


class DiskCache:
    """
    SQLite-backed cache with connection pooling and TTL.

    OPTIMIZATION: Connection TTL prevents stale connections in long-running
    processes. Default TTL is 1 hour.
    """

    # Connection TTL in seconds (1 hour default)
    _CONN_TTL: float = 3600.0
    # Connection timeout in seconds (prevents infinite hangs)
    _CONN_TIMEOUT: float = 10.0

    def __init__(self, db_path: Path = DEFAULT_CACHE_PATH):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = str(db_path)
        self._conn: aiosqlite.Connection | None = None
        self._schema_ready = False
        self._lock: asyncio.Lock | None = None  # lazy — created inside event loop
        self._conn_created_at: float | None = None  # OPTIMIZATION: Connection TTL tracking

    async def _get_conn(self) -> aiosqlite.Connection:
        """
        Return persistent connection, initializing schema once.

        OPTIMIZATION: Connection TTL prevents stale connections.
        If connection is older than _CONN_TTL, it's recreated.

        FIX CACHE-001: Added timeout and proper error handling to prevent
        connection leaks and infinite hangs on DB initialization failure.
        """
        if self._lock is None:
            self._lock = asyncio.Lock()

        if self._conn is None:
            async with self._lock:
                # Double-check after acquiring lock
                if self._conn is None:
                    # Check if connection needs refresh (TTL expired)
                    if (
                        self._conn_created_at is not None
                        and time.time() - self._conn_created_at > self._CONN_TTL
                    ):
                        logger.debug("Cache connection TTL expired, refreshing")
                        # Close old connection if it exists
                        if self._conn is not None:
                            try:
                                await self._conn.close()
                            except Exception:
                                pass
                        self._conn = None
                        self._conn_created_at = None

                    # Create new connection if needed
                    if self._conn is None:
                        try:
                            # FIX CACHE-001: Timeout prevents infinite hang
                            self._conn = await asyncio.wait_for(
                                aiosqlite.connect(self._db_path), timeout=self._CONN_TIMEOUT
                            )
                            self._conn_created_at = time.time()
                            await self._conn.execute("PRAGMA journal_mode=WAL")
                            await self._conn.execute("""
                                CREATE TABLE IF NOT EXISTS cache (
                                    hash          TEXT PRIMARY KEY,
                                    model         TEXT NOT NULL,
                                    response      TEXT NOT NULL,
                                    tokens_input  INTEGER DEFAULT 0,
                                    tokens_output INTEGER DEFAULT 0,
                                    created_at    REAL NOT NULL
                                )
                            """)
                            await self._conn.commit()
                            self._schema_ready = True
                            logger.debug("Cache connection established, schema ready")
                        except asyncio.TimeoutError:
                            logger.error("Cache connection timed out after %ds", self._CONN_TIMEOUT)
                            self._conn = None
                            self._conn_created_at = None
                            raise
                        except Exception as e:
                            logger.error("Cache connection failed: %s", e)
                            # FIX CACHE-001: Reset ALL state on error
                            self._conn = None
                            self._conn_created_at = None
                            self._schema_ready = False
                            raise
        return self._conn

    async def get(
        self, model: str, prompt: str, max_tokens: int, system: str = "", temperature: float = 0.3
    ) -> dict | None:
        h = prompt_hash(model, prompt, max_tokens, system, temperature)
        db = await self._get_conn()
        async with db.execute(
            "SELECT response, tokens_input, tokens_output FROM cache WHERE hash = ?", (h,)
        ) as cursor:
            row = await cursor.fetchone()
        if row:
            return {
                "response": row[0],
                "tokens_input": row[1],
                "tokens_output": row[2],
                "cached": True,
            }
        return None

    async def put(
        self,
        model: str,
        prompt: str,
        max_tokens: int,
        response: str,
        tokens_input: int = 0,
        tokens_output: int = 0,
        system: str = "",
        temperature: float = 0.3,
    ):
        h = prompt_hash(model, prompt, max_tokens, system, temperature)
        db = await self._get_conn()
        await db.execute(
            """INSERT OR REPLACE INTO cache
               (hash, model, response, tokens_input, tokens_output, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (h, model, response, tokens_input, tokens_output, time.time()),
        )
        await db.commit()

    async def clear(self):
        db = await self._get_conn()
        await db.execute("DELETE FROM cache")
        await db.commit()

    async def stats(self) -> dict:
        db = await self._get_conn()
        async with db.execute("SELECT COUNT(*) FROM cache") as cursor:
            row = await cursor.fetchone()
        return {"entries": row[0] if row else 0}

    async def close(self):
        """Close persistent connection."""
        if self._conn is not None:
            try:
                await self._conn.close()
                # Yield control so the aiosqlite background thread can finish
                # its final callbacks before asyncio.run() closes the loop.
                await asyncio.sleep(0)
            except Exception:
                pass
            finally:
                self._conn = None
                self._conn_created_at = None
                self._schema_ready = False
                logger.debug("Cache connection closed")
