"""
Disk Cache — prompt hash → response persistence
================================================
SQLite-backed cache for deduplicating identical API calls.
Uses aiosqlite for non-blocking async I/O and WAL mode for
concurrent read/write safety under multi-task execution.

Counterfactual: Without disk caching → vulnerability Ψ: identical sub-tasks
across iterations consume budget for zero quality gain.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import aiosqlite

from .models import prompt_hash

DEFAULT_CACHE_PATH = Path.home() / ".orchestrator_cache" / "cache.db"


class DiskCache:
    def __init__(self, db_path: Path = DEFAULT_CACHE_PATH):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = str(db_path)

    async def _ensure_schema(self, db: aiosqlite.Connection):
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("""
            CREATE TABLE IF NOT EXISTS cache (
                hash          TEXT PRIMARY KEY,
                model         TEXT NOT NULL,
                response      TEXT NOT NULL,
                tokens_input  INTEGER DEFAULT 0,
                tokens_output INTEGER DEFAULT 0,
                created_at    REAL NOT NULL
            )
        """)
        await db.commit()

    async def get(self, model: str, prompt: str, max_tokens: int,
                  system: str = "", temperature: float = 0.3) -> Optional[dict]:
        h = prompt_hash(model, prompt, max_tokens, system, temperature)
        async with aiosqlite.connect(self._db_path) as db:
            await self._ensure_schema(db)
            async with db.execute(
                "SELECT response, tokens_input, tokens_output FROM cache WHERE hash = ?",
                (h,)
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

    async def put(self, model: str, prompt: str, max_tokens: int,
                  response: str, tokens_input: int = 0, tokens_output: int = 0,
                  system: str = "", temperature: float = 0.3):
        h = prompt_hash(model, prompt, max_tokens, system, temperature)
        async with aiosqlite.connect(self._db_path) as db:
            await self._ensure_schema(db)
            await db.execute(
                """INSERT OR REPLACE INTO cache
                   (hash, model, response, tokens_input, tokens_output, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (h, model, response, tokens_input, tokens_output, time.time())
            )
            await db.commit()

    async def clear(self):
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("DELETE FROM cache")
            await db.commit()

    async def stats(self) -> dict:
        async with aiosqlite.connect(self._db_path) as db:
            await self._ensure_schema(db)
            async with db.execute("SELECT COUNT(*) FROM cache") as cursor:
                row = await cursor.fetchone()
        return {"entries": row[0] if row else 0}

    def close(self):
        # No persistent connection to close; connections are per-operation.
        pass
