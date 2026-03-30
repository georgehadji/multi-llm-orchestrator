"""
Secure JSON Cache — Safe serialization without pickle
======================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Secure cache implementation using JSON instead of pickle.
Eliminates remote code execution risk from pickle deserialization.

Features:
- JSON serialization (safe)
- Optional compression
- TTL support
- No pickle usage

USAGE:
    from orchestrator.secure_cache import SecureCache

    cache = SecureCache()
    await cache.set("key", {"data": "value"})
    value = await cache.get("key")
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiosqlite

logger = logging.getLogger("orchestrator.secure_cache")


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    ttl_seconds: int | None = None
    compressed: bool = False

    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.ttl_seconds is None:
            return False
        return time.time() > (self.created_at + self.ttl_seconds)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "key": self.key,
            "value": self.value,
            "created_at": self.created_at,
            "ttl_seconds": self.ttl_seconds,
            "compressed": self.compressed,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CacheEntry:
        """Create from dictionary."""
        return cls(
            key=data["key"],
            value=data["value"],
            created_at=data.get("created_at", time.time()),
            ttl_seconds=data.get("ttl_seconds"),
            compressed=data.get("compressed", False),
        )


class SecureCache:
    """
    Secure cache using JSON serialization.

    SECURITY: Uses JSON instead of pickle to prevent RCE.
    """

    # Connection timeout in seconds
    _CONN_TIMEOUT: float = 10.0

    def __init__(self, db_path: Path | None = None):
        """
        Initialize secure cache.

        Args:
            db_path: Path to SQLite database
        """
        if db_path is None:
            db_path = Path.home() / ".orchestrator_cache" / "secure_cache.db"

        db_path.parent.mkdir(parents=True, exist_ok=True)

        self._db_path = str(db_path)
        self._conn: aiosqlite.Connection | None = None
        self._lock: asyncio.Lock | None = None
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "evictions": 0,
        }

    async def _get_conn(self) -> aiosqlite.Connection:
        """Get or create database connection."""
        if self._lock is None:
            self._lock = asyncio.Lock()

        if self._conn is None:
            async with self._lock:
                if self._conn is None:
                    self._conn = await asyncio.wait_for(
                        aiosqlite.connect(self._db_path),
                        timeout=self._CONN_TIMEOUT
                    )
                    await self._conn.execute("PRAGMA journal_mode=WAL")
                    await self._conn.execute("PRAGMA synchronous=FULL")
                    await self._conn.execute("""
                        CREATE TABLE IF NOT EXISTS cache (
                            key         TEXT PRIMARY KEY,
                            value       TEXT NOT NULL,
                            created_at  REAL NOT NULL,
                            ttl_seconds INTEGER,
                            compressed  INTEGER DEFAULT 0
                        )
                    """)
                    await self._conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_cache_ttl ON cache(ttl_seconds)"
                    )
                    await self._conn.commit()

        return self._conn

    def _serialize(self, value: Any) -> str:
        """
        Serialize value to JSON string.

        SECURITY: Uses JSON instead of pickle.
        """
        return json.dumps(value, default=str)

    def _deserialize(self, data: str) -> Any:
        """
        Deserialize JSON string to value.

        SECURITY: Uses JSON instead of pickle - no RCE risk.
        """
        return json.loads(data)

    def _compress(self, data: str) -> bytes:
        """Compress data."""
        return zlib.compress(data.encode('utf-8'), level=6)

    def _decompress(self, data: bytes) -> str:
        """Decompress data."""
        return zlib.decompress(data).decode('utf-8')

    async def get(self, key: str, default: Any = None) -> Any:
        """
        Get value from cache.

        Args:
            key: Cache key
            default: Default value if not found

        Returns:
            Cached value or default
        """
        try:
            db = await self._get_conn()
            async with db.execute(
                "SELECT value, ttl_seconds, compressed FROM cache WHERE key = ?",
                (key,)
            ) as cursor:
                row = await cursor.fetchone()

            if row is None:
                self._stats["misses"] += 1
                return default

            value_blob, ttl_seconds, compressed = row

            # Check TTL
            if ttl_seconds:
                # Get created_at from separate query
                async with db.execute(
                    "SELECT created_at FROM cache WHERE key = ?",
                    (key,)
                ) as cursor:
                    row2 = await cursor.fetchone()

                if row2:
                    created_at = row2[0]
                    if time.time() > (created_at + ttl_seconds):
                        # Entry expired
                        await self.delete(key)
                        self._stats["misses"] += 1
                        return default

            # Deserialize
            value_str = self._decompress(value_blob) if compressed else value_blob

            value = self._deserialize(value_str)

            self._stats["hits"] += 1
            return value

        except Exception as e:
            logger.error(f"Cache get failed: {e}")
            return default

    async def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: int | None = None,
        compress: bool = False,
    ) -> None:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time-to-live in seconds
            compress: Compress value
        """
        try:
            # Serialize to JSON (SECURE - no pickle)
            value_str = self._serialize(value)

            # Compress if requested and value is large enough
            compressed = False
            value_blob = value_str

            if compress and len(value_str) > 1024:  # Only compress if > 1KB
                compressed_blob = self._compress(value_str)
                if len(compressed_blob) < len(value_str.encode('utf-8')):
                    value_blob = compressed_blob
                    compressed = True

            db = await self._get_conn()
            await db.execute(
                """INSERT OR REPLACE INTO cache
                   (key, value, created_at, ttl_seconds, compressed)
                   VALUES (?, ?, ?, ?, ?)""",
                (key, value_blob, time.time(), ttl_seconds, 1 if compressed else 0)
            )
            await db.commit()

            self._stats["sets"] += 1

        except Exception as e:
            logger.error(f"Cache set failed: {e}")

    async def delete(self, key: str) -> bool:
        """
        Delete value from cache.

        Args:
            key: Cache key

        Returns:
            True if deleted
        """
        try:
            db = await self._get_conn()
            await db.execute("DELETE FROM cache WHERE key = ?", (key,))
            await db.commit()

            self._stats["deletes"] += 1
            return True

        except Exception as e:
            logger.error(f"Cache delete failed: {e}")
            return False

    async def clear(self) -> None:
        """Clear all cache entries."""
        try:
            db = await self._get_conn()
            await db.execute("DELETE FROM cache")
            await db.commit()

            self._stats["evictions"] += self._stats["sets"]
            self._stats["sets"] = 0

            logger.info("Cache cleared")

        except Exception as e:
            logger.error(f"Cache clear failed: {e}")

    async def cleanup_expired(self) -> int:
        """
        Remove expired entries.

        Returns:
            Number of entries removed
        """
        try:
            db = await self._get_conn()

            # Find expired entries
            now = time.time()
            expired_keys = []

            async with db.execute(
                "SELECT key, created_at, ttl_seconds FROM cache WHERE ttl_seconds IS NOT NULL"
            ) as cursor:
                async for row in cursor:
                    key, created_at, ttl_seconds = row
                    if now > (created_at + ttl_seconds):
                        expired_keys.append(key)

            # Delete expired entries
            for key in expired_keys:
                await db.execute("DELETE FROM cache WHERE key = ?", (key,))

            await db.commit()

            if expired_keys:
                self._stats["evictions"] += len(expired_keys)
                logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")

            return len(expired_keys)

        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")
            return 0

    async def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        try:
            db = await self._get_conn()

            async with db.execute(
                "SELECT COUNT(*), SUM(length(value)) FROM cache"
            ) as cursor:
                row = await cursor.fetchone()

            count = row[0] if row else 0
            size_bytes = row[1] if row else 0

            return {
                "entries": count,
                "size_bytes": size_bytes,
                "size_mb": round(size_bytes / 1024 / 1024, 2),
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "sets": self._stats["sets"],
                "deletes": self._stats["deletes"],
                "evictions": self._stats["evictions"],
                "hit_rate": (
                    self._stats["hits"] / (self._stats["hits"] + self._stats["misses"])
                    if (self._stats["hits"] + self._stats["misses"]) > 0
                    else 0.0
                ),
            }

        except Exception as e:
            logger.error(f"Cache stats failed: {e}")
            return {}

    async def close(self) -> None:
        """Close database connection."""
        if self._conn:
            try:
                await self._conn.close()
            except Exception:
                pass
            finally:
                self._conn = None


# ─────────────────────────────────────────────
# Convenience Functions
# ─────────────────────────────────────────────

_default_cache: SecureCache | None = None


def get_secure_cache(db_path: Path | None = None) -> SecureCache:
    """Get or create default secure cache."""
    global _default_cache
    if _default_cache is None:
        _default_cache = SecureCache(db_path)
    return _default_cache


def reset_secure_cache() -> None:
    """Reset default cache (for testing)."""
    global _default_cache
    if _default_cache:
        _default_cache.clear()
    _default_cache = None
