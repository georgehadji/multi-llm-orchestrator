"""
Multi-Layer Caching System
==========================

Hierarchical cache with automatic promotion/demotion:
- L1: In-memory (fastest, smallest)
- L2: Redis (shared, medium speed)
- L3: Disk (slowest, largest)

Usage:
    from orchestrator.caching import MultiLayerCache, CacheLevel

    cache = MultiLayerCache()

    # Cache with specific level
    await cache.set(key, value, level=CacheLevel.L1_MEMORY, ttl=timedelta(minutes=5))

    # Automatic fetch from any level
    value = await cache.get(key)  # Tries L1 → L2 → L3
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import pickle
import sqlite3
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = logging.getLogger("orchestrator.caching")


# ═══════════════════════════════════════════════════════════════════════════════
# Cache Level Enum
# ═══════════════════════════════════════════════════════════════════════════════

class CacheLevel(Enum):
    """Cache levels in order of speed (fastest first)."""
    L1_MEMORY = 1    # In-process dict (fastest, ~1MB)
    L2_REDIS = 2     # Shared memory (fast, ~100MB)
    L3_DISK = 3      # Local disk (slow, ~1GB)
    L4_S3 = 4        # Object storage (slowest, unlimited)

    @property
    def is_local(self) -> bool:
        """Check if this level is local (not requiring network)."""
        return self in (CacheLevel.L1_MEMORY, CacheLevel.L3_DISK)


# ═══════════════════════════════════════════════════════════════════════════════
# Cache Entry
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CacheEntry:
    """A cached value with metadata."""
    key: str
    value: Any
    level: CacheLevel
    created_at: datetime
    expires_at: datetime | None
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    size_bytes: int = 0

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    @property
    def age(self) -> timedelta:
        return datetime.utcnow() - self.created_at

    def touch(self) -> None:
        """Update access metadata."""
        self.access_count += 1
        self.last_accessed = datetime.utcnow()


# ═══════════════════════════════════════════════════════════════════════════════
# Cache Backend Interface
# ═══════════════════════════════════════════════════════════════════════════════

class CacheBackend(ABC):
    """Abstract cache backend."""

    @property
    @abstractmethod
    def level(self) -> CacheLevel:
        pass

    @abstractmethod
    async def get(self, key: str) -> Any | None:
        pass

    @abstractmethod
    async def set(
        self,
        key: str,
        value: Any,
        ttl: timedelta | None = None,
    ) -> None:
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        pass

    @abstractmethod
    async def clear(self) -> None:
        pass

    @abstractmethod
    async def keys(self, pattern: str = "*") -> list[str]:
        pass

    @abstractmethod
    async def close(self) -> None:
        pass

    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# L1: In-Memory Cache (LRU)
# ═══════════════════════════════════════════════════════════════════════════════

class InMemoryCache(CacheBackend):
    """Thread-safe in-memory LRU cache."""

    def __init__(self, max_size: int = 1000, max_memory_mb: float = 100):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()
        self._stats = {"hits": 0, "misses": 0, "evictions": 0}
        self._current_memory = 0

    @property
    def level(self) -> CacheLevel:
        return CacheLevel.L1_MEMORY

    async def get(self, key: str) -> Any | None:
        async with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._stats["misses"] += 1
                return None

            if entry.is_expired:
                del self._cache[key]
                self._current_memory -= entry.size_bytes
                self._stats["misses"] += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.touch()
            self._stats["hits"] += 1

            return entry.value

    async def set(
        self,
        key: str,
        value: Any,
        ttl: timedelta | None = None,
    ) -> None:
        # Calculate size
        try:
            size = len(pickle.dumps(value))
        except:
            size = 1024  # Estimate

        async with self._lock:
            # Check if we need to evict
            while (
                len(self._cache) >= self.max_size or
                self._current_memory + size > self.max_memory_bytes
            ):
                if not self._cache:
                    break
                # Evict oldest
                oldest_key, oldest_entry = self._cache.popitem(last=False)
                self._current_memory -= oldest_entry.size_bytes
                self._stats["evictions"] += 1

            expires = None
            if ttl:
                expires = datetime.utcnow() + ttl

            entry = CacheEntry(
                key=key,
                value=value,
                level=self.level,
                created_at=datetime.utcnow(),
                expires_at=expires,
                size_bytes=size,
            )

            # Remove old entry if exists
            if key in self._cache:
                self._current_memory -= self._cache[key].size_bytes

            self._cache[key] = entry
            self._current_memory += size

    async def delete(self, key: str) -> bool:
        async with self._lock:
            if key in self._cache:
                self._current_memory -= self._cache[key].size_bytes
                del self._cache[key]
                return True
            return False

    async def clear(self) -> None:
        async with self._lock:
            self._cache.clear()
            self._current_memory = 0

    async def keys(self, pattern: str = "*") -> list[str]:
        import fnmatch
        async with self._lock:
            return [k for k in self._cache if fnmatch.fnmatch(k, pattern)]

    async def close(self) -> None:
        await self.clear()

    def get_stats(self) -> dict[str, Any]:
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total if total > 0 else 0

        return {
            "level": "L1_MEMORY",
            "entries": len(self._cache),
            "memory_bytes": self._current_memory,
            "memory_mb": round(self._current_memory / (1024 * 1024), 2),
            "hit_rate": round(hit_rate, 3),
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "evictions": self._stats["evictions"],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# L2: Redis Cache
# ═══════════════════════════════════════════════════════════════════════════════

class RedisCache(CacheBackend):
    """Redis-based distributed cache."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        prefix: str = "orch:",
    ):
        self.host = host
        self.port = port
        self.db = db
        self.prefix = prefix
        self._client = None
        self._stats = {"hits": 0, "misses": 0}

    def _get_key(self, key: str) -> str:
        return f"{self.prefix}{key}"

    async def _get_client(self):
        if self._client is None:
            import aioredis
            self._client = await aioredis.from_url(
                f"redis://{self.host}:{self.port}/{self.db}"
            )
        return self._client

    @property
    def level(self) -> CacheLevel:
        return CacheLevel.L2_REDIS

    async def get(self, key: str) -> Any | None:
        try:
            client = await self._get_client()
            data = await client.get(self._get_key(key))

            if data is None:
                self._stats["misses"] += 1
                return None

            self._stats["hits"] += 1
            return pickle.loads(data)
        except Exception as e:
            logger.error(f"Redis get failed: {e}")
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: timedelta | None = None,
    ) -> None:
        try:
            client = await self._get_client()
            data = pickle.dumps(value)

            if ttl:
                await client.setex(
                    self._get_key(key),
                    int(ttl.total_seconds()),
                    data
                )
            else:
                await client.set(self._get_key(key), data)
        except Exception as e:
            logger.error(f"Redis set failed: {e}")

    async def delete(self, key: str) -> bool:
        try:
            client = await self._get_client()
            result = await client.delete(self._get_key(key))
            return result > 0
        except Exception as e:
            logger.error(f"Redis delete failed: {e}")
            return False

    async def clear(self) -> None:
        try:
            client = await self._get_client()
            pattern = self._get_key("*")
            keys = await client.keys(pattern)
            if keys:
                await client.delete(*keys)
        except Exception as e:
            logger.error(f"Redis clear failed: {e}")

    async def keys(self, pattern: str = "*") -> list[str]:
        try:
            client = await self._get_client()
            full_pattern = self._get_key(pattern)
            keys = await client.keys(full_pattern)
            prefix_len = len(self.prefix)
            return [k.decode()[prefix_len:] if isinstance(k, bytes) else k[prefix_len:] for k in keys]
        except Exception as e:
            logger.error(f"Redis keys failed: {e}")
            return []

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None

    def get_stats(self) -> dict[str, Any]:
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total if total > 0 else 0

        return {
            "level": "L2_REDIS",
            "host": self.host,
            "hit_rate": round(hit_rate, 3),
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# L3: Disk Cache
# ═══════════════════════════════════════════════════════════════════════════════

class DiskCache(CacheBackend):
    """SQLite-based disk cache."""

    def __init__(
        self,
        db_path: str = ".cache/disk_cache.db",
        max_size_mb: float = 1000,
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self._stats = {"hits": 0, "misses": 0, "evictions": 0}
        self._initialized = False

    def _init(self):
        if self._initialized:
            return

        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value BLOB,
                    created_at TEXT,
                    expires_at TEXT,
                    size_bytes INTEGER
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_expires ON cache(expires_at)
            """)
            conn.commit()

        self._initialized = True

    @property
    def level(self) -> CacheLevel:
        return CacheLevel.L3_DISK

    async def get(self, key: str) -> Any | None:
        self._init()

        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute(
                    "SELECT value, expires_at FROM cache WHERE key = ?",
                    (key,)
                )
                row = cursor.fetchone()

                if row is None:
                    self._stats["misses"] += 1
                    return None

                value_blob, expires_at = row

                # Check expiration
                if expires_at:
                    expires = datetime.fromisoformat(expires_at)
                    if datetime.utcnow() > expires:
                        conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                        conn.commit()
                        self._stats["misses"] += 1
                        return None

                self._stats["hits"] += 1
                return pickle.loads(value_blob)

        except Exception as e:
            logger.error(f"Disk cache get failed: {e}")
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: timedelta | None = None,
    ) -> None:
        self._init()

        try:
            # Serialize value
            value_blob = pickle.dumps(value)
            size = len(value_blob)

            # Check if we need to evict
            with sqlite3.connect(str(self.db_path)) as conn:
                # Get current size
                cursor = conn.execute("SELECT SUM(size_bytes) FROM cache")
                current_size = cursor.fetchone()[0] or 0

                # Evict if necessary (remove oldest)
                while current_size + size > self.max_size_bytes:
                    cursor = conn.execute(
                        "SELECT key, size_bytes FROM cache ORDER BY created_at ASC LIMIT 1"
                    )
                    row = cursor.fetchone()
                    if row is None:
                        break

                    conn.execute("DELETE FROM cache WHERE key = ?", (row[0],))
                    current_size -= row[1]
                    self._stats["evictions"] += 1

                # Calculate expiration
                expires_at = None
                if ttl:
                    expires_at = (datetime.utcnow() + ttl).isoformat()

                # Insert or replace
                conn.execute(
                    """
                    INSERT OR REPLACE INTO cache (key, value, created_at, expires_at, size_bytes)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (key, value_blob, datetime.utcnow().isoformat(), expires_at, size)
                )
                conn.commit()

        except Exception as e:
            logger.error(f"Disk cache set failed: {e}")

    async def delete(self, key: str) -> bool:
        self._init()

        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Disk cache delete failed: {e}")
            return False

    async def clear(self) -> None:
        self._init()

        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute("DELETE FROM cache")
                conn.commit()
        except Exception as e:
            logger.error(f"Disk cache clear failed: {e}")

    async def keys(self, pattern: str = "*") -> list[str]:
        self._init()

        try:
            import fnmatch
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute("SELECT key FROM cache")
                all_keys = [row[0] for row in cursor.fetchall()]
                return [k for k in all_keys if fnmatch.fnmatch(k, pattern)]
        except Exception as e:
            logger.error(f"Disk cache keys failed: {e}")
            return []

    async def close(self) -> None:
        pass

    def get_stats(self) -> dict[str, Any]:
        self._init()

        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total if total > 0 else 0

        # Get entry count
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute("SELECT COUNT(*), SUM(size_bytes) FROM cache")
                count, size = cursor.fetchone()
        except:
            count, size = 0, 0

        return {
            "level": "L3_DISK",
            "entries": count or 0,
            "size_bytes": size or 0,
            "size_mb": round((size or 0) / (1024 * 1024), 2),
            "hit_rate": round(hit_rate, 3),
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "evictions": self._stats["evictions"],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-Layer Cache
# ═══════════════════════════════════════════════════════════════════════════════

class MultiLayerCache:
    """
    Hierarchical cache with automatic promotion/demotion.

    Reads: L1 → L2 → L3 (fastest first)
    Writes: Specified level and all slower levels
    """

    def __init__(
        self,
        backends: list[CacheBackend] | None = None,
        default_ttl: timedelta = timedelta(hours=1),
    ):
        self.backends = backends or self._create_default_backends()
        self.default_ttl = default_ttl
        self._promotion_lock = asyncio.Lock()

    def _create_default_backends(self) -> list[CacheBackend]:
        """Create default L1 and L3 backends."""
        return [
            InMemoryCache(max_size=1000, max_memory_mb=100),
            DiskCache(max_size_mb=1000),
        ]

    async def get(self, key: str) -> Any | None:
        """
        Get value from cache, trying each level in order.

        If found in slower cache, promotes to faster caches.
        """
        value = None
        found_at = None

        # Try each level
        for i, backend in enumerate(self.backends):
            value = await backend.get(key)
            if value is not None:
                found_at = i
                break

        if value is None:
            return None

        # Promote to faster caches (async, don't block)
        if found_at and found_at > 0:
            asyncio.create_task(self._promote(key, value, found_at))

        return value

    async def _promote(self, key: str, value: Any, from_level: int) -> None:
        """Promote value to faster caches."""
        async with self._promotion_lock:
            for i in range(from_level):
                try:
                    await self.backends[i].set(key, value, ttl=self.default_ttl)
                except Exception as e:
                    logger.warning(f"Failed to promote to L{i+1}: {e}")

    async def set(
        self,
        key: str,
        value: Any,
        ttl: timedelta | None = None,
        level: CacheLevel | None = None,
    ) -> None:
        """
        Set value in cache.

        Writes to specified level and all slower levels.
        """
        ttl = ttl or self.default_ttl

        # Determine which backends to write to
        start_index = 0
        if level:
            start_index = next(
                (i for i, b in enumerate(self.backends) if b.level == level),
                0
            )

        # Write to specified level and all slower levels
        for i in range(start_index, len(self.backends)):
            try:
                await self.backends[i].set(key, value, ttl=ttl)
            except Exception as e:
                logger.warning(f"Failed to write to L{i+1}: {e}")

    async def delete(self, key: str) -> bool:
        """Delete from all cache levels."""
        results = await asyncio.gather(*[
            backend.delete(key)
            for backend in self.backends
        ])
        return any(results)

    async def invalidate(self, pattern: str = "*") -> int:
        """Invalidate all keys matching pattern."""
        total = 0
        for backend in self.backends:
            try:
                keys = await backend.keys(pattern)
                for key in keys:
                    if await backend.delete(key):
                        total += 1
            except Exception as e:
                logger.error(f"Failed to invalidate in {backend.level}: {e}")
        return total

    async def clear(self) -> None:
        """Clear all cache levels."""
        await asyncio.gather(*[b.clear() for b in self.backends])

    async def close(self) -> None:
        """Close all cache backends."""
        await asyncio.gather(*[b.close() for b in self.backends])

    def get_stats(self) -> dict[str, Any]:
        """Get combined stats from all levels."""
        return {
            "levels": [b.get_stats() for b in self.backends],
            "total_hits": sum(b.get_stats().get("hits", 0) for b in self.backends),
            "total_misses": sum(b.get_stats().get("misses", 0) for b in self.backends),
        }

    async def warmup(self, keys: list[str], loader: Callable[[str], Awaitable[Any]]) -> int:
        """
        Warm up cache by pre-loading keys.

        Returns number of keys loaded.
        """
        loaded = 0
        for key in keys:
            value = await self.get(key)
            if value is None:
                try:
                    value = await loader(key)
                    if value is not None:
                        await self.set(key, value)
                        loaded += 1
                except Exception as e:
                    logger.error(f"Failed to warm up key {key}: {e}")
        return loaded


# ═══════════════════════════════════════════════════════════════════════════════
# Global Cache Instance
# ═══════════════════════════════════════════════════════════════════════════════

_cache: MultiLayerCache | None = None


def get_cache() -> MultiLayerCache:
    """Get global cache instance."""
    global _cache
    if _cache is None:
        _cache = MultiLayerCache()
    return _cache


def reset_cache() -> None:
    """Reset global cache (for testing)."""
    global _cache
    _cache = None


# Convenience decorator for caching function results

def cached(
    key_prefix: str = "",
    ttl: timedelta | None = None,
    level: CacheLevel | None = None,
):
    """Decorator to cache function results."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            cache = get_cache()

            # Build cache key
            key_parts = [key_prefix, func.__name__]
            key_parts.extend(str(a) for a in args)
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            key = hashlib.sha256(":".join(key_parts).encode()).hexdigest()

            # Try cache
            cached_value = await cache.get(key)
            if cached_value is not None:
                return cached_value

            # Compute and cache
            result = await func(*args, **kwargs)
            await cache.set(key, result, ttl=ttl, level=level)
            return result

        return wrapper
    return decorator
