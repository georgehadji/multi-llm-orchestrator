"""
Meta-Optimization Performance Utilities
========================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Performance optimizations for meta-optimization at scale:
- Async batch processing for large archives
- LRU caching for frequently accessed data
- Connection pooling for database operations

USAGE:
    from orchestrator.meta_performance import AsyncBatchProcessor, LRUCache

    # Batch processing
    processor = AsyncBatchProcessor(batch_size=100, concurrency=10)
    results = await processor.process_batch(items, process_fn)

    # Caching
    cache = LRUCache(max_size=1000, ttl=3600)
    await cache.put("key", "value")
    value = await cache.get("key")
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import aiosqlite

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

logger = logging.getLogger("orchestrator.meta_performance")


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────


@dataclass
class PerformanceConfig:
    """Configuration for performance optimizations."""

    # Batch processing
    batch_size: int = 100
    max_concurrency: int = 10
    batch_timeout: float = 30.0

    # Caching
    cache_max_size: int = 1000
    cache_ttl_seconds: int = 3600
    embedding_cache_ttl: int = 7200  # 2 hours for embeddings

    # Connection pooling
    pool_min_size: int = 1
    pool_max_size: int = 5
    connection_timeout: float = 10.0

    # Performance monitoring
    enable_stats: bool = True
    slow_threshold: float = 1.0  # Log operations slower than this


# ─────────────────────────────────────────────
# Async Batch Processor
# ─────────────────────────────────────────────

T = TypeVar("T")
U = TypeVar("U")


class AsyncBatchProcessor:
    """
    Process large archives in batches with controlled concurrency.

    Prevents memory exhaustion and improves throughput for
    large-scale meta-optimization operations.
    """

    def __init__(self, config: PerformanceConfig | None = None):
        self.config = config or PerformanceConfig()
        self._semaphore = asyncio.Semaphore(self.config.max_concurrency)
        self._stats = {
            "batches_processed": 0,
            "items_processed": 0,
            "total_time": 0.0,
            "errors": 0,
        }

    async def process_batch(
        self,
        items: list[T],
        processor: Callable[[T], U],
        batch_size: int | None = None,
        concurrency: int | None = None,
    ) -> list[U]:
        """
        Process items in batches with controlled concurrency.

        Args:
            items: Items to process
            processor: Async function to process each item
            batch_size: Override default batch size
            concurrency: Override default concurrency

        Returns:
            List of processed results
        """
        batch_size = batch_size or self.config.batch_size
        concurrency = concurrency or self.config.max_concurrency

        start_time = time.time()
        semaphore = asyncio.Semaphore(concurrency)

        async def process_with_semaphore(item: T) -> U:
            async with semaphore:
                try:
                    return await processor(item)
                except Exception as e:
                    logger.warning(f"Batch processor error: {e}")
                    self._stats["errors"] += 1
                    raise

        # Split into batches
        batches = [items[i : i + batch_size] for i in range(0, len(items), batch_size)]

        # Process all batches concurrently
        tasks = []
        for batch in batches:
            batch_tasks = [process_with_semaphore(item) for item in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            tasks.append(batch_results)
            self._stats["batches_processed"] += 1
            self._stats["items_processed"] += len(batch)

        # Flatten results
        results = []
        for batch_results in tasks:
            for result in batch_results:
                if not isinstance(result, Exception):
                    results.append(result)

        elapsed = time.time() - start_time
        self._stats["total_time"] += elapsed

        if elapsed > self.config.slow_threshold:
            logger.info(
                f"Batch processing completed: {len(items)} items in {elapsed:.2f}s "
                f"({len(items)/elapsed:.1f} items/sec)"
            )

        return results

    async def process_batch_with_retry(
        self,
        items: list[T],
        processor: Callable[[T], U],
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> list[U]:
        """
        Process batch with automatic retry on failure.

        Args:
            items: Items to process
            processor: Async function to process each item
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries (exponential backoff)

        Returns:
            List of processed results
        """

        async def process_with_retry(item: T) -> U:
            last_error = None
            for attempt in range(max_retries):
                try:
                    return await processor(item)
                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        delay = retry_delay * (2**attempt)  # Exponential backoff
                        await asyncio.sleep(delay)

            if last_error:
                raise last_error
            raise RuntimeError("Retry failed without error")  # Should never reach here

        return await self.process_batch(items, process_with_retry)

    def get_stats(self) -> dict[str, Any]:
        """Get batch processing statistics."""
        stats = self._stats.copy()
        if stats["items_processed"] > 0:
            stats["avg_time_per_item"] = stats["total_time"] / stats["items_processed"]
        return stats


# ─────────────────────────────────────────────
# LRU Cache
# ─────────────────────────────────────────────


@dataclass
class CacheEntry(Generic[T]):
    """Cache entry with metadata."""

    value: T
    created_at: float
    last_accessed: float
    size_bytes: int = 0


class LRUCache(Generic[T]):
    """
    LRU cache with TTL and size limits.

    Optimized for frequently accessed meta-optimization data
    like embeddings and pattern matches.
    """

    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: int = 3600,
        max_memory_mb: float | None = None,
    ):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024) if max_memory_mb else None

        self._cache: OrderedDict[str, CacheEntry[T]] = OrderedDict()
        self._lock = asyncio.Lock()
        self._current_memory = 0

        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expirations": 0,
        }

    async def get(self, key: str) -> T | None:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value, or None if not found/expired
        """
        async with self._lock:
            if key not in self._cache:
                self._stats["misses"] += 1
                return None

            entry = self._cache[key]

            # Check TTL
            if time.time() - entry.created_at > self.ttl_seconds:
                await self._evict(key)
                self._stats["expirations"] += 1
                return None

            # Update access time and move to end (most recently used)
            entry.last_accessed = time.time()
            self._cache.move_to_end(key)
            self._stats["hits"] += 1

            return entry.value

    async def put(self, key: str, value: T, size_bytes: int = 0):
        """
        Put value in cache.

        Args:
            key: Cache key
            value: Value to cache
            size_bytes: Estimated size in bytes (for memory limiting)
        """
        async with self._lock:
            now = time.time()

            # If key exists, remove old entry first
            if key in self._cache:
                old_entry = self._cache[key]
                self._current_memory -= old_entry.size_bytes
                del self._cache[key]

            # Check memory limit
            if self.max_memory_bytes:
                while self._current_memory + size_bytes > self.max_memory_bytes and self._cache:
                    # Evict least recently used
                    oldest_key = next(iter(self._cache))
                    await self._evict(oldest_key)

            # Check size limit
            while len(self._cache) >= self.max_size and self._cache:
                oldest_key = next(iter(self._cache))
                await self._evict(oldest_key)

            # Add new entry
            entry = CacheEntry(
                value=value,
                created_at=now,
                last_accessed=now,
                size_bytes=size_bytes,
            )
            self._cache[key] = entry
            self._current_memory += size_bytes

    async def _evict(self, key: str):
        """Evict a key from cache."""
        if key in self._cache:
            entry = self._cache[key]
            self._current_memory -= entry.size_bytes
            del self._cache[key]
            self._stats["evictions"] += 1

    async def clear(self):
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()
            self._current_memory = 0

    async def cleanup_expired(self) -> int:
        """
        Remove expired entries.

        Returns:
            Number of entries removed
        """
        async with self._lock:
            now = time.time()
            expired = [
                key
                for key, entry in self._cache.items()
                if now - entry.created_at > self.ttl_seconds
            ]

            for key in expired:
                await self._evict(key)

            return len(expired)

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        hit_rate = (
            self._stats["hits"] / (self._stats["hits"] + self._stats["misses"])
            if (self._stats["hits"] + self._stats["misses"]) > 0
            else 0.0
        )

        return {
            **self._stats,
            "hit_rate": hit_rate,
            "size": len(self._cache),
            "max_size": self.max_size,
            "memory_bytes": self._current_memory,
            "max_memory_bytes": self.max_memory_bytes,
        }


# ─────────────────────────────────────────────
# Connection Pool
# ─────────────────────────────────────────────


class ConnectionPool:
    """
    Pool for SQLite connections.

    Reduces connection overhead for high-frequency database operations
    in meta-optimization.
    """

    def __init__(
        self,
        db_path: Path,
        config: PerformanceConfig | None = None,
    ):
        self.db_path = db_path
        self.config = config or PerformanceConfig()

        self._pool: asyncio.Queue[aiosqlite.Connection] = asyncio.Queue(
            maxsize=self.config.pool_max_size
        )
        self._lock = asyncio.Lock()
        self._initialized = False

        self._stats = {
            "connections_created": 0,
            "connections_borrowed": 0,
            "connections_returned": 0,
            "wait_time": 0.0,
        }

    async def initialize(self):
        """Initialize connection pool with minimum connections."""
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return

            # Create minimum connections
            for _ in range(self.config.pool_min_size):
                conn = await self._create_connection()
                await self._pool.put(conn)

            self._initialized = True
            logger.debug(
                f"Connection pool initialized with {self.config.pool_min_size} connections"
            )

    async def _create_connection(self) -> aiosqlite.Connection:
        """Create a new database connection."""
        start_time = time.time()

        conn = await aiosqlite.connect(
            str(self.db_path),
            timeout=self.config.connection_timeout,
        )

        # Optimize for concurrent reads
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.execute("PRAGMA synchronous=NORMAL")
        await conn.execute("PRAGMA cache_size=-64000")  # 64MB cache

        self._stats["connections_created"] += 1

        elapsed = time.time() - start_time
        logger.debug(f"Created database connection in {elapsed:.3f}s")

        return conn

    async def acquire(self) -> aiosqlite.Connection:
        """
        Acquire a connection from the pool.

        Returns:
            Database connection
        """
        if not self._initialized:
            await self.initialize()

        start_time = time.time()

        # Try to get existing connection
        try:
            conn = await asyncio.wait_for(
                self._pool.get(),
                timeout=self.config.connection_timeout,
            )
            self._stats["connections_borrowed"] += 1

            # Verify connection is still valid
            try:
                await conn.execute("SELECT 1")
                return conn
            except aiosqlite.Error:
                # Connection invalid, create new one
                logger.warning("Acquired invalid connection, creating new one")
                return await self._create_connection()

        except asyncio.TimeoutError:
            # Pool exhausted, create new connection if under limit
            if self._pool.qsize() < self.config.pool_max_size:
                return await self._create_connection()

            # Wait for available connection
            elapsed = time.time() - start_time
            self._stats["wait_time"] += elapsed
            return await self._pool.get()

    async def release(self, conn: aiosqlite.Connection):
        """
        Return a connection to the pool.

        Args:
            conn: Connection to return
        """
        if self._pool.qsize() < self.config.pool_max_size:
            await self._pool.put(conn)
            self._stats["connections_returned"] += 1
        else:
            # Pool full, close connection
            try:
                await conn.close()
            except Exception:
                pass

    async def close(self):
        """Close all connections in the pool."""
        async with self._lock:
            while not self._pool.empty():
                conn = await self._pool.get()
                try:
                    await conn.close()
                except Exception:
                    pass

            self._initialized = False

    def get_stats(self) -> dict[str, Any]:
        """Get connection pool statistics."""
        return {
            **self._stats,
            "pool_size": self._pool.qsize(),
            "max_pool_size": self.config.pool_max_size,
            "initialized": self._initialized,
        }

    async def __aenter__(self) -> ConnectionPool:
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# ─────────────────────────────────────────────
# Context Manager for Pooled Connections
# ─────────────────────────────────────────────


class PooledConnection:
    """Context manager for acquiring pooled connections."""

    def __init__(self, pool: ConnectionPool):
        self.pool = pool
        self.conn: aiosqlite.Connection | None = None

    async def __aenter__(self) -> aiosqlite.Connection:
        self.conn = await self.pool.acquire()
        return self.conn

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            await self.pool.release(self.conn)


# ─────────────────────────────────────────────
# Convenience Functions
# ─────────────────────────────────────────────

_default_cache: LRUCache | None = None


def get_default_cache(max_size: int = 1000, ttl: int = 3600) -> LRUCache:
    """Get or create default LRU cache."""
    global _default_cache
    if _default_cache is None:
        _default_cache = LRUCache(max_size=max_size, ttl_seconds=ttl)
    return _default_cache


def reset_default_cache() -> None:
    """Reset default cache (for testing)."""
    global _default_cache
    _default_cache = None
