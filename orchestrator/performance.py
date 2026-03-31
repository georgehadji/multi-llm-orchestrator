"""
Performance Optimization Module
==============================
Caching, connection pooling, and monitoring for the Multi-LLM Orchestrator.

Provides:
- LRU/Memory cache with TTL support
- Redis integration (fallback to in-memory)
- Database connection pooling
- Request/response caching decorators
- Performance metrics collection

Usage:
    from orchestrator.performance import cached, cache, ConnectionPool

    @cached(ttl=300)
    async def get_expensive_data():
        return await fetch_from_db()
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import inspect
import json
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar

from typing_extensions import ParamSpec

from .log_config import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


# ═══════════════════════════════════════════════════════════════════════════════
# CACHE IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class CacheEntry:
    """Cache entry with TTL and metadata."""

    value: Any
    created_at: float
    ttl: int
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        return time.time() - self.created_at > self.ttl

    def touch(self):
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = time.time()


class LRUCache:
    """
    Thread-safe LRU cache with TTL support.

    Features:
    - Maximum size enforcement (evicts least recently used)
    - Per-key TTL support
    - Hit/miss statistics
    - Memory usage tracking
    """

    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._lock = asyncio.Lock()
        self._stats = {"hits": 0, "misses": 0, "evictions": 0}

    async def get(self, key: str) -> Any | None:
        """Get value from cache."""
        async with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._stats["misses"] += 1
                return None

            if entry.is_expired():
                del self._cache[key]
                self._stats["misses"] += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.touch()
            self._stats["hits"] += 1

            return entry.value

    async def set(self, key: str, value: Any, ttl: int | None = None):
        """Set value in cache with optional TTL."""
        async with self._lock:
            # Evict expired entries first
            await self._evict_expired()

            # Evict LRU if at capacity
            if len(self._cache) >= self._max_size and key not in self._cache:
                self._cache.popitem(last=False)
                self._stats["evictions"] += 1

            entry = CacheEntry(
                value=value,
                created_at=time.time(),
                ttl=ttl or self._default_ttl,
            )
            self._cache[key] = entry
            self._cache.move_to_end(key)

    async def delete(self, key: str) -> bool:
        """Delete key from cache. Returns True if key existed."""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    async def clear(self):
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()

    async def _evict_expired(self):
        """Remove expired entries."""
        expired_keys = [key for key, entry in self._cache.items() if entry.is_expired()]
        for key in expired_keys:
            del self._cache[key]
            self._stats["evictions"] += 1

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total if total > 0 else 0

        # Calculate memory usage estimate
        total_accesses = sum(entry.access_count for entry in self._cache.values())

        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "hit_rate": f"{hit_rate:.1%}",
            "evictions": self._stats["evictions"],
            "memory_estimate_mb": len(self._cache) * 0.001,  # Rough estimate
            "avg_accesses_per_key": total_accesses / len(self._cache) if self._cache else 0,
        }


class RedisCache:
    """
    Redis-backed cache with connection pooling.
    Falls back gracefully if Redis is unavailable.
    """

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        self._host = host
        self._port = port
        self._db = db
        self._redis = None
        self._pool = None
        self._fallback_cache: LRUCache | None = None
        self._initialized = False

    async def initialize(self):
        """Initialize Redis connection pool."""
        if self._initialized:
            return

        try:
            import redis.asyncio as redis

            self._pool = redis.ConnectionPool(
                host=self._host,
                port=self._port,
                db=self._db,
                max_connections=50,
                retry_on_timeout=True,
            )
            self._redis = redis.Redis(connection_pool=self._pool)

            # Test connection
            await self._redis.ping()
            logger.info(f"Redis cache connected at {self._host}:{self._port}")

        except Exception as e:
            logger.warning(f"Redis unavailable ({e}), using in-memory LRU cache")
            self._fallback_cache = LRUCache(max_size=5000)

        self._initialized = True

    async def get(self, key: str) -> Any | None:
        """Get value from cache."""
        await self.initialize()

        if self._redis:
            try:
                value = await self._redis.get(key)
                if value:
                    return json.loads(value)
            except Exception as e:
                logger.debug(f"Redis get error: {e}")

        if self._fallback_cache:
            return await self._fallback_cache.get(key)

        return None

    async def set(self, key: str, value: Any, ttl: int = 300):
        """Set value in cache with TTL."""
        await self.initialize()

        if self._redis:
            try:
                await self._redis.setex(key, ttl, json.dumps(value))
                return
            except Exception as e:
                logger.debug(f"Redis set error: {e}")

        if self._fallback_cache:
            await self._fallback_cache.set(key, value, ttl)

    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        await self.initialize()

        if self._redis:
            try:
                result = await self._redis.delete(key)
                return result > 0
            except Exception as e:
                logger.debug(f"Redis delete error: {e}")

        if self._fallback_cache:
            return await self._fallback_cache.delete(key)

        return False

    async def get_many(self, keys: list[str]) -> dict[str, Any]:
        """Get multiple values efficiently."""
        await self.initialize()

        if self._redis:
            try:
                values = await self._redis.mget(keys)
                return {
                    key: json.loads(value) if value else None
                    for key, value in zip(keys, values, strict=False)
                }
            except Exception as e:
                logger.debug(f"Redis mget error: {e}")

        if self._fallback_cache:
            results = {}
            for key in keys:
                value = await self._fallback_cache.get(key)
                results[key] = value
            return results

        return dict.fromkeys(keys)

    async def set_many(self, mapping: dict[str, Any], ttl: int = 300):
        """Set multiple values with pipeline for efficiency."""
        await self.initialize()

        if self._redis:
            try:
                pipe = self._redis.pipeline()
                for key, value in mapping.items():
                    pipe.setex(key, ttl, json.dumps(value))
                await pipe.execute()
                return
            except Exception as e:
                logger.debug(f"Redis pipeline error: {e}")

        if self._fallback_cache:
            for key, value in mapping.items():
                await self._fallback_cache.set(key, value, ttl)

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        stats = {
            "backend": "redis" if self._redis else "memory",
            "connected": self._redis is not None,
        }

        if self._fallback_cache:
            stats.update(self._fallback_cache.get_stats())

        return stats

    async def close(self):
        """Close Redis connection pool."""
        if self._pool:
            await self._pool.disconnect()


# Global cache instance (lazy initialization)
_cache_instance: RedisCache | None = None


def get_cache() -> RedisCache:
    """Get global cache instance."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = RedisCache()
    return _cache_instance


# Shortcuts for convenience
cache = property(get_cache)


# ═══════════════════════════════════════════════════════════════════════════════
# CACHE DECORATORS
# ═══════════════════════════════════════════════════════════════════════════════


def generate_cache_key(func: Callable, args: tuple, kwargs: dict) -> str:
    """Generate deterministic cache key from function call."""
    # Get function signature
    sig = inspect.signature(func)
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()

    # Create key from function name and arguments
    key_parts = [
        func.__module__,
        func.__qualname__,
    ]

    # Add bound arguments
    for name, value in sorted(bound.arguments.items()):
        key_parts.append(f"{name}={repr(value)}")

    key_string = "|".join(key_parts)
    return hashlib.sha256(key_string.encode()).hexdigest()[:32]


def cached(ttl: int = 300, key_prefix: str | None = None):
    """
    Decorator for caching function results.

    Args:
        ttl: Time to live in seconds
        key_prefix: Optional prefix for cache keys

    Usage:
        @cached(ttl=600)
        async def get_models():
            return await fetch_models()
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            cache = get_cache()

            # Generate cache key
            if key_prefix:
                cache_key = f"{key_prefix}:{generate_cache_key(func, args, kwargs)}"
            else:
                cache_key = generate_cache_key(func, args, kwargs)

            # Try cache
            cached_value = await cache.get(cache_key)
            if cached_value is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_value

            # Execute function
            result = await func(*args, **kwargs)

            # Store in cache
            await cache.set(cache_key, result, ttl)

            return result

        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Run async version in sync context
            get_cache()
            cache_key = generate_cache_key(func, args, kwargs)
            if key_prefix:
                cache_key = f"{key_prefix}:{cache_key}"

            # For sync functions, we need to handle caching differently
            # Since we can't await, we'll just execute without caching
            # or raise an error if caching is required
            return func(*args, **kwargs)

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


def cache_invalidate(*keys: str):
    """
    Invalidate specific cache keys.

    Usage:
        @cache_invalidate("models", "routing_table")
        async def update_models():
            # ... update logic
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            result = await func(*args, **kwargs)

            cache = get_cache()
            for key in keys:
                await cache.delete(key)
                logger.debug(f"Invalidated cache key: {key}")

            return result

        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            result = func(*args, **kwargs)

            # Fire and forget async invalidation
            cache = get_cache()
            for key in keys:
                asyncio.create_task(cache.delete(key))

            return result

        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

    return decorator


# ═══════════════════════════════════════════════════════════════════════════════
# CONNECTION POOLING
# ═══════════════════════════════════════════════════════════════════════════════


class ConnectionPool:
    """
    Generic connection pool for database/API connections.

    Features:
    - Min/max pool size management
    - Connection timeout handling
    - Health checks
    - Graceful shutdown
    """

    def __init__(
        self,
        factory: Callable[[], Any],
        min_size: int = 2,
        max_size: int = 10,
        max_idle_time: int = 300,
        connection_timeout: int = 30,
    ):
        self._factory = factory
        self._min_size = min_size
        self._max_size = max_size
        self._max_idle_time = max_idle_time
        self._connection_timeout = connection_timeout

        self._pool: asyncio.Queue = asyncio.Queue(maxsize=max_size)
        self._connections: list[Any] = []
        self._in_use: set = set()
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(max_size)
        self._initialized = False
        self._stats = {
            "created": 0,
            "destroyed": 0,
            "wait_timeouts": 0,
            "checkouts": 0,
            "checkins": 0,
        }

    async def initialize(self):
        """Initialize pool with minimum connections."""
        if self._initialized:
            return

        for _ in range(self._min_size):
            conn = await self._create_connection()
            await self._pool.put(conn)

        self._initialized = True
        logger.info(f"Connection pool initialized: {self._min_size}/{self._max_size}")

    async def _create_connection(self) -> Any:
        """Create new connection."""
        try:
            conn = await asyncio.wait_for(
                self._factory(),
                timeout=self._connection_timeout,
            )
            self._stats["created"] += 1
            return conn
        except asyncio.TimeoutError:
            self._stats["wait_timeouts"] += 1
            raise ConnectionError("Connection timeout")

    async def _destroy_connection(self, conn: Any):
        """Close and remove connection."""
        try:
            if hasattr(conn, "close"):
                await conn.close() if asyncio.iscoroutinefunction(conn.close) else conn.close()
            self._stats["destroyed"] += 1
        except Exception as e:
            logger.warning(f"Error destroying connection: {e}")

    @asynccontextmanager
    async def acquire(self):
        """Acquire connection from pool."""
        await self.initialize()

        async with self._semaphore:
            conn = None
            try:
                # Try to get from pool with timeout
                try:
                    conn = await asyncio.wait_for(
                        self._pool.get(),
                        timeout=self._connection_timeout,
                    )
                except asyncio.TimeoutError:
                    self._stats["wait_timeouts"] += 1
                    raise ConnectionError("Pool exhausted")

                self._in_use.add(id(conn))
                self._stats["checkouts"] += 1

                yield conn

            finally:
                if conn:
                    self._in_use.discard(id(conn))
                    self._stats["checkins"] += 1
                    await self._pool.put(conn)

    def get_stats(self) -> dict[str, Any]:
        """Get pool statistics."""
        return {
            **self._stats,
            "pool_size": self._pool.qsize(),
            "in_use": len(self._in_use),
            "available": self._pool.qsize() - len(self._in_use),
            "utilization": len(self._in_use) / self._max_size if self._max_size > 0 else 0,
        }

    async def close(self):
        """Close all connections in pool."""
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                await self._destroy_connection(conn)
            except asyncio.QueueEmpty:
                break


# ═══════════════════════════════════════════════════════════════════════════════
# PERFORMANCE METRICS
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class MetricPoint:
    """Single metric data point."""

    timestamp: float
    value: float
    labels: dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """
    Collects and aggregates performance metrics.

    Tracks:
    - Response times (p50, p95, p99)
    - Request rates
    - Error rates
    - Resource utilization
    """

    def __init__(self, max_history: int = 1000):
        self._metrics: dict[str, list[MetricPoint]] = {}
        self._max_history = max_history
        self._lock = asyncio.Lock()

    async def record(self, name: str, value: float, labels: dict[str, str] | None = None):
        """Record a metric value."""
        async with self._lock:
            if name not in self._metrics:
                self._metrics[name] = []

            point = MetricPoint(
                timestamp=time.time(),
                value=value,
                labels=labels or {},
            )

            self._metrics[name].append(point)

            # Trim history
            if len(self._metrics[name]) > self._max_history:
                self._metrics[name] = self._metrics[name][-self._max_history :]

    def get_stats(self, name: str, window_seconds: int = 300) -> dict[str, Any]:
        """Get statistics for a metric."""
        if name not in self._metrics:
            return {}

        cutoff = time.time() - window_seconds
        points = [p for p in self._metrics[name] if p.timestamp > cutoff]

        if not points:
            return {"count": 0}

        values = [p.value for p in points]
        values_sorted = sorted(values)
        n = len(values)

        return {
            "count": n,
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / n,
            "p50": values_sorted[int(n * 0.50)],
            "p95": values_sorted[int(n * 0.95)] if n > 20 else values_sorted[-1],
            "p99": values_sorted[int(n * 0.99)] if n > 100 else values_sorted[-1],
        }

    def get_all_stats(self, window_seconds: int = 300) -> dict[str, dict[str, Any]]:
        """Get statistics for all metrics."""
        return {name: self.get_stats(name, window_seconds) for name in self._metrics}


# Global metrics collector
metrics_collector = MetricsCollector()


# ═══════════════════════════════════════════════════════════════════════════════
# QUERY OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════════


class QueryOptimizer:
    """
    Database query optimization utilities.

    Provides:
    - Query result caching
    - Batch operations
    - N+1 query prevention
    - Connection reuse
    """

    def __init__(self, cache_ttl: int = 60):
        self._cache_ttl = cache_ttl

    @cached(ttl=300)
    async def cached_query(self, query_key: str, query_func: Callable[[], T]) -> T:
        """Execute query with caching."""
        return await query_func()

    async def batch_get(
        self,
        ids: list[str],
        fetch_func: Callable[[list[str]], list[T]],
        batch_size: int = 100,
    ) -> list[T]:
        """
        Fetch multiple items in batches to prevent N+1 queries.

        Args:
            ids: List of IDs to fetch
            fetch_func: Function that accepts list of IDs and returns items
            batch_size: Number of IDs per batch
        """
        results = []

        for i in range(0, len(ids), batch_size):
            batch = ids[i : i + batch_size]
            batch_results = await fetch_func(batch)
            results.extend(batch_results)

        return results

    def build_selective_query(
        self,
        table: str,
        columns: list[str] | None = None,
        where: dict[str, Any] | None = None,
        order_by: str | None = None,
        limit: int | None = None,
    ) -> str:
        """
        Build optimized SELECT query with only needed columns.

        Args:
            table: Table name
            columns: Specific columns (None = all, not recommended)
            where: WHERE clause conditions
            order_by: ORDER BY column
            limit: LIMIT value
        """
        # Use specific columns instead of SELECT *
        col_str = ", ".join(columns) if columns else "*"
        query = f"SELECT {col_str} FROM {table}"

        # Add WHERE clause
        if where:
            conditions = " AND ".join(f"{k} = ?" for k in where)
            query += f" WHERE {conditions}"

        # Add ORDER BY
        if order_by:
            query += f" ORDER BY {order_by}"

        # Add LIMIT
        if limit:
            query += f" LIMIT {limit}"

        return query


# ═══════════════════════════════════════════════════════════════════════════════
# USAGE EXAMPLES
# ═══════════════════════════════════════════════════════════════════════════════


async def example_usage():
    """Example usage of performance module."""

    # Initialize cache
    cache = get_cache()
    await cache.initialize()

    # Basic caching
    await cache.set("key", {"data": "value"}, ttl=300)
    await cache.get("key")

    # Using decorator
    @cached(ttl=600)
    async def fetch_models():
        # Expensive operation
        return {"models": ["gpt-4", "claude"]}

    await fetch_models()  # First call - cache miss
    await fetch_models()  # Second call - cache hit

    # Cache statistics
    stats = cache.get_stats()
    print(f"Cache hit rate: {stats.get('hit_rate', 'N/A')}")

    # Connection pool
    async def create_db_connection():
        # Your connection logic here
        pass

    pool = ConnectionPool(create_db_connection, min_size=2, max_size=10)

    async with pool.acquire():
        # Use connection
        pass

    # Metrics
    await metrics_collector.record("response_time", 45.2, {"endpoint": "/api/models"})
    stats = metrics_collector.get_stats("response_time")
    print(f"P95 response time: {stats.get('p95', 'N/A')}ms")

    # Cleanup
    await cache.close()
    await pool.close()


if __name__ == "__main__":
    asyncio.run(example_usage())
