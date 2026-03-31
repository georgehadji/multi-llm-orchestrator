"""
Nexus Search — Query Cache
===========================
Author: Georgios-Chrysovalantis Chatzivantsidis

TTL-based query result caching for improved performance.

Features:
- MD5-based cache key generation
- Configurable TTL (time-to-live)
- Automatic expiration
- Cache metrics tracking

Usage:
    from orchestrator.nexus_search.optimization import QueryCache

    cache = QueryCache(ttl_seconds=3600)

    # Try to get cached results
    results = await cache.get(query, sources)
    if results is None:
        # Cache miss, perform search
        results = await search(query, sources)
        # Cache the results
        await cache.set(query, sources, results)
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from orchestrator.nexus_search.models import SearchResults, SearchSource

logger = logging.getLogger("orchestrator.nexus_search")


@dataclass
class CacheEntry:
    """A single cache entry."""

    results: SearchResults
    timestamp: datetime
    ttl_seconds: int
    hits: int = 0

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        age = datetime.now() - self.timestamp
        return age.total_seconds() > self.ttl_seconds

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "results": self.results.to_dict(),
            "timestamp": self.timestamp.isoformat(),
            "ttl_seconds": self.ttl_seconds,
            "hits": self.hits,
        }


class QueryCache:
    """
    TTL-based query result cache.

    Features:
    - MD5-based cache key generation
    - Configurable TTL
    - Automatic expiration
    - Hit/miss metrics

    Usage:
        cache = QueryCache(ttl_seconds=3600)
        results = await cache.get(query, sources)
    """

    def __init__(self, ttl_seconds: int = 3600, max_size: int = 1000):
        """
        Initialize query cache.

        Args:
            ttl_seconds: Time-to-live for cache entries (default: 1 hour)
            max_size: Maximum number of entries (default: 1000)
        """
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self._cache: dict[str, CacheEntry] = {}

        # Metrics
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def _compute_cache_key(self, query: str, sources: list[SearchSource]) -> str:
        """
        Compute cache key from query and sources.

        Args:
            query: Search query
            sources: List of search sources

        Returns:
            MD5 hash of query + sources
        """
        # Create canonical representation
        sources_str = ",".join(sorted(s.value for s in sources))
        key = f"{query.lower().strip()}:{sources_str}"

        # Return MD5 hash
        return hashlib.md5(key.encode("utf-8")).hexdigest()

    async def get(
        self,
        query: str,
        sources: list[SearchSource],
    ) -> SearchResults | None:
        """
        Get cached results if available and fresh.

        Args:
            query: Search query
            sources: Search sources

        Returns:
            Cached SearchResults or None if not found/expired
        """
        cache_key = self._compute_cache_key(query, sources)

        if cache_key in self._cache:
            entry = self._cache[cache_key]

            # Check if expired
            if entry.is_expired():
                logger.debug(f"Cache entry expired: {query[:50]}...")
                del self._cache[cache_key]
                self.misses += 1
                return None

            # Cache hit
            entry.hits += 1
            self.hits += 1
            logger.debug(
                f"Cache hit: {query[:50]}... (age: {(datetime.now() - entry.timestamp).seconds}s, hits: {entry.hits})"
            )
            return entry.results

        # Cache miss
        self.misses += 1
        logger.debug(f"Cache miss: {query[:50]}...")
        return None

    async def set(
        self,
        query: str,
        sources: list[SearchSource],
        results: SearchResults,
    ) -> None:
        """
        Cache search results.

        Args:
            query: Search query
            sources: Search sources
            results: Search results to cache
        """
        cache_key = self._compute_cache_key(query, sources)

        # Check if at max size
        if len(self._cache) >= self.max_size:
            # Evict oldest entry
            self._evict_oldest()

        # Create cache entry
        entry = CacheEntry(
            results=results,
            timestamp=datetime.now(),
            ttl_seconds=self.ttl_seconds,
        )

        self._cache[cache_key] = entry
        logger.debug(f"Cached: {query[:50]}... (TTL: {self.ttl_seconds}s)")

    def _evict_oldest(self) -> None:
        """Evict oldest cache entry."""
        if not self._cache:
            return

        # Find oldest entry
        oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k].timestamp)

        del self._cache[oldest_key]
        self.evictions += 1
        logger.debug(f"Evicted oldest entry (cache size: {len(self._cache)}/{self.max_size})")

    async def clear(self) -> None:
        """Clear all cached results."""
        self._cache.clear()
        logger.info("Cache cleared")

    async def cleanup_expired(self) -> int:
        """
        Remove expired entries.

        Returns:
            Number of entries removed
        """
        expired_keys = [key for key, entry in self._cache.items() if entry.is_expired()]

        for key in expired_keys:
            del self._cache[key]

        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")

        return len(expired_keys)

    def get_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache metrics
        """
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0

        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": hit_rate,
            "total_requests": total_requests,
        }

    def reset_stats(self) -> None:
        """Reset cache statistics."""
        self.hits = 0
        self.misses = 0
        self.evictions = 0


# Global cache instance
_cache: QueryCache | None = None


def get_query_cache(ttl_seconds: int = 3600, max_size: int = 1000) -> QueryCache:
    """
    Get or create global query cache instance.

    Args:
        ttl_seconds: Cache TTL (default: 1 hour)
        max_size: Maximum cache size (default: 1000)

    Returns:
        QueryCache instance
    """
    global _cache
    if _cache is None:
        _cache = QueryCache(ttl_seconds=ttl_seconds, max_size=max_size)
    return _cache


def reset_cache() -> None:
    """Reset global cache instance."""
    global _cache
    if _cache:
        _cache._cache.clear()  # Clear synchronously
