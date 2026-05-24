"""
AgentCache — Hash-based response caching for agent calls
===========================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Optimization D-10: Caches agent responses for identical task patterns
using hash(task.goal + task.context) as key. TTL-based expiration.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("orchestrator.learning.agent_cache")

CACHE_TTL_SECONDS = 3600  # 1 hour


@dataclass
class CachedResponse:
    """A cached agent response with metadata."""
    key: str
    output: str
    score: float
    timestamp: float = field(default_factory=time.time)

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.timestamp) > CACHE_TTL_SECONDS


class AgentCache:
    """Hash-based cache for agent responses."""

    def __init__(self) -> None:
        self._cache: dict[str, CachedResponse] = {}

    def make_key(self, goal: str, context: str = "", model: str = "") -> str:
        """Create a deterministic cache key from task parameters."""
        raw = f"{goal}:::{context}:::{model}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    def get(self, key: str) -> CachedResponse | None:
        """Get a cached response. Returns None on miss or expiry."""
        entry = self._cache.get(key)
        if entry is None:
            return None
        if entry.is_expired:
            del self._cache[key]
            return None
        return entry

    def put(self, key: str, output: str, score: float) -> CachedResponse:
        """Store a response in the cache."""
        entry = CachedResponse(key=key, output=output, score=score)
        self._cache[key] = entry
        return entry

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()

    @property
    def size(self) -> int:
        return len(self._cache)
