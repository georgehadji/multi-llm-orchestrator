"""Prompt cache for NadirClaw — in-memory LRU cache for chat completions.

Caches LLM responses keyed by (model + messages hash) to skip redundant calls.
Configurable via environment variables:
  NADIRCLAW_CACHE_ENABLED   — enable/disable (default: true)
  NADIRCLAW_CACHE_TTL       — seconds before entries expire (default: 300)
  NADIRCLAW_CACHE_MAX_SIZE  — max cached entries (default: 1000)
"""

import hashlib
import json
import logging
import os
import time
from collections import OrderedDict
from threading import Lock
from typing import Any, Dict, Optional

logger = logging.getLogger("nadirclaw.cache")


def _cache_enabled() -> bool:
    return os.getenv("NADIRCLAW_CACHE_ENABLED", "true").lower() in ("true", "1", "yes")


def _cache_ttl() -> int:
    return int(os.getenv("NADIRCLAW_CACHE_TTL", "300"))


def _cache_max_size() -> int:
    return int(os.getenv("NADIRCLAW_CACHE_MAX_SIZE", "1000"))


def _make_cache_key(model: str, messages: list) -> str:
    """Build a deterministic cache key from model + messages (ignoring temperature/stream)."""
    # Normalize messages to just role + content
    normalized = []
    for m in messages:
        if hasattr(m, "role"):
            normalized.append({"role": m.role, "content": m.text_content() if hasattr(m, "text_content") else str(m.content)})
        elif isinstance(m, dict):
            normalized.append({"role": m.get("role", ""), "content": m.get("content", "")})
        else:
            normalized.append(str(m))

    blob = json.dumps({"model": model or "", "messages": normalized}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


class PromptCache:
    """Thread-safe in-memory LRU cache with TTL for chat completions."""

    def __init__(self, max_size: int | None = None, ttl: int | None = None):
        self.max_size = max_size if max_size is not None else _cache_max_size()
        self.ttl = ttl if ttl is not None else _cache_ttl()
        self._cache: OrderedDict[str, tuple[float, Dict[str, Any]]] = OrderedDict()
        self._lock = Lock()
        self._hits = 0
        self._misses = 0

    def get(self, model: str, messages: list) -> Optional[Dict[str, Any]]:
        """Look up a cached response. Returns None on miss or expiry."""
        key = _make_cache_key(model, messages)
        with self._lock:
            if key in self._cache:
                ts, data = self._cache[key]
                if time.time() - ts < self.ttl:
                    # Move to end (most recently used)
                    self._cache.move_to_end(key)
                    self._hits += 1
                    logger.debug("Cache HIT: %s", key[:12])
                    return data
                else:
                    # Expired
                    del self._cache[key]
            self._misses += 1
            return None

    def put(self, model: str, messages: list, response: Dict[str, Any]) -> None:
        """Store a response in the cache."""
        key = _make_cache_key(model, messages)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = (time.time(), response)
            # Evict oldest if over max size
            while len(self._cache) > self.max_size:
                self._cache.popitem(last=False)

    def get_stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "enabled": _cache_enabled(),
                "entries": len(self._cache),
                "max_size": self.max_size,
                "ttl": self.ttl,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 4) if total > 0 else 0.0,
                "total_lookups": total,
            }

    def clear(self) -> None:
        """Clear all cached entries and reset stats."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0


# ---------------------------------------------------------------------------
# Global prompt cache (lazy singleton)
# ---------------------------------------------------------------------------

_prompt_cache: Optional[PromptCache] = None
_cache_init_lock = Lock()


def get_prompt_cache() -> PromptCache:
    """Get the global prompt cache singleton."""
    global _prompt_cache
    if _prompt_cache is None:
        with _cache_init_lock:
            if _prompt_cache is None:
                _prompt_cache = PromptCache()
    return _prompt_cache
