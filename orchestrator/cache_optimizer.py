"""
Multi-Level Cache Optimizer — Backward-compatibility shim
===========================================================
Canonical location: orchestrator/infrastructure/cache_optimizer.py
Import from there directly for new code; this shim exists for existing callers.
"""

from .infrastructure.cache_optimizer import (  # noqa: F401
    CacheConfig,
    CacheOptimizer,
    L1MemoryCache,
    L2DiskCache,
    MemoryCacheEntry,
    SmartCacheKeyGenerator,
    WarmPattern,
)

_optimizer_singleton: "CacheOptimizer | None" = None


def get_cache_optimizer() -> "CacheOptimizer":
    """Return the process-wide CacheOptimizer singleton."""
    global _optimizer_singleton
    if _optimizer_singleton is None:
        _optimizer_singleton = CacheOptimizer()
    return _optimizer_singleton


__all__ = [
    "CacheConfig",
    "CacheOptimizer",
    "L1MemoryCache",
    "L2DiskCache",
    "MemoryCacheEntry",
    "SmartCacheKeyGenerator",
    "WarmPattern",
    "get_cache_optimizer",
]
