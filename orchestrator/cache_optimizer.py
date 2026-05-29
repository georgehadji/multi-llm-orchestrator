"""
Multi-Level Cache Optimizer — Backward-compatibility shim
===========================================================
Canonical location: orchestrator/infrastructure/cache_optimizer.py
Import from there directly for new code; this shim exists for existing callers.
"""
from .infrastructure.cache_optimizer import (  # noqa: F401
    CacheConfig,
    L1MemoryCache,
    L2DiskCache,
    MemoryCacheEntry,
    SmartCacheKeyGenerator,
    WarmPattern,
)

__all__ = [
    "CacheConfig",
    "L1MemoryCache",
    "L2DiskCache",
    "MemoryCacheEntry",
    "SmartCacheKeyGenerator",
    "WarmPattern",
]
