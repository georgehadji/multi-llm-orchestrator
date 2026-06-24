"""
Cache — Backward-compatibility shim
======================================
The canonical DiskCache now lives in orchestrator/infrastructure/cache.py.
"""

from .infrastructure.cache import DiskCache  # noqa: F401
from .infrastructure.cache import DEFAULT_CACHE_PATH  # noqa: F401

__all__ = ["DiskCache", "DEFAULT_CACHE_PATH"]
