"""
Cache — Backward-compatibility shim
======================================
The canonical DiskCache now lives in orchestrator/infrastructure/cache.py.
"""

from .infrastructure.cache import *  # noqa: F401, F403
