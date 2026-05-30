"""
secure_cache — Backward-compatibility shim
The canonical implementation lives in orchestrator/infrastructure/secure_cache.py.
New code should import from `orchestrator.infrastructure.secure_cache` directly.
"""

from .infrastructure.secure_cache import *  # noqa: F401, F403
