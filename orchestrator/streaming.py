"""
streaming — Backward-compatibility shim
The canonical implementation lives in orchestrator/infrastructure/streaming.py.
New code should import from `orchestrator.infrastructure.streaming` directly.
"""

from .infrastructure.streaming import *  # noqa: F401, F403
