"""
triggers — Backward-compatibility shim
The canonical implementation lives in orchestrator/events/triggers.py.
New code should import from `orchestrator.events.triggers` directly.
"""

from .events.triggers import *  # noqa: F401, F403
