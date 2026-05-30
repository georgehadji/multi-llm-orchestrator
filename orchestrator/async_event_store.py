"""
async_event_store — Backward-compatibility shim
The canonical implementation lives in orchestrator/events/async_event_store.py.
New code should import from `orchestrator.events.async_event_store` directly.
"""

from .events.async_event_store import *  # noqa: F401, F403
