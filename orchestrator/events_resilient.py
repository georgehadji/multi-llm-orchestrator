"""
events_resilient — Backward-compatibility shim
The canonical implementation lives in orchestrator/events/events_resilient.py.
New code should import from `orchestrator.events.events_resilient` directly.
"""

from .events.events_resilient import *  # noqa: F401, F403
