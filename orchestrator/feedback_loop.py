"""
feedback_loop — Backward-compatibility shim
The canonical implementation lives in orchestrator/operations/feedback_loop.py.
New code should import from `orchestrator.operations.feedback_loop` directly.
"""

from .operations.feedback_loop import *  # noqa: F401, F403
