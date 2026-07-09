"""
diagnostics — Backward-compatibility shim
The canonical implementation lives in orchestrator/operations/diagnostics.py.
New code should import from `orchestrator.operations.diagnostics` directly.
"""

from .operations.diagnostics import *  # noqa: F401, F403
