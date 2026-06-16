"""
Autonomous Debugger — Backward-compatibility shim.
====================================================
The canonical implementation is in orchestrator/autonomous_debugger.py.
Import from there directly for new code; this module exists only for callers
that reference orchestrator.safety.autonomous_debugger.
"""

from orchestrator.autonomous_debugger import *  # noqa: F401, F403
