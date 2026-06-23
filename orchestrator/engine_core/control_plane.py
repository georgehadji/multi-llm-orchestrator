"""
Control Plane Service — Backward-compatibility shim.
=====================================================
The canonical implementation is in orchestrator/control_plane.py.
Import from there directly for new code; this module exists only for callers
that reference orchestrator.engine_core.control_plane.
"""

from orchestrator.control_plane import *  # noqa: F401, F403
