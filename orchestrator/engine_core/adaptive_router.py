"""
Adaptive Router — Backward-compatibility shim.
=============================================
The canonical implementation is in orchestrator/adaptive_router.py.
Import from there directly for new code; this module exists only for callers
that reference orchestrator.engine_core.adaptive_router.
"""

from orchestrator.adaptive_router import AdaptiveRouter, ModelState  # noqa: F401
