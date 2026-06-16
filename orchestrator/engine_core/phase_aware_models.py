"""
Phase-Aware Model Selection — Backward-compatibility shim.
===========================================================
The canonical implementation is in orchestrator/phase_aware_models.py.
Import from there directly for new code; this module exists only for callers
that reference orchestrator.engine_core.phase_aware_models.
"""

from orchestrator.phase_aware_models import *  # noqa: F401, F403
