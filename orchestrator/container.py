"""
ServiceContainer — Backward-compatibility shim.
================================================
The canonical implementation is in orchestrator/engine_core/container.py.
Import from there directly for new code; this module exists only for callers
that reference orchestrator.container.
"""

from orchestrator.engine_core.container import (  # noqa: F401
    ServiceContainer,
)
