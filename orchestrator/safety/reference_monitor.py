"""
reference_monitor — Backward-compatibility shim
The canonical implementation lives in orchestrator/reference_monitor.py
(the one actually wired into ControlPlane).
New code should import from `orchestrator.reference_monitor` directly.

This copy was dead (zero live callers, confirmed by exhaustive repo-wide
grep — hunt T9) and carried a broken relative import
(`from ..specs import ...`, one dot too many) inside a TYPE_CHECKING-only
block, so it had no runtime impact — but would raise ImportError the moment
anything outside a type checker actually reached it.
"""

from ..reference_monitor import *  # noqa: F401, F403
