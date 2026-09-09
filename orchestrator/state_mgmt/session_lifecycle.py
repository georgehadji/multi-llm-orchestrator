"""Re-export shim.

SessionLifecycleManager's canonical implementation lives in
orchestrator.session_lifecycle (root, the module engine.py actually wires
up). This used to be an independent full copy that could silently diverge
from the live class; it is now just a re-export so there is a single source
of truth.
"""

from __future__ import annotations

from ..session_lifecycle import SessionLifecycleManager

__all__ = ["SessionLifecycleManager"]
