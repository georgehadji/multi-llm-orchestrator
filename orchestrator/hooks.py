"""
HookRegistry — Re-export shim
==============================
Unified into unified_events/core.py.
This file kept as backward-compat shim.
"""

from .unified_events.core import (  # noqa: F401
    EventType,
    HookRegistry,
)

# Dashboard registry remains for legacy integration until Phase 2
from .events.core import DashboardHookRegistry  # noqa: F401
