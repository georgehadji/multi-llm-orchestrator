"""
HookRegistry — Re-export shim
==============================
Unified with AgentMessageBus into events/core.py.
This file kept as backward-compat shim.
"""

from .core import (  # noqa: F401
    DashboardHookRegistry,
    EventType,
    HookRegistry,
)
