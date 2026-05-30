"""
Legacy Event Bus — Re-export shim
==================================
Unified into unified_events/core.py.
This file kept as backward-compat shim.
"""

from ..unified_events.core import (  # noqa: F401
    AgentMessageEvent as AgentMessage,
    EventType,
    UnifiedEventBus as EventBus,
    UnifiedEventBus as RedisEventBus,
    HookRegistry,
)

# Placeholder for MessageType until fully unified
from enum import Enum


class MessageType(str, Enum):
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    QUERY = "query"
    ALERT = "alert"
    PROGRESS = "progress"
    CONFLICT = "conflict"
    APPROVAL_REQUEST = "approval_request"
