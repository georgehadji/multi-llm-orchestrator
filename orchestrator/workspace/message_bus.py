"""
AgentMessage — Structured inter-agent communication protocol
==============================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Capability 5 of the Agentic System Implementation Plan.
Provides message types, publish/subscribe routing, and delivery
guarantees for inter-agent communication.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger("orchestrator.workspace.message_bus")


class MessageType(str, Enum):
    """Types of messages agents can exchange."""
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    QUERY = "query"
    ALERT = "alert"
    PROGRESS = "progress"
    CONFLICT = "conflict"
    APPROVAL_REQUEST = "approval"


@dataclass
class AgentMessage:
    """A structured message between agents."""
    id: str
    sender: str
    content: str
    msg_type: MessageType = MessageType.QUERY
    recipient: str | None = None
    reply_to: str | None = None
    priority: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


class AgentMessageBus:
    """Publish/subscribe message bus for agent communication.

    Agents subscribe to message types they're interested in.
    Messages are delivered to inboxes for asynchronous processing.
    """

    def __init__(self) -> None:
        self.subscriptions: dict[str, list[MessageType]] = {}
        self.message_history: list[AgentMessage] = []
        self._inboxes: dict[str, list[AgentMessage]] = {}

    def subscribe(self, agent_id: str, message_types: list[MessageType]) -> None:
        """Register an agent's interest in specific message types."""
        self.subscriptions[agent_id] = message_types
        if agent_id not in self._inboxes:
            self._inboxes[agent_id] = []

    def publish(self, message: AgentMessage) -> None:
        """Publish a message to the bus.

        If recipient is specified, delivers directly.
        Otherwise, broadcasts to all subscribers of that message type.
        """
        self.message_history.append(message)
        if message.recipient:
            self._deliver(message.recipient, message)
        else:
            for agent_id, types in self.subscriptions.items():
                if message.msg_type in types and agent_id != message.sender:
                    self._deliver(agent_id, message)

    def _deliver(self, agent_id: str, message: AgentMessage) -> None:
        """Deliver a message to an agent's inbox."""
        if agent_id in self._inboxes:
            self._inboxes[agent_id].append(message)

    def read_inbox(self, agent_id: str) -> list[AgentMessage]:
        """Read and clear an agent's inbox."""
        messages = self._inboxes.get(agent_id, [])
        self._inboxes[agent_id] = []
        return messages

    def get_history(self, limit: int = 50) -> list[AgentMessage]:
        """Get recent message history."""
        return self.message_history[-limit:]
