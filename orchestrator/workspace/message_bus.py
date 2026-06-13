"""
Agent Message Bus — Capability 5
=================================
Publish/subscribe message bus for agent communication.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from collections import defaultdict


class MessageType(str, Enum):
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    QUERY = "query"
    ALERT = "alert"
    PROGRESS = "progress"
    CONFLICT = "conflict"
    APPROVAL_REQUEST = "approval"


@dataclass
class AgentMessage:
    id: str
    sender: str
    content: str
    recipient: str | None = None
    msg_type: MessageType = MessageType.QUERY
    priority: int = 0


class AgentMessageBus:
    """Publish/subscribe message bus for agent communication."""

    def __init__(self):
        self.subscriptions: dict[str, list[MessageType]] = defaultdict(list)
        self.message_history: list[AgentMessage] = []
        self._inboxes: dict[str, list[AgentMessage]] = defaultdict(list)

    def subscribe(self, agent_id: str, message_types: list[MessageType]) -> None:
        """Register an agent's interest in specific message types."""
        self.subscriptions[agent_id].extend(message_types)

    def publish(self, message: AgentMessage) -> None:
        """Deliver a message to the recipient agent's inbox or broadcast to subscribers."""
        self.message_history.append(message)
        if message.recipient:
            # Direct delivery
            self._inboxes[message.recipient].append(message)
        else:
            # Broadcast to all subscribed agents
            for agent_id, sub_types in self.subscriptions.items():
                if message.msg_type in sub_types:
                    self._inboxes[agent_id].append(message)

    def read_inbox(self, agent_id: str) -> list[AgentMessage]:
        """Read the inbox of a specific agent."""
        return self._inboxes[agent_id]

    def get_history(self) -> list[AgentMessage]:
        """Get the complete history of published messages."""
        return self.message_history
