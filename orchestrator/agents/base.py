"""
AgentBase — Abstract base class for all specialized agents
================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Capability 1 of the Agentic System Implementation Plan.
Defines the core Agent interface that all specialized agents implement.

Each agent has:
- A role description (system prompt)
- A set of tools it can use
- Access to the shared workspace
- A message inbox for inter-agent communication
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from ..models import Model, TaskStatus, TaskType

if TYPE_CHECKING:
    from ..api_clients import UnifiedClient
    from .tools.base import Tool
    from ..workspace.workspace import ProjectWorkspace

logger = logging.getLogger("orchestrator.agents.base")


class AgentRole(str, Enum):
    """Specialized agent roles."""

    ARCHITECT = "architect"
    DEVELOPER = "developer"
    REVIEWER = "reviewer"
    TESTER = "tester"
    DEVOPS = "devops"
    RESEARCHER = "researcher"
    USER = "user"
    PRODUCT_MANAGER = "product_manager"
    QA = "qa"
    INVESTIGATOR = "investigator"


@dataclass
class AgentTask:
    """A task assigned to an agent."""

    id: str
    goal: str
    context: str = ""
    target_role: AgentRole | None = None
    dependencies: list[str] = field(default_factory=list)
    max_iterations: int = 3
    artifacts: list[str] = field(default_factory=list)  # Paths to created files
    status: TaskStatus = TaskStatus.PENDING
    score: float = 0.0


@dataclass
class AgentTaskResult:
    """Result of an agent executing a task."""

    task_id: str
    success: bool
    output: str = ""
    score: float = 0.0
    artifacts: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)


class AgentBase(ABC):
    """Base class for all specialized agents.

    Args:
        role: The agent's role identifier.
        tools: List of tools the agent can use.
        workspace: Shared project workspace (blackboard).
        client: LLM client for generation.
        model_preferences: Optional per-task-type model preferences.
    """

    def __init__(
        self,
        role: AgentRole,
        tools: list[Tool] = None,
        workspace: "ProjectWorkspace | None" = None,
        client: "UnifiedClient | None" = None,
        model_preferences: dict[TaskType, Model] = None,
        event_bus: Any = None,
    ) -> None:
        self.role = role
        self.tools: dict[str, Tool] = {t.name: t for t in (tools or [])}
        self.workspace = workspace
        self.client = client
        self._event_bus = event_bus
        # Lazy-load from centralised registry when no explicit preferences given.
        # The try/except avoids a circular import between agent_model_registry
        # (which imports AgentRole from here) and agents/base.py.
        if model_preferences is not None:
            self.model_preferences = model_preferences
        else:
            try:
                from ..agent_model_registry import get_default_model_preferences

                self.model_preferences = get_default_model_preferences(role)
            except Exception:
                self.model_preferences = {}
        self.inbox: list[AgentTask] = []
        self.memory = None
        try:
            from ..learning.agent_memory import AgentMemory

            self.memory = AgentMemory(agent_id=role.value)
        except ImportError:
            pass

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """The agent's core identity and instructions."""

    @abstractmethod
    async def handle_task(self, task: AgentTask) -> AgentTaskResult:
        """Execute an assigned task.

        Args:
            task: The task to execute.

        Returns:
            AgentTaskResult with output, score, and artifacts.
        """

    def get_tool(self, name: str) -> "Tool | None":
        """Get a tool by name."""
        return self.tools.get(name)

    async def send_message(self, recipient: AgentRole, content: str) -> bool:
        """Send a message to another agent via the unified event bus.

        Requires ``event_bus`` to have been injected at construction time.
        Raises RuntimeError if no event bus is available.
        """
        if self._event_bus is None:
            raise RuntimeError(
                f"Agent {self.role.value} has no event bus — cannot send messages. "
                "Inject event_bus at construction time."
            )
        try:
            from ..unified_events.core import AgentMessageEvent

            await self._event_bus.publish(
                AgentMessageEvent(
                    aggregate_id=self.role.value,
                    sender=self.role.value,
                    content=content,
                    msg_type="query",
                    recipient=recipient.value,
                )
            )
            return True
        except Exception:
            logger.debug("Event bus publish failed for agent message", exc_info=True)
            return False
