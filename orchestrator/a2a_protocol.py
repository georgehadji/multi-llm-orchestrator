"""
A2A Protocol — Agent-to-Agent Communication
=============================================

Implements A2A (Agent-to-Agent) protocol:
- Agent registration and discovery
- Task delegation between agents
- Message passing between agents
- Task status tracking

Usage:
    from orchestrator.a2a_protocol import A2AManager, AgentCard, TaskSendRequest
    
    manager = A2AManager()
    
    # Register an agent
    agent_card = AgentCard(
        agent_id="code_writer",
        name="Code Writer Agent",
        description="Writes code based on specifications",
        capabilities=["code_generation", "refactoring"],
    )
    await manager.register_agent(agent_card)
    
    # Send task to another agent
    task_request = TaskSendRequest(
        task_id="task_001",
        target_agent="code_writer",
        message="Write a fibonacci function",
        context={"language": "python"},
    )
    result = await manager.send_task(task_request)
    
    # Get task status
    status = await manager.get_task_status("task_001")
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Callable

from .log_config import get_logger

logger = get_logger(__name__)


class AgentState(Enum):
    """Agent state."""
    IDLE = "idle"
    BUSY = "busy"
    UNAVAILABLE = "unavailable"


class TaskStatus(Enum):
    """Task status."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input_required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


@dataclass
class AgentCapability:
    """Agent capability."""
    name: str
    description: str
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None


@dataclass
class AgentCard:
    """Agent card for discovery (A2A specification)."""
    agent_id: str
    name: str
    description: str
    capabilities: List[str] = field(default_factory=list)
    agent_state: AgentState = AgentState.IDLE
    version: str = "1.0"
    provider: Optional[Dict[str, str]] = None
    skills: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agentId": self.agent_id,
            "name": self.name,
            "description": self.description,
            "capabilities": self.capabilities,
            "agentState": self.agent_state.value,
            "version": self.version,
            "provider": self.provider,
            "skills": self.skills,
            "metadata": self.metadata,
        }


@dataclass
class MessagePart:
    """A part of a message."""
    type: str  # text, image, file
    content: str
    mime_type: Optional[str] = None


@dataclass
class A2AMessage:
    """A2A message between agents."""
    id: str
    sender: str
    receiver: str
    message_type: str
    parts: List[MessagePart] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskSendRequest:
    """Request to send a task to an agent."""
    task_id: str
    target_agent: str
    message: str
    context: Dict[str, Any] = field(default_factory=dict)
    priority: int = 5  # 1-10
    timeout_seconds: float = 60.0


@dataclass
class TaskResult:
    """Result of a task."""
    task_id: str
    status: TaskStatus
    result: Optional[Any] = None
    error: Optional[str] = None
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class A2AManager:
    """
    Manages A2A (Agent-to-Agent) communication.

    Implements:
    - Agent registration and discovery
    - Task delegation
    - Message passing
    - Task status tracking
    
    FIX BUG-DEADLOCK-003: Added response tracking and cleanup to prevent deadlocks.
    """

    def __init__(self):
        # Registered agents
        self._agents: Dict[str, AgentCard] = {}

        # Agent task handlers (functions that process tasks)
        self._handlers: Dict[str, Callable] = {}

        # Task tracking
        self._tasks: Dict[str, TaskResult] = {}
        self._task_events: Dict[str, asyncio.Queue] = {}

        # Message queues per agent
        self._message_queues: Dict[str, asyncio.Queue] = {}
        
        # FIX BUG-DEADLOCK-003: Track pending responses for cleanup on timeout
        self._pending_responses: Dict[str, asyncio.Task] = {}
        self._response_timeouts: Dict[str, float] = {}
        
        # FIX: Limit queue size to prevent unbounded growth
        self._max_queue_size: int = 1000

    async def register_agent(
        self,
        agent_card: AgentCard,
        handler: Optional[Callable] = None,
    ) -> None:
        """
        Register an agent in the A2A network.
        
        Args:
            agent_card: Agent card with metadata
            handler: Optional async function to handle tasks
        """
        self._agents[agent_card.agent_id] = agent_card
        
        if handler:
            self._handlers[agent_card.agent_id] = handler
        
        # Create message queue for this agent
        self._message_queues[agent_card.agent_id] = asyncio.Queue()
        
        logger.info(f"Registered agent: {agent_card.agent_id}")

    async def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent."""
        self._agents.pop(agent_id, None)
        self._handlers.pop(agent_id, None)
        self._message_queues.pop(agent_id, None)
        logger.info(f"Unregistered agent: {agent_id}")

    def get_agent(self, agent_id: str) -> Optional[AgentCard]:
        """Get agent by ID."""
        return self._agents.get(agent_id)

    def list_agents(
        self,
        capability: Optional[str] = None,
        state: Optional[AgentState] = None,
    ) -> List[AgentCard]:
        """List registered agents, optionally filtered."""
        agents = list(self._agents.values())
        
        if capability:
            agents = [a for a in agents if capability in a.capabilities]
        
        if state:
            agents = [a for a in agents if a.agent_state == state]
        
        return agents

    async def find_agent(
        self,
        capability: str,
        prefer_idle: bool = True,
    ) -> Optional[AgentCard]:
        """
        Find an agent with the required capability.
        
        Args:
            capability: Required capability
            prefer_idle: Prefer idle agents
            
        Returns:
            Best matching agent or None
        """
        candidates = self.list_agents(capability=capability)
        
        if not candidates:
            return None
        
        if prefer_idle:
            # Sort by state (idle first)
            candidates.sort(key=lambda a: (
                a.agent_state != AgentState.IDLE,
                a.agent_state != AgentState.BUSY,
            ))
        
        return candidates[0]

    async def send_task(
        self,
        request: TaskSendRequest,
    ) -> TaskResult:
        """
        Send a task to another agent.
        
        FIX BUG-DEADLOCK-003: Added response tracking and cleanup on timeout
        to prevent queue deadlocks from orphaned responses.

        Args:
            request: Task send request

        Returns:
            Task result
        """
        # Check if target agent exists
        target_agent = self._agents.get(request.target_agent)
        if not target_agent:
            return TaskResult(
                task_id=request.task_id,
                status=TaskStatus.FAILED,
                error=f"Agent {request.target_agent} not found",
            )

        # Check if agent is available
        if target_agent.agent_state == AgentState.UNAVAILABLE:
            return TaskResult(
                task_id=request.task_id,
                status=TaskStatus.FAILED,
                error=f"Agent {request.target_agent} is unavailable",
            )

        # Update task status
        task_result = TaskResult(
            task_id=request.task_id,
            status=TaskStatus.SUBMITTED,
        )
        self._tasks[request.task_id] = task_result

        # Create task event queue
        self._task_events[request.task_id] = asyncio.Queue()

        # Update agent state
        target_agent.agent_state = AgentState.BUSY

        # Send message to agent
        message = A2AMessage(
            id=str(uuid.uuid4()),
            sender="orchestrator",
            receiver=request.target_agent,
            message_type="task",
            parts=[MessagePart(type="text", content=request.message)],
            metadata=request.context,
        )

        queue = self._message_queues.get(request.target_agent)
        if queue:
            # FIX BUG-DEADLOCK-003: Check queue size to prevent unbounded growth
            if queue.qsize() >= self._max_queue_size:
                logger.warning(f"Queue for {request.target_agent} is full (max={self._max_queue_size})")
                target_agent.agent_state = AgentState.IDLE
                return TaskResult(
                    task_id=request.task_id,
                    status=TaskStatus.FAILED,
                    error=f"Agent {request.target_agent} queue is full",
                )
            
            await queue.put(message)

        logger.info(f"Sent task {request.task_id} to agent {request.target_agent}")

        # Check for handler
        handler = self._handlers.get(request.target_agent)
        if handler:
            try:
                # Execute handler
                task_result.status = TaskStatus.WORKING

                # FIX BUG-DEADLOCK-003: Wrap handler with cleanup tracking
                async def run_handler_with_cleanup():
                    """Run handler and cleanup tracking on completion."""
                    try:
                        return await handler(request.message, request.context)
                    finally:
                        # Clean up tracking on completion (success or failure)
                        self._pending_responses.pop(request.task_id, None)
                        self._response_timeouts.pop(request.task_id, None)
                
                # Create tracked task
                response_task = asyncio.create_task(run_handler_with_cleanup())
                self._pending_responses[request.task_id] = response_task
                self._response_timeouts[request.task_id] = (
                    asyncio.get_event_loop().time() + request.timeout_seconds
                )

                # Run with timeout
                result = await asyncio.wait_for(
                    response_task,
                    timeout=request.timeout_seconds,
                )

                task_result.status = TaskStatus.COMPLETED
                task_result.result = result

            except asyncio.TimeoutError:
                # FIX BUG-DEADLOCK-003: Cleanup on timeout
                logger.warning(
                    f"Task {request.task_id} timed out after {request.timeout_seconds}s"
                )
                
                # Remove from tracking
                self._pending_responses.pop(request.task_id, None)
                self._response_timeouts.pop(request.task_id, None)
                
                # Cancel handler task if still running
                if request.task_id in self._pending_responses:
                    response_task = self._pending_responses.pop(request.task_id)
                    if not response_task.done():
                        response_task.cancel()
                        try:
                            await response_task
                        except asyncio.CancelledError:
                            pass  # Expected
                
                task_result.status = TaskStatus.FAILED
                task_result.error = f"Task timed out after {request.timeout_seconds}s"

            except Exception as e:
                # FIX: Also cleanup on exception
                self._pending_responses.pop(request.task_id, None)
                self._response_timeouts.pop(request.task_id, None)
                
                task_result.status = TaskStatus.FAILED
                task_result.error = str(e)

        # Reset agent state
        target_agent.agent_state = AgentState.IDLE

        return task_result

    async def send_message(
        self,
        sender: str,
        receiver: str,
        message: str,
        message_type: str = "message",
    ) -> bool:
        """
        Send a message from one agent to another.
        
        Args:
            sender: Sender agent ID
            receiver: Receiver agent ID
            message: Message content
            message_type: Type of message
            
        Returns:
            True if sent successfully
        """
        # Check if receiver exists
        if receiver not in self._agents:
            logger.warning(f"Receiver {receiver} not found")
            return False
        
        # Create message
        a2a_message = A2AMessage(
            id=str(uuid.uuid4()),
            sender=sender,
            receiver=receiver,
            message_type=message_type,
            parts=[MessagePart(type="text", content=message)],
        )
        
        # Add to receiver's queue
        queue = self._message_queues.get(receiver)
        if queue:
            await queue.put(a2a_message)
            logger.debug(f"Message sent from {sender} to {receiver}")
            return True
        
        return False

    async def receive_message(self, agent_id: str, timeout: float = 5.0) -> Optional[A2AMessage]:
        """
        Receive a message for an agent.
        
        Args:
            agent_id: Agent to receive message for
            timeout: Timeout in seconds
            
        Returns:
            Message or None
        """
        queue = self._message_queues.get(agent_id)
        if not queue:
            return None
        
        try:
            message = await asyncio.wait_for(queue.get(), timeout=timeout)
            return message
        except asyncio.TimeoutError:
            return None

    async def get_task_status(self, task_id: str) -> Optional[TaskResult]:
        """Get status of a task."""
        return self._tasks.get(task_id)

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task."""
        task = self._tasks.get(task_id)
        if not task:
            return False
        
        if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELED):
            return False
        
        task.status = TaskStatus.CANCELED
        logger.info(f"Canceled task {task_id}")
        return True

    def get_pending_tasks(self, agent_id: Optional[str] = None) -> List[TaskResult]:
        """Get pending tasks."""
        tasks = [
            t for t in self._tasks.values()
            if t.status in (TaskStatus.PENDING, TaskStatus.SUBMITTED, TaskStatus.WORKING)
        ]
        
        if agent_id:
            # Filter by agent - this would need more context
            pass
        
        return tasks

    def get_agent_stats(self) -> Dict[str, Any]:
        """Get statistics about agents and tasks."""
        return {
            "total_agents": len(self._agents),
            "idle_agents": sum(1 for a in self._agents.values() if a.agent_state == AgentState.IDLE),
            "busy_agents": sum(1 for a in self._agents.values() if a.agent_state == AgentState.BUSY),
            "total_tasks": len(self._tasks),
            "pending_tasks": sum(1 for t in self._tasks.values() if t.status == TaskStatus.PENDING),
            "completed_tasks": sum(1 for t in self._tasks.values() if t.status == TaskStatus.COMPLETED),
            "failed_tasks": sum(1 for t in self._tasks.values() if t.status == TaskStatus.FAILED),
        }

    async def cleanup_orphaned_responses(self) -> int:
        """
        FIX BUG-DEADLOCK-003: Clean up orphaned responses from timed-out requests.
        
        This method should be called periodically to clean up any responses that
        were orphaned due to timeout handling issues.
        
        Returns:
            Number of responses cleaned up
        """
        import time
        
        current_time = asyncio.get_event_loop().time()
        cleaned = 0
        
        # Find expired timeouts
        expired_task_ids = [
            task_id for task_id, timeout in self._response_timeouts.items()
            if current_time > timeout
        ]
        
        for task_id in expired_task_ids:
            # Cancel pending task
            if task_id in self._pending_responses:
                task = self._pending_responses[task_id]
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass  # Expected
            
            # Remove from tracking
            self._pending_responses.pop(task_id, None)
            self._response_timeouts.pop(task_id, None)
            cleaned += 1
        
        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} orphaned responses")
        
        return cleaned


# Global manager instance
_default_manager: Optional[A2AManager] = None


def get_a2a_manager() -> A2AManager:
    """Get the default A2A manager instance."""
    global _default_manager
    if _default_manager is None:
        _default_manager = A2AManager()
    return _default_manager