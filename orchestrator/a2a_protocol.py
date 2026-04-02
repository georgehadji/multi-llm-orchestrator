"""
A2AProtocol — Agent-to-Agent external agent client
================================================
Module for invoking external agents (LangGraph, Vertex AI, Azure AI Foundry)
using the A2A protocol for inter-agent communication.

Pattern: Adapter
Async: Yes — for I/O-bound network operations
Layer: L3 Agents

Usage:
    from orchestrator.a2a_protocol import A2AClient
    client = A2AClient(agent_endpoint="https://external-agent.example.com")
    result = await client.invoke_agent(task="summarize", data={"text": "..."})
"""

from __future__ import annotations

import asyncio
import json
import logging
import time as _time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import aiohttp

logger = logging.getLogger("orchestrator.a2a_protocol")


class AgentType(Enum):
    """Types of external agents supported."""

    LANGGRAPH = "langgraph"
    VERTEX_AI = "vertex_ai"
    AZURE_AI = "azure_ai"
    CUSTOM_HTTP = "custom_http"
    OPENAI_ASSISTANTS = "openai_assistants"


@dataclass
class A2AMessage:
    """Message in the A2A protocol."""

    role: str  # "user", "assistant", "system", "tool"
    content: str
    name: str | None = None  # Optional name for the message
    tool_calls: list[dict[str, Any]] | None = None  # Tool calls if role is "assistant"
    tool_call_id: str | None = None  # ID of tool call result if role is "tool"


@dataclass
class A2ATask:
    """Task definition for A2A protocol."""

    id: str
    type: str  # "summarization", "classification", "qa", "generation", etc.
    data: dict[str, Any]
    metadata: dict[str, Any] | None = None
    timeout: float = 30.0


@dataclass
class A2AResponse:
    """Response from an A2A agent."""

    success: bool
    content: str | None = None
    tool_results: list[dict[str, Any]] | None = None
    error: str | None = None
    metadata: dict[str, Any] | None = None
    execution_time: float | None = None


class A2AClient:
    """Client for communicating with external agents using the A2A protocol."""

    def __init__(
        self,
        agent_endpoint: str,
        agent_type: AgentType = AgentType.CUSTOM_HTTP,
        api_key: str | None = None,
        timeout: float = 30.0,
    ):
        """
        Initialize the A2A client.

        Args:
            agent_endpoint: Endpoint URL of the external agent
            agent_type: Type of external agent
            api_key: API key for authentication (if required)
            timeout: Request timeout in seconds
        """
        self.agent_endpoint = agent_endpoint
        self.agent_type = agent_type
        self.api_key = api_key
        self.timeout = timeout
        self.session: Any = None  # aiohttp.ClientSession, lazy-loaded

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def connect(self):
        """Establish connection to the external agent."""
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        import aiohttp as _aiohttp  # Lazy import to avoid hang at module load time

        timeout_config = _aiohttp.ClientTimeout(total=self.timeout)
        self.session = _aiohttp.ClientSession(timeout=timeout_config, headers=headers)
        logger.info(f"A2A client connected to {self.agent_endpoint}")

    async def close(self):
        """Close the connection to the external agent."""
        if self.session:
            await self.session.close()
        logger.info("A2A client disconnected")

    async def invoke_agent(
        self, task: A2ATask | str, data: dict[str, Any] | None = None
    ) -> A2AResponse:
        """
        Invoke an external agent with a task.

        Args:
            task: Either an A2ATask object or a task type string
            data: Task data (required if task is a string)

        Returns:
            A2AResponse: Response from the external agent
        """
        if not self.session:
            raise RuntimeError("A2A client not connected. Call connect() first.")

        # Create task object if only string provided
        if isinstance(task, str):
            if not data:
                raise ValueError("Data is required when task is a string")
            import uuid

            task = A2ATask(id=str(uuid.uuid4()), type=task, data=data)

        start_time = asyncio.get_event_loop().time()

        try:
            if self.agent_type == AgentType.LANGGRAPH:
                response = await self._invoke_langgraph_agent(task)
            elif self.agent_type == AgentType.VERTEX_AI:
                response = await self._invoke_vertex_ai_agent(task)
            elif self.agent_type == AgentType.AZURE_AI:
                response = await self._invoke_azure_ai_agent(task)
            elif self.agent_type == AgentType.OPENAI_ASSISTANTS:
                response = await self._invoke_openai_assistant(task)
            else:  # CUSTOM_HTTP
                response = await self._invoke_custom_http_agent(task)

            execution_time = asyncio.get_event_loop().time() - start_time
            response.execution_time = execution_time

            return response
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            logger.error(f"A2A agent invocation failed: {e}")
            return A2AResponse(success=False, error=str(e), execution_time=execution_time)

    async def _invoke_langgraph_agent(self, task: A2ATask) -> A2AResponse:
        """Invoke a LangGraph agent."""
        # LangGraph typically uses a state graph that can be invoked via HTTP
        url = f"{self.agent_endpoint}/invoke"

        payload = {"input": task.data, "config": {"recursion_limit": 50}}

        try:
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return A2AResponse(
                        success=True,
                        content=json.dumps(result.get("output", result)),
                        metadata=result.get("metadata", {}),
                    )
                else:
                    error_text = await response.text()
                    return A2AResponse(
                        success=False,
                        error=f"LangGraph agent returned status {response.status}: {error_text}",
                    )
        except Exception as e:
            return A2AResponse(success=False, error=str(e))

    async def _invoke_vertex_ai_agent(self, task: A2ATask) -> A2AResponse:
        """Invoke a Vertex AI agent."""
        # Vertex AI typically uses a predict endpoint
        url = f"{self.agent_endpoint}:predict"

        # Format the task for Vertex AI
        instance = {"prompt": self._format_task_for_vertex_ai(task)}

        payload = {
            "instances": [instance],
            "parameters": {"temperature": 0.7, "maxOutputTokens": 1024},
        }

        try:
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    predictions = result.get("predictions", [])
                    if predictions:
                        content = predictions[0].get("content", predictions[0])
                        return A2AResponse(success=True, content=str(content))
                    else:
                        return A2AResponse(success=False, error="No predictions returned")
                else:
                    error_text = await response.text()
                    return A2AResponse(
                        success=False,
                        error=f"Vertex AI agent returned status {response.status}: {error_text}",
                    )
        except Exception as e:
            return A2AResponse(success=False, error=str(e))

    async def _invoke_azure_ai_agent(self, task: A2ATask) -> A2AResponse:
        """Invoke an Azure AI agent."""
        # Azure AI Foundry typically uses a chat completions endpoint
        url = f"{self.agent_endpoint}/chat/completions"

        # Format the task as a conversation
        messages = [
            {"role": "system", "content": "You are an AI assistant that helps with various tasks."},
            {"role": "user", "content": self._format_task_for_azure(task)},
        ]

        payload = {"messages": messages, "temperature": 0.7, "max_tokens": 1024}

        try:
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    choices = result.get("choices", [])
                    if choices:
                        content = choices[0].get("message", {}).get("content", "")
                        return A2AResponse(success=True, content=content)
                    else:
                        return A2AResponse(success=False, error="No choices returned")
                else:
                    error_text = await response.text()
                    return A2AResponse(
                        success=False,
                        error=f"Azure AI agent returned status {response.status}: {error_text}",
                    )
        except Exception as e:
            return A2AResponse(success=False, error=str(e))

    async def _invoke_openai_assistant(self, task: A2ATask) -> A2AResponse:
        """Invoke an OpenAI Assistant."""
        # Create a thread for the task
        thread_url = f"{self.agent_endpoint}/threads"
        thread_payload = {
            "messages": [{"role": "user", "content": self._format_task_for_openai_assistant(task)}]
        }

        try:
            async with self.session.post(thread_url, json=thread_payload) as thread_response:
                if thread_response.status != 200:
                    error_text = await thread_response.text()
                    return A2AResponse(
                        success=False,
                        error=f"Failed to create thread: {thread_response.status} - {error_text}",
                    )

                thread_data = await thread_response.json()
                thread_id = thread_data["id"]

                # Run the assistant
                run_url = f"{self.agent_endpoint}/threads/{thread_id}/runs"
                run_payload = {
                    "assistant_id": task.data.get(
                        "assistant_id"
                    )  # Expect assistant_id in task data
                }

                async with self.session.post(run_url, json=run_payload) as run_response:
                    if run_response.status != 200:
                        error_text = await run_response.text()
                        return A2AResponse(
                            success=False,
                            error=f"Failed to start run: {run_response.status} - {error_text}",
                        )

                    run_data = await run_response.json()
                    run_id = run_data["id"]

                    # Poll for completion
                    status = "queued"
                    while status in ["queued", "in_progress", "cancelling"]:
                        await asyncio.sleep(1)  # Wait 1 second before polling again

                        status_url = f"{self.agent_endpoint}/threads/{thread_id}/runs/{run_id}"
                        async with self.session.get(status_url) as status_response:
                            if status_response.status != 200:
                                error_text = await status_response.text()
                                return A2AResponse(
                                    success=False,
                                    error=f"Failed to get run status: {status_response.status} - {error_text}",
                                )

                            status_data = await status_response.json()
                            status = status_data["status"]

                    if status == "completed":
                        # Get the messages from the thread
                        messages_url = f"{self.agent_endpoint}/threads/{thread_id}/messages"
                        async with self.session.get(messages_url) as messages_response:
                            if messages_response.status != 200:
                                error_text = await messages_response.text()
                                return A2AResponse(
                                    success=False,
                                    error=f"Failed to get messages: {messages_response.status} - {error_text}",
                                )

                            messages_data = await messages_response.json()
                            # Get the latest assistant message
                            assistant_messages = [
                                msg for msg in messages_data["data"] if msg["role"] == "assistant"
                            ]

                            if assistant_messages:
                                content = (
                                    assistant_messages[0]["content"][0]
                                    .get("text", {})
                                    .get("value", "")
                                )
                                return A2AResponse(success=True, content=content)
                            else:
                                return A2AResponse(
                                    success=False, error="No assistant messages found"
                                )
                    else:
                        return A2AResponse(success=False, error=f"Run failed with status: {status}")
        except Exception as e:
            return A2AResponse(success=False, error=str(e))

    async def _invoke_custom_http_agent(self, task: A2ATask) -> A2AResponse:
        """Invoke a custom HTTP agent."""
        # Send the task to the custom endpoint
        url = self.agent_endpoint

        payload = {
            "task_id": task.id,
            "task_type": task.type,
            "data": task.data,
            "metadata": task.metadata or {},
        }

        try:
            async with self.session.post(url, json=payload) as response:
                if response.status in [200, 201]:
                    result = await response.json()
                    return A2AResponse(
                        success=True,
                        content=result.get("content", json.dumps(result)),
                        tool_results=result.get("tool_results"),
                        metadata=result.get("metadata"),
                    )
                else:
                    error_text = await response.text()
                    return A2AResponse(
                        success=False,
                        error=f"Custom agent returned status {response.status}: {error_text}",
                    )
        except Exception as e:
            return A2AResponse(success=False, error=str(e))

    def _format_task_for_vertex_ai(self, task: A2ATask) -> str:
        """Format a task for Vertex AI."""
        if task.type == "summarization":
            text = task.data.get("text", "")
            return f"Please summarize the following text: {text}"
        elif task.type == "classification":
            text = task.data.get("text", "")
            labels = task.data.get("labels", [])
            return f"Classify the following text into one of these categories: {', '.join(labels)}. Text: {text}"
        elif task.type == "qa":
            context = task.data.get("context", "")
            question = task.data.get("question", "")
            return f"Based on the following context, answer the question: {question}\n\nContext: {context}"
        else:
            # Generic formatting
            return f"Perform the following task: {task.type}. Data: {json.dumps(task.data)}"

    def _format_task_for_azure(self, task: A2ATask) -> str:
        """Format a task for Azure AI."""
        if task.type == "summarization":
            text = task.data.get("text", "")
            return f"Summarize this text in 2-3 sentences: {text}"
        elif task.type == "classification":
            text = task.data.get("text", "")
            labels = task.data.get("labels", [])
            return f"Categorize this text as one of: {', '.join(labels)}\nText: {text}"
        elif task.type == "qa":
            context = task.data.get("context", "")
            question = task.data.get("question", "")
            return f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        else:
            # Generic formatting
            return f"Task: {task.type}\nInstructions: {task.data}"

    def _format_task_for_openai_assistant(self, task: A2ATask) -> str:
        """Format a task for OpenAI Assistant."""
        if task.type == "summarization":
            text = task.data.get("text", "")
            return f"Please summarize the following text: {text}"
        elif task.type == "classification":
            text = task.data.get("text", "")
            labels = task.data.get("labels", [])
            return f"Classify this text into one of these categories: {', '.join(labels)}\n\nText: {text}"
        elif task.type == "qa":
            context = task.data.get("context", "")
            question = task.data.get("question", "")
            return f"Using this context: {context}\n\nPlease answer: {question}"
        else:
            # Generic formatting
            return (
                f"Please help with this task: {task.type}. Here's the data: {json.dumps(task.data)}"
            )

    async def batch_invoke_agents(self, tasks: list[A2ATask]) -> list[A2AResponse]:
        """
        Invoke multiple agents in parallel.

        Args:
            tasks: List of tasks to invoke

        Returns:
            List of responses from the agents
        """
        if not self.session:
            raise RuntimeError("A2A client not connected. Call connect() first.")

        # Create tasks for concurrent execution
        coroutines = [self.invoke_agent(task) for task in tasks]
        responses = await asyncio.gather(*coroutines, return_exceptions=True)

        # Handle any exceptions that occurred during execution
        processed_responses = []
        for response in responses:
            if isinstance(response, Exception):
                processed_responses.append(A2AResponse(success=False, error=str(response)))
            else:
                processed_responses.append(response)

        return processed_responses

    async def health_check(self) -> bool:
        """
        Perform a health check on the external agent.

        Returns:
            bool: True if the agent is reachable and responding, False otherwise
        """
        if not self.session:
            return False

        try:
            # Try to make a simple request to the endpoint
            url = (
                f"{self.agent_endpoint}/health"
                if not self.agent_endpoint.endswith("/health")
                else self.agent_endpoint
            )

            async with self.session.get(url) as response:
                return response.status == 200
        except Exception:
            return False

    def get_agent_info(self) -> dict[str, Any]:
        """
        Get information about the configured agent.

        Returns:
            Dict with agent information
        """
        return {
            "endpoint": self.agent_endpoint,
            "type": self.agent_type.value,
            "connected": self.session is not None,
        }


class A2ACoordinator:
    """Coordinates multiple A2A clients for distributed agent execution."""

    def __init__(self):
        """Initialize the A2A coordinator."""
        self.clients: list[A2AClient] = []
        self.agent_registry: dict[str, A2AClient] = {}  # agent_id -> client

    def register_agent(self, agent_id: str, client: A2AClient):
        """
        Register an agent with the coordinator.

        Args:
            agent_id: Unique identifier for the agent
            client: A2A client for the agent
        """
        self.clients.append(client)
        self.agent_registry[agent_id] = client
        logger.info(f"Registered agent {agent_id} with coordinator")

    async def distribute_task(
        self, task: A2ATask, agent_filter: list[str] | None = None
    ) -> dict[str, A2AResponse]:
        """
        Distribute a task to multiple agents and collect responses.

        Args:
            task: Task to distribute
            agent_filter: Optional list of agent IDs to limit distribution

        Returns:
            Dict mapping agent IDs to their responses
        """
        # Determine which agents to use
        target_agents = self.agent_registry
        if agent_filter:
            target_agents = {
                aid: client for aid, client in self.agent_registry.items() if aid in agent_filter
            }

        # Create tasks for each agent
        agent_tasks = []
        for agent_id, client in target_agents.items():
            agent_task = asyncio.create_task(client.invoke_agent(task), name=f"agent_{agent_id}")
            agent_tasks.append((agent_id, agent_task))

        # Execute all tasks concurrently
        results = {}
        for agent_id, task_coro in agent_tasks:
            try:
                response = await task_coro
                results[agent_id] = response
            except Exception as e:
                results[agent_id] = A2AResponse(success=False, error=str(e))

        return results

    async def get_coordinator_stats(self) -> dict[str, Any]:
        """
        Get statistics about the coordinator.

        Returns:
            Dict with coordinator statistics
        """
        connected_agents = 0
        for client in self.clients:
            try:
                if await client.health_check():
                    connected_agents += 1
            except:
                continue

        return {
            "total_agents": len(self.clients),
            "connected_agents": connected_agents,
            "agent_ids": list(self.agent_registry.keys()),
        }


# Global coordinator for convenience
_global_a2a_coordinator = A2ACoordinator()


def get_global_a2a_coordinator() -> A2ACoordinator:
    """
    Get the global A2A coordinator instance.

    Returns:
        A2ACoordinator instance
    """
    return _global_a2a_coordinator


# ---------------------------------------------------------------------------
# Queue-based A2A types — used by engine.py send_agent_task() and
# tests/test_reliability_regression.py.
# ---------------------------------------------------------------------------


class TaskStatus(str, Enum):
    """Status of a dispatched inter-agent task."""

    SUBMITTED = "submitted"
    PROCESSING = "processing"
    COMPLETED = "completed"
    TIMEOUT = "timeout"
    FAILED = "failed"


class AgentState(str, Enum):
    """Lifecycle state of a registered agent."""

    IDLE = "idle"
    BUSY = "busy"
    OFFLINE = "offline"


@dataclass
class AgentCard:
    """Metadata card for a registered agent."""

    agent_id: str
    name: str = ""
    description: str = ""
    capabilities: list[str] = field(default_factory=list)
    endpoint: str = ""
    version: str = "1.0"


@dataclass
class MessagePart:
    """Single content part inside a routing A2AMessage."""

    content: str
    type: str = "text"
    metadata: dict[str, Any] = field(default_factory=dict)


# Redefine A2AMessage as a routing/envelope message.
# The original LLM-conversation dataclass is kept as _LLMConversationMessage
# for any internal use that still references the old fields.
_LLMConversationMessage = A2AMessage  # preserve old class under a private name


@dataclass
class A2AMessage:  # noqa: F811  (intentional re-definition for routing format)
    """Routing message passed through agent message queues."""

    id: str
    sender: str
    receiver: str
    message_type: str
    parts: list[MessagePart] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskSendRequest:
    """Request to dispatch a task to a target agent."""

    task_id: str
    target_agent: str = ""
    message: str = ""
    timeout_seconds: float = 30.0
    context: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    """Result returned by A2AQueueManager.send_task().

    Optional fields (output, score, model_used) mirror orchestrator.models.TaskResult
    so that tests sharing this type can construct instances with those fields.
    """

    task_id: str
    status: TaskStatus = TaskStatus.COMPLETED
    result: Any = None
    content: str = ""
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    # Optional fields for compatibility with orchestrator result storage
    output: str = ""
    score: float = 0.0
    model_used: Any = None


class A2AQueueManager:
    """
    Queue-based inter-agent message-passing manager.

    Supports registering agents with optional async handlers, dispatching
    tasks with timeouts, and cleaning up orphaned in-flight responses.
    """

    def __init__(self, max_queue_size: int = 100) -> None:
        self._agents: dict[str, AgentCard] = {}
        self._handlers: dict[str, Any] = {}
        # defaultdict so direct queue access before register_agent() works in tests
        self._message_queues: dict[str, asyncio.Queue] = defaultdict(asyncio.Queue)
        self._pending_responses: dict[str, Any] = {}
        self._response_timeouts: dict[str, float] = {}
        self._max_queue_size = max_queue_size

    async def register_agent(self, card: AgentCard, handler=None) -> None:
        """Register an agent with an optional async message handler."""
        self._agents[card.agent_id] = card
        # Touch queue to ensure it exists (defaultdict creates it on access)
        _ = self._message_queues[card.agent_id]
        if handler is not None:
            self._handlers[card.agent_id] = handler
        logger.debug("Registered agent %r (handler=%s)", card.agent_id, handler is not None)

    async def send_task(self, request: TaskSendRequest) -> TaskResult:
        """Dispatch a task to the target agent and await its result."""
        agent_id = request.target_agent

        # Queue-capacity check first (queue may exist via direct test access)
        queue = self._message_queues.get(agent_id)
        if queue is not None and queue.qsize() >= self._max_queue_size:
            return TaskResult(
                task_id=request.task_id,
                status=TaskStatus.FAILED,
                error=f"queue is full for agent {agent_id!r}",
            )

        # Agent-registration check
        if agent_id not in self._agents:
            return TaskResult(
                task_id=request.task_id,
                status=TaskStatus.FAILED,
                error=f"agent {agent_id!r} not found",
            )

        handler = self._handlers.get(agent_id)
        if handler is None:
            return TaskResult(task_id=request.task_id, status=TaskStatus.SUBMITTED)

        # Reserve tracking state before yielding the event loop
        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()
        self._pending_responses[request.task_id] = future
        self._response_timeouts[request.task_id] = _time.time() + request.timeout_seconds
        t0 = _time.monotonic()

        async def _run_handler() -> None:
            try:
                res = await handler(request.message, {})
                if not future.done():
                    future.set_result(res)
            except Exception as exc:
                if not future.done():
                    future.set_exception(exc)

        handler_task = asyncio.create_task(_run_handler())

        try:
            result_value = await asyncio.wait_for(
                asyncio.shield(future),
                timeout=request.timeout_seconds,
            )
            return TaskResult(
                task_id=request.task_id,
                status=TaskStatus.COMPLETED,
                result=result_value,
                execution_time=_time.monotonic() - t0,
            )
        except asyncio.TimeoutError:
            handler_task.cancel()
            try:
                await handler_task
            except (asyncio.CancelledError, Exception):
                pass
            # Return FAILED (not TIMEOUT) — callers treat timeout as a failure path
            return TaskResult(
                task_id=request.task_id,
                status=TaskStatus.FAILED,
                error="task timed out",
                execution_time=_time.monotonic() - t0,
            )
        except Exception as exc:
            return TaskResult(
                task_id=request.task_id,
                status=TaskStatus.FAILED,
                error=str(exc),
                execution_time=_time.monotonic() - t0,
            )
        finally:
            self._pending_responses.pop(request.task_id, None)
            self._response_timeouts.pop(request.task_id, None)

    async def cleanup_orphaned_responses(self) -> int:
        """
        Cancel and remove responses whose deadline has passed.
        Returns the number of entries cleaned up.
        """
        now = _time.time()
        orphaned = [
            tid for tid, deadline in list(self._response_timeouts.items()) if now >= deadline
        ]
        for tid in orphaned:
            item = self._pending_responses.pop(tid, None)
            if item is not None and hasattr(item, "cancel"):
                done = item.done() if hasattr(item, "done") else False
                if not done:
                    item.cancel()
            self._response_timeouts.pop(tid, None)
        return len(orphaned)


# ---------------------------------------------------------------------------
# Public aliases
# ---------------------------------------------------------------------------

# engine.py + tests import these names
A2AManager = A2AQueueManager
get_a2a_manager = get_global_a2a_coordinator
