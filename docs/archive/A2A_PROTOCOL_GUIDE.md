# A2A Protocol Guide — Agent-to-Agent Communication

**Version:** 1.0.0 | **Updated:** 2026-03-25 | **Author:** Georgios-Chrysovalantis Chatzivantsidis

> **Connect external AI agents to the Orchestrator.** LangGraph, Vertex AI, Azure AI Foundry, OpenAI Assistants — unified via A2A protocol.

---

## Quick Start

### 1. Basic Usage

```python
from orchestrator.a2a_protocol import A2AClient, AgentType

# Create client for external agent
client = A2AClient(
    agent_endpoint="https://external-agent.example.com",
    agent_type=AgentType.CUSTOM_HTTP,
    api_key="your-api-key",
    timeout=30.0,
)

# Invoke agent
async with client:
    result = await client.invoke_agent(
        task="summarize",
        data={"text": "Long document to summarize..."}
    )
    
    print(result.content)
```

### 2. Multi-Agent Orchestration

```python
from orchestrator.a2a_protocol import A2AMultiAgent

# Create multi-agent orchestrator
orchestrator = A2AMultiAgent()

# Register agents
orchestrator.register_agent(
    name="summarizer",
    endpoint="https://summarizer.example.com",
    agent_type=AgentType.LANGGRAPH,
)

orchestrator.register_agent(
    name="classifier",
    endpoint="https://classifier.example.com",
    agent_type=AgentType.VERTEX_AI,
)

# Route task to appropriate agent
result = await orchestrator.route_task(
    task_type="summarization",
    data={"document": "Long text..."}
)
```

---

## Table of Contents

1. [Overview](#overview)
2. [Agent Types](#agent-types)
3. [API Reference](#api-reference)
4. [Integration Examples](#integration-examples)
5. [Multi-Agent Orchestration](#multi-agent-orchestration)
6. [Error Handling](#error-handling)
7. [Best Practices](#best-practices)

---

## Overview

The A2A (Agent-to-Agent) Protocol enables seamless communication between the AI Orchestrator and external AI agents. It provides:

- **Unified interface** for multiple agent platforms
- **Automatic protocol translation** (LangGraph, Vertex AI, Azure AI, OpenAI)
- **Async execution** for non-blocking operations
- **Circuit breaker** for fault tolerance
- **Retry logic** with exponential backoff

### Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Orchestrator  │────▶│   A2A Protocol   │────▶│  External Agent │
│                 │     │    Client        │     │  (LangGraph,    │
└─────────────────┘     └──────────────────┘     │   Vertex AI,    │
                                                  │   Azure AI)     │
                                                  └─────────────────┘
```

### Use Cases

| Use Case | Description | Example |
|----------|-------------|---------|
| **Specialized Tasks** | Delegate specific tasks to specialized agents | Summarization, translation, classification |
| **Multi-Model Routing** | Route to optimal model based on task | GPT-4 for reasoning, Claude for writing |
| **Fallback Chains** | Fallback to alternative agents on failure | Primary fails → Secondary agent |
| **Parallel Execution** | Run multiple agents in parallel | Compare outputs from different agents |

---

## Agent Types

### LangGraph

```python
from orchestrator.a2a_protocol import A2AClient, AgentType

client = A2AClient(
    agent_endpoint="https://langgraph-agent.example.com/invoke",
    agent_type=AgentType.LANGGRAPH,
    api_key="your-api-key",
)

result = await client.invoke_agent(
    task="graph_execution",
    data={
        "graph_name": "research_graph",
        "input": {"query": "Research topic..."}
    }
)
```

### Vertex AI

```python
from orchestrator.a2a_protocol import A2AClient, AgentType

client = A2AClient(
    agent_endpoint="https://us-central1-aiplatform.googleapis.com/v1/projects/PROJECT_ID/locations/us-central1/endpoints/ENDPOINT_ID:predict",
    agent_type=AgentType.VERTEX_AI,
    api_key="your-gcp-token",
)

result = await client.invoke_agent(
    task="classification",
    data={
        "instances": [{"content": "Text to classify..."}]
    }
)
```

### Azure AI Foundry

```python
from orchestrator.a2a_protocol import A2AClient, AgentType

client = A2AClient(
    agent_endpoint="https://YOUR_RESOURCE.openai.azure.com/openai/deployments/YOUR_DEPLOYMENT/chat/completions",
    agent_type=AgentType.AZURE_AI,
    api_key="your-azure-key",
)

result = await client.invoke_agent(
    task="chat",
    data={
        "messages": [
            {"role": "user", "content": "Hello!"}
        ]
    }
)
```

### OpenAI Assistants

```python
from orchestrator.a2a_protocol import A2AClient, AgentType

client = A2AClient(
    agent_endpoint="https://api.openai.com/v1/assistants/ASSISTANT_ID/runs",
    agent_type=AgentType.OPENAI_ASSISTANTS,
    api_key="sk-...",
)

result = await client.invoke_agent(
    task="assistant_run",
    data={
        "thread_id": "thread_123",
        "instructions": "Analyze the document..."
    }
)
```

### Custom HTTP

```python
from orchestrator.a2a_protocol import A2AClient, AgentType

client = A2AClient(
    agent_endpoint="https://your-custom-agent.com/api/v1/invoke",
    agent_type=AgentType.CUSTOM_HTTP,
    api_key="your-api-key",
    timeout=60.0,
)

result = await client.invoke_agent(
    task="custom_task",
    data={"custom_field": "value"}
)
```

---

## API Reference

### A2AClient

```python
class A2AClient:
    """Client for communicating with external agents."""
    
    def __init__(
        self,
        agent_endpoint: str,
        agent_type: AgentType = AgentType.CUSTOM_HTTP,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    )
    
    async def connect() -> None:
        """Establish connection to agent."""
    
    async def disconnect() -> None:
        """Close connection."""
    
    async def invoke_agent(
        self,
        task: str,
        data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> A2AResponse:
        """
        Invoke external agent.
        
        Args:
            task: Task type (summarize, classify, generate, etc.)
            data: Task input data
            metadata: Optional metadata
            
        Returns:
            A2AResponse with success, content, error
        """
    
    async def health_check() -> bool:
        """Check if agent is healthy."""
```

### A2AMultiAgent

```python
class A2AMultiAgent:
    """Orchestrator for multiple external agents."""
    
    def register_agent(
        self,
        name: str,
        endpoint: str,
        agent_type: AgentType,
        api_key: Optional[str] = None,
        priority: int = 1,
    ) -> None:
        """Register an external agent."""
    
    def unregister_agent(self, name: str) -> None:
        """Unregister an agent."""
    
    async def route_task(
        self,
        task_type: str,
        data: Dict[str, Any],
        preferred_agent: Optional[str] = None,
    ) -> A2AResponse:
        """Route task to appropriate agent."""
    
    async def broadcast(
        self,
        task: str,
        data: Dict[str, Any],
    ) -> Dict[str, A2AResponse]:
        """Send task to all agents, collect responses."""
    
    async def compare_agents(
        self,
        task: str,
        data: Dict[str, Any],
        metrics: List[str] = ["quality", "latency", "cost"],
    ) -> AgentComparison:
        """Compare agent performance."""
```

### A2AMessage

```python
@dataclass
class A2AMessage:
    """Message in A2A protocol."""
    
    role: str  # "user", "assistant", "system", "tool"
    content: str
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
```

### A2ATask

```python
@dataclass
class A2ATask:
    """Task definition."""
    
    id: str
    type: str  # "summarization", "classification", "qa", "generation"
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
    timeout: float = 30.0
```

### A2AResponse

```python
@dataclass
class A2AResponse:
    """Response from external agent."""
    
    success: bool
    content: Optional[str] = None
    tool_results: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None
    
    @property
    def is_success(self) -> bool:
        return self.success and self.error is None
```

---

## Integration Examples

### Example 1: Summarization Pipeline

```python
from orchestrator.a2a_protocol import A2AClient, AgentType

# Create summarization client
summarizer = A2AClient(
    agent_endpoint="https://summarizer.example.com",
    agent_type=AgentType.LANGGRAPH,
)

async def summarize_document(text: str) -> str:
    async with summarizer:
        result = await summarizer.invoke_agent(
            task="summarize",
            data={
                "text": text,
                "max_length": 500,
                "style": "concise"
            }
        )
        
        if result.success:
            return result.content
        else:
            raise Exception(f"Summarization failed: {result.error}")

# Usage
summary = await summarize_document(long_document)
print(summary)
```

### Example 2: Classification with Fallback

```python
from orchestrator.a2a_protocol import A2AMultiAgent, AgentType

# Create multi-agent with fallback
agents = A2AMultiAgent()

# Primary classifier
agents.register_agent(
    name="primary_classifier",
    endpoint="https://classifier-primary.example.com",
    agent_type=AgentType.VERTEX_AI,
    priority=1,
)

# Fallback classifier
agents.register_agent(
    name="fallback_classifier",
    endpoint="https://classifier-fallback.example.com",
    agent_type=AgentType.AZURE_AI,
    priority=2,
)

async def classify_with_fallback(text: str):
    # Try primary first
    result = await agents.route_task(
        task_type="classification",
        data={"text": text},
        preferred_agent="primary_classifier",
    )
    
    # Fallback if primary fails
    if not result.success:
        print(f"Primary failed: {result.error}, trying fallback...")
        result = await agents.route_task(
            task_type="classification",
            data={"text": text},
            preferred_agent="fallback_classifier",
        )
    
    return result

# Usage
classification = await classify_with_fallback("Text to classify...")
```

### Example 3: Parallel Agent Execution

```python
from orchestrator.a2a_protocol import A2AMultiAgent

agents = A2AMultiAgent()

# Register multiple agents for comparison
agents.register_agent("gpt4", "https://gpt4.example.com", AgentType.OPENAI_ASSISTANTS)
agents.register_agent("claude", "https://claude.example.com", AgentType.CUSTOM_HTTP)
agents.register_agent("gemini", "https://gemini.example.com", AgentType.CUSTOM_HTTP)

async def compare_outputs(prompt: str):
    # Broadcast to all agents
    results = await agents.broadcast(
        task="generate",
        data={"prompt": prompt}
    )
    
    # Compare results
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"Success: {result.success}")
        print(f"Time: {result.execution_time}s")
        print(f"Content: {result.content[:200]}...")
    
    return results

# Usage
results = await compare_outputs("Explain quantum computing")
```

### Example 4: Integration with Orchestrator Tasks

```python
from orchestrator import Orchestrator
from orchestrator.a2a_protocol import A2AClient

# Create orchestrator
orch = Orchestrator()

# Create external agent client
external_agent = A2AClient(
    agent_endpoint="https://specialized-agent.example.com",
    agent_type=AgentType.LANGGRAPH,
)

# Use external agent within orchestrator task
async def run_project_with_external_agent():
    state = await orch.run_project(
        project_description="Build a web scraper",
        success_criteria="All tests pass",
    )
    
    # Use external agent for code review
    async with external_agent:
        review = await external_agent.invoke_agent(
            task="code_review",
            data={
                "code": state.generated_code,
                "criteria": ["security", "performance", "readability"]
            }
        )
        
        print(f"External review: {review.content}")
    
    return state

# Run
state = await run_project_with_external_agent()
```

---

## Multi-Agent Orchestration

### Agent Selection Strategy

```python
from orchestrator.a2a_protocol import A2AMultiAgent, SelectionStrategy

agents = A2AMultiAgent()

# Configure selection strategy
agents.configure_selection(
    strategy=SelectionStrategy.ROUND_ROBIN,  # or LEAST_LOADED, BEST_PERFORMING
    health_check_interval=60,  # seconds
)

# Register agents with different capabilities
agents.register_agent(
    name="fast_agent",
    endpoint="https://fast.example.com",
    agent_type=AgentType.CUSTOM_HTTP,
    capabilities=["summarization", "classification"],
    avg_latency=0.5,
)

agents.register_agent(
    name="quality_agent",
    endpoint="https://quality.example.com",
    agent_type=AgentType.LANGGRAPH,
    capabilities=["summarization", "generation"],
    avg_latency=2.0,
)
```

### Agent Comparison

```python
from orchestrator.a2a_protocol import A2AMultiAgent

agents = A2AMultiAgent()
# ... register agents ...

# Compare agent performance
comparison = await agents.compare_agents(
    task="summarize",
    data={"text": "Sample document..."},
    metrics=["quality", "latency", "cost"],
)

print(f"Best quality: {comparison.best_by_quality}")
print(f"Fastest: {comparison.fastest}")
print(f"Cheapest: {comparison.cheapest}")
```

---

## Error Handling

### Retry Logic

```python
from orchestrator.a2a_protocol import A2AClient, RetryConfig

client = A2AClient(
    agent_endpoint="https://agent.example.com",
    retry_config=RetryConfig(
        max_retries=3,
        initial_delay=1.0,
        max_delay=30.0,
        exponential_base=2,
        retryable_errors=["timeout", "rate_limit", "server_error"],
    ),
)

try:
    result = await client.invoke_agent(task="generate", data={...})
except Exception as e:
    print(f"All retries failed: {e}")
```

### Circuit Breaker

```python
from orchestrator.a2a_protocol import A2AClient, CircuitBreakerConfig

client = A2AClient(
    agent_endpoint="https://agent.example.com",
    circuit_breaker=CircuitBreakerConfig(
        failure_threshold=5,  # Open circuit after 5 failures
        recovery_timeout=60,  # Wait 60s before trying again
        half_open_requests=3,  # Allow 3 test requests in half-open state
    ),
)

# Circuit breaker automatically opens on repeated failures
# and closes after recovery timeout
```

### Error Types

```python
from orchestrator.a2a_protocol import (
    A2AError,
    A2ATimeoutError,
    A2AConnectionError,
    A2AAuthenticationError,
    A2ARateLimitError,
)

try:
    result = await client.invoke_agent(task="generate", data={...})
except A2ATimeoutError:
    print("Agent timed out")
except A2AConnectionError:
    print("Connection failed")
except A2AAuthenticationError:
    print("Invalid API key")
except A2ARateLimitError:
    print("Rate limited, retry later")
except A2AError as e:
    print(f"General error: {e}")
```

---

## Best Practices

### 1. Connection Pooling

```python
# Reuse client instances
client = A2AClient(...)
await client.connect()

# Use for multiple requests
for task in tasks:
    result = await client.invoke_agent(task=task, data=...)

# Close when done
await client.disconnect()
```

### 2. Async Context Manager

```python
# Preferred pattern
async with A2AClient(...) as client:
    result = await client.invoke_agent(task="generate", data={...})
# Automatically disconnects
```

### 3. Timeout Configuration

```python
# Set appropriate timeouts
client = A2AClient(
    agent_endpoint="https://agent.example.com",
    timeout=30.0,  # General timeout
    connect_timeout=5.0,  # Connection timeout
    read_timeout=25.0,  # Read timeout
)
```

### 4. Health Monitoring

```python
# Periodic health checks
async def monitor_agent_health(client):
    while True:
        healthy = await client.health_check()
        if not healthy:
            print("Agent unhealthy!")
            # Trigger alert or failover
        await asyncio.sleep(60)  # Check every minute
```

### 5. Logging

```python
import logging

logging.getLogger("orchestrator.a2a_protocol").setLevel(logging.DEBUG)

# Logs all requests/responses for debugging
```

---

## Related Documentation

- [USAGE_GUIDE.md](./USAGE_GUIDE.md) — Main usage guide
- [CAPABILITIES.md](./CAPABILITIES.md) — Full capabilities
- [INTEGRATIONS_COMPLETE.md](./INTEGRATIONS_COMPLETE.md) — All integrations

---

**License:** MIT | **Author:** Georgios-Chrysovalantis Chatzivantsidis
