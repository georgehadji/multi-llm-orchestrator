# Complete Integrations Guide

**Version:** 1.0.0 | **Updated:** 2026-03-25 | **Author:** Georgios-Chrysovalantis Chatzivantsidis

> **Complete guide to all external integrations** — LiteLLM, Mnemo Cortex, RTK, MCP, BM25, and more.

---

## Overview

The AI Orchestrator integrates with multiple external systems to extend its capabilities. This guide covers all integrations, their setup, and usage.

### Integration Summary

| Integration | Status | Description | Guide |
|-------------|--------|-------------|-------|
| **Nexus Search** | ✅ Built-in | Self-hosted web search | [NEXUS_SEARCH_README.md](./NEXUS_SEARCH_README.md) |
| **A2A Protocol** | ✅ Built-in | External agent communication | [A2A_PROTOCOL_GUIDE.md](./A2A_PROTOCOL_GUIDE.md) |
| **Preflight/Session** | ✅ Built-in | Mnemo Cortex features | [PREFLIGHT_SESSION_GUIDE.md](./PREFLIGHT_SESSION_GUIDE.md) |
| **Token Optimizer** | ✅ Built-in | RTK-style compression | [TOKEN_OPTIMIZER_GUIDE.md](./TOKEN_OPTIMIZER_GUIDE.md) |
| **MCP Server** | ✅ Built-in | Model Context Protocol | This guide |
| **BM25 Search** | ✅ Built-in | Full-text search | This guide |
| **LLM Reranker** | ✅ Built-in | Search result reranking | This guide |
| **LiteLLM** | 🔌 External | 100+ LLM providers | This guide |
| **LangGraph** | 🔌 External | Agent workflows | This guide |

---

## Table of Contents

1. [MCP Server Integration](#1-mcp-server-integration)
2. [BM25 Search Integration](#2-bm25-search-integration)
3. [LLM Reranker Integration](#3-llm-reranker-integration)
4. [LiteLLM Integration](#4-litellm-integration)
5. [LangGraph Integration](#5-langgraph-integration)
6. [Integration Examples](#6-integration-examples)

---

## 1. MCP Server Integration

Model Context Protocol (MCP) enables Claude Desktop and other MCP clients to interact with the AI Orchestrator.

### Setup

```bash
# Start MCP Server
python -m orchestrator.mcp_server --http --port 8181

# Or with configuration
python -m orchestrator.mcp_server \
  --http \
  --port 8181 \
  --host 0.0.0.0 \
  --auth-token your-secret-token
```

### Claude Desktop Configuration

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "orchestrator": {
      "command": "python",
      "args": [
        "-m",
        "orchestrator.mcp_server",
        "--http",
        "--port",
        "8181"
      ],
      "env": {
        "ORCHESTRATOR_API_KEY": "your-api-key"
      }
    }
  }
}
```

### Available Tools

| Tool | Description | Parameters |
|------|-------------|------------|
| `run_project` | Execute AI project | `description`, `criteria`, `budget` |
| `search_knowledge` | Search knowledge base | `query`, `limit` |
| `get_project_status` | Get project status | `project_id` |
| `analyze_code` | Analyze code quality | `code`, `language` |

### Usage from Claude Desktop

```
Claude: Run a project to build a FastAPI authentication service

MCP: Starting project...
  Description: Build a FastAPI authentication service
  Success Criteria: All endpoints tested, OpenAPI docs complete
  Budget: $5.00
  
Project started: proj_abc123
Status: Running (3/10 tasks complete)
```

### Python API

```python
from orchestrator.mcp_server import MCPServer, MCPConfig

# Create server
server = MCPServer(
    host="localhost",
    port=8181,
    auth_token="your-token",
)

# Register tools
server.register_tool("run_project", run_project_handler)
server.register_tool("search_knowledge", search_knowledge_handler)

# Start server
await server.start()
```

---

## 2. BM25 Search Integration

BM25 (Best Matching 25) provides fast full-text search for codebases and documents.

### Setup

```python
from orchestrator.bm25_search import BM25Search, get_bm25_search

# Initialize
bm25 = BM25Search(
    index_path="./bm25_index",
    language="en",  # or "el" for Greek
)

# Or get singleton
bm25 = get_bm25_search()
```

### Indexing Documents

```python
from orchestrator.bm25_search import BM25Search

bm25 = BM25Search()

# Index single document
await bm25.index_document(
    doc_id="doc_001",
    content="This is the document content...",
    metadata={
        "title": "Authentication Guide",
        "type": "documentation",
        "project_id": "proj_123",
    },
)

# Index multiple documents
documents = [
    {
        "doc_id": "doc_001",
        "content": "Content 1...",
        "metadata": {"title": "Doc 1"},
    },
    {
        "doc_id": "doc_002",
        "content": "Content 2...",
        "metadata": {"title": "Doc 2"},
    },
]

await bm25.index_documents(documents)

# Index codebase
from pathlib import Path
await bm25.index_codebase(
    root_path=Path("/path/to/project"),
    extensions=[".py", ".ts", ".js"],
    exclude_patterns=["__pycache__", "node_modules", ".git"],
)
```

### Searching

```python
from orchestrator.bm25_search import BM25Search

bm25 = BM25Search()

# Basic search
results = await bm25.search(
    query="authentication JWT token",
    limit=10,
)

for result in results:
    print(f"• {result.doc_id} (score: {result.score:.2f})")
    print(f"  {result.snippet[:200]}...")

# Search with filters
results = await bm25.search(
    query="database connection",
    limit=10,
    filters={
        "project_id": "proj_123",
        "type": "code",
    },
)

# Search with highlighting
results = await bm25.search(
    query="async rate limiter",
    limit=10,
    highlight=True,
    highlight_pre="<mark>",
    highlight_post="</mark>",
)
```

### Hybrid Search (BM25 + Vector)

```python
from orchestrator.hybrid_search import HybridSearch

hybrid = HybridSearch(
    bm25_index="./bm25_index",
    vector_index="./vector_index",
)

# Hybrid search with RRF (Reciprocal Rank Fusion)
results = await hybrid.search(
    query="authentication patterns",
    k=60,  # RRF constant
    limit=20,
    use_reranking=True,
)

# Results are fused from both BM25 and vector search
for result in results:
    print(f"• {result.doc_id}")
    print(f"  BM25 rank: {result.bm25_rank}")
    print(f"  Vector rank: {result.vector_rank}")
    print(f"  Fused score: {result.fused_score}")
```

---

## 3. LLM Reranker Integration

LLM-based reranking improves search result relevance by using an LLM to score results.

### Setup

```python
from orchestrator.reranker import LLMReranker, get_reranker

# Initialize
reranker = LLMReranker(
    model="gpt-4o-mini",  # or any supported model
    max_tokens=4000,
)

# Or get singleton
reranker = get_reranker()
```

### Reranking Results

```python
from orchestrator.reranker import LLMReranker

reranker = LLMReranker()

# Search results to rerank
results = [
    {"doc_id": "doc_1", "content": "Authentication with JWT..."},
    {"doc_id": "doc_2", "content": "OAuth 2.0 implementation..."},
    {"doc_id": "doc_3", "content": "Session management basics..."},
]

# Rerank
reranked = await reranker.rerank(
    query="JWT token authentication",
    results=results,
    limit=5,
)

print("Reranked results:")
for i, result in enumerate(reranked, 1):
    print(f"{i}. {result['doc_id']} (relevance: {result['relevance_score']:.2f})")
```

### Batch Reranking

```python
from orchestrator.reranker import LLMReranker

reranker = LLMReranker()

# Rerank multiple queries
queries = [
    "authentication",
    "rate limiting",
    "caching strategies",
]

all_results = {}
for query in queries:
    results = await bm25.search(query, limit=10)
    reranked = await reranker.rerank(query, results, limit=5)
    all_results[query] = reranked

# Access results
for query, results in all_results.items():
    print(f"\n{query}:")
    for r in results:
        print(f"  • {r['doc_id']}")
```

---

## 4. LiteLLM Integration

LiteLLM provides access to 100+ LLM providers through a unified API.

### Setup

```bash
# Install LiteLLM
pip install litellm

# Configure providers
cat > litellm_config.yaml << 'EOF'
model_list:
  - model_name: "gpt-4"
    litellm_params:
      model: "gpt-4"
      api_key: os.environ/OPENAI_API_KEY
  
  - model_name: "claude-3"
    litellm_params:
      model: "anthropic/claude-3-sonnet-20240229"
      api_key: os.environ/ANTHROPIC_API_KEY
  
  - model_name: "gemini-pro"
    litellm_params:
      model: "gemini/gemini-pro"
      api_key: os.environ/GEMINI_API_KEY
EOF
```

### Usage

```python
from orchestrator.litellm_integration import LiteLLMClient

client = LiteLLMClient(
    config_path="./litellm_config.yaml",
    default_model="gpt-4",
)

# Generate text
response = await client.generate(
    prompt="Explain quantum computing",
    model="gpt-4",
    max_tokens=500,
)

print(response.content)

# Route to optimal model
response = await client.route_and_generate(
    prompt="Write authentication code",
    task_type="code_generation",  # Auto-routes to best coding model
)
```

### Provider Routing

```python
from orchestrator.litellm_integration import ProviderRouter

router = ProviderRouter()

# Register providers
router.register_provider(
    name="openai",
    models=["gpt-4", "gpt-4o", "gpt-4o-mini"],
    cost_per_1m_tokens=0.03,
    capabilities=["chat", "code", "vision"],
)

router.register_provider(
    name="anthropic",
    models=["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"],
    cost_per_1m_tokens=0.15,
    capabilities=["chat", "code", "long-context"],
)

# Get optimal provider for task
provider = router.get_optimal_provider(
    task_type="code_generation",
    budget=1.0,
    requirements=["low-latency"],
)

print(f"Recommended: {provider.name}")
```

---

## 5. LangGraph Integration

LangGraph enables complex agent workflows with the AI Orchestrator.

### Setup

```bash
# Install LangGraph
pip install langgraph langchain
```

### Basic Workflow

```python
from orchestrator.langgraph_integration import LangGraphClient
from langgraph.graph import StateGraph, END

# Create LangGraph client
client = LangGraphClient()

# Define workflow
workflow = StateGraph(dict)

# Add nodes
workflow.add_node("research", client.run_agent("research_agent"))
workflow.add_node("write", client.run_agent("writing_agent"))
workflow.add_node("review", client.run_agent("review_agent"))

# Add edges
workflow.add_edge("research", "write")
workflow.add_edge("write", "review")
workflow.add_edge("review", END)

# Compile
app = workflow.compile()

# Run workflow
result = await app.ainvoke({
    "topic": "AI architecture patterns",
})

print(result["output"])
```

### Multi-Agent Orchestration

```python
from orchestrator.langgraph_integration import MultiAgentOrchestrator

orchestrator = MultiAgentOrchestrator()

# Register agents
orchestrator.register_agent(
    name="researcher",
    agent_type="langgraph",
    config={"graph_id": "research_graph"},
)

orchestrator.register_agent(
    name="writer",
    agent_type="langgraph",
    config={"graph_id": "writing_graph"},
)

orchestrator.register_agent(
    name="reviewer",
    agent_type="langgraph",
    config={"graph_id": "review_graph"},
)

# Run multi-agent workflow
result = await orchestrator.run_workflow(
    task="Write comprehensive guide on AI architecture",
    agents=["researcher", "writer", "reviewer"],
    mode="sequential",  # or "parallel", "iterative"
)
```

---

## 6. Integration Examples

### Example 1: Full Search Pipeline

```python
from orchestrator.bm25_search import BM25Search
from orchestrator.reranker import LLMReranker
from orchestrator.nexus_search import search as web_search

async def full_search_pipeline(query: str):
    # 1. Web search for context
    web_results = await web_search(query, num_results=5)
    
    # 2. Local BM25 search
    bm25 = BM25Search()
    local_results = await bm25.search(query, limit=10)
    
    # 3. Rerank local results
    reranker = LLMReranker()
    reranked = await reranker.rerank(query, local_results, limit=5)
    
    # 4. Combine results
    combined = {
        "web": web_results,
        "local": reranked,
    }
    
    return combined

# Usage
results = await full_search_pipeline("Python async best practices")
```

### Example 2: MCP + LangGraph

```python
from orchestrator.mcp_server import MCPServer
from orchestrator.langgraph_integration import LangGraphClient

# Create MCP server
mcp = MCPServer(port=8181)

# Create LangGraph client
langgraph = LangGraphClient()

# Register MCP tool that uses LangGraph
@mcp.tool("run_workflow")
async def run_workflow_handler(query: str):
    result = await langgraph.run_agent("research_agent", {"query": query})
    return result["output"]

# Start server
await mcp.start()
```

### Example 3: Hybrid Search with Reranking

```python
from orchestrator.hybrid_search import HybridSearch
from orchestrator.reranker import LLMReranker

async def enhanced_search(query: str):
    # Hybrid search
    hybrid = HybridSearch(
        bm25_index="./index",
        vector_index="./vectors",
    )
    
    results = await hybrid.search(
        query=query,
        k=60,
        limit=20,
    )
    
    # Rerank top results
    reranker = LLMReranker(model="gpt-4o-mini")
    reranked = await reranker.rerank(
        query=query,
        results=results,
        limit=10,
    )
    
    return reranked

# Usage
results = await enhanced_search("authentication patterns")
```

---

## Configuration

### Environment Variables

```bash
# MCP Server
export MCP_SERVER_PORT=8181
export MCP_SERVER_AUTH_TOKEN=your-token

# BM25 Search
export BM25_INDEX_PATH=./bm25_index
export BM25_LANGUAGE=en

# LLM Reranker
export RERANKER_MODEL=gpt-4o-mini
export RERANKER_MAX_TOKENS=4000

# LiteLLM
export LITELLM_CONFIG_PATH=./litellm_config.yaml
export LITELLM_DEFAULT_MODEL=gpt-4

# LangGraph
export LANGGRAPH_DEFAULT_GRAPH=research_graph
```

---

## Troubleshooting

### MCP Server Not Connecting

```bash
# Check server status
curl http://localhost:8181/health

# Check logs
python -m orchestrator.mcp_server --http --port 8181 2>&1 | tee mcp.log
```

### BM25 Search Returns No Results

```python
# Check index
bm25 = BM25Search()
stats = await bm25.get_index_stats()
print(f"Indexed documents: {stats.document_count}")

# Reindex if needed
await bm25.reindex_all()
```

### Reranker Timeout

```python
# Use faster model
reranker = LLMReranker(model="gpt-4o-mini")

# Reduce batch size
reranked = await reranker.rerank(
    query=query,
    results=results,
    batch_size=5,  # Process 5 at a time
)
```

---

## Related Documentation

- [NEXUS_SEARCH_README.md](./NEXUS_SEARCH_README.md) — Web search integration
- [A2A_PROTOCOL_GUIDE.md](./A2A_PROTOCOL_GUIDE.md) — External agent protocol
- [USAGE_GUIDE.md](./USAGE_GUIDE.md) — Main usage guide

---

**License:** MIT | **Author:** Georgios-Chrysovalantis Chatzivantsidis
