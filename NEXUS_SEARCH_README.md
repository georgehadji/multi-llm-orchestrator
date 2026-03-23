# Nexus Search

**Private, self-hosted web search for AI Orchestrator.**

Nexus Search provides intelligent web search capabilities powered by self-hosted search infrastructure. All searches are private, tracked-free, and integrated directly into the AI Orchestrator.

## Features

- 🔍 **Multi-Source Search** — Web, academic, tech, news, and code
- 🧠 **Query Classification** — Automatic optimal source selection
- 📚 **Deep Research** — Multi-step research with synthesis
- 🔒 **Privacy-First** — No tracking, no profiling
- 💰 **Zero Cost** — Self-hosted, no API fees
- ⚡ **Fast** — Local deployment, minimal latency

## Quick Start

### 1. Start Nexus Search

```bash
# Using Docker Compose
docker-compose -f nexus-search.docker-compose.yml up -d

# Check status
docker ps | grep nexus-search
```

### 2. Configure AI Orchestrator

```bash
# Add to .env
export NEXUS_SEARCH_ENABLED=true
export NEXUS_API_URL=http://localhost:8080
```

### 3. Use in Code

```python
from orchestrator.nexus_search import search, research

# Simple search
results = await search("Python async best practices")
for result in results.top:
    print(f"{result.title}: {result.url}")

# Deep research
report = await research("Microservices architecture patterns 2026")
print(f"Found {report.source_count} sources")
print(f"Summary: {report.summary[:200]}...")
```

## Available Sources

| Source | Description | Examples |
|--------|-------------|----------|
| **Web** | General web search | Google, Bing, DuckDuckGo |
| **Academic** | Scholarly articles | Google Scholar, arXiv, PubMed |
| **Tech** | Technology content | HackerNews, tech blogs |
| **News** | News articles | Google News, Bing News |
| **Code** | Code repositories | GitHub, Stack Overflow |

## Configuration

### Environment Variables

```bash
# Enable/disable Nexus Search
NEXUS_SEARCH_ENABLED=true

# Nexus API URL
NEXUS_API_URL=http://localhost:8080

# Request timeout (seconds)
NEXUS_TIMEOUT=30

# Maximum results per query
NEXUS_MAX_RESULTS=20

# Rate limit (queries per minute)
NEXUS_RATE_LIMIT=60

# Enable caching
NEXUS_CACHE_ENABLED=true

# Cache TTL (seconds)
NEXUS_CACHE_TTL=3600
```

### Docker Configuration

```yaml
# nexus-search.docker-compose.yml
services:
  nexus-search:
    image: searxng/searxng:latest
    ports:
      - "8080:8080"
    volumes:
      - ./nexus_config:/etc/searxng:rw
    environment:
      - SEARXNG_SECRET_KEY=your-secret-key
```

## API Reference

### Simple Search

```python
from orchestrator.nexus_search import search, SearchSource, OptimizationMode

results = await search(
    query="Python async",
    sources=[SearchSource.WEB, SearchSource.TECH],
    optimization=OptimizationMode.BALANCED,
    num_results=10,
)
```

### Deep Research

```python
from orchestrator.nexus_search import research

report = await research(
    query="Microservices patterns",
    depth=3,  # Number of iterations
)

print(f"Findings: {len(report.findings)}")
print(f"Sources: {report.source_count}")
print(f"Summary: {report.summary}")
```

### Query Classification

```python
from orchestrator.nexus_search import classify, QueryType

query_type = await classify("Python async best practices")
# Returns: QueryType.RESEARCH
```

## Integration Points

### Project Enhancer

Automatically searches latest best practices when enhancing project descriptions.

```python
orch = Orchestrator(nexus_search_enabled=True)
state = await orch.run_project(
    project_description="Build FastAPI service",
    enhance_with_web_search=True,
)
```

### Architecture Advisor

Searches latest architecture patterns.

```python
advisor = ArchitectureAdvisor(nexus_search_enabled=True)
decision = await advisor.analyze(
    description="Real-time analytics",
    include_web_research=True,
)
```

### ARA Research Pipeline

Uses Nexus for real web research.

```python
from orchestrator.ara_pipelines import PipelineFactory, ReasoningMethod

pipeline = PipelineFactory.create(
    method=ReasoningMethod.RESEARCH,
    nexus_enabled=True,
)
result = await pipeline.execute(task)
```

## Troubleshooting

### Nexus Search Not Available

```bash
# Check if container is running
docker ps | grep nexus-search

# Check logs
docker logs nexus-search

# Test health endpoint
curl http://localhost:8080/healthz
```

### Slow Searches

```bash
# Increase timeout
export NEXUS_TIMEOUT=60

# Reduce results
export NEXUS_MAX_RESULTS=10
```

### Rate Limiting

```bash
# Increase rate limit
export NEXUS_RATE_LIMIT=120
```

## Architecture

```
┌─────────────────┐
│  AI Orchestrator│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Nexus Search    │
│ Orchestrator    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Nexus Provider  │
│ (SearXNG API)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ SearXNG Engine  │
│ - Google        │
│ - Scholar       │
│ - GitHub        │
│ - StackOverflow │
│ - News          │
└─────────────────┘
```

## Privacy & Security

- **Self-hosted** — All searches run locally
- **No tracking** — Queries are not logged
- **No profiling** — Each query is independent
- **HTTPS support** — Optional TLS for production
- **Rate limiting** — Built-in protection

## Performance

| Metric | Target | Actual |
|--------|--------|--------|
| Search latency | <500ms | ~200ms |
| Research time | <10s | ~5s |
| Concurrent searches | 100/min | 60/min (default) |
| Cache hit rate | >50% | ~70% |

## License

MIT License — Same as AI Orchestrator

## Credits

Powered by [SearXNG](https://github.com/searxng/searxng) (AGPL-3.0).
All SearXNG references are branded as "Nexus Search" for seamless integration.
