# Nexus Search — Web Search Integration

**Version:** 1.0.0 | **Updated:** 2026-03-25 | **Author:** Georgios-Chrysovalantis Chatzivantsidis

> **Private, self-hosted web search for AI Orchestrator.** Zero tracking, zero profiling, zero third-party API costs.

---

## Quick Start

### 1. Start Nexus Search

```bash
# Using Docker Compose
docker-compose -f nexus-search.docker-compose.yml up -d

# Verify it's running
docker ps | grep nexus-search
```

### 2. Configure Environment

```bash
# .env or export
export NEXUS_SEARCH_ENABLED=true
export NEXUS_API_URL=http://localhost:8080
export NEXUS_TIMEOUT=30
export NEXUS_MAX_RESULTS=20
```

### 3. Use in Code

```python
from orchestrator.nexus_search import search, research

# Simple search
results = await search("Python async best practices")
for r in results:
    print(f"• {r.title} - {r.url}")

# Deep research
report = await research("Microservices architecture patterns 2026")
print(report.summary)
print(f"Sources: {len(report.findings)}")
```

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [API Reference](#api-reference)
5. [CLI Usage](#cli-usage)
6. [Integration with Orchestrator](#integration-with-orchestrator)
7. [Troubleshooting](#troubleshooting)

---

## Overview

Nexus Search provides intelligent web search capabilities for the AI Orchestrator. It replaces expensive third-party APIs (Perplexity, Tavily) with a self-hosted solution that:

- **Searches multiple sources**: Web, academic papers, tech docs, news, code repositories
- **Classifies queries automatically**: Determines optimal search strategy
- **Performs deep research**: Multi-step agent for comprehensive analysis
- **Respects privacy**: Zero tracking, zero profiling, zero data retention

### Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Orchestrator  │────▶│  Nexus Search    │────▶│  Search Providers│
│                 │     │  Orchestrator    │     │  (Self-hosted)   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │
                               ▼
                        ┌──────────────────┐
                        │  Query Classifier│
                        │  Research Agent  │
                        └──────────────────┘
```

### Key Components

| Component | Description |
|-----------|-------------|
| **NexusSearchOrchestrator** | Main entry point for search operations |
| **QueryClassifier** | Auto-detects query type and optimal search strategy |
| **ResearchAgent** | Multi-step deep research agent |
| **NexusProvider** | Abstracts search provider implementation |
| **SearchSource** | Enum for search sources (web, academic, tech, news, code) |

---

## Installation

### Prerequisites

- Docker & Docker Compose
- Python 3.10+
- At least 2GB RAM for Nexus Search

### Step 1: Start Nexus Search

```bash
# Clone or navigate to orchestrator directory
cd "E:\Documents\Vibe-Coding\Ai Orchestrator"

# Start Nexus Search
docker-compose -f nexus-search.docker-compose.yml up -d

# Check status
docker ps | grep nexus-search

# View logs
docker logs nexus-search -f
```

### Step 2: Install Python Dependencies

```bash
# Nexus Search is included with orchestrator
pip install -e .

# Or install just the search module
pip install aiohttp httpx
```

### Step 3: Verify Installation

```bash
# Test connectivity
python -c "from orchestrator.nexus_search import get_nexus_orchestrator; print('OK')"

# Run health check
python -m orchestrator nexus status
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NEXUS_SEARCH_ENABLED` | `false` | Enable/disable Nexus Search |
| `NEXUS_API_URL` | `http://localhost:8080` | Nexus Search API URL |
| `NEXUS_TIMEOUT` | `30` | Request timeout in seconds |
| `NEXUS_MAX_RESULTS` | `20` | Maximum search results |
| `NEXUS_RATE_LIMIT` | `60` | Requests per minute |
| `NEXUS_CACHE_ENABLED` | `true` | Enable response caching |
| `NEXUS_CACHE_TTL` | `3600` | Cache TTL in seconds |

### Python Configuration

```python
from orchestrator.nexus_search import configure, NexusConfig

# Configure globally
configure(
    api_url="http://localhost:8080",
    timeout=30,
    max_results=20,
    cache_enabled=True,
    cache_ttl=3600,
)

# Or create custom config
config = NexusConfig(
    api_url="http://nexus.internal:8080",
    timeout=60,
    max_results=50,
    optimization_mode="quality",  # or "speed", "balanced"
)
```

---

## API Reference

### Search

```python
from orchestrator.nexus_search import search, SearchSource, OptimizationMode

# Basic search
results = await search("Python async best practices")

# With options
results = await search(
    query="Python async best practices",
    sources=[SearchSource.WEB, SearchSource.TECH],  # Filter sources
    optimization=OptimizationMode.BALANCED,  # or SPEED, QUALITY
    num_results=10,
)

# Iterate results
for result in results:
    print(f"Title: {result.title}")
    print(f"URL: {result.url}")
    print(f"Snippet: {result.snippet}")
    print(f"Source: {result.source.value}")
    print(f"Score: {result.relevance_score}")
    print("---")
```

### Research

```python
from orchestrator.nexus_search import research

# Deep research report
report = await research("Microservices architecture patterns 2026")

# Access findings
print(f"Summary: {report.summary}")
print(f"Total findings: {len(report.findings)}")

for finding in report.findings:
    print(f"\n### {finding.title}")
    print(finding.content)
    print(f"Sources: {len(finding.sources)}")
```

### Classification

```python
from orchestrator.nexus_search import classify, QueryType

# Classify a query
classification = await classify("How to build FastAPI service")
print(f"Type: {classification.type.value}")  # QueryType.TECHNICAL
print(f"Recommended sources: {classification.recommended_sources}")

# Use classification to optimize search
classification = await classify(query)
results = await search(
    query=query,
    sources=classification.recommended_sources,
)
```

### Advanced: Direct Orchestrator Access

```python
from orchestrator.nexus_search import NexusSearchOrchestrator, SearchSource

# Create orchestrator
nexus = NexusSearchOrchestrator(
    auto_classify=True,  # Auto-classify queries
)

# Initialize
await nexus.initialize()

# Search
results = await nexus.search(
    query="Python async",
    sources=[SearchSource.WEB, SearchSource.CODE],
    optimization="balanced",
    num_results=15,
)

# Research
report = await nexus.research(
    query="Event-driven architecture patterns",
    depth=5,  # Number of research iterations
)

# Classify
classification = await nexus.classify("Best React hooks")
```

---

## CLI Usage

### Search

```bash
# Basic search
python -m orchestrator nexus search "Python async best practices"

# With sources
python -m orchestrator nexus search "Microservices patterns" --sources tech,academic

# JSON output
python -m orchestrator nexus search "CVE 2026" --json
```

### Research

```bash
# Deep research
python -m orchestrator nexus research "AI architecture patterns 2026"

# With depth
python -m orchestrator nexus research "Serverless best practices" --depth 5

# Save report
python -m orchestrator nexus research "Kubernetes security" --output report.md
```

### Status & Classification

```bash
# Check Nexus status
python -m orchestrator nexus status

# JSON status
python -m orchestrator nexus status --json

# Classify query
python -m orchestrator nexus classify "How to build FastAPI service"

# JSON classification
python -m orchestrator nexus classify "Python async" --json
```

---

## Integration with Orchestrator

### Project Enhancer with Web Context

```python
from orchestrator.enhancer import ProjectEnhancer

# Enable Nexus integration
enhancer = ProjectEnhancer(nexus_enabled=True)

# Enhance project with web context
enhanced = await enhancer.enhance(
    project_description="Build a FastAPI authentication service",
    use_web_context=True,  # Uses Nexus Search
)

print(enhanced.web_context)  # Real web research results
```

### Architecture Advisor with Web Context

```python
from orchestrator.architecture_advisor import ArchitectureAdvisor

# Enable Nexus integration
advisor = ArchitectureAdvisor(nexus_enabled=True)

# Get architecture decision with web research
decision = await advisor.advise(
    project_description="Real-time collaborative whiteboard",
    use_web_context=True,  # Uses Nexus Search
)

print(decision.web_research_summary)
```

### ARA Pipeline with Research

```python
from orchestrator.ara_pipelines import PipelineFactory, ReasoningMethod
from orchestrator.nexus_search import research

# Create pipeline with research
pipeline = PipelineFactory.create(
    method=ReasoningMethod.RESEARCH,
    nexus_enabled=True,  # Uses real web search
)

result = await pipeline.execute(task)
print(f"Sources found: {result.metadata.get('nexus_search', False)}")
```

### Project Analyzer

```python
from orchestrator.analyzer import ProjectAnalyzer

# Enable Nexus integration
analyzer = ProjectAnalyzer(nexus_enabled=True)

# Analyze with web comparison
analysis = await analyzer.analyze(
    project_id="proj_123",
    compare_with_web=True,  # Uses Nexus Search
)

print(analysis.web_comparison)
```

---

## Troubleshooting

### Nexus Search Not Available

```bash
# 1. Check if Docker container is running
docker ps | grep nexus-search

# 2. Check container logs
docker logs nexus-search

# 3. Test connectivity
curl http://localhost:8080/health

# 4. Restart container
docker-compose -f nexus-search.docker-compose.yml restart
```

### Slow Search Performance

```bash
# Increase timeout
export NEXUS_TIMEOUT=60

# Reduce max results
export NEXUS_MAX_RESULTS=10

# Enable caching
export NEXUS_CACHE_ENABLED=true
export NEXUS_CACHE_TTL=3600
```

### Search Returns No Results

```python
# Try different sources
from orchestrator.nexus_search import SearchSource

results = await search(
    query="your query",
    sources=[SearchSource.WEB, SearchSource.ACADEMIC, SearchSource.TECH],
)

# Increase max results
results = await search(query, num_results=50)

# Disable optimization for comprehensive results
from orchestrator.nexus_search import OptimizationMode
results = await search(query, optimization=OptimizationMode.QUALITY)
```

### Rate Limiting

```python
# Configure rate limiting
from orchestrator.nexus_search import configure

configure(
    rate_limit=30,  # requests per minute
    retry_after=60,  # seconds to wait on rate limit
)
```

---

## Search Sources

| Source | Description | Best For |
|--------|-------------|----------|
| `WEB` | General web search | Broad queries, general knowledge |
| `ACADEMIC` | Academic papers, arXiv | Research, scientific topics |
| `TECH` | Technical documentation | Programming, frameworks, APIs |
| `NEWS` | News articles | Current events, recent developments |
| `CODE` | Code repositories (GitHub, GitLab) | Code examples, libraries |

---

## Optimization Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `SPEED` | Fastest results, fewer sources | Quick lookups, simple queries |
| `BALANCED` | Good balance of speed/quality | Default for most queries |
| `QUALITY` | Comprehensive search, all sources | Research, complex topics |

---

## Examples

### Example 1: Quick Fact Check

```python
from orchestrator.nexus_search import search

results = await search("Python 3.12 new features")
top_result = results[0]
print(f"Found: {top_result.title}")
print(f"URL: {top_result.url}")
```

### Example 2: Competitive Analysis

```python
from orchestrator.nexus_search import research

report = await research("FastAPI vs Django performance 2026")
print(report.summary)

for finding in report.findings:
    print(f"\n{finding.title}")
    print(finding.content)
```

### Example 3: Security Research

```python
from orchestrator.nexus_search import search, SearchSource

# Search for CVEs
results = await search(
    query="CVE-2026 Python vulnerabilities",
    sources=[SearchSource.WEB, SearchSource.TECH],
    num_results=20,
)

for r in results:
    if "CVE" in r.title:
        print(f"⚠️ {r.title}")
        print(f"   {r.url}")
```

### Example 4: Academic Research

```python
from orchestrator.nexus_search import research, SearchSource

report = await research(
    query="Transformer architecture improvements 2026",
    sources=[SearchSource.ACADEMIC, SearchSource.TECH],
)

print(f"Academic papers found: {len(report.findings)}")
for finding in report.findings:
    print(f"• {finding.title}")
```

---

## Related Documentation

- [USAGE_GUIDE.md](./USAGE_GUIDE.md) — Main usage guide with Nexus integration examples
- [CAPABILITIES.md](./CAPABILITIES.md) — Full capabilities overview
- [ARA_PIPELINE_GUIDE.md](./ARA_PIPELINE_GUIDE.md) — ARA Pipeline with research method

---

## API Reference Summary

| Function | Description | Returns |
|----------|-------------|---------|
| `search(query, sources, optimization, num_results)` | Perform web search | `SearchResults` |
| `research(query, depth, sources)` | Deep research report | `ResearchReport` |
| `classify(query)` | Classify query type | `QueryClassification` |
| `configure(**kwargs)` | Configure Nexus Search | `None` |
| `get_nexus_orchestrator()` | Get singleton orchestrator | `NexusSearchOrchestrator` |

---

**License:** MIT | **Author:** Georgios-Chrysovalantis Chatzivantsidis
