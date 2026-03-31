# AI Orchestrator Documentation Index

**Version:** 1.0.0 | **Updated:** 2026-03-25 | **Status:** Complete ✅

> **Central index for all documentation** — Find the right guide for your needs.

---

## Quick Navigation

| I want to... | Start Here | Time |
|--------------|------------|------|
| **Get started** | [README.md](#getting-started) | 5 min |
| **Learn CLI usage** | [USAGE_GUIDE.md](#usage-guides) | 15 min |
| **Understand features** | [CAPABILITIES.md](#feature-documentation) | 30 min |
| **Use ARA Pipeline** | [METHODS.md](#ara-pipeline) | 20 min |
| **Setup web search** | [NEXUS_SEARCH_README.md](#integrations) | 10 min |
| **Connect external agents** | [A2A_PROTOCOL_GUIDE.md](#integrations) | 15 min |
| **Understand architecture** | [ARCHITECTURE_OVERVIEW.md](#architecture) | 25 min |

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Usage Guides](#usage-guides)
3. [Feature Documentation](#feature-documentation)
4. [ARA Pipeline](#ara-pipeline)
5. [Integrations](#integrations)
6. [Management Systems](#management-systems)
7. [Architecture](#architecture)
8. [Migration Guides](#migration-guides)
9. [Debugging & Troubleshooting](#debugging--troubleshooting)
10. [Development](#development)

---

## Getting Started

### Essential Reading

| Document | Description | Read Time |
|----------|-------------|-----------|
| **[README.md](./README.md)** | Installation, quick start, first project | 5 min |
| **[USAGE_GUIDE.md](./USAGE_GUIDE.md)** | CLI reference, Python API, examples | 15 min |
| **[CAPABILITIES.md](./CAPABILITIES.md)** | Feature overview, version history | 20 min |

### Quick Start Path

1. Read [README.md](./README.md) — Install and run first project
2. Read [USAGE_GUIDE.md](./USAGE_GUIDE.md) — Learn CLI and Python API
3. Explore [CAPABILITIES.md](./CAPABILITIES.md) — Discover advanced features

---

## Usage Guides

### Core Usage

| Document | Description | Read Time |
|----------|-------------|-----------|
| **[USAGE_GUIDE.md](./USAGE_GUIDE.md)** | Complete CLI & Python API reference | 30 min |
| **[METHODS.md](./METHODS.md)** | ARA Pipeline methods (12 reasoning strategies) | 25 min |
| **[INTEGRATIONS_COMPLETE.md](./INTEGRATIONS_COMPLETE.md)** | All external integrations | 20 min |

### Examples by Use Case

| Use Case | Documents |
|----------|-----------|
| Build a REST API | [USAGE_GUIDE.md](./USAGE_GUIDE.md#1-build-a-fastapi-service-simplest) |
| Build a Next.js App | [USAGE_GUIDE.md](./USAGE_GUIDE.md#2-build-a-nextjs-app) |
| Use ARA Pipeline | [USAGE_GUIDE.md](./USAGE_GUIDE.md#ara-pipeline--advanced-reasoning-methods) |
| Web Search | [NEXUS_SEARCH_README.md](./NEXUS_SEARCH_README.md) |
| External Agents | [A2A_PROTOCOL_GUIDE.md](./A2A_PROTOCOL_GUIDE.md) |

---

## Feature Documentation

### Core Features

| Document | Description | Read Time |
|----------|-------------|-----------|
| **[CAPABILITIES.md](./CAPABILITIES.md)** | All features, version history, SaaS mapping | 30 min |
| **[ARCHITECTURE_OVERVIEW.md](./ARCHITECTURE_OVERVIEW.md)** | System architecture, data flow, deployment | 25 min |

### v6.2 New Features

| Feature | Document | Description |
|---------|----------|-------------|
| 🧠 **ARA Pipeline** | [METHODS.md](./METHODS.md) | 12 Advanced Reasoning Methods |
| 🔮 **Nexus Search** | [NEXUS_SEARCH_README.md](./NEXUS_SEARCH_README.md) | Self-hosted web search |
| 🛡️ **Preflight** | [PREFLIGHT_SESSION_GUIDE.md](./PREFLIGHT_SESSION_GUIDE.md#1-preflight-validation) | Response quality control |
| 📊 **Session Watcher** | [PREFLIGHT_SESSION_GUIDE.md](./PREFLIGHT_SESSION_GUIDE.md#2-session-watcher) | Memory management (HOT/WARM/COLD) |
| 🎭 **Persona Modes** | [PREFLIGHT_SESSION_GUIDE.md](./PREFLIGHT_SESSION_GUIDE.md#3-persona-modes) | Behavior customization |
| 💾 **Token Optimizer** | [TOKEN_OPTIMIZER_GUIDE.md](./TOKEN_OPTIMIZER_GUIDE.md) | 60-90% token reduction |

### v6.0-v6.1 Features

| Feature | Document | Description |
|---------|----------|-------------|
| **Black Swan Resilience** | [CAPABILITIES.md](./CAPABILITIES.md#v60-black-swan-resilience) | Event store, plugin sandbox, backpressure |
| **Mission-Critical Command Center** | [CAPABILITIES.md](./CAPABILITIES.md#v60-mission-critical-command-center) | Real-time monitoring dashboard |
| **Production Optimizations** | [CAPABILITIES.md](./CAPABILITIES.md#v61-production-optimizations) | -35% cost reduction |

---

## ARA Pipeline

### Methods Documentation

| Document | Description | Read Time |
|----------|-------------|-----------|
| **[METHODS.md](./METHODS.md)** | Complete guide to all 12 reasoning methods | 30 min |

### Reasoning Methods

#### Standard Methods (7)

| Method | Use Case | Cost Multiplier | Quality Gain |
|--------|----------|-----------------|--------------|
| Multi-Perspective | Complex decisions | 2.0× | +30% |
| Iterative | Refinement tasks | 1.5× | +20% |
| Debate | Architecture, trade-offs | 2.5× | +40% |
| Research | Deep analysis | 3.0× | +45% |
| Jury | High-stakes code | 5.0× | +50% |
| Scientific | Hypothesis testing | 2.5× | +40% |
| Socratic | Philosophical questions | 2.0× | +25% |

#### Specialized Methods (5)

| Method | Use Case | Cost Multiplier | Quality Gain |
|--------|----------|-----------------|--------------|
| Pre-Mortem ⭐ | Risk assessment | 1.8× | +45% |
| Bayesian | Uncertainty quantification | 2.2× | +50% |
| Dialectical | Strategic decisions | 2.5× | +40% |
| Analogical | Innovation, creativity | 2.0× | +35% |
| Delphi | Expert consensus | 3.5× | +55% |

---

## Integrations

### Built-in Integrations

| Integration | Document | Description |
|-------------|----------|-------------|
| **Nexus Search** | [NEXUS_SEARCH_README.md](./NEXUS_SEARCH_README.md) | Self-hosted web search (zero tracking) |
| **A2A Protocol** | [A2A_PROTOCOL_GUIDE.md](./A2A_PROTOCOL_GUIDE.md) | External agent communication |
| **MCP Server** | [INTEGRATIONS_COMPLETE.md](./INTEGRATIONS_COMPLETE.md#1-mcp-server-integration) | Model Context Protocol |
| **BM25 Search** | [INTEGRATIONS_COMPLETE.md](./INTEGRATIONS_COMPLETE.md#2-bm25-search-integration) | Full-text search |
| **LLM Reranker** | [INTEGRATIONS_COMPLETE.md](./INTEGRATIONS_COMPLETE.md#3-llm-reranker-integration) | Search result reranking |
| **Preflight/Session** | [PREFLIGHT_SESSION_GUIDE.md](./PREFLIGHT_SESSION_GUIDE.md) | Mnemo Cortex features |
| **Token Optimizer** | [TOKEN_OPTIMIZER_GUIDE.md](./TOKEN_OPTIMIZER_GUIDE.md) | RTK-style compression |

### External Integrations

| Integration | Document | Description |
|-------------|----------|-------------|
| **LiteLLM** | [INTEGRATIONS_COMPLETE.md](./INTEGRATIONS_COMPLETE.md#4-litellm-integration) | 100+ LLM providers |
| **LangGraph** | [INTEGRATIONS_COMPLETE.md](./INTEGRATIONS_COMPLETE.md#5-langgraph-integration) | Agent workflows |

### Integration Quick Starts

```bash
# Nexus Search
docker-compose -f nexus-search.docker-compose.yml up -d

# MCP Server (for Claude Desktop)
python -m orchestrator.mcp_server --http --port 8181

# Dashboard
python -m orchestrator dashboard
```

---

## Management Systems

### Enterprise Features

| System | Document | Description |
|--------|----------|-------------|
| **Knowledge Management** | [MANAGEMENT_SYSTEMS_GUIDE.md](./MANAGEMENT_SYSTEMS_GUIDE.md#1-knowledge-management) | Capture and retrieve learnings |
| **Project Management** | [MANAGEMENT_SYSTEMS_GUIDE.md](./MANAGEMENT_SYSTEMS_GUIDE.md#2-project-management) | Scheduling, resource allocation |
| **Product Management** | [MANAGEMENT_SYSTEMS_GUIDE.md](./MANAGEMENT_SYSTEMS_GUIDE.md#3-product-management) | RICE scoring, backlog, roadmap |
| **Quality Control** | [MANAGEMENT_SYSTEMS_GUIDE.md](./MANAGEMENT_SYSTEMS_GUIDE.md#4-quality-control) | Testing, static analysis, compliance |
| **Project Analyzer** | [MANAGEMENT_SYSTEMS_GUIDE.md](./MANAGEMENT_SYSTEMS_GUIDE.md#5-project-analyzer) | Post-project analysis |

---

## Architecture

### System Documentation

| Document | Description | Read Time |
|----------|-------------|-----------|
| **[ARCHITECTURE_OVERVIEW.md](./ARCHITECTURE_OVERVIEW.md)** | Complete system architecture | 25 min |
| **[CAPABILITIES.md](./CAPABILITIES.md#architecture-overview)** | High-level architecture | 10 min |

### Architecture Topics

| Topic | Section |
|-------|---------|
| Layer Architecture | [ARCHITECTURE_OVERVIEW.md#1-layer-architecture](./ARCHITECTURE_OVERVIEW.md#1-layer-architecture) |
| Component Details | [ARCHITECTURE_OVERVIEW.md#2-component-details](./ARCHITECTURE_OVERVIEW.md#2-component-details) |
| Data Flow | [ARCHITECTURE_OVERVIEW.md#3-data-flow](./ARCHITECTURE_OVERVIEW.md#3-data-flow) |
| Deployment | [ARCHITECTURE_OVERVIEW.md#4-deployment-architecture](./ARCHITECTURE_OVERVIEW.md#4-deployment-architecture) |
| Security | [ARCHITECTURE_OVERVIEW.md#5-security-architecture](./ARCHITECTURE_OVERVIEW.md#5-security-architecture) |

---

## Migration Guides

### Dashboard Migration

| Document | Description | Read Time |
|----------|-------------|-----------|
| **[DASHBOARD_MIGRATION_GUIDE.md](./DASHBOARD_MIGRATION_GUIDE.md)** | Migrate 9 legacy dashboards → unified dashboard_core | 20 min |

### Migration Path

1. Read [DASHBOARD_MIGRATION_GUIDE.md](./DASHBOARD_MIGRATION_GUIDE.md)
2. Update imports
3. Migrate custom views
4. Test and verify
5. Remove legacy files

---

## Debugging & Troubleshooting

### Debug Documentation

| Document | Description | Read Time |
|----------|-------------|-----------|
| **[docs/debugging/DEBUGGING_GUIDE.md](./docs/debugging/DEBUGGING_GUIDE.md)** | Comprehensive debugging manual | 20 min |
| **[docs/debugging/TROUBLESHOOTING_CHEATSHEET.md](./docs/debugging/TROUBLESHOOTING_CHEATSHEET.md)** | Quick fixes for common issues | 5 min |

### Common Issues

| Issue | Solution |
|-------|----------|
| API connection fails | [USAGE_GUIDE.md](./USAGE_GUIDE.md#troubleshooting) |
| Dashboard won't start | [DASHBOARD_MIGRATION_GUIDE.md](./DASHBOARD_MIGRATION_GUIDE.md#troubleshooting) |
| Nexus Search unavailable | [NEXUS_SEARCH_README.md](./NEXUS_SEARCH_README.md#troubleshooting) |
| A2A timeout | [A2A_PROTOCOL_GUIDE.md](./A2A_PROTOCOL_GUIDE.md#error-handling) |

---

## Development

### Development Guides

| Guide | Location | Description |
|-------|----------|-------------|
| **Testing** | `tests/` directory | 152+ tests |
| **Linting** | `pyproject.toml` | Ruff, Black, MyPy config |
| **Pre-commit** | `.pre-commit-config.yaml` | Git hooks |

### Development Commands

```bash
# Install for development
pip install -e ".[dev]"

# Run tests
pytest tests/

# Lint
ruff check orchestrator/

# Type check
mypy orchestrator/

# Format
black orchestrator/
```

---

## Documentation Changelog

### 2026-03-25 — Major Documentation Update

**New Documents (12):**
- ✅ NEXUS_SEARCH_README.md
- ✅ A2A_PROTOCOL_GUIDE.md
- ✅ PREFLIGHT_SESSION_GUIDE.md
- ✅ DASHBOARD_MIGRATION_GUIDE.md
- ✅ MANAGEMENT_SYSTEMS_GUIDE.md
- ✅ TOKEN_OPTIMIZER_GUIDE.md
- ✅ INTEGRATIONS_COMPLETE.md
- ✅ ARCHITECTURE_OVERVIEW.md
- ✅ DOCUMENTATION_INDEX.md (this file)

**Updated Documents:**
- ✅ README.md — Added v6.2 features, new documentation links
- ✅ CAPABILITIES.md — Updated with all v6.2 features
- ✅ USAGE_GUIDE.md — Added complete examples

---

## Quick Reference

### Environment Variables

```bash
# API Keys (required: at least one)
export OPENAI_API_KEY="sk-..."
export DEEPSEEK_API_KEY="sk-..."  # Recommended
export GOOGLE_API_KEY="AIzaSy..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Features
export NEXUS_SEARCH_ENABLED=true
export ARA_PIPELINE_ENABLED=true
export PREFLIGHT_ENABLED=true
```

### Key Commands

```bash
# Run project
python -m orchestrator --project "..." --criteria "..." --budget 5.0

# Launch dashboard
python -m orchestrator dashboard

# Start MCP Server
python -m orchestrator.mcp_server --http --port 8181

# Run tests
pytest tests/ -v
```

---

## Related Resources

- **GitHub Repository:** https://github.com/georrgehadji/multi-llm-orchestrator
- **Issues:** https://github.com/georrgehadji/multi-llm-orchestrator/issues
- **PyPI Package:** https://pypi.org/project/multi-llm-orchestrator/

---

**License:** MIT | **Author:** Georgios-Chrysovalantis Chatzivantsidis
