# multi-llm-orchestrator

Multi-provider LLM orchestrator with intelligent routing, cost optimization, and cross-provider resilience.

Decomposes project specs → Routes tasks to optimal providers → Executes generate→critique→revise cycles → Evaluates quality.

**Key Features:** 7 LLM providers • Cost-optimized routing • Budget hierarchy • Resume capability • Deterministic validation • Policy enforcement • Real-time telemetry

---

## Quick Start

### Install

```bash
pip install -e .                    # Core + aiosqlite
pip install pytest ruff jsonschema  # Optional validators
```

### Environment Setup

Set API keys for providers you'll use (at least one required):

```bash
# OpenAI (GPT-4o, GPT-4o-mini)
export OPENAI_API_KEY="sk-..."

# DeepSeek (Coder, Reasoner R1) — RECOMMENDED
export DEEPSEEK_API_KEY="sk-..."

# Google (Gemini 2.5 Pro, Flash)
export GOOGLE_API_KEY="AIzaSy..."
# or
export GEMINI_API_KEY="AIzaSy..."

# Kimi/Moonshot (K2.5 — 8K/32K/128K variants)
export KIMI_API_KEY="sk-..."
# or
export MOONSHOT_API_KEY="sk-..."

# Minimax (Minimax-3)
export MINIMAX_API_KEY="..."

# Zhipu (GLM-4)
export ZHIPUAI_API_KEY="..."

# Optional: OpenTelemetry tracing
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4318"
export ORCHESTRATOR_LOG_LEVEL="INFO"
```

**Quickest setup:** Only set `DEEPSEEK_API_KEY` (best cost/quality) + at least one other for fallback.

Or create `.env` file:
```bash
cat > .env << 'EOF'
DEEPSEEK_API_KEY=sk-...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AIzaSy...
EOF

source .env
```

### First Project

```bash
python -m orchestrator \
  --project "Build a FastAPI REST API with JWT authentication" \
  --criteria "All endpoints tested, OpenAPI docs complete" \
  --budget 2.0
```

Output saved to `./results/`

---

## Development Setup

### Prerequisites

- Python 3.10+
- Make (optional, for convenience commands)
- Docker (optional, for containerized runs)

### Install for Development

```bash
# Clone repository
git clone https://github.com/gchatz22/multi-llm-orchestrator.git
cd multi-llm-orchestrator

# Install with all development dependencies
make install-dev
# Or: pip install -e ".[dev,security,tracing]"

# Set up pre-commit hooks
pre-commit install
```

### Development Commands

```bash
# Run tests
make test              # All tests with coverage
make test-unit         # Unit tests only
make test-integration  # Integration tests only

# Code quality
make lint              # Run ruff linter
make format            # Format with black
make type-check        # Type check with mypy
make security-check    # Security scan with bandit

# Run all CI checks locally
make ci
```

### Project Structure

```
├── orchestrator/          # Main package
│   ├── domain/           # Business logic (models, policies)
│   ├── infrastructure/   # External services (API clients, cache)
│   ├── exceptions.py     # Exception hierarchy
│   └── logging.py        # Structured logging
├── tests/                # Test suite
│   ├── unit/            # Unit tests
│   └── integration/     # Integration tests
├── docs/                 # Documentation
├── pyproject.toml        # Package config & tool settings
├── Makefile             # Development commands
└── Dockerfile           # Multi-stage container build
```

### Environment Configuration

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
# Edit .env with your API keys
```

### Docker

```bash
# Build image
make docker-build

# Run container
make docker-run

# Run tests in container
make docker-test
```

---

## Providers & Models

| Provider | Models | Cost (per 1M tokens input) |
|----------|--------|--------------------------|
| **DeepSeek** | Coder, Reasoner (R1) | $0.27–$0.55 ⭐ |
| **Kimi** | K2.5 (8K/32K/128K) | $0.14 |
| **Minimax** | Minimax-3 (frontier reasoning) | $0.50 |
| **Zhipu** | GLM-4 Plus (general purpose) | $0.50 |
| **OpenAI** | GPT-4o, GPT-4o-mini | $0.15–$2.50 |
| **Google** | Gemini 2.5 Pro, Flash | $0.15–$1.25 |

---

## Core Capabilities

- **Multi-Provider Routing:** Task-aware model selection with fallback chains
- **Cost Optimization:** EMA-tracked pricing, budget hierarchy (org/team/job)
- **Quality Assurance:** Deterministic validators + multi-round critique + LLM scoring
- **Resilience:** Circuit breaker (3-strike) + cross-provider fallback chains
- **Resume:** Auto-detect similar projects, resumable by project ID
- **Policy Control:** HARD/SOFT/MONITOR enforcement for compliance
- **Observability:** OpenTelemetry tracing, real-time telemetry, event hooks

### 🆕 v5.1 Management Systems

Enterprise-grade management suite for large-scale operations:

| System | Key Features |
|--------|-------------|
| **Knowledge Management** | Semantic search, pattern recognition, auto-learning from projects |
| **Project Management** | Critical path analysis, resource scheduling, risk assessment |
| **Product Management** | RICE prioritization, feature flags, sentiment analysis |
| **Quality Control** | Multi-level testing, static analysis, compliance gates |
| **Project Analyzer** | Automatic post-project analysis & improvement suggestions |
| **Real-Time Dashboard** | Live metrics from orchestrator telemetry |

### 🆕 v5.0 Performance Optimization

Production-ready performance enhancements:

| Feature | Benefit |
|---------|---------|
| **Dual-Layer Caching** | Redis + LRU fallback, sub-millisecond hits |
| **Dashboard v5.0** | 5x faster load, <100ms FCP, gzip compression |
| **Connection Pooling** | Bounded resource management |
| **KPI Monitoring** | Real-time performance tracking with alerts |

**Performance Targets:**
- First Contentful Paint: <100ms
- Cache Hit Rate: >85%
- P95 Response Time: <300ms

---

## CLI Examples

### 1. Build a FastAPI Service (Default)

```bash
python -m orchestrator \
  --project "FastAPI authentication with JWT tokens" \
  --criteria "All endpoints tested, OpenAPI docs complete" \
  --budget 3.0
```

### 2. Build a Next.js App

```bash
python -m orchestrator \
  --project "Next.js e-commerce storefront with product listing and cart" \
  --criteria "npm build succeeds, no console errors" \
  --output-dir ./storefront
```

### 3. Resume Interrupted Run

```bash
python -m orchestrator --resume <project_id>
```

### 4. Skip Enhancement & Resume Detection

```bash
python -m orchestrator \
  --project "Build a React dashboard" \
  --criteria "npm build succeeds" \
  --no-enhance --new-project
```

### 5. Launch Mission Control Dashboard

```bash
# Run real-time dashboard with live data
python run_dashboard_realtime.py --port 8888

# Run optimized dashboard
python run_optimized_dashboard.py --port 8888

# Or with Redis caching
python run_optimized_dashboard.py --redis-host localhost --port 8888

# View at http://localhost:8888
```

---

## Python API

### Basic Usage

```python
import asyncio
from orchestrator import Orchestrator, Budget

async def main():
    budget = Budget(max_usd=5.0, max_time_seconds=3600)
    orch = Orchestrator(budget=budget)

    state = await orch.run_project(
        project_description="Implement a Python rate limiter using decorators",
        success_criteria="pytest suite passes, ruff linting clean",
    )

    print(f"Status: {state.status.value}")
    print(f"Spent: ${state.budget.spent_usd:.4f}")
    print(f"Score: {state.overall_quality_score:.3f}")

asyncio.run(main())
```

### Management Systems API

```python
from orchestrator import (
    get_knowledge_base, get_project_manager,
    get_product_manager, get_quality_controller,
    KnowledgeType, RICEScore, FeaturePriority, TestLevel
)

# Knowledge Management
kb = get_knowledge_base()
await kb.add_artifact(
    type=KnowledgeType.SOLUTION,
    title="Race condition fix",
    content="Use asyncio.Lock()...",
    tags=["async", "python"],
)
similar = await kb.find_similar("async race condition")

# Project Management
pm = get_project_manager()
timeline = await pm.create_schedule(
    project_id="my_project",
    tasks=tasks,
    resources=resources,
)
print(f"Critical path: {timeline.critical_path}")

# Product Management
pm = get_product_manager()
feature = await pm.add_feature(
    name="AI Assistant",
    rice_score=RICEScore(500, 3, 80, 2),  # Score = 600
    priority=FeaturePriority.P0_CRITICAL,
)
backlog = pm.get_prioritized_backlog(limit=10)

# Quality Control
qc = get_quality_controller()
report = await qc.run_quality_gate(
    project_path=Path("."),
    levels=[TestLevel.UNIT, TestLevel.PERFORMANCE],
)
print(f"Quality Score: {report.quality_score:.1f}/100")
```

---

## Documentation

### Main Documentation
- **[USAGE_GUIDE.md](./USAGE_GUIDE.md)** — CLI reference, Python API examples, best practices
- **[CAPABILITIES.md](./CAPABILITIES.md)** — Feature deep-dive, architecture details, advanced recipes

### Debugging & Troubleshooting
- **[docs/debugging/DEBUGGING_GUIDE.md](./docs/debugging/DEBUGGING_GUIDE.md)** — Comprehensive debugging manual
- **[docs/debugging/TROUBLESHOOTING_CHEATSHEET.md](./docs/debugging/TROUBLESHOOTING_CHEATSHEET.md)** — Quick fixes

### Performance & Management
- **[docs/performance/PERFORMANCE_OPTIMIZATION.md](./docs/performance/PERFORMANCE_OPTIMIZATION.md)** — Performance optimization
- **[docs/performance/MANAGEMENT_SYSTEMS.md](./docs/performance/MANAGEMENT_SYSTEMS.md)** — Management systems guide

---

## Architecture Overview

```
Project Description
       ↓
Auto-Resume Detect → Project Enhancer → Architecture Advisor
       ↓
Decompose into Tasks
       ↓
Route → Generate → Critique → Revise → Evaluate
       ↓
Cross-Provider Fallback Chain (quality escalation)
       ↓
Deterministic Validation (python_syntax, pytest, ruff, json_schema)
       ↓
Store Results + Telemetry + State Checkpoint (SQLite)
```

---

## Testing

```bash
pytest tests/                  # Full suite (152+ tests)
pytest tests/test_opus_removal.py -v  # Verify Opus removal
pytest --tb=short -q          # Summary
```

---

## Configuration

All configuration can be set via environment variables or Python API:

```bash
# Required: at least one provider key
export OPENAI_API_KEY="sk-..."
export DEEPSEEK_API_KEY="sk-..."
export GOOGLE_API_KEY="AIzaSy..."
export KIMI_API_KEY="sk-..."
export MINIMAX_API_KEY="..."
export ZHIPU_API_KEY="...":

# Optional
export ORCHESTRATOR_CACHE_DIR="~/.orchestrator_cache"
export ORCHESTRATOR_LOG_LEVEL="INFO"
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4318"
```

---

## Project Status

- **Baseline:** 152+ tests passing, 0 regressions
- **Last Updated:** 2026-02-26

---

## Next Steps

1. Install & set API key (see Quick Start above)
2. Run first project: `python -m orchestrator --project "..."`
3. Check [USAGE_GUIDE.md](./USAGE_GUIDE.md) for CLI reference
4. See [CAPABILITIES.md](./CAPABILITIES.md) for advanced features

---

**Questions?** Check the docs or run `python -m orchestrator --help`
