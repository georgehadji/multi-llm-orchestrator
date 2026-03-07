# multi-llm-orchestrator

Multi-provider LLM orchestrator with intelligent routing, cost optimization, and cross-provider resilience.

Decomposes project specs → Routes tasks to optimal providers → Executes generate→critique→revise cycles → Evaluates quality.

**Key Features:** 6 LLM providers • Cost-optimized routing (-35%) • Budget hierarchy • Resume capability • Deterministic validation • Policy enforcement • Real-time telemetry • Mission-Critical Command Center

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
# OpenAI (GPT-4o, GPT-4o-mini, o4-mini)
export OPENAI_API_KEY="sk-..."

# DeepSeek (Chat V3.2, Reasoner R1) — RECOMMENDED ⭐
export DEEPSEEK_API_KEY="sk-..."

# Google (Gemini 2.5 Pro, Flash, Flash Lite)
export GOOGLE_API_KEY="AIzaSy..."
# or
export GEMINI_API_KEY="AIzaSy..."

# Anthropic (Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku)
export ANTHROPIC_API_KEY="sk-ant-..."

# Minimax (MiniMax-Text-01)
export MINIMAX_API_KEY="..."

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

## Dashboard (Web UI)

Launch the web dashboard for visual project management:

### Windows
```bash
# Double-click in Explorer, or run in CMD:
start_dashboard.bat
```

### Linux/Mac
```bash
python start_dashboard.py
```

### Features
- ✅ **New Project** - Create from scratch with natural language
- ✅ **Improve Codebase** - Refactor existing projects
- ✅ **Upload Spec** - YAML/JSON project specifications
- ✅ **Real-time Updates** - Live progress, logs, and status
- ✅ **Multiple Projects** - Run New + Improve + Upload simultaneously

**URL:** http://localhost:8888

---

## Development Setup

### Prerequisites

- Python 3.10+
- Make (optional, for convenience commands)
- Docker (optional, for containerized runs)

### Install for Development

```bash
# Clone repository
git clone https://github.com/georgehadji/multi-llm-orchestrator.git
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

| Provider | Models | Cost (per 1M tokens input/output) |
|----------|--------|-----------------------------------|
| **DeepSeek** | Chat, Reasoner (R1) | $0.28/$0.42 ⭐ **Best Value** |
| **Google** | Gemini Flash Lite | $0.075/$0.30 |
| **Google** | Gemini Flash | $0.15/$0.60 |
| **OpenAI** | GPT-4o-mini | $0.15/$0.60 |
| **Anthropic** | Claude 3 Haiku | $0.25/$1.25 |
| **MiniMax** | MiniMax-Text-01 | $0.50/$1.50 |
| **OpenAI** | o4-mini | $1.50/$6.00 |
| **Anthropic** | Claude 3.5 Sonnet | $3.00/$15.00 ⭐ **Best Coding** |
| **Google** | Gemini Pro | $1.25/$10.00 |
| **OpenAI** | GPT-4o | $2.50/$10.00 |

---

## Core Capabilities

- **Multi-Provider Routing:** Task-aware model selection with fallback chains
- **Cost Optimization:** EMA-tracked pricing, budget hierarchy (org/team/job)
- **Quality Assurance:** Deterministic validators + multi-round critique + LLM scoring
- **Resilience:** Circuit breaker (3-strike) + cross-provider fallback chains
- **Resume:** Auto-detect similar projects, resumable by project ID
- **Policy Control:** HARD/SOFT/MONITOR enforcement for compliance
- **Observability:** OpenTelemetry tracing, real-time telemetry, event hooks

### 🆕 v6.0 Black Swan Resilience

Production-hardened defenses against catastrophic failures:

| Feature | Protection | Risk Reduction |
|---------|------------|----------------|
| **Resilient Event Store** | WAL + replication + checksums | Data corruption: $155k → $500 (99.7%) |
| **Secure Plugin Runtime** | seccomp + Landlock + capabilities | Sandbox escape: $1.15M → $1k (99.9%) |
| **Streaming Backpressure** | Bounded queues + circuit breaker | Memory exhaustion: $30k → $500 (98.3%) |

**Total Risk Reduction:** 99.85% ($1.3M → $2k potential loss)

**Quick Start:**
```python
# Opt-in to resilient features (backward compatible)
from orchestrator.events_resilient import ResilientEventStore
from orchestrator.plugin_isolation_secure import SecureIsolatedRuntime
from orchestrator.streaming_resilient import ResilientStreamingPipeline
```

See [MIGRATION_GUIDE_v6.md](MIGRATION_GUIDE_v6.md) for details.

---

### 🆕 v6.1 Production Optimizations

Cost and performance optimizations based on adversarial stress testing:

| Optimization | Mechanism | Impact |
|--------------|-----------|--------|
| **Confidence-Based Early Exit** | Exit when stable high performance detected | -25% iterations |
| **Tiered Model Selection** | CHEAP→BALANCED→PREMIUM escalation | -22% cost |
| **Semantic Sub-Result Caching** | Pattern-based caching (not exact match) | -15% cost, -50% latency |
| **Fast Regression Detection** | EMA α=0.2 (was 0.1) | 2× faster response |
| **Tool Safety Validation** | Blocks hallucinated shell/code execution | Security hardening |

**Total Cost Reduction:** 35% ($2.40 → $1.55 per project)

**Quick Start:**
```python
# Optimizations enabled by default in v6.1+
from orchestrator import Orchestrator

orch = Orchestrator()  # All optimizations active

# Check semantic cache stats
print(orch._semantic_cache.get_stats())

# View tier escalation history
print(orch._tier_escalation_count)
```

See [OPTIMIZATION_IMPLEMENTATION_SUMMARY.md](OPTIMIZATION_IMPLEMENTATION_SUMMARY.md) for details.

---

### 🆕 v6.0 Mission-Critical Command Center

Real-time operational dashboard for production monitoring:

| Feature | Specification |
|---------|---------------|
| **Latency** | < 500ms end-to-end, 100ms batching |
| **Reliability** | WebSocket → SSE → polling fallback |
| **Alerting** | 5-level severity, ACK required for Critical |
| **Security** | RBAC (viewer/operator/admin), immutable audit log |
| **Layout** | Fixed KPIs, no reflow on alert, spatial stability |

**Dashboard Layout:**
```
┌─ Header (60px) ───────────────────────────┐
│  ◈ LLM ORCHESTRATOR      COST $1.23/hr ▲2  │
├─ KPI Row (200px) ─────────────────────────┤
│  [MODEL HEALTH] [TASK QUEUE] [QUALITY]    │
├─ Main Content ────────────────────────────┤
│  ⚠️ ACTIVE CRITICAL ALERTS (2)            │
│     • Model gpt-4o unhealthy    [ACK]     │
│  ℹ️ SYSTEM EVENTS                         │
│     • Project completed                   │
├─ Status Bar (40px) ───────────────────────┤
│  ● Connected | Latency: 45ms              │
└───────────────────────────────────────────┘
```

**Quick Start:**
```python
from orchestrator import Orchestrator
from orchestrator.command_center_integration import enable_command_center

orch = Orchestrator()
cc = enable_command_center(orch)

# Open dashboard: orchestrator/CommandCenter.html
# Or: python -m http.server 8080 --directory orchestrator
```

See [COMMAND_CENTER_IMPLEMENTATION.md](COMMAND_CENTER_IMPLEMENTATION.md) for details.

---

### 🆕 v5.2 Code Quality & Attribution

Enhanced code generation with automatic documentation:

| Feature | Description |
|---------|-------------|
| **Author Attribution** | Every file includes `Author: Georgios-Chrysovalantis Chatzivantsidis` |
| **Thorough Comments** | All functions, classes, and complex logic documented |
| **Smart Validator Filtering** | Python validators auto-removed for HTML/CSS/JS tasks |
| **Code Output Cleaning** | Strips markdown fences and placeholder comments |
| **Temperature Optimization** | 0.0 for code (deterministic), 0.2 for review |

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
| **Architecture Rules** | Auto-select optimal architecture & generate rules |
| **Output Organization** | Auto-move tasks, generate & run tests |
| **Cache Suppression** | Clean output without verbose cache messages |
| **WordPress Plugin Rules** | Professional WP plugin development guidelines |
| **InDesign Plugin Rules** | Professional InDesign plugin development (UXP/C++) |

### 🆕 v5.0 Performance Optimization

Production-ready performance enhancements:

| Feature | Benefit |
|---------|---------|
| **Dual-Layer Caching** | Redis + LRU fallback, sub-millisecond hits |
| **Dashboard v5.0** | 5x faster load, <100ms FCP, gzip compression |
| **Enhanced Dashboard v2.0** | Architecture visibility, task progress, model status |
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
# 🎮 Run LIVE Dashboard v4.0 (RECOMMENDED) - Gamified, real-time WebSocket
python -c "from orchestrator.dashboard_live import run_live_dashboard; run_live_dashboard()"

# Or run Ant Design dashboard v3.0 - Modern professional UI
python scripts/run_dashboard.py

# Or run enhanced dashboard v2.0
python -c "from orchestrator.dashboard_enhanced import run_enhanced_dashboard; run_enhanced_dashboard()"

# View at http://localhost:8888
```

**🎮 Mission Control LIVE v4.0 Features:**
- ⚡ **WebSocket Real-time** - True live updates (no polling!)
- 🎮 **Gamification** - XP, levels, achievements
- 🔔 **Toast Notifications** - Instant alerts for all events
- 🎊 **Celebration Effects** - Confetti on project completion
- 🎵 **Sound Effects** - Audio feedback
- 🔥 **Live Task Monitor** - Watch tasks execute in real-time
- 🧪 **Test Tracking** - Live test execution monitoring

**Achievements to Unlock:**
🎯 Task Master • ⚡ Speed Demon • 💯 Perfectionist • 💰 Budget Master • 🧪 Test Champion • 🔥 On Fire • 🏗️ Architect

**Ant Design Dashboard v3.0 Features:**
- 🎨 **Modern UI**: Ant Design component library
- 📊 **Real-time Visualization**: Live metrics and charts
- 🏗️ **Architecture Panel**: Complete architecture decisions
- 🤖 **Model Health Table**: Detailed status with metrics
- ⚡ **Task Progress**: Iteration tracking with scores
- 🔄 **Auto-refresh**: Every 3 seconds
- 📱 **Responsive**: Works on all screen sizes

**Enhanced Dashboard v2.0 Features:**
- 🏗️ **Architecture Decisions**: Style, paradigm, technology stack
- ⚡ **Real-time Task Progress**: Current/total tasks, iteration, score
- 📋 **Project Details**: Description, success criteria, budget

### 6. Output Organization & Test Automation

After project completion, files are automatically organized:

```bash
python -m orchestrator \
  --project "Build a REST API" \
  --criteria "All endpoints tested" \
  --budget 5.0

# Output:
# 📁 Organizing project output...
#   ✅ Tasks moved: 8
#   ✅ Tests generated: 2
#   ✅ Tests: 5/6 passed
#   📈 Coverage: 78.5%
```

**Organization Features:**
- 📂 Task files moved to `tasks/` folder
- 🧪 Missing tests auto-generated
- ✅ Tests automatically executed
- 📊 Coverage report generated
- 🔇 Cache messages suppressed for clean output
- 🤖 **Model Status**: Available/unavailable with reasons
- 🔄 **Auto-refresh**: Live updates every 3 seconds

---

## Utility Scripts

The `scripts/` folder contains utility scripts for common tasks:

```bash
# Start Ant Design dashboard
python scripts/run_dashboard.py

# Run tests with coverage
python scripts/run_tests.py

# Organize project output
python scripts/organize_output.py ./outputs/project_123

# Check model availability
python scripts/check_models.py

# Clean cache files
python scripts/cleanup_cache.py

# Create new project
python scripts/create_project.py \
  -p "Build a REST API" \
  -c "All endpoints tested" \
  -b 5.0
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
export ANTHROPIC_API_KEY="sk-ant-..."
export MINIMAX_API_KEY="..."

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
