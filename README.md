# Multi-LLM Orchestrator

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-141%2F141%20passing-brightgreen.svg)]()
[![Coverage](https://img.shields.io/badge/coverage-12.2%25-blue.svg)]()
[![Code Quality](https://img.shields.io/badge/code%20quality-ruff%20%7C%20black%20%7C%20mypy-blue.svg)]()

> **Autonomous Software Development Platform** with multi-provider LLM orchestration, intelligent routing, cross-provider resilience, and comprehensive observability.

Decomposes project specifications into tasks, routes them to optimal LLM providers, and executes iterative generate-critique-revise-evaluate cycles with deterministic validation.

---

## Key Features

| Feature | Description | Status |
|---------|-------------|--------|
| **Smart Routing** | Task-aware model selection with cost optimization (-35%) | Stable |
| **Budget Management** | Hierarchical budgets (org/team/job) with mid-task enforcement | Stable |
| **Cross-Provider Fallback** | Automatic failover with native OpenRouter fallbacks | Stable |
| **Real-time Telemetry** | Cost tracking, latency monitoring, quality metrics | Stable |
| **Deterministic Validation** | Syntax checkers, test runners, security scans | Stable |
| **OpenRouter Optimizations** | JSON Schema, model variants, provider sorting | Stable |
| **Nexus Search** | Self-hosted web search with hybrid RRF ranking | Stable |
| **ARA Pipeline** | 12 Advanced Reasoning Methods (Debate, Jury, etc.) | Stable |
| **iOS Suite** | App Store compliance with 6 enhancement modules | Stable |
| **Service Architecture** | Extracted executor, evaluator, generator services | Phase 6 Complete |
| **Resilience Layer** | Circuit breaker, per-model registry, cascade fallback | Phase 6 Complete |
| **Port Interfaces** | CachePort, StatePort, EventPort + NullAdapters (DI) | Phase 7 Complete |
| **Observability** | Per-model metrics: latency, error rate, cost tracking | Phase 6 Complete |
| **MVOS Audit** | 7 invariants verified at runtime (100% passing) | Phase 8 Complete |
| **Architectural Validation** | Full audit closure with health grades updated | Phase 9 Complete |

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/georgehadji/multi-llm-orchestrator.git
cd multi-llm-orchestrator

# Install with all dependencies
pip install -e ".[dev,security,tracing]"

# Or minimal install
pip install -e .
```

### Configuration

Create a `.env` file with your API keys:

```bash
# Required: At least one provider
OPENAI_API_KEY="sk-..."
DEEPSEEK_API_KEY="sk-..."
GOOGLE_API_KEY="AIzaSy..."
ANTHROPIC_API_KEY="sk-ant-..."

# Optional: OpenRouter optimizations
USE_JSON_SCHEMA_RESPONSES="true"
USE_MODEL_VARIANTS="true"
USE_NATIVE_FALLBACKS="true"
```

### First Run

```bash
# Simple project
python -m orchestrator \
  --project "Build a FastAPI REST API with JWT auth" \
  --budget 2.0

# With success criteria
python -m orchestrator \
  --project "Create a React dashboard" \
  --criteria "TypeScript, responsive, dark mode" \
  --budget 5.0
```

Results are saved to `./results/`.

---

## Architecture

```
Project Description
       |
       v
+-----------------+     +-----------------+
| Auto-Resume     |---->|  Enhancement    |
| Detection       |     |  & Advisory     |
+--------|--------+     +-----------------+
         |
         v
+-----------------------------------------+
|         DECOMPOSITION ENGINE            |
|  Breaks projects into atomic tasks      |
+---------------|-------------------------+
                |
                v
+-----------------------------------------+
|           EXECUTION PIPELINE            |
|  Route -> Generate -> Critique -> Revise|
|         |                               |
|  Evaluate + Deterministic Validation    |
|         |                               |
|  Cross-Provider Fallback Chain          |
+---------------|-------------------------+
                |
                v
+-----------------------------------------+
|         RESULTS + TELEMETRY             |
|  Code | Tests | Docs | Metrics          |
+-----------------------------------------+
```

See [docs/CODEBASE_MINDMAP.md](docs/CODEBASE_MINDMAP.md) for the complete architecture.

---

## Service Architecture (Phase 6)

The engine uses a service-oriented architecture with injected callbacks and observability:

```python
from orchestrator.services.executor import ExecutorService
from orchestrator.services.evaluator import EvaluatorService
from orchestrator.services.generator import GeneratorService

# Wired in Orchestrator.__init__ with tracer + telemetry injection
executor = ExecutorService(execute_fn=..., guard=..., tracer=..., telemetry=...)
evaluator = EvaluatorService(client=..., budget=..., tracer=..., telemetry=...)
generator = GeneratorService(decompose_fn=..., tracer=...)
```

- **ExecutorService**: Task execution with timing, error normalization, tracer spans
- **EvaluatorService**: 2-pass self-consistency evaluation with score parsing
- **GeneratorService**: Project decomposition with guard wrapping and tracing

All services accept an optional `ResiliencePolicy` for retry, fallback, and circuit-breaker behavior.

---

## Resilience & Observability (Phase 6+)

**Per-Model Circuit Breaker Isolation:**
```python
from orchestrator.circuit_breaker import CircuitBreakerRegistry
from orchestrator.resilience import run_with_resilience

registry = CircuitBreakerRegistry(failure_threshold=3, reset_timeout=60.0)
result = await run_with_resilience(
    callables=[primary_fn, fallback_fn],
    policy=ResiliencePolicy(retries=3, timeout=30.0),
    model_ids=["gpt-4o", "claude-3.5-sonnet"],
    registry=registry,  # Tripped models are auto-skipped
)
```

**Observability Metrics (per model):**
```python
from orchestrator.services.observability import ObservabilityService

obs = ObservabilityService()
await obs.record_call("gpt-4o", latency_ms=250.0, cost_usd=0.01, success=True)

summary = obs.model_summary("gpt-4o")
print(f"Error rate: {summary.error_rate}, Total cost: {obs.total_cost_usd()}")
```

**Cost-Tier Cascade:**
```python
from orchestrator.resilience import CascadePolicy, classify_model_tier

policy = CascadePolicy.for_model(Model.GPT_4O)
# Generates fallback chain: FREE → BUDGET → PREMIUM
rp = policy.to_policy()  # Returns ResiliencePolicy with ordered models
```

---

## Port-Based Architecture (Phase 7)

Hexagonal architecture with Protocol-based dependency injection:

```python
from orchestrator.ports import CachePort, StatePort, NullCache, NullState

# Test with null adapters (no SQLite, no I/O):
orch = Orchestrator(cache=NullCache(), state_manager=NullState())

# Production with real adapters:
from orchestrator.cache import DiskCache
from orchestrator.state import StateManager
orch = Orchestrator(cache=DiskCache(), state_manager=StateManager())
```

Port implementations are automatically validated via structural subtyping.

---

## OpenRouter Optimizations

| Optimization | Impact | Environment Variable |
|--------------|--------|----------------------|
| **JSON Schema** | -50% parsing errors | `USE_JSON_SCHEMA_RESPONSES=true` |
| **Model Variants** | Cost savings with `:free` tier | `USE_MODEL_VARIANTS=true` |
| **Native Fallbacks** | +30% fallback success | `USE_NATIVE_FALLBACKS=true` |
| **Provider Sorting** | +40% throughput | `USE_PROVIDER_SORTING=true` |

### Canary Deployment

```python
from orchestrator.canary_deployment import get_canary_deployment

canary = get_canary_deployment()
await canary.start_rollout(
    optimization="json_schema",
    stages=[0.01, 0.05, 0.10, 0.25, 0.50, 1.0],
    thresholds={"max_error_rate": 0.05}
)
```

See [OPENROUTER_IMPLEMENTATION_COMPLETE.md](OPENROUTER_IMPLEMENTATION_COMPLETE.md) for details.

---

## Testing

```bash
# Run all tests (124 tests, ~22s)
pytest

# Run without coverage (faster)
pytest --no-cov

# With coverage report
pytest --cov=orchestrator --cov-report=html

# Specific test suites
pytest tests/integration/ -v
pytest tests/smoke/ -v
pytest tests/test_circuit_breaker.py -v
```

### MVOS Audit

Verify operational readiness after deployment:

```bash
python scripts/mvos_audit.py --verbose
```

Checks 6 critical invariants: CLI health, run project, API execution, state resume, persistence latency, and circuit breaker behavior.

---

## Dashboard

Launch the web UI for visual project management:

```bash
# Windows
start-ide.bat

# Direct Python
python -c "from orchestrator.dashboard_live import run_live_dashboard; run_live_dashboard()"
```

**Features:**
- Create projects from natural language
- Refactor existing codebases
- Upload YAML/JSON specifications
- Real-time progress and logs
- Multiple concurrent projects

**URL:** http://localhost:8888

---

## Documentation

| Document | Description |
|----------|-------------|
| [USAGE_GUIDE.md](USAGE_GUIDE.md) | Comprehensive usage examples |
| [CAPABILITIES.md](CAPABILITIES.md) | Feature capabilities matrix |
| [METHODS.md](METHODS.md) | API reference and methods |
| [DESIGN.md](DESIGN.md) | Architecture and design decisions |
| [AGENTS.md](AGENTS.md) | AI agent guidelines |
| [docs/CODEBASE_MINDMAP.md](docs/CODEBASE_MINDMAP.md) | Complete architecture mindmap |
| [docs/MVOS_CHECKLIST.md](docs/MVOS_CHECKLIST.md) | MVOS audit runbook |

---

## Contributing

We welcome contributions. Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Install dev dependencies
pip install -e ".[dev,security,tracing]"

# Run linting
ruff check orchestrator/
black orchestrator/ tests/
mypy orchestrator/

# Run security scan
bandit -r orchestrator/
```

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

**Author:** Georgios-Chrysovalantis Chatzivantsidis  
**Version:** 6.0.0  
**Last Updated:** 2026-04-20
