<!-- From: E:\Documents\Vibe-Coding\Ai Orchestrator\AGENTS.md -->
# AGENTS.md — AI Coding Agent Guide for Multi-LLM Orchestrator

> **Author:** Georgios-Chrysovalantis Chatzivantis Chatzivantsidis
> **Version:** v6.3 (2026-04-27)
> **Language:** English (project uses English for all documentation)

---

## 1. Project Overview

The **Multi-LLM Orchestrator** is a Python-based autonomous software development platform that decomposes project specifications into tasks, routes them to optimal LLM providers, and executes iterative generate→critique→revise→evaluate cycles.

### Key Capabilities

| Capability | Description |
|------------|-------------|
| **Smart Routing** | Task-aware model selection with cost optimization |
| **Multi-Provider** | Unified OpenRouter API with cross-provider fallback chains |
| **Budget Management** | Hierarchical budgets with mid-task enforcement |
| **Quality Assurance** | Deterministic validators + multi-round critique + LLM scoring |
| **Resilience** | Circuit breaker (3-strike), automatic failover, resume capability |
| **Nexus Search** | Self-hosted web search integration |
| **ARA Pipeline** | Advanced Reasoning Methods (Debate, Jury, Pre-Mortem, etc.) |
| **Service Architecture** | Extracted Executor, Evaluator, Generator, Observability services |
| **Port Interfaces** | Hexagonal DI with CachePort, StatePort, EventPort + NullAdapters |
| **MVOS Audit** | Runtime invariant verification for operational readiness |

### Technology Stack

- **Language:** Python 3.10+ (strict type hints required)
- **Async:** asyncio, aiohttp, aiosqlite
- **LLM Integration:** OpenRouter (unified API for 300+ models)
- **Data Validation:** Pydantic v2
- **Testing:** pytest, pytest-asyncio, pytest-cov
- **Linting:** ruff, black, mypy
- **Security:** bandit, safety, pip-audit, gitleaks, trufflehog

---

## 2. Project Structure

```
├── orchestrator/              # Main package (~351 Python files)
│   ├── __init__.py           # Lazy-loading entry point, version 6.0.0
│   ├── __main__.py           # Module execution entry point
│   ├── cli.py                # CLI entry point (main command)
│   ├── cli_dashboard.py      # Dashboard CLI
│   ├── cli_nash.py           # Nash stability CLI
│   ├── cli_website.py        # Website generator CLI
│   ├── engine.py             # Core orchestration engine (~5,100 lines)
│   ├── models.py             # Core data models, enums, routing tables, cost tables
│   ├── budget.py             # Async budget tracking with atomic reserve pattern
│   ├── exceptions.py         # Exception hierarchy (ApplicationError base)
│   ├── api_clients.py        # Unified LLM client via OpenRouter
│   ├── model_selector.py     # Intelligent model routing
│   ├── model_registry.py     # Model registry and metadata
│   ├── validators.py         # Deterministic validation (syntax, tests, etc.)
│   ├── state.py              # Project state persistence (SQLite)
│   ├── cache.py              # Disk-based caching
│   ├── semantic_cache.py     # Semantic similarity caching
│   ├── services/             # Application-layer service modules
│   │   ├── executor.py       # Task execution service
│   │   ├── evaluator.py      # 2-pass self-consistency evaluation
│   │   ├── generator.py      # Project decomposition service
│   │   └── observability.py  # Per-model latency/cost/error-rate tracker
│   ├── cost_optimization/    # Cost optimization modules
│   │   ├── batch_client.py
│   │   ├── model_cascading.py
│   │   ├── prompt_cache.py
│   │   ├── speculative_gen.py
│   │   └── ...
│   ├── nexus_search/         # Web search integration
│   │   ├── core.py
│   │   ├── models.py
│   │   ├── nexus_client.py
│   │   ├── server_manager.py
│   │   └── config.py
│   ├── engine_core/          # Core engine components
│   │   ├── core.py
│   │   ├── task_executor.py
│   │   ├── critique_cycle.py
│   │   ├── fallback_handler.py
│   │   ├── budget_enforcer.py
│   │   └── dependency_resolver.py
│   ├── dashboard_core/       # Unified dashboard components
│   │   ├── core.py
│   │   └── mission_control.py
│   ├── unified_events/       # Event bus system
│   │   └── core.py
│   ├── scaffold/             # Project scaffolding templates
│   │   ├── cli.py
│   │   ├── fastapi.py
│   │   ├── nextjs.py
│   │   ├── react_vite.py
│   │   └── ...
│   ├── plugins/              # Plugin system base classes
│   │   ├── base.py
│   │   ├── cost_optimization.py
│   │   └── nash_stability.py
│   └── ide_backend/          # IDE backend integration
│       ├── server.py
│       ├── ide_orchestrator_server.py
│       ├── websocket_manager.py
│       ├── session_manager.py
│       └── ...
├── tests/                    # Test suite (19 Python files)
│   ├── integration/          # Integration tests with shared conftest.py
│   │   ├── conftest.py       # Shared fixtures (orchestrator_fixture, mock tasks)
│   │   ├── test_full_run.py
│   │   ├── test_resume_after_crash.py
│   │   └── test_circuit_breaker_fail_fast.py
│   ├── smoke/                # Smoke tests (CLI contracts, API health)
│   │   ├── test_cli.py
│   │   └── test_api_contracts.py
│   └── test_*.py             # Unit tests (circuit breaker, concurrency,
│                               services, resilience, ports, MVOS, etc.)
├── docs/                     # Documentation
│   ├── CODEBASE_MINDMAP.md   # Complete architecture mindmap
│   ├── ARCHITECTURAL_AUDIT_V5.md
│   ├── MVOS_CHECKLIST.md     # MVOS audit runbook
│   └── README.md
├── scripts/                  # Utility scripts
│   ├── analyze_models_detail.py
│   ├── audit_openrouter_models.py
│   ├── fetch_openrouter_models.py
│   ├── validate_models.py
│   └── mvos_audit.py
├── pyproject.toml            # Package configuration (hatchling build)
├── requirements.txt          # Production dependencies (pinned)
├── requirements-dev.txt      # Development dependencies (pinned)
├── .env.example              # Environment variables template
├── .github/workflows/ci.yml  # GitHub Actions CI
└── .gitignore                # Comprehensive ignore rules
```

### Entry Points

| Entry Point | Command | Description |
|-------------|---------|-------------|
| CLI | `python -m orchestrator --project "..." --budget 5.0` | Main orchestrator CLI |
| Console Script | `orchestrator` or `mllm` | Installed console script alias |
| Dashboard | `dashboard` | Launch dashboard CLI |
| Dashboard (Python) | `python -c "from orchestrator.dashboard_live import run_live_dashboard; run_live_dashboard()"` | Direct Python execution |
| IDE Backend | `python -m orchestrator.ide_backend.server` | IDE backend server |

---

## 3. Build and Development Commands

### Installation

```bash
# Development install (recommended)
pip install -e ".[dev,security,tracing,dashboard,docs]"

# Or using requirements files
pip install -r requirements-dev.txt
```

### Code Quality (configured in pyproject.toml)

| Command | Purpose | Configuration |
|---------|---------|---------------|
| `black orchestrator/ tests/` | Code formatting | Line length: 100, target: py310+ |
| `ruff check orchestrator/` | Fast linting | See pyproject.toml [tool.ruff] |
| `ruff check --fix orchestrator/` | Auto-fix issues | Applies safe fixes |
| `mypy orchestrator/` | Type checking | Strict mode enabled |
| `bandit -r orchestrator/` | Security scan | Skips B101 (assert warnings) |
| `safety check` | Dependency vulnerabilities | - |
| `pip-audit` | Python package vulnerability scanning | - |
| `gitleaks detect` | Secret detection in git | - |
| `trufflehog git file://.` | Deep secret scanning | - |

### Testing

```bash
# Run all tests with coverage (baseline: ~12% coverage)
pytest

# Run specific test file
pytest tests/test_circuit_breaker.py -v

# Run without coverage (faster)
pytest --no-cov

# Run only unit tests (exclude slow/integration)
pytest -m "not slow and not integration"

# Run with parallel execution
pytest -x

# Run specific test pattern
pytest -k "test_model" -v
```

**Note:** Pytest configuration is centralized in `pyproject.toml` under `[tool.pytest.ini_options]`. The `tests/integration/conftest.py` provides shared fixtures for integration tests.

---

## 4. Code Style Guidelines

### Python Style

- **Line Length:** 100 characters (enforced by black and ruff)
- **Target Python:** 3.10, 3.11, 3.12, 3.13
- **Import Style:** Absolute imports preferred (relative imports exist in codebase)
- **Type Hints:** Strict typing required (mypy strict mode)
- **Docstrings:** Google convention (configured in ruff)

### File Header Template

All Python files should include author attribution:

```python
"""
Module Name — Brief Description
================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Longer description if needed.
"""
```

### Import Organization

```python
# 1. Standard library
import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

# 2. Third-party
from pydantic import BaseModel

# 3. First-party (orchestrator)
from orchestrator.models import Model, Task
from orchestrator.budget import Budget
```

### Type Hints (Strict)

```python
# Required everywhere
def process_task(task: Task, budget: Budget | None = None) -> TaskResult:
    ...

# Use TYPE_CHECKING for circular imports
if TYPE_CHECKING:
    from orchestrator.engine import Orchestrator
```

### Error Handling

Use the custom exception hierarchy in `orchestrator/exceptions.py`:

```python
from orchestrator.exceptions import ModelUnavailableError, TaskTimeoutError, BudgetExceededError

try:
    result = await execute_task(task)
except ModelUnavailableError as e:
    # Retry with fallback
    logger.warning(f"Model {e.details['model']} unavailable")
except BudgetExceededError:
    # Non-retriable, propagate
    raise
```

### Async Patterns

- All I/O operations must be async
- Use `asyncio.gather()` for parallel operations
- StateManager methods are async (must be awaited)
- All StateManager calls must be awaited

---

## 5. Testing Instructions

### Test Markers (defined in pyproject.toml)

| Marker | Description |
|--------|-------------|
| `@pytest.mark.slow` | Slow tests (deselect with `-m "not slow"`) |
| `@pytest.mark.integration` | Integration tests |
| `@pytest.mark.unit` | Unit tests |
| `@pytest.mark.requires_api` | Tests requiring API keys |
| `@pytest.mark.e2e` | End-to-end tests |
| `@pytest.mark.load` | Load tests |
| `@pytest.mark.stress` | Stress tests |
| `@pytest.mark.benchmark` | Benchmark tests |
| `@pytest.mark.mock` | Tests with mocked dependencies |

### Writing Tests

```python
# tests/test_feature.py
import pytest
from orchestrator.models import Model, Task, TaskType

@pytest.mark.unit
async def test_task_creation():
    task = Task(description="Test", task_type=TaskType.CODE_GEN)
    assert task.status == TaskStatus.PENDING

@pytest.mark.requires_api
async def test_api_call():
    # Only runs when API keys available
    ...
```

### Integration Test Fixtures

The `tests/integration/conftest.py` provides:

- `temp_state_manager` — StateManager backed by temporary SQLite DB
- `mock_task` / `mock_tasks` — Generic code-gen tasks
- `ok_result_t1` / `ok_result_t2` — Sample TaskResult objects
- `orchestrator_fixture` — Orchestrator with tiny budget, temp state, mocked cache

---

## 6. Security Considerations

### API Keys and Secrets

- **NEVER commit `.env` files** — they are in `.gitignore`
- Copy `.env.example` to `.env` and fill in your keys
- Use environment variables in production (secrets managers preferred)

Required provider keys (at least one):

```bash
export OPENAI_API_KEY="sk-..."
export DEEPSEEK_API_KEY="sk-..."
export GOOGLE_API_KEY="AIzaSy..."
export ANTHROPIC_API_KEY="sk-ant-..."
export MISTRAL_API_KEY="..."
export XAI_API_KEY="..."
export ALIBABA_API_KEY="..."
export ZHIPU_API_KEY="..."
export MOONSHOT_API_KEY="..."
export MINIMAX_API_KEY="..."
```

### Input Validation

All user inputs must be validated to prevent path traversal:

```python
from pathlib import Path
from orchestrator.secure_execution import InputValidator

# Validate paths
base_path = Path.cwd()
path = (base_path / user_input).resolve()
try:
    path.relative_to(base_path)
except ValueError:
    raise SecurityError("Path traversal detected")
```

### Security Tools

| Tool | Purpose |
|------|---------|
| bandit | Static security analysis |
| safety | Dependency vulnerability scanning |
| pip-audit | Python package vulnerability scanning |
| gitleaks | Secret detection in git history |
| trufflehog | Deep secret scanning |

### Runtime Security

- Plugin sandboxing: `PLUGIN_SANDBOX_ENABLED=true`
- Audit logging: `AUDIT_LOG_ENABLED=true`
- Rate limiting: `RATE_LIMIT_PER_MINUTE=60`

---

## 7. Configuration

### Environment Variables (`.env`)

```bash
# Required: At least one LLM provider
OPENAI_API_KEY=sk-...
DEEPSEEK_API_KEY=sk-...
GOOGLE_API_KEY=...
ANTHROPIC_API_KEY=...

# Optional: Other providers
MISTRAL_API_KEY=...
XAI_API_KEY=...
ALIBABA_API_KEY=...
ZHIPU_API_KEY=...
MINIMAX_API_KEY=...
MOONSHOT_API_KEY=...

# Optional: OpenTelemetry tracing
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
OTEL_SERVICE_NAME=ai-orchestrator

# Optional: Orchestrator settings
ORCH_MAX_CONCURRENCY=3
ORCH_MAX_PARALLEL_TASKS=3
ORCH_DEFAULT_BUDGET_USD=10.0
ORCH_DEFAULT_TIMEOUT_SECONDS=10800

# Optional: Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Optional: Cache settings
CACHE_TTL_HOURS=48
SEMANTIC_CACHE_THRESHOLD=0.85

# Optional: Dashboard
DASHBOARD_PORT=8000
DASHBOARD_HOST=127.0.0.1

# Optional: MCP Server
MCP_HTTP_MODE=false
MCP_PORT=8181
MCP_HOST=0.0.0.0

# Optional: Security
PLUGIN_SANDBOX_ENABLED=true
AUDIT_LOG_ENABLED=true
AUDIT_LOG_PATH=~/.orchestrator_cache/audit.log
RATE_LIMIT_PER_MINUTE=60

# Optional: OpenRouter optimizations
USE_JSON_SCHEMA_RESPONSES=true
USE_MODEL_VARIANTS=true
USE_NATIVE_FALLBACKS=true
USE_PROVIDER_SORTING=true
```

### Key Configuration Files

| File | Purpose |
|------|---------|
| `pyproject.toml` | Package metadata, tool configs (black, ruff, mypy, pytest, coverage, bandit) |
| `.env` | Runtime environment variables (gitignored) |
| `.env.example` | Template for `.env` |
| `.gitignore` | Prevents committing secrets, cache, outputs |
| `orchestrator/models.py` | Routing tables, cost tables, model definitions |
| `orchestrator/config.py` | OpenRouter optimization configuration |

### Model Routing (in `models.py`)

- `ROUTING_TABLE`: Maps TaskType → preferred Model
- `COST_TABLE`: Per-token pricing for cost optimization
- `FALLBACK_CHAIN`: Cross-provider fallback sequences
- `MODEL_MAX_TOKENS`: Context window limits

---

## 8. Architecture Overview

### Core Pipeline

```
Project Description
       |
       ▼
Auto-Resume Detect → Project Enhancer → Architecture Advisor
       |
       ▼
Decompose into Tasks
       |
       ▼
Route → Generate → Critique → Revise → Evaluate
       |
       ▼
Cross-Provider Fallback Chain (quality escalation)
       |
       ▼
Deterministic Validation (python_syntax, pytest, ruff, json_schema)
       |
       ▼
Store Results + Telemetry + State Checkpoint (SQLite)
```

### Key Components

| Module | Responsibility |
|--------|---------------|
| `engine.py` | Main orchestration loop, task execution (~5,100 lines) |
| `model_selector.py` | Intelligent provider selection |
| `api_clients.py` | Unified OpenRouter client |
| `validators.py` | Deterministic output validation |
| `state.py` | SQLite-based state persistence |
| `budget.py` | Budget tracking and enforcement |
| `semantic_cache.py` | Pattern-based result caching |
| `policy_engine.py` | Policy enforcement (HARD/SOFT/MONITOR) |
| `nexus_search/` | Web search integration |
| `ara_pipelines.py` | Advanced Reasoning Methods |
| `ios_hig_prompts.py` | iOS HIG compliance prompts |
| `canary_deployment.py` | Canary rollout for optimizations |

### Service Architecture (Phase 6+)

Extracted services from the engine:

- **ExecutorService** — Task execution with timing, error normalization, tracer spans
- **EvaluatorService** — 2-pass self-consistency evaluation with score parsing
- **GeneratorService** — Project decomposition with guard wrapping and tracing
- **ObservabilityService** — Per-model metrics: latency, error rate, cost tracking

### Port-Based Architecture (Phase 7)

Hexagonal architecture with Protocol-based dependency injection:

```python
from orchestrator.ports import CachePort, StatePort, NullCache, NullState

# Test with null adapters (no SQLite, no I/O):
orch = Orchestrator(cache=NullCache(), state_manager=NullState())
```

### Async Architecture

- All StateManager methods are async
- Task execution uses `asyncio.gather()` for parallelism
- Circuit breaker tracks model health asynchronously
- Budget checks happen mid-task (not just pre-task)

### Subpackages

| Subpackage | Purpose |
|------------|---------|
| `services/` | Executor, Evaluator, Generator, Observability services |
| `cost_optimization/` | Batch client, model cascading, prompt caching, speculative generation |
| `nexus_search/` | Web search client, models, server manager |
| `engine_core/` | Core execution, critique cycle, fallback handling, budget enforcement |
| `dashboard_core/` | Unified dashboard components, mission control |
| `unified_events/` | Event bus system for decoupled communication |
| `scaffold/` | Project templates (FastAPI, Next.js, React, CLI, etc.) |
| `plugins/` | Plugin system base classes and implementations |
| `ide_backend/` | IDE integration server, WebSocket handlers, session manager |

---

## 9. Common Development Tasks

### Adding a New Model

1. Add to `Model` enum in `orchestrator/models.py`
2. Add pricing to `COST_TABLE`
3. Add to `ROUTING_TABLE` if it has specific strengths
4. Update `FALLBACK_CHAIN` for cross-provider resilience
5. Add context window to `MODEL_MAX_TOKENS`

### Adding a New Validator

1. Create validator function in `orchestrator/validators.py`
2. Add to `all_validators_pass()` or `async_run_validators()`
3. Add tests in `tests/test_validators.py` (or create it)

### Adding a New Test

1. Create `tests/test_feature.py`
2. Use appropriate markers: `@pytest.mark.unit`, `@pytest.mark.integration`
3. Import from `orchestrator` package (not relative)
4. Run with `pytest tests/test_feature.py -v`

### Adding a New Scaffold Template

1. Create template in `orchestrator/scaffold/templates/`
2. Inherit from base template class
3. Register in `orchestrator/scaffold/__init__.py`

### Debugging Tips

- Use `LOG_LEVEL=DEBUG` for verbose logs
- Check `orchestrator.log` for execution traces
- Use `python -m orchestrator --dry-run` to preview without execution
- Use `--resume <project_id>` to resume interrupted projects
- Check SQLite state database for persisted state

---

## 10. Important Notes for AI Agents

### Circular Import Warnings

- The codebase has some circular import challenges
- Use `TYPE_CHECKING` for type-only imports
- The `__init__.py` uses lazy-loading (`__getattr__`) to prevent issues
- Some modules are wrapped in try/except in `engine.py` to allow CLI to load
- When adding new imports, test with `python -c "from orchestrator import Orchestrator"`

### Pre-existing Code Style

- Many files have long lines (enforced limit: 100, but historical code may exceed)
- ruff ignores many pre-existing violations (see pyproject.toml ignore list)
- Focus on new code quality; don't mass-reformat existing files
- There are many pre-existing unused import warnings (F401) and other baseline violations — ignore these
- MyPy strict mode is enabled, but many legacy modules have `ignore_errors = true` in pyproject.toml overrides

### Testing Reality

- Current coverage baseline: ~12% (target: 50%)
- Pytest config in pyproject.toml includes `--ignore` flags for files that may not exist
- Run `pytest --collect-only` to verify test discovery

### Documentation

- Project uses Markdown for all documentation
- English is the primary language
- Author attribution required on all files
- Architecture documentation in `docs/CODEBASE_MINDMAP.md`

### Version Information

- Current version: 6.0.0 (in `orchestrator/__init__.py`)
- Build system: hatchling
- Version is dynamically extracted from `__init__.py`

---

## 11. Quick Reference

```bash
# Install
pip install -e ".[dev,security,tracing]"

# Format & Lint
black orchestrator/ tests/
ruff check --fix orchestrator/
mypy orchestrator/

# Security Scan
bandit -r orchestrator/
safety check
gitleaks detect

# Test
pytest                              # Full suite
pytest -m "not slow"                # Exclude slow tests
pytest tests/test_circuit_breaker.py -v  # Specific file
pytest -k "test_name_pattern"       # Pattern match

# Run CLI
python -m orchestrator --project "Build API" --budget 5.0
python -m orchestrator --resume <project_id>

# Run Dashboard
python -c "from orchestrator.dashboard_live import run_live_dashboard; run_live_dashboard()"

# IDE Backend
python -m orchestrator.ide_backend.server

# MVOS Audit
python scripts/mvos_audit.py --verbose

# Check State
python -c "from orchestrator.state import StateManager; import asyncio; sm = StateManager(); asyncio.run(sm.list_projects())"
```

---

**Last Updated:** 2026-04-27
**Maintainer:** Georgios-Chrysovalantis Chatzivantsidis
