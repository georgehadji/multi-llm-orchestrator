# Multi-LLM Orchestrator - AI Agent Guide

> **Note:** This file is intended for AI coding agents. For human contributors, see [README.md](./README.md) and [docs/PROJECT_STRUCTURE.md](./docs/PROJECT_STRUCTURE.md).

---

## Project Overview

**Multi-LLM Orchestrator** is a Python-based autonomous project completion system that decomposes project specifications into tasks, routes them to optimal LLM providers, and executes generate→critique→revise cycles with quality evaluation.

**Current Version:** 6.0.0  
**Python Requirement:** >=3.10  
**License:** MIT

### Key Capabilities
- Multi-provider LLM routing (6+ providers: OpenAI, DeepSeek, Google, Anthropic, MiniMax, Mistral, etc.)
- Cost-optimized routing (~35% cost reduction)
- Budget hierarchy with resume capability
- Deterministic validation + multi-round critique
- Policy enforcement (HARD/SOFT/MONITOR modes)
- Real-time telemetry and observability
- Web-based Mission Control dashboard

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

### Module Organization

```
orchestrator/
├── __init__.py              # Package exports (687 lines, comprehensive API)
├── __main__.py              # CLI entry point: python -m orchestrator
├── cli.py                   # Command-line interface with argparse
├── cli_dashboard.py         # Dashboard CLI commands
├── cli_nash.py             # Nash stability CLI commands
│
# ═══════════════════════════════════════════════════════════════════════════════
# CORE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
├── engine.py               # Main orchestration loop (generate→critique→revise→evaluate)
├── models.py               # Data models: Task, ProjectState, Budget, Model enum, routing tables
├── state.py                # SQLite-based state persistence with WAL
├── cache.py                # Disk-based response caching
├── semantic_cache.py       # Pattern-based semantic caching (L3)
├── cache_optimizer.py      # Multi-level cache (L1 memory, L2 disk, L3 semantic)
│
# ═══════════════════════════════════════════════════════════════════════════════
# API & ROUTING
# ═══════════════════════════════════════════════════════════════════════════════
├── api_clients.py          # Unified client for all LLM providers
├── adaptive_router.py      # Smart routing with health tracking and circuit breaker
├── outcome_router.py       # Outcome-weighted routing
├── routing.py              # Static routing tables (ROUTING_TABLE, FALLBACK_CHAIN)
│
# ═══════════════════════════════════════════════════════════════════════════════
# TASK MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════
├── planner.py              # Constraint-based task planning
├── policy.py               # Policy definitions (ModelProfile, PolicySet, JobSpec)
├── policy_engine.py        # Policy enforcement engine
├── policy_dsl.py           # Policy domain-specific language
├── validators.py           # Deterministic output validation
├── dep_resolver.py         # Dependency resolution for tasks
│
# ═══════════════════════════════════════════════════════════════════════════════
# QUALITY & ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
├── project_analyzer.py     # Post-project analysis and insights
├── improvement_suggester.py # AI-powered improvement suggestions
├── codebase_analyzer.py    # Multi-LLM codebase understanding
├── codebase_reader.py      # Secure file reading with path traversal protection
├── analyzer.py             # Analysis orchestration
├── aggregator.py           # Profile aggregation across runs
│
# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT & ORGANIZATION
# ═══════════════════════════════════════════════════════════════════════════════
├── output_writer.py        # Write task outputs to filesystem
├── output_organizer.py     # Auto-organize output, generate/run tests
├── progress_writer.py      # Progressive output during execution
├── output_writer_trimmed.py # Optimized output writer
├── assembler.py            # Project assembly from tasks
├── project_assembler.py    # Enhanced project assembly
│
# ═══════════════════════════════════════════════════════════════════════════════
# DASHBOARD SYSTEMS (Legacy + v6.0 Unified)
# ═══════════════════════════════════════════════════════════════════════════════
├── dashboard_core/         # v6.0 unified dashboard (NEW)
│   ├── __init__.py
│   ├── core.py
│   └── mission_control.py
├── dashboard.py            # Legacy v1.0
├── dashboard_real.py       # Real-time v2.0
├── dashboard_optimized.py  # Performance-optimized
├── dashboard_enhanced.py   # Enhanced v2.0
├── dashboard_antd.py       # Ant Design v3.0
├── dashboard_live.py       # Gamified LIVE v4.0 (WebSocket)
├── dashboard_mission_control.py  # Mission Control v5.0
├── unified_dashboard.py    # Unified dashboard attempts
│
# ═══════════════════════════════════════════════════════════════════════════════
# ARCHITECTURE & RULES
# ═══════════════════════════════════════════════════════════════════════════════
├── architecture_rules.py       # Rules engine with constraints
├── architecture_rules_fixed.py # Bug-fixed version
├── architecture_advisor.py     # Architecture recommendations
├── frontend_rules.py           # Frontend-specific rules
├── wordpress_plugin_rules.py   # WordPress plugin guidelines
├── indesign_plugin_rules.py    # InDesign plugin guidelines
│
# ═══════════════════════════════════════════════════════════════════════════════
# MANAGEMENT SYSTEMS (v5.1+)
# ═══════════════════════════════════════════════════════════════════════════════
├── knowledge_base.py       # Knowledge management system
├── knowledge_graph.py      # Performance knowledge graph (Nash)
├── project_manager.py      # Project management (critical path, scheduling)
├── product_manager.py      # Product management (RICE scoring, features)
├── quality_control.py      # Quality management (multi-level testing)
│
# ═══════════════════════════════════════════════════════════════════════════════
# NASH STABILITY FEATURES (v6.1 - Strategic Competitive Moat)
# ═══════════════════════════════════════════════════════════════════════════════
├── nash_stable_orchestrator.py   # Nash-stable orchestrator integration
├── nash_events.py                # Nash event system
├── nash_backup.py                # Backup/restore system
├── nash_auto_tuning.py           # Auto-tuning with multi-armed bandit
├── nash_monitor.py               # Stability monitoring
├── nash_infrastructure_v2.py     # Infrastructure resilience
├── pareto_frontier.py            # Cost-quality frontier optimization
├── federated_learning.py         # Cross-org federated learning
├── adaptive_templates.py         # Adaptive prompt templates
│
# ═══════════════════════════════════════════════════════════════════════════════
# TELEMETRY & OBSERVABILITY
# ═══════════════════════════════════════════════════════════════════════════════
├── telemetry.py            # Real-time telemetry collection
├── telemetry_store.py      # Persistent telemetry with WAL
├── metrics.py              # Metrics export (Console, JSON, Prometheus)
├── monitoring.py           # KPI monitoring with alerts
├── tracing.py              # OpenTelemetry tracing integration
├── audit.py                # Immutable audit logging
│
# ═══════════════════════════════════════════════════════════════════════════════
# EVENT SYSTEMS (Legacy + v6.0 Unified)
# ═══════════════════════════════════════════════════════════════════════════════
├── unified_events/         # v6.0 unified event bus (NEW)
│   ├── __init__.py
│   └── core.py
├── events.py               # Legacy event system
├── events_resilient.py     # Resilient event store with WAL
├── streaming.py            # Streaming event bus
├── streaming_resilient.py  # Resilient streaming
├── hooks.py                # Event hooks registry
│
# ═══════════════════════════════════════════════════════════════════════════════
# RESILIENCE & SECURITY
# ═══════════════════════════════════════════════════════════════════════════════
├── integration_circuit_breaker.py  # Circuit breaker pattern
├── plugin_isolation.py             # Plugin process isolation
├── plugin_isolation_secure.py      # Secure plugin runtime (seccomp/Landlock)
├── secure_execution.py             # Secure execution context
├── reference_monitor.py            # Reference monitor pattern
├── remediation.py                  # Auto-remediation engine
├── sagas.py                        # Saga pattern for distributed transactions
│
# ═══════════════════════════════════════════════════════════════════════════════
# AGENTS & ORCHESTRATION
# ═══════════════════════════════════════════════════════════════════════════════
├── agents.py               # Agent pool and task channels
├── orchestration_agent.py  # Orchestration agent implementation
├── control_plane.py        # Control plane for job specs
├── agent_orchestration.py  # Agent orchestration patterns
│
# ═══════════════════════════════════════════════════════════════════════════════
# SCAFFOLDING & APP BUILDING
# ═══════════════════════════════════════════════════════════════════════════════
├── scaffold/               # Project scaffolding templates
│   ├── __init__.py
│   └── templates/
│       ├── __init__.py
│       ├── cli.py
│       ├── fastapi.py
│       ├── generic.py
│       ├── html.py
│       ├── library.py
│       ├── nextjs.py
│       └── react_vite.py
├── app_builder.py          # Application builder
├── app_assembler.py        # App assembly
├── app_detector.py         # App type detection
├── app_verifier.py         # App verification
│
# ═══════════════════════════════════════════════════════════════════════════════
# OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════════
├── optimization.py         # Optimization backends (Greedy, Pareto)
├── cost.py                 # Cost prediction and forecasting
├── performance.py          # Performance optimization
│
# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY MODULES
# ═══════════════════════════════════════════════════════════════════════════════
├── config.py               # Pydantic-based configuration
├── exceptions.py           # Exception hierarchy
├── logging.py              # Structured logging with correlation IDs
├── log_config.py           # Logging configuration
├── compat.py               # Backward compatibility layer
├── resume_detector.py      # Auto-resume project detection
├── enhancer.py             # Project description enhancement
├── specs.py                # Specification handling
├── project_file.py         # Project file I/O
├── pricing_cache.py        # Pricing cache with EMA tracking
├── slash_commands.py       # Slack-style slash commands
├── git_integration.py      # Git auto-commit integration
├── git_service.py          # Git service abstraction
├── secrets_manager.py      # Secrets management
├── visualization.py        # DAG visualization
├── leaderboard.py          # Model performance leaderboard
├── health.py               # Health checks
├── diagnostics.py          # Diagnostic tools
├── quick_self_test.py      # Quick self-test on startup
├── dry_run.py              # Dry-run mode
├── brain.py                # AI reasoning and cognitive layer
├── evaluation.py           # LLM-based evaluation scoring
├── escalation.py           # Automatic escalation to higher-capability models
├── checkpoints.py          # Intermediate state checkpoints
├── modes.py                # Per-request behavioral modes
├── prompt_enhancer.py      # Prompt enhancement and optimization
├── cost_analytics.py       # Cost analytics and forecasting
├── competitive.py          # Competitive routing intelligence
├── tracing.py              # OpenTelemetry tracing integration
├── plan_then_build.py      # Plan-first, then execute pattern
├── memory_bank.py          # Persistent cross-run memory
├── context_condensing.py   # Context compression for long runs
├── hierarchy.py            # Multi-level org/team hierarchy
├── triggers.py             # Event-driven triggers
├── workspace.py            # Workspace isolation
├── gateway.py              # API gateway functionality
├── connectors.py           # External system connectors
├── sandbox.py              # Secure code execution sandbox
├── context_sources.py      # Multiple context sources
├── api_server.py           # REST API server
├── skills.py               # Claude skills system
├── drift.py                # Drift detection
├── browser_testing.py      # Browser-based testing
├── token_optimizer.py      # Command-specific token compression
├── a2a_protocol.py         # A2A external agent client
├── persona_modes.py        # Persona-based behavioral modes
├── learning_aggregator.py  # Persistent cross-run learning
├── multi_tenant_gateway.py # Multi-tenant API gateway
└── ... (additional utilities)
```

---

## Technology Stack

### Core Dependencies
```toml
[project.dependencies]
openai>=1.30          # OpenAI API client
google-genai>=1.0     # Google Gemini client
aiosqlite>=0.19       # Async SQLite
pydantic>=2.0         # Data validation
pydantic-settings>=2.0 # Settings management
typing-extensions>=4.0 # Type hints
```

### Optional Dependencies
```toml
dev = [               # Development
    pytest>=8.0,
    pytest-cov>=4.1,
    pytest-asyncio>=0.21,
    pytest-xdist>=3.3,
    black>=23.7,
    ruff>=0.1.0,
    mypy>=1.5,
    pre-commit>=3.4,
]
security = [          # Security scanning
    bandit[toml]>=1.7.0,
    safety>=2.3.0,
]
tracing = [           # OpenTelemetry
    opentelemetry-api>=1.20,
    opentelemetry-sdk>=1.20,
    opentelemetry-exporter-otlp-proto-grpc>=1.20,
]
dashboard = [         # Web dashboard
    fastapi>=0.100.0,
    uvicorn[standard]>=0.23.0,
    websockets>=11.0,
    httpx>=0.24.0,
]
```

---

## Build and Test Commands

### Setup
```bash
# Basic install
pip install -e .

# Development install (with all dev tools)
make install-dev
# Or: pip install -e ".[dev,security]"

# Full install (all optional dependencies)
make install-all
# Or: pip install -e ".[dev,security,tracing,dashboard,docs]"

# Pre-commit hooks
pre-commit install
```

### Testing
```bash
# Run all tests with coverage
make test
# Or: pytest -xvs

# Unit tests only
make test-unit
# Or: pytest tests/unit -xvs -m unit

# Integration tests only
make test-integration
# Or: pytest tests/integration -xvs -m integration --ignore=tests/integration/test_api_clients.py

# Fast parallel tests
make test-fast
# Or: pytest -x --forked -n auto

# Coverage report
make test-cov
# Or: pytest --cov=orchestrator --cov-report=html --cov-report=term
```

### Code Quality
```bash
# Format code with black
make format
# Or: black orchestrator/ tests/

# Check formatting
make format-check
# Or: black --check orchestrator/ tests/

# Lint with ruff
make lint
# Or: ruff check orchestrator/ tests/

# Auto-fix lint issues
make lint-fix
# Or: ruff check --fix orchestrator/ tests/

# Type check with mypy
make type-check
# Or: mypy orchestrator/

# Security scan
make security-check
# Runs bandit and safety

# Run all CI checks
make ci
# Runs: format-check, lint, type-check, test-ci, security-check
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

## Code Style Guidelines

### Formatting (Black)
- Line length: **100 characters**
- Target Python versions: 3.10, 3.11, 3.12, 3.13
- Run: `black orchestrator/ tests/`

### Linting (Ruff)
- Target Python: 3.10+
- Enabled rules: E, F, I, W, UP, B, C4, SIM, TCH, TID
- Ignored: E501 (line too long - handled by black), B008, SIM105
- Max complexity: 15
- Docstring convention: Google style

### Type Checking (MyPy)
- Strict mode enabled
- `disallow_untyped_defs = true`
- `no_implicit_optional = true`
- Ignore missing imports (for external libs)
- Exclude: tests/, docs/

### Import Organization
```python
# 1. Standard library
import asyncio
import json
from typing import Optional

# 2. Third-party
import pydantic
from openai import AsyncOpenAI

# 3. First-party (orchestrator)
from .models import Task, ProjectState
from .exceptions import ModelUnavailableError
```

### File Header Template
```python
"""
Module Name — Short Description
================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Longer description of what this module does.

Key Features:
- Feature 1
- Feature 2

FIX #X: Description of bug fix
FEAT: Description of feature
"""
```

### Naming Conventions
- **Modules:** `snake_case.py`
- **Classes:** `PascalCase`
- **Functions/Methods:** `snake_case`
- **Constants:** `UPPER_SNAKE_CASE`
- **Private members:** `_leading_underscore`
- **Abstract base classes:** `Base` prefix or `ABC` suffix

### Documentation Style
- Use Google-style docstrings
- All public functions/classes must have docstrings
- Include type hints for all function signatures
- Document exceptions raised with `Raises:` section

---

## Testing Instructions

### Test Organization
```
tests/
├── __init__.py
├── test_*.py              # Individual test files (123+ files)
├── unit/                  # Unit tests (if organized)
└── integration/           # Integration tests (if organized)
```

### Test Markers
```python
# Available pytest markers:
@pytest.mark.slow          # Slow tests (skip with -m "not slow")
@pytest.mark.integration   # Integration tests
@pytest.mark.unit          # Unit tests
@pytest.mark.requires_api  # Tests requiring API keys
```

### Running Specific Tests
```bash
# Run specific test file
pytest tests/test_engine_e2e.py -v

# Run tests matching pattern
pytest -k "test_router" -v

# Run with API (requires keys)
pytest tests/test_api_clients.py -v --no-cov

# Skip slow tests
pytest -m "not slow" -v
```

### Test Coverage Requirements
- Minimum coverage: **70%**
- Coverage reports: `coverage_html/` directory
- Excluded from coverage:
  - `*/tests/*`
  - `*/test_*`
  - `orchestrator/__main__.py`
  - `orchestrator/_vendor/*`

### Writing Tests
```python
import pytest
from orchestrator.models import Task, TaskType
from orchestrator.engine import Orchestrator

@pytest.mark.unit
async def test_task_creation():
    """Test that tasks are created correctly."""
    task = Task(
        id="test_001",
        type=TaskType.CODE_GEN,
        instruction="Write a function",
    )
    assert task.id == "test_001"
    assert task.type == TaskType.CODE_GEN

@pytest.mark.integration
@pytest.mark.requires_api
async def test_api_call():
    """Test actual API call (requires API key)."""
    # Test implementation
    pass
```

---

## Configuration and Environment

### Required API Keys (at least one)
```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# DeepSeek (recommended - best cost/quality)
export DEEPSEEK_API_KEY="sk-..."

# Google Gemini
export GOOGLE_API_KEY="AIzaSy..."

# Anthropic Claude
export ANTHROPIC_API_KEY="sk-ant-..."

# MiniMax
export MINIMAX_API_KEY="..."
```

### Optional Environment Variables
```bash
# Logging
export ORCHESTRATOR_LOG_LEVEL="INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL

# Cache
export ORCHESTRATOR_CACHE_DIR="~/.orchestrator_cache"

# Tracing
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4318"

# Git Integration
export ORCHESTRATOR_GIT_ENABLED="false"
export ORCHESTRATOR_GIT_STRATEGY="manual"  # after_each_task | after_phase | after_project

# Security
export ORCHESTRATOR_SECURE_MODE="true"
export ORCHESTRATOR_MAX_UPLOAD_SIZE="10"
export ORCHESTRATOR_ALLOWED_EXTENSIONS=".py,.js,.ts,.yaml,.yml,.json,.md,.txt"
```

### Configuration File (.env)
Copy `.env.example` to `.env` and configure:
```bash
cp .env.example .env
# Edit .env with your API keys
```

---

## Security Considerations

### Input Validation
- All file paths are validated to prevent path traversal attacks
- User input is sanitized before use in shell commands
- File extensions are validated against allowlists

### API Key Handling
- API keys are NEVER logged
- Keys are loaded from environment variables or `.env` file
- Use `MissingAPIKeyError` for missing key handling

### Secure Execution
```python
from orchestrator.secure_execution import InputValidator, SecurityContext

# Validate paths
validator = InputValidator(base_path=Path.cwd())
safe_path = validator.validate_path(user_input)

# Security context
context = SecurityContext(
    allow_shell_execution=False,
    allowed_extensions={".py", ".js"},
)
```

### Running Security Scans
```bash
# Bandit (static analysis)
bandit -r orchestrator/

# Safety (dependency vulnerabilities)
safety check

# Pre-commit security hooks
detect-private-key  # Detects committed secrets
```

---

## Common Development Tasks

### Running the Orchestrator
```bash
# Basic usage
python -m orchestrator \
  --project "Build a FastAPI REST API" \
  --criteria "All endpoints tested, OpenAPI docs complete" \
  --budget 2.0

# Resume interrupted project
python -m orchestrator --resume <project_id>

# List all projects
python -m orchestrator --list-projects

# Analyze codebase
python -m orchestrator analyze ./my_project --budget 1.0
```

### Running Dashboards
```bash
# v6.0 Unified Dashboard (RECOMMENDED)
python -c "from orchestrator import run_dashboard; run_dashboard()"

# Legacy dashboards (backward compatible)
python -c "from orchestrator.dashboard_live import run_live_dashboard; run_live_dashboard()"
python -c "from orchestrator.dashboard_antd import run_ant_design_dashboard; run_ant_design_dashboard()"
```

### Organizing Output
```python
from orchestrator import organize_project_output

report = organize_project_output("./outputs/project_123")
print(f"Tasks moved: {report.tasks_moved}")
print(f"Tests generated: {report.tests_generated}")
print(f"Tests passed: {report.tests_passed}/{report.tests_total}")
```

---

## Key Architectural Patterns

### Exception Hierarchy
All exceptions inherit from `ApplicationError`:
```
ApplicationError
├── ConfigurationError
│   └── MissingAPIKeyError
├── OrchestratorError
│   ├── BudgetExceededError
│   └── TimeoutError
├── ModelError
│   ├── ModelUnavailableError
│   ├── RateLimitError
│   ├── TokenLimitError
│   └── AuthenticationError
├── TaskError
│   ├── TaskValidationError
│   ├── TaskTimeoutError
│   └── TaskRetryExhaustedError
├── PolicyError
│   └── PolicyViolationError
├── CacheError
└── StateError
```

### Circuit Breaker Pattern
Models are marked unhealthy after 3 consecutive failures:
```python
# In engine.py
_CIRCUIT_BREAKER_THRESHOLD: int = 3

# Check model health
if not self.api_health[model]:
    logger.warning(f"{model.value}: circuit breaker open")
    # Use fallback
```

### Event-Driven Architecture (v6.0)
```python
from orchestrator import get_event_bus, ProjectStartedEvent

# Get unified event bus
event_bus = get_event_bus()

# Subscribe to events
@event_bus.on(EventType.PROJECT_STARTED)
async def handle_project_start(event: ProjectStartedEvent):
    print(f"Project started: {event.project_id}")

# Emit events
event_bus.emit(ProjectStartedEvent(project_id="abc123"))
```

---

## SRE Bug-Fix Pass (v5.1 → v6.0 Hardening)

Six bugs discovered and fixed via systematic SRE audit:

| Bug | File | Root Cause | Fix |
|-----|------|------------|-----|
| BUG-001 | `cost.py` | `can_afford_job()` read stale remaining budget (didn't subtract reservations) | Deduct `_team_reserved` in `remaining(level="team")` |
| BUG-002 | `engine.py` | `asyncio.gather(return_exceptions=True)` → exceptions stored as results, never re-raised | Check `isinstance(r, Exception)` before indexing results |
| BUG-003 | `hybrid_search_pipeline.py` | RRF scoring mutated `SearchResult` objects in-place, corrupting shared state | Create new `SearchResult` copies instead of mutating |
| BUG-004 | `cost.py` | Same as BUG-001 — `_team_reserved` missing from `remaining()` calculation | One-line fix: subtract `reserved` alongside `spent` |
| BUG-005 | `rate_limiter.py` | TOCTOU race: two concurrent asyncio coroutines both passed limit check before either called `record()` | Atomic in-flight reservation in `check()`; `release()` on error |
| OpenAI-temp | `api_clients.py` | `temperature` parameter passed to OpenAI models — newer o1/o3/o4 series reject it (fixed at 1) | Remove `temperature` from `_call_openai()` entirely |

**Additional fixes:**
- `a2a_protocol.py`: Full `A2AQueueManager` implementation with bounded queues, timeout-to-FAILED mapping, and orphan cleanup
- `engine.py`: `_cleanup_background_tasks()` guarded against `CancelledError` from `task.exception()` on cancelled tasks

**Test coverage:** `tests/test_bug_fixes_v5_1.py` (10), `tests/test_bug_fixes_v5_1_round2.py` (10), `tests/test_reliability_regression.py` (22)

---

## Version History & Deprecations

### v6.0 Major Changes
- **Dashboard consolidation:** 7 dashboards → 1 core + plugins
- **Event unification:** 4 event systems → 1 unified bus
- **Plugin extraction:** Core-only + optional plugins

### Deprecated (to be removed in v7.0)
- `orchestrator.dashboard_*` (individual dashboards) → Use `dashboard_core`
- `orchestrator.streaming` (legacy events) → Use `unified_events`
- `orchestrator.events` (old events) → Use `unified_events`

### Backward Compatibility
The `orchestrator.compat` module provides backward-compatible aliases:
```python
from orchestrator.compat import (
    run_live_dashboard,      # Maps to new unified dashboard
    ProjectEventBus,         # Maps to UnifiedEventBus
    # ... etc
)
```

---

## Troubleshooting

### Common Issues

**Import errors:**
```bash
# Ensure package is installed in editable mode
pip install -e ".[dev]"
```

**Missing API keys:**
```python
from orchestrator.exceptions import MissingAPIKeyError

try:
    orch = Orchestrator()
except MissingAPIKeyError as e:
    print(f"Set {e.details['provider']} API key")
```

**Database locked (SQLite):**
- State manager uses WAL mode for better concurrency
- If still locked, check for zombie processes

**Dashboard not loading:**
- Check if required dependencies installed: `pip install -e ".[dashboard]"`
- Verify port 8888 is available

---

## Additional Resources

- [README.md](./README.md) - Human-oriented documentation
- [docs/PROJECT_STRUCTURE.md](./docs/PROJECT_STRUCTURE.md) - Detailed project structure
- [USAGE_GUIDE.md](./USAGE_GUIDE.md) - CLI reference and Python API
- [CAPABILITIES.md](./CAPABILITIES.md) - Feature deep-dive
- [MIGRATION_GUIDE_v6.md](./MIGRATION_GUIDE_v6.md) - v6.0 migration guide
- [SECURITY.md](./SECURITY.md) - Security policies

---

*Last updated: 2026-03-06*  
*Author: Georgios-Chrysovalantis Chatzivantsidis*
