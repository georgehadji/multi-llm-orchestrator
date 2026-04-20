# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Multi-LLM Orchestrator** — Production-grade orchestrator for coordinating multiple LLM providers (Anthropic, OpenAI, Google, DeepSeek) with intelligent routing, budget hierarchy enforcement, and circuit breaker resilience.

**Core Capabilities:**
- Multi-provider routing with quality-aware model selection
- Cross-run budget hierarchy with pre-flight checks
- Resume capability with auto-detection (prevents infinite loops)
- Policy-driven enforcement (compliance, latency, cost constraints)
- Deterministic validation + LLM-based evaluation scoring
- Telemetry collection and circuit breaker health tracking

**Repository:** https://github.com/georgehadji/multi-llm-orchestrator

---

## Architecture & Design Patterns

> **Master reference:** [`docs/CODEBASE_MINDMAP.md`](docs/CODEBASE_MINDMAP.md) — read this file **before any architectural decision or implementation**.

### Hexagonal Architecture (Ports & Adapters)
- **Driving adapters:** `cli.py`, `api_server.py`, webhooks, tests
- **Application core:** `engine.py` (Mediator), domain services
- **Driven adapters:** LLM providers, databases, cache, telemetry
- **Domain models:** `models.py` (pure dataclasses/enums, no I/O)

### Pattern Summary
| Scope | Pattern | Key Files |
|-------|---------|-----------|
| Overall | Hexagonal Architecture | — |
| Orchestration | **Mediator** | `engine.py` |
| LLM Routing | **Strategy** | `model_routing.py`, `planner.py` |
| Optional Features | **Decorator** | `verification.py`, `prompt_enhancer.py` |
| Validation | **Chain of Responsibility** | `validators.py`, `preflight.py` |
| Persistence | **Repository + Memento** | `state.py`, `checkpoints.py` |
| Cross-cutting | **Observer / EventBus** | `events.py`, `hooks.py` |
| LLM Provider Abstraction | **Adapter** | `api_clients.py` (`UnifiedClient`) |
| HTTP API Gateway | **Facade** | `gateway.py` |
| Budget Hierarchy | **Composite** | `cost.py` |
| Resilience | **State Machine** | `resilience.py`, `rate_limiter.py` |

> **Important naming distinction:** `api_clients.py::UnifiedClient` is the LLM provider adapter (normalizes OpenAI/Google/Anthropic/DeepSeek SDKs into a single `call_model()` interface). `gateway.py` is the HTTP API gateway for routing external requests — these are separate concerns.

### Three Unbreakable Rules
1. **`engine.py` = Mediator** — New business logic goes into a new service module, **not** into `engine.py`. The engine only wires services together.
2. **`models.py` = Pure data** — No I/O, no asyncio, no behavior. Only dataclasses and enums.
3. **TDD without exceptions** — First failing test (RED), then implementation (GREEN), then commit.

---

## Core Execution Pipeline

The primary control loop in `engine.py` follows this pipeline for each task:

```
decompose project → [for each task]:
  generate → critique → revise → evaluate
  ↑_____________________________________|  (iterate up to max_iterations)
```

Key data flows:
- **`models.py`** defines `ROUTING_TABLE` (task type → preferred model) and `FALLBACK_CHAIN` (fallback order on failure).
- **`api_clients.py::UnifiedClient`** wraps all provider SDKs; returns normalized `APIResponse` with `text`, `input_tokens`, `output_tokens`, `cost_usd`.
- **`state.py::StateManager`** persists `ProjectState` to `~/.orchestrator_cache/state.db` (async SQLite via `aiosqlite`) after each task — enables crash recovery.
- **`planner.py::ConstraintPlanner`** selects models based on policy constraints before each task.

### Dual-Budget System
There are two independent budget mechanisms that can be used together:

| Component | Location | Scope | Purpose |
|-----------|----------|-------|---------|
| `Budget` dataclass | `models.py` | Per-run | Tracks spend/time within a single `run_project()` call |
| `BudgetHierarchy` | `cost.py` | Cross-run | Org → Team → Job caps that persist across multiple runs |

A `BudgetHierarchy` instance is passed into `Orchestrator` alongside the per-run `Budget`. Do not confuse them — they serve different purposes and both can be active simultaneously.

---

## Configuration

**Environment variables:** Required API keys: `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`. Optional OpenRouter optimizations: `USE_JSON_SCHEMA_RESPONSES`, `USE_MODEL_VARIANTS`, `USE_NATIVE_FALLBACKS`, `USE_PROVIDER_SORTING`. Copy `.env.example` to `.env`.

---

## Development Workflow

### Test-Driven Development (TDD)
1. Write a failing test (RED phase) — verify it fails with the expected error.
2. Implement minimal code to pass (GREEN phase).
3. Run full test suite to verify no regressions.
4. Commit with a detailed message.

### Tests Directory Note
The `tests/` directory contains 200+ files, many being one-off verification/migration scripts (e.g., `verify_fix.py`, `move_tests.py`). Actual test modules follow the `test_*.py` pattern. Use `pytest tests/ -m unit` or `pytest tests/ -m integration` to target meaningful tests rather than running the full directory.

### Git Worktrees for Isolation
This project uses git worktrees for isolated feature branches:
```bash
# Create new worktree for feature
git worktree add .claude/worktrees/feature-name -b feature-name

# Work in worktree, test thoroughly
cd .claude/worktrees/feature-name
pytest tests/

# Commit, push, etc.
git commit -m "fix: description"
git push -u origin feature-name

# After merge, clean up
cd ../..
git worktree remove .claude/worktrees/feature-name
```
Note: `.claude/worktrees/` is in `.gitignore` for safety.

### Plan Mode for Non-trivial Tasks
- Enter plan mode for any task with 3+ steps or architectural decisions.
- Write detailed specs upfront; verify plan with the user before implementation.
- If something goes sideways, **STOP and re-plan immediately** — don't keep pushing.

---

## Common Commands

### Setup
```bash
pip install -e ".[dev,security,tracing]"          # Install with development dependencies
pip install -e ".[dev,security,tracing,dashboard,docs]"   # Install all optional dependencies
pre-commit install        # Set up pre-commit hooks (requires .pre-commit-config.yaml)
```

### Testing
```bash
pytest tests/ -v --cov=orchestrator --cov-report=term-missing  # Run all tests with coverage
pytest tests/ -m unit -v           # Unit tests only
pytest tests/ -m integration -v    # Integration tests only
pytest -n auto tests/              # Run tests in parallel (faster)
pytest tests/ -v -m "not slow"     # All tests (skip slow markers)
pytest tests/test_rate_limiter.py -v     # Single test module
pytest tests/test_rate_limiter.py::test_check_within_limit  # Single test function
pytest --tb=short -q               # Summary output
```

### Code Quality
```bash
ruff check orchestrator/          # Run ruff linter
ruff check orchestrator/ --fix    # Run ruff with auto-fix
black orchestrator/               # Format code with black
black orchestrator/ --check       # Check formatting without changes
mypy orchestrator/                # Run mypy type checker
bandit -r orchestrator/           # Run bandit security scan
safety check                      # Check dependencies for vulnerabilities
pre-commit run --all-files        # Run pre-commit hooks on all files
```

### CLI Usage
```bash
python -m orchestrator --project "Build a FastAPI REST API" --criteria "All endpoints tested" --budget 2.0
python -m orchestrator --resume <project_id>
python -m orchestrator --analyze-codebase /path/to/project
```

### Dashboard (Web UI)
```bash
# Windows
start_dashboard.bat

# Linux/Mac
python start_dashboard.py

# URL: http://localhost:8888
```

---

## Testing Strategy

- **Test markers:** `unit`, `integration`, `slow`, `requires_api`, `e2e`, `load`, `stress`, `benchmark`
- **Coverage:** Configured in `pyproject.toml`; `fail_under = 0` (temporarily relaxed)
- **Pytest configuration:** See `[tool.pytest.ini_options]` in `pyproject.toml`
- **Stress tests:** Pre-existing failures in `tests/stress_test.py` (S2, S6, S7) — documented, not blocking

```bash
pytest -m unit            # Only unit tests
pytest -m "not slow"      # Skip slow tests
pytest -m integration     # Only integration tests
```

---

## Known Limitations

- **Resume detection:** Uses file modification time heuristic; could be more robust.
- **Policy system:** Enforcement mode selection (HARD/SOFT/MONITOR) not yet fully integrated.
- **Stress tests:** Pre-existing failures in `tests/stress_test.py` (S2, S6, S7) — documented, not blocking.

---

## Key References

- **Architecture overview:** [`docs/CODEBASE_MINDMAP.md`](docs/CODEBASE_MINDMAP.md) — complete architecture mind map
- **Usage Guide:** [`USAGE_GUIDE.md`](USAGE_GUIDE.md)
- **Debugging Guide:** [`docs/debugging/DEBUGGING_GUIDE.md`](docs/debugging/DEBUGGING_GUIDE.md)
- **Tool configuration:** [`pyproject.toml`](pyproject.toml) — pytest, ruff, black, mypy, coverage settings
