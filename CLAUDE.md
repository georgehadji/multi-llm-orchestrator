# CLAUDE.md

Guidance for Claude Code (claude.ai/code) in this repository.

## Project Overview

**Multi-LLM Orchestrator** — Production-grade orchestrator coordinating multiple LLM providers (Anthropic, OpenAI, Google, DeepSeek) with intelligent routing, budget hierarchy enforcement, circuit breaker resilience.

**Core Capabilities:**
- Multi-provider routing with quality-aware model selection
- Cross-run budget hierarchy with pre-flight checks
- Resume capability with auto-detection (prevents infinite loops)
- Policy-driven enforcement (compliance, latency, cost constraints)
- Deterministic validation + LLM-based evaluation scoring
- Telemetry collection + circuit breaker health tracking

**Repository:** https://github.com/georgehadji/multi-llm-orchestrator

---

## Architecture & Design Patterns

> **Master reference:** [`docs/CODEBASE_MINDMAP.md`](docs/CODEBASE_MINDMAP.md) — read **before any architectural decision or implementation**.

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

> `api_clients.py::UnifiedClient` = LLM provider adapter (normalizes OpenAI/Google/Anthropic/DeepSeek SDKs into single `call_model()` interface). `gateway.py` = HTTP API gateway for external request routing — separate concerns.

### Four Unbreakable Rules
1. **`engine.py` = Mediator** — New logic → new service module, **not** `engine.py`. Engine only wires services.
2. **`models.py` = Pure data** — No I/O, no asyncio, no behavior. Only dataclasses + enums.
3. **TDD without exceptions** — First failing test (RED), then impl (GREEN), then commit.
4. **No new root-level modules** — All new code in existing subpackages (`orchestrator/domain/`, `application/`, `engine_core/`, `infrastructure/`, `commands/`, `generators/`, etc.). No new `orchestrator/*.py` at depth 1.

---

## Core Execution Pipeline

Primary control loop in `engine.py`:

```
decompose project → [for each task]:
  generate → critique → revise → evaluate
  ↑_____________________________________|  (iterate up to max_iterations)
```

Key data flows:
- **`models.py`** defines `ROUTING_TABLE` (task type → preferred model) + `FALLBACK_CHAIN` (fallback order on failure).
- **`api_clients.py::UnifiedClient`** wraps all provider SDKs; returns normalized `APIResponse` with `text`, `input_tokens`, `output_tokens`, `cost_usd`.
- **`state.py::StateManager`** persists `ProjectState` to `~/.orchestrator_cache/state.db` (async SQLite via `aiosqlite`) after each task — enables crash recovery.
- **`planner.py::ConstraintPlanner`** selects models based on policy constraints before each task.

### Dual-Budget System
Two independent budget mechanisms usable together:

| Component | Location | Scope | Purpose |
|-----------|----------|-------|---------|
| `Budget` dataclass | `models.py` | Per-run | Tracks spend/time within single `run_project()` call |
| `BudgetHierarchy` | `cost.py` | Cross-run | Org → Team → Job caps persisting across runs |

`BudgetHierarchy` passed into `Orchestrator` alongside per-run `Budget`. Don't confuse them — different purposes, both can be active simultaneously.

---

## Configuration

**Env vars:** Required API keys: `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`. Optional OpenRouter optimizations: `USE_JSON_SCHEMA_RESPONSES`, `USE_MODEL_VARIANTS`, `USE_NATIVE_FALLBACKS`, `USE_PROVIDER_SORTING`, `USE_RESPONSE_HEALING` (server-side JSON repair for non-streaming structured-output requests). Copy `.env.example` to `.env`.

---

## Development Workflow

### Test-Driven Development (TDD)
1. Write failing test (RED) — verify fails with expected error.
2. Implement minimal code to pass (GREEN).
3. Run full test suite, verify no regressions.
4. Commit with detailed message.

### Tests Directory Note
`tests/` has 200+ files, many one-off verification/migration scripts (e.g., `verify_fix.py`, `move_tests.py`). Actual test modules follow `test_*.py` pattern. Use `pytest tests/ -m unit` or `pytest tests/ -m integration` — don't run full directory.

### Git Worktrees for Isolation
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
`.claude/worktrees/` in `.gitignore` for safety.

### Plan Mode for Non-trivial Tasks
- Enter plan mode for tasks with 3+ steps or architectural decisions.
- Write detailed specs upfront; verify plan before impl.
- If something goes sideways, **STOP + re-plan immediately** — don't keep pushing.

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
- **Pytest config:** See `[tool.pytest.ini_options]` in `pyproject.toml`
- **Stress tests:** No pytest-based stress suite is committed. `projects/stress_test/*.yaml` are 8 manually-run stress scenarios (`python -m orchestrator --file <yaml>`); see `projects/stress_test/README.md`. The `load`/`stress` markers below are registered for that future work and currently unused.

```bash
pytest -m unit            # Only unit tests
pytest -m "not slow"      # Skip slow tests
pytest -m integration     # Only integration tests
```

---

## Known Limitations

- **Resume detection:** Uses file mod time heuristic; could be more robust.
- **Policy system:** Enforcement mode selection (HARD/SOFT/MONITOR) not fully integrated.
- **Stress tests:** `tests/stress_test.py` was documented here but never committed (see `projects/stress_test/README.md`). No automated stress suite currently runs in CI; `projects/stress_test/*.yaml` are manual scenarios.

---

## Key References

- **Architecture overview:** [`docs/CODEBASE_MINDMAP.md`](docs/CODEBASE_MINDMAP.md) — complete architecture mind map
- **Usage Guide:** [`USAGE_GUIDE.md`](USAGE_GUIDE.md)
- **Debugging Guide:** [`docs/debugging/DEBUGGING_GUIDE.md`](docs/debugging/DEBUGGING_GUIDE.md)
- **Tool configuration:** [`pyproject.toml`](pyproject.toml) — pytest, ruff, black, mypy, coverage settings
