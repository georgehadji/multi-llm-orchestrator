# AGENTS.md

Multi-LLM Orchestrator — Python 3.10+ autonomous dev platform that decomposes projects into tasks, routes to LLMs, runs generate→critique→revise→evaluate cycles.

## Setup

```bash
pip install -e ".[dev,security,tracing]"
cp .env.example .env  # add at least one provider API key
```

## CI Pipeline (must pass in this order)

1. `black --check orchestrator/ tests/` — format check (line-length 100)
2. `ruff check orchestrator/ tests/` — lint (many pre-existing violations are suppressed; see pyproject.toml)
3. `lint-imports` — import boundary enforcement (5 contracts, see below)
4. `mypy orchestrator/domain/ orchestrator/application/ orchestrator/engine_core/container.py --ignore-missing-imports --no-strict-optional` — strict on domain+app layers; rest is informational
5. `pytest -m "not slow and not requires_api and not stress and not e2e" --tb=short -q --cov=orchestrator --cov-report=xml --cov-fail-under=6` — unit+integration tests
6. `pytest tests/contracts/ -v --tb=short --no-cov` — contract tests (run with empty API keys)
7. `bandit -r orchestrator/ --severity-level high --confidence-level medium -x orchestrator/graphify-out` — security (currently warn-only)

## Architecture: 5 Import Boundary Contracts

These **block merge**. Enforced by `lint-imports` via `.importlinter`:

1. **Domain purity** — `orchestrator.domain`, `orchestrator.models`, `orchestrator.exceptions` must NOT import from application, infrastructure, engine_core, or engine.
2. **Application no concrete infra** — `orchestrator.application` must NOT import from `orchestrator.infrastructure`.
3. **Application services no engine** — `orchestrator.application` must NOT import from `orchestrator.engine` (except `chat_cli.py` and `project_runner.py` which are documented shims).
4. **Engine core no loose infra** — `orchestrator.engine_core.pipeline`, `pipeline_runner`, `project_planner`, `state_coordinator`, `stages` must NOT import infrastructure. (`container.py` is exempt — it's the composition root.)
5. **Root modules no infra** — Root-level `orchestrator/*.py` must NOT import from `orchestrator.infrastructure` (13 documented backward-compat shims are exempt).

## Three Unbreakable Rules

1. **`engine.py` = Mediator** — New business logic goes into new service modules, NOT into engine.py. Engine only wires services.
2. **`models.py` = Pure data** — No I/O, no asyncio, no behavior. Only dataclasses and enums.
3. **TDD without exceptions** — RED (failing test) → GREEN (implementation) → commit.

## Key Architectural Layers

```
orchestrator/domain/      → Pure value objects, enums, exceptions (zero deps)
orchestrator/application/ → Use-cases, extracted services (no infra, no engine)
orchestrator/engine_core/ → Pipeline orchestration (no infra except container.py)
orchestrator/infrastructure/ → Concrete adapters (DB, LLM clients, cache, search)
orchestrator/engine.py    → Top-level mediator (~5,100 lines), wires everything
orchestrator/models.py    → ROUTING_TABLE, COST_TABLE, FALLBACK_CHAIN, MODEL_MAX_TOKENS
orchestrator/ports.py     → Protocol-based DI (CachePort, StatePort, EventPort + NullAdapters)
```

## Testing

- **Coverage floor:** `fail_under = 15` in pyproject.toml (CI uses 6% — never lower either)
- **Many test files are ignored** in pytest config — they're one-off debug/migration scripts. See `[tool.pytest.ini_options]` addopts `--ignore` list.
- **Key test directories:** `tests/integration/`, `tests/contracts/`, `tests/unit/`, `tests/smoke/`, `tests/regression/`
- **Markers:** `unit`, `integration`, `slow`, `requires_api`, `e2e`, `load`, `stress`, `benchmark`, `mock`
- **Quick single-file test:** `pytest tests/test_circuit_breaker.py -v`
- **Skip slow stuff:** `pytest -m "not slow"`
- **No pre-commit config** in this repo (the file only exists inside `openevolve-main/`)

## Code Style

- Line length: 100 (black + ruff)
- Strict type hints (mypy strict mode), but ~30 legacy modules have `ignore_errors = true` in pyproject.toml — DO NOT add new modules to that list
- ruff ignores ~40 pre-existing violation categories (F401, F811, E402, etc.) — new code must not trigger these
- Google-style docstrings
- Author attribution header required on all Python files
- All I/O must be async; `StateManager` methods are all async
- Use `TYPE_CHECKING` for circular imports

## Non-obvious Gotchas

- `orchestrator/__init__.py` uses lazy-loading (`__getattr__`) to avoid circular imports — test with `python -c "from orchestrator import Orchestrator"`
- `engine.py` wraps some imports in try/except to let CLI load without all deps
- `grimp` pinned to <3.4 because 3.4+ has a Rust panic on Windows with this codebase
- Version is dynamic from `orchestrator/__init__.py` (currently 6.0.0)
- State persistence: SQLite via `aiosqlite` at `~/.orchestrator_cache/state.db`
- Console scripts: `orchestrator`, `mllm`, `dashboard`

## Quick Reference

```bash
# Run CLI
python -m orchestrator --project "Build API" --budget 5.0
python -m orchestrator --resume <project_id>
python -m orchestrator --dry-run  # plan only

# Code quality
black orchestrator/ tests/
ruff check --fix orchestrator/
mypy orchestrator/

# Security
bandit -r orchestrator/
safety check
gitleaks detect

# Architecture check
lint-imports
```

See also: `CLAUDE.md`, `docs/CODEBASE_MINDMAP.md`, `pyproject.toml`
