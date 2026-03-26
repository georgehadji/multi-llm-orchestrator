# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Multi-LLM Orchestrator** — Production-grade orchestrator for coordinating multiple LLM providers (Anthropic, OpenAI, Google, DeepSeek) with intelligent routing, budget hierarchy enforcement, and circuit breaker resilience.

**Core Capabilities:**
- Multi-provider routing with quality-aware model selection
- Cross-run budget hierarchy with pre‑flight checks
- Resume capability with auto‑detection (prevents infinite loops)
- Policy‑driven enforcement (compliance, latency, cost constraints)
- Deterministic validation + LLM‑based evaluation scoring
- Telemetry collection and circuit breaker health tracking

**Repository:** https://github.com/georgehadji/multi-llm-orchestrator
**Status:** Active development (v6.0 complete)

---

## Architecture & Design Patterns

> **Master reference:** [`docs/ARCHITECTURE_ROADMAP.md`](docs/ARCHITECTURE_ROADMAP.md) — read this file **before any architectural decision or implementation**.

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
| Cross‑cutting | **Observer / EventBus** | `events.py`, `hooks.py` |
| LLM Providers | **Adapter + Protocol** | `gateway.py` |
| Budget Hierarchy | **Composite** | `cost.py` |
| Resilience | **State Machine** | `resilience.py`, `rate_limiter.py` |

### Three Unbreakable Rules
1. **`engine.py` = Mediator** — New business logic goes into a new service module, **not** into `engine.py`. The engine only wires services together.
2. **`models.py` = Pure data** — No I/O, no asyncio, no behavior. Only dataclasses and enums.
3. **TDD without exceptions** — First failing test (RED), then implementation (GREEN), then commit.

---

## Development Workflow

### Test‑Driven Development (TDD)
1. Write a failing test (RED phase) – verify it fails with the expected error.
2. Implement minimal code to pass (GREEN phase).
3. Run full test suite to verify no regressions.
4. Commit with a detailed message.

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

### Plan Mode for Non‑trivial Tasks
- Enter plan mode (via Claude Code) for any task with 3+ steps or architectural decisions.
- Write detailed specs upfront; verify plan with the user before implementation.
- If something goes sideways, **STOP and re‑plan immediately** – don’t keep pushing.

---

## Common Commands

### Setup
```bash
make install-dev          # Install with development dependencies
make install-all          # Install all optional dependencies (dev, security, tracing, docs)
pre‑commit install        # Set up pre‑commit hooks
```

### Testing
```bash
make test                 # Run all tests with coverage
make test‑unit            # Unit tests only
make test‑integration     # Integration tests only
make test‑fast            # Run tests in parallel (faster)
pytest tests/ -v -m "not slow"           # All tests (skip slow markers)
pytest tests/test_rate_limiter.py -v     # Single test module
pytest tests/test_rate_limiter.py::test_check_within_limit  # Single test function
pytest --tb=short -q                     # Summary output
```

### Code Quality
```bash
make lint                 # Run ruff linter
make lint‑fix             # Run ruff with auto‑fix
make format               # Format code with black
make format‑check         # Check formatting without changes
make type‑check           # Run mypy type checker
make security‑check       # Run bandit + safety security scans
make precommit            # Install and run pre‑commit hooks
```

### Maintenance
```bash
make clean                # Remove build artifacts, caches
make clean‑all            # Remove all generated files (including outputs/, results/, logs/)
make ci                   # Run all CI checks locally (format‑check, lint, type‑check, test‑ci, security‑check)
```

### Docker
```bash
make docker‑build         # Build Docker image
make docker‑run           # Run Docker container (requires .env file)
make docker‑test          # Run tests in Docker
```

### CLI Usage
```bash
python -m orchestrator --project "Build a FastAPI REST API" --criteria "All endpoints tested" --budget 2.0
python -m orchestrator --resume <project_id>
python -m orchestrator --analyze‑codebase /path/to/project
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

- **Test markers:** `unit`, `integration`, `slow`, `requires_api`
- **Coverage:** Configured in `pyproject.toml`; target 0% (temporarily relaxed)
- **Pytest configuration:** See `[tool.pytest.ini_options]` in `pyproject.toml`
- **Stress tests:** 4 pre‑existing failures in `tests/stress_test.py` (S2, S6, S7) – documented, not blocking

**Run a subset of tests:**
```bash
pytest -m unit            # Only unit tests
pytest -m "not slow"      # Skip slow tests
pytest -m integration     # Only integration tests
```

---

## Implementation Status (v6.0)

**All Phases Complete ✅**

| Phase | Status | Count | Features |
|-------|--------|-------|----------|
| P0 | ✅ Complete | 3/3 | autonomy, model_routing, verification |
| P1 | ✅ Complete | 12/12 | brain, evaluation, escalation, checkpoints, modes, prompt_enhancer, cost_analytics, competitive, tracing, plan‑then‑build, memory_bank, context_condensing |
| P2 | ✅ Complete | 10/10 | hierarchy, triggers, workspace, gateway, connectors, sandbox, context_sources, api_server, skills, drift |
| P3 | ✅ Complete | 1/1 | browser_testing |

**Overall: 26 Complete ✅ | 0 Partial 🟡 | 0 Missing ❌**

---

## Recent Critical Fixes

### v5.1 SRE Bug‑Finding Pass
- **BUG‑001:** Budget reservation leaked on `run_project()` failure – fixed in `cost.py` and `engine.py`
- **BUG‑002:** `asyncio.gather` without `return_exceptions=True` left orphan tasks – fixed in `engine.py`
- **BUG‑003:** `SearchResult` mutated in‑place during reranking – fixed in `hybrid_search_pipeline.py`
- **BUG‑004:** `BudgetHierarchy.remaining(level="team")` ignored pending reservations – fixed in `cost.py`
- **BUG‑005:** RateLimiter TOCTOU between `check()` and `record()` – fixed in `rate_limiter.py`
- **OPENAI‑T:** `temperature` parameter forwarded to OpenAI newer models – removed from `api_clients.py`

### v1.0 Resilience Hardening
- **Fix #1:** Terminal `COMPLETED_DEGRADED` status added to prevent infinite resume loops
- **Fix #2:** Critique resilience with graduated circuit breaker (3‑strike threshold)
- **Fix #3:** `BudgetHierarchy.charge_job()` integration verified for cross‑run budget enforcement

---

## Known Limitations & TODOs

- **Stress tests:** 4 pre‑existing failures in `tests/stress_test.py` (S2, S6, S7) – documented, not blocking
- **Resume detection:** Uses file modification time heuristic; could be more robust
- **Policy system:** Enforcement mode selection (HARD/SOFT/MONITOR) not yet fully integrated

---

## Code Quality Standards

- **Testing:** TDD required; all features must have a failing test first
- **Commits:** Atomic, descriptive messages with context
- **Documentation:** Docstrings on all public methods
- **Code Review:** Via GitHub PR; resilience fixes approved before merge

---

## Contact & References

- **Repository:** https://github.com/georgehadji/multi-llm-orchestrator
- **Architecture Roadmap:** [`docs/ARCHITECTURE_ROADMAP.md`](docs/ARCHITECTURE_ROADMAP.md)
- **Usage Guide:** [`USAGE_GUIDE.md`](USAGE_GUIDE.md)
- **Capabilities Deep‑dive:** [`CAPABILITIES.md`](CAPABILITIES.md)
- **Debugging Guide:** [`docs/debugging/DEBUGGING_GUIDE.md`](docs/debugging/DEBUGGING_GUIDE.md)

For questions about architecture, strategy, or development approach, refer to:
- PR #5: Resilience black‑swan fixes (detailed rationale)
- Issue discussions: Architecture and design rationale documented there
- Commit messages: Detailed technical context in each commit

---

**Last Updated:** 2026‑03‑21 (v6.0 – all phases complete)
