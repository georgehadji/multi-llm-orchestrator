# REASONIX.md — Multi-LLM Orchestrator

## Stack
- **Python 3.10+** — [pyproject.toml](pyproject.toml:14)
- **Build:** hatchling — [pyproject.toml](pyproject.toml:1-2)
- **LLM SDKs:** openai≥1.30, google-genai≥1.0 — [pyproject.toml](pyproject.toml:29-30)
- **Data:** pydantic≥2.0, aiosqlite≥0.19 — [pyproject.toml](pyproject.toml:31-32)
- **HTTP:** httpx, aiohttp — [pyproject.toml](pyproject.toml:37-38)
- **Retry:** tenacity≥8.2 — [pyproject.toml](pyproject.toml:39)

## Layout
- `orchestrator/` — main package (~300 flat modules + subpackages)
- `orchestrator/domain/` — ports, exceptions, model_registry, services/
- `orchestrator/application/` — services: executor, evaluator, skill_optimizer, project_runner, chat_cli
- `orchestrator/engine_core/` — pipeline, pipeline_runner, project_planner, state_coordinator, stages/, container
- `orchestrator/infrastructure/` — concrete adapters: llm_client, state, cache, telemetry, streaming
- `orchestrator/agents/` — specialized agent modules
- `orchestrator/design/skills/` — per-TaskType skill documents (`.SKILL.md`)
- `tests/` — unit/, integration/, contracts/, smoke/
- `docs/` — ADRs, plans, architectural audits, CODEBASE_MINDMAP.md
- `scripts/` — model analysis and git utilities

## Commands
```bash
# Install (editable + dev extras)
pip install -e ".[dev,security,tracing]"

# Run CLI
python -m orchestrator --project "Build a FastAPI todo app" --budget 2.0
python -m orchestrator --resume <project_id>
python -m orchestrator dashboard --days 30

# Lint / format
ruff check orchestrator/ tests/
ruff check orchestrator/ --fix
black orchestrator/ tests/
black --check orchestrator/ tests/

# Type check (strict on domain/application; full codebase warn-only)
mypy orchestrator/domain/ orchestrator/application/ orchestrator/engine_core/container.py --ignore-missing-imports --no-strict-optional
mypy orchestrator/ --ignore-missing-imports --no-strict-optional

# Tests
pytest -m "not slow and not requires_api and not stress and not e2e" --tb=short -q --cov=orchestrator
pytest -m unit -v
pytest -m integration -v
pytest -n auto tests/

# Import-linter (4 architectural contracts)
lint-imports

# Security
bandit -r orchestrator/ --severity-level high
safety check
```

## Conventions
- **4 import-linter contracts** enforced at CI: domain-purity, application-no-concrete-infra, application-services-no-engine, engine-core-no-loose-infra — [.importlinter](.importlinter:1-110)
- **Black** line-length 100, target py310 — [pyproject.toml](pyproject.toml:111-116)
- **Ruff** isort with `known-first-party = ["orchestrator"]`, docstring convention google — [pyproject.toml](pyproject.toml:118-155)
- **mypy strict** on core layers; 26 legacy modules in `ignore_errors` override (do NOT add new code there) — [pyproject.toml](pyproject.toml:157-211)
- **Coverage ratchet floor: 6%** — never lower; raise in ~5% steps — [pyproject.toml](pyproject.toml:244)
- **Test markers:** unit, integration, slow, requires_api, e2e, load, stress, benchmark, mock, asyncio, edge_case — [pyproject.toml](pyproject.toml:226-239)

## Watch out for
- **engine.py is the Mediator** — new business logic goes into a new service module, NOT into engine.py. models.py must stay pure (no I/O, no asyncio). — [CLAUDE.md](CLAUDE.md:77-79)
- **api_clients.py::UnifiedClient** is the LLM provider adapter (not an HTTP gateway). **gateway.py** is the HTTP API gateway — separate concerns.
- **300+ flat modules** in orchestrator/ — most modules live at package root, not in deep subdirectories.
- **Ruff ignores 40+ rule codes** as pre-existing baseline violations (F401, F811, F821, E402, I001, etc.) — new code should NOT rely on these ignores.
- **mypy overrides list** in pyproject.toml must not grow — modules should be removed from it as they are typed.
- **.claude/worktrees/**, **openevolve-main/**, **Search/**, **ide_outputs/** are external dependencies or worktrees — not part of the orchestrator source.
