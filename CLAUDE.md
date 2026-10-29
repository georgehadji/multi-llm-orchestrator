# CLAUDE.md

Multi-LLM Orchestrator: Python 3.10+ platform that decomposes a project into tasks, routes each to an LLM, and loops generate → critique → revise → evaluate. Repo: https://github.com/georgehadji/multi-llm-orchestrator

This file routes. It holds no content. Full former text: [docs/DEV_GUIDE.md](docs/DEV_GUIDE.md).

## Rules (never break)
1. `orchestrator/engine.py` is the Mediator. New logic goes in a new service module, not in `engine.py`.
2. `orchestrator/models.py` is pure data: dataclasses and enums, no I/O, no asyncio.
3. TDD: failing test (RED), implementation (GREEN), commit.
4. No new `orchestrator/*.py` at depth 1. Use an existing subpackage.

Gates: `lint-imports` (`.importlinter`), root-file freeze, coverage ratchet. Never weaken one to get past it. Load skill `orchestrator-change-control` before any change.

## Where do I go for...
| Task | Read |
|---|---|
| Architecture, how modules connect | `docs/CODEBASE_MINDMAP.md`, [orchestrator/CONTEXT.md](orchestrator/CONTEXT.md) |
| What is X (a noun), what else moves if I change it | [docs/map/CLAUDE.md](docs/map/CLAUDE.md) |
| Write or place a test, markers | [tests/CONTEXT.md](tests/CONTEXT.md) |
| Run a check or audit script | [scripts/CONTEXT.md](scripts/CONTEXT.md) |
| Find a plan, audit, ADR, design doc | [docs/CONTEXT.md](docs/CONTEXT.md) |
| CI order, import contracts, gotchas | [docs/CI_AND_CONTRACTS.md](docs/CI_AND_CONTRACTS.md) |
| Setup, commands, workflow, TDD, worktrees | [docs/DEV_GUIDE.md](docs/DEV_GUIDE.md) |
| Config and flags | skill `orchestrator-config-and-flags`, `.env.example` |
| Something is broken | skill `orchestrator-debugging-playbook` |
| Run the app | skill `orchestrator-run`, `USAGE_GUIDE.md` |

## Top level
- `orchestrator/` package (the product). `tests/` pytest suite. `scripts/` gate and audit tooling. `docs/` all documentation.
- Skills: `.claude/skills/`. Tool config: `pyproject.toml`, `.importlinter`.
- Not the product (separate or vendored projects, not confirmed individually): `Search/`, `openevolve-main/`, `gadini-barberia/`, `ophthalmologist-thessaloniki/`, `thessaloniki-dental/`, `ide_frontend/`, `ide_outputs/`, `uploads/`.
- Root `*_PLAN.md`, `*_audit*.md`, `implementation_*.md` files are historical. See [docs/CONTEXT.md](docs/CONTEXT.md).
