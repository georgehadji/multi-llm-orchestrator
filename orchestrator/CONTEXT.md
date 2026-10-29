# orchestrator/ — the product package

**Job:** the engine. Decompose → route → generate → critique → revise → evaluate, with budgets, resume, and resilience.

## Layers (dependency direction: top may import below, never up)
| Path | Role | Must not import |
|---|---|---|
| `domain/`, `models.py`, `exceptions.py` | Pure data, enums, policies | application, infrastructure, engine_core, engine |
| `application/` | Use-cases and services | `infrastructure/` (concrete), `engine.py` |
| `engine_core/` | Pipeline orchestration; `container.py` is the composition root | infrastructure (except `container.py`) |
| `infrastructure/` | Concrete adapters: DB, LLM clients, cache, search | — |
| `engine.py` | Mediator. Wires services only | — |
| `ports.py` | Protocol DI (Cache, State, Event + Null adapters) | — |

Other packages (`quality/`, `reasoning/`, `generators/`, `nash/`, ...): application-tier, same no-concrete-infra rule. Authoritative list: `.importlinter`.

## Inputs
`models.py` (`ROUTING_TABLE`, `FALLBACK_CHAIN`), `.env` keys, `orchestrator_config.json`.

## Output
Project artifacts and `ProjectState` persisted to `~/.orchestrator_cache/state.db` (aiosqlite).

## Before you edit here
- Load skill `orchestrator-change-control` and `orchestrator-architecture-contract`.
- New code goes in an existing subpackage. No new `orchestrator/*.py` at depth 1 (`scripts/check_new_root_files.py`).
- Verify layering: `lint-imports`. Verify lazy imports: `python -c "from orchestrator import Orchestrator"`.

## Human check
Change touches `engine.py`, `models.py`, or a gate config: review the diff against the 4 rules in `CLAUDE.md`.
