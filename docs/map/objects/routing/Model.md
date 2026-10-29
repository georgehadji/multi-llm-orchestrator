---
type: object
cluster: routing
universe: live
status: verified
verified: 2026-10-22 at 26fbe258 (source unchanged since d74829e6)
entity: orchestrator/models.py
---

# Model, with ROUTING_TABLE, FALLBACK_CHAIN and COST_TABLE

The enum of every routable LLM id, plus the three tables that say which model runs which task type, what replaces it on failure, and what it costs.

## Why this shape

`models.py` may not do I/O at import (rule 2), so the JSON-backed tables are built on first access by a module `__getattr__` and then cached in module globals (`models.py:914-939`). Config lives in `orchestrator/config/*.json` (`costs.json`, `routing.json`, `fallbacks.json`, `thresholds.json`, `limits.json`).

## Shape

- `Model(Enum)`: values are `provider/name` strings. Retired free-tier members keep a leading underscore (`models.py:138-146`). About 155 members by a regex count.
- `PROVIDER_MAP` (prefix → provider) and `get_provider(model)` (`models.py:779`, `:807`).
- `ROUTING_TABLE`: task type → list of models, from `routing.json` (13 task types). `FALLBACK_CHAIN`: model → model, from `fallbacks.json` (67 entries) (`models.py:886-901`).
- Unknown ids in the JSON are silently dropped when the table is built (`models.py:889`, `:900`).
- `COST_TABLE` exists twice in this file: a literal at `models.py:397` (about 155 entries) and a JSON loader registered at `models.py:924` (`_build_cost_table`, `:881`).

Citations: `orchestrator/models.py:138`, `:397`, `:779-939`

## Connected to

- **owns:** the tables above.
- **owned-by:** imported across the package: `ROUTING_TABLE` is referenced in 22 files, `FALLBACK_CHAIN` in 11.
- **joins:** `application/fallback_handler.py:103-106` (reads `ROUTING_TABLE`), `application/decomposer.py:704-707` and `application/testing/repair_policy.py:76-78` (read `FALLBACK_CHAIN`).
- **looks-like-but-is-not:** [ModelRegistry](ModelRegistry.md), which holds model ids as plain strings with its own cost table.

## If you change this

- **Hits:** the JSON configs, `application/model_profile_builder.py` and every file reading the tables; saved state stores `model_used` as `Model.value` and restores with `Model(...)` (`infrastructure/state.py:155`, `:172-182`), so removing a member breaks old checkpoints (restore falls back to `GPT_4O_MINI`).
- **Does not hit:** `ModelRegistry` constants. Adding a model here does not add it there.
- **Three cost sources:** the `models.py` literal, `config/costs.json` (148 entries) and `ModelRegistry.COST_TABLE` (60 entries). Comparing json to registry: 54 ids in common, 2 with different prices (`qwen/qwen3.6-flash`, `deepseek/deepseek-v3.2`), 94 only in json, 6 only in the registry. The `models.py` literal was not compared with the json, because the package would not import here (missing `aiosqlite`).
- **Probably dead loader:** a module `__getattr__` only runs for missing attributes, and `COST_TABLE` is assigned at import (`models.py:397`), so the `costs.json` loader would never run. Read from the code, not run.

## Surfaces

| Surface | Role |
|---|---|
| `application/model_profile_builder.py`, routing code | reads |
| `config/*.json` | source for routing, fallbacks and the unused cost loader |

## See

- Source: `orchestrator/models.py:138`; configs `orchestrator/config/`
