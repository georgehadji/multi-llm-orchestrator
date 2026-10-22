---
type: object
cluster: routing
universe: ghost
status: verified
verified: 2026-10-22 at 26fbe258 (source unchanged since d74829e6)
entity: orchestrator/domain/phase_policy.py
---

# PhasePolicy (with Phase and ReasoningEffort)

A table of per-phase temperature, thinking and reasoning-effort settings, with lookup functions.

## Why this shape

The module says it replaces temperature literals and reasoning checks scattered across services, so a tuning change touches one table (`phase_policy.py:7-12`). Values are justified in `docs/REASONING_AND_TEMPERATURE.md`, which names this file as the source of truth.

## Shape

- `Phase` enum: decompose, generate, critique, revise, evaluate, extract, summarize, creative, sampling (`phase_policy.py:33-49`).
- `ReasoningEffort`: none, low, medium, high (`:52-58`).
- `PhasePolicy` frozen: `temperature`, `use_thinking`, `reasoning_effort`, `prefer_reasoning`, `exclude_reasoning` (`:61-80`).
- `_PHASE_POLICY` table (`:87-101`); per-`TaskType` temperature overrides built by enum value (`:107-126`).
- Helpers: `policy_for`, `temperature_for`, `use_thinking_for`, `reasoning_effort_for`, `prefer_reasoning_for` (`:132-160`).

Citations: `orchestrator/domain/phase_policy.py:33-160`

## Why ghost

A grep across `orchestrator/` and `tests/` for `phase_policy`, `temperature_for`, `policy_for` and `use_thinking_for` found importers only in `tests/unit/test_phase_policy.py`. No production module reads the table, so it describes intended policy, not behaviour. A dynamic import would not show.

## Connected to

- **owns:** the policy table.
- **owned-by:** none found in production.
- **joins:** imports `TaskType` from `..models` (`phase_policy.py:30`, see [WorkEnums](../work/WorkEnums.md)).
- **looks-like-but-is-not:** other places that set temperature directly.

## If you change this

- **Hits:** `tests/unit/test_phase_policy.py` and `docs/REASONING_AND_TEMPERATURE.md` (the module says to change both together, `:84-85`).
- **Does not hit:** actual request temperature or reasoning settings today, since nothing calls it.
- **Enum coupling:** the per-task overrides look up `TaskType` by value, so renaming a `TaskType` value silently drops its override (`:109-122`).

## Surfaces

| Surface | Role |
|---|---|
| tests | reads |

## See

- Source: `orchestrator/domain/phase_policy.py:33`
