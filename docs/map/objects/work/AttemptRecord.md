---
type: object
cluster: work
universe: live
status: verified
verified: 2026-10-22 at 80161d2c (source unchanged since d74829e6)
entity: orchestrator/models.py
---

# AttemptRecord

One failed iteration, kept so the next retry sees why the last one failed.

## Why this shape

Strings only (`model_used` is `Model.value`, `output_snippet` is the first 200 characters) so it serializes without conversion (`models.py:1182-1184`).

## Shape

- `attempt_num` (1-based), `model_used`, `output_snippet`, `failure_reason`, `validators_failed` (`models.py:1180-1188`).

Citations: `orchestrator/models.py:1176-1188`

## Connected to

- **owns:** nothing.
- **owned-by:** [TaskResult](TaskResult.md) (`attempt_history`, `models.py:1151`).
- **joins:** built in `application/critique_cycle.py:249`; also referenced by `engine_core/stages/self_consistency.py`, `prompt_builder.py`, `unified_events/core.py`.
- **looks-like-but-is-not:** none found.

## If you change this

- **Hits:** `_attempt_to_dict` / `_attempt_from_dict` (`infrastructure/state.py:119`, `:139`); `critique_cycle.py`, `prompt_builder.py`.
- **Does not hit:** `Verdict` (a pass/fail from verifiers, not a failed attempt).
- **Persistence is tolerant:** `_attempt_to_dict` also accepts a raw dict and reads `attempt` as a fallback key for `attempt_num` (`state.py:131`).

## Surfaces

| Surface | Role |
|---|---|
| `application/critique_cycle.py` | writes |
| `infrastructure/state.py` | reads and writes |
| `prompt_builder.py` | reads |

## See

- Source: `orchestrator/models.py:1176`
