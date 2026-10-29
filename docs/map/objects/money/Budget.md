---
type: object
cluster: money
universe: live
status: verified
verified: 2026-10-22 at 3b5f746d (source unchanged since d74829e6)
entity: orchestrator/budget.py
---

# Budget

The per-run spend and time tracker. Concurrent tasks reserve, then charge, against it.

## Why this shape

It has async behaviour (an `asyncio.Lock` and a reserve / commit / release pattern), so it lives outside `models.py`, which must stay pure data (`budget.py:6-9`). `models.py` only re-exports it (`models.py:1054`). The lock is created in `__post_init__` to avoid a race on lazy creation (`budget.py:72-77`).

## Shape

- Caps: `max_usd=8.0`, `max_time_seconds=5400.0` (`budget.py:44-45`).
- Spend: `spent_usd`, private `_reserved_usd`, `phase_spent`, `phase_tokens`, `start_time`, `original_start_time` (`budget.py:46-70`).
- `BUDGET_PARTITIONS` splits the cap by phase: decomposition 0.05, generation 0.45, cross_review 0.25, evaluation 0.15, reserve 0.10 (`budget.py:21-27`).
- Methods: `charge`, `reserve`, `commit_reservation`, `release_reservation`, `track_tokens`, `can_afford`, `phase_remaining` (`budget.py:98-162`).

Citations: `orchestrator/budget.py:21-189`

## Connected to

- **owns:** nothing.
- **owned-by:** [ProjectState](../work/ProjectState.md) (`budget: Budget | None`, `models.py:1199`).
- **joins:** `application/budget_enforcer.py:235` charges it; stages charge directly (`engine_core/stages/generate.py:160`, `critique.py:123`, `application/evaluator.py:205`).
- **looks-like-but-is-not:** [BudgetHierarchy](BudgetHierarchy.md) (cross-run, synchronous). The two can both be active.

## If you change this

- **Hits:** `_budget_to_dict` / `_budget_from_dict` (`infrastructure/state.py:52`, `:64`); `budget_enforcer.py`; every stage that calls `charge`; about 39 files import the name.
- **Does not hit:** `BudgetHierarchy` caps.
- **Phase accounting gap:** `charge` adds to `phase_spent` only when the phase name is one of the five partition keys (`budget.py:128`). Callers pass `"critique"` (`critique.py:123`) and `"verbalized_sampling"` (`verbalized_sampling.py:138`), which are not keys. Those charges raise `spent_usd` but not any `phase_spent` entry.
- **Silent loss on resume:** the saved dict has no `phase_tokens` or `_reserved_usd` (`state.py:52-61`).
- **None budget:** `ProjectState.budget` may be `None`, but `_budget_to_dict` reads `b.max_usd` with no check, and `_state_to_dict` calls it unguarded (`state.py:206`). Read from the code, not run.

## Surfaces

| Surface | Role |
|---|---|
| stages, evaluator, `budget_enforcer.py` | writes (charge) |
| `infrastructure/state.py` | reads (checkpoint) |
| dashboards, writers | reads (`to_dict`, `budget.py:174`) |

## See

- Source: `orchestrator/budget.py:36`
