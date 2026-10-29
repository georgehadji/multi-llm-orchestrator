---
type: object
cluster: money
universe: ghost
status: verified
verified: 2026-10-22 at 3b5f746d (source unchanged since d74829e6)
entity: orchestrator/infrastructure/token_budget.py
---

# TokenBudgetManager

Splits a token allowance across conversation turns (equal, weighted or dynamic). It counts tokens, not dollars.

## Why this shape

A separate manager with `AllocationStrategy`, `TurnAllocation` and `BudgetStats`, plus module-level helpers (`infrastructure/token_budget.py:49-135`, `:407-423`).

## Shape

- `allocate_budget`, `record_usage`, `get_remaining_budget`, `get_stats`, `reset` (`token_budget.py:165`, `:303`, `:364`, `:374`, `:392`).
- `get_token_budget_manager` returns a shared instance (`token_budget.py:407`).

Citations: `orchestrator/infrastructure/token_budget.py:49-446`

## Why ghost

A grep for `TokenBudgetManager`, `get_token_budget_manager` and `token_budget import` across `orchestrator/` and `tests/` found only the canonical file, the root `orchestrator/token_budget.py` copy and `tests/unit/test_hunt_t10_cost_optimization.py`. The root file's own header says there are zero live callers of either copy (hunt T10). `cost_optimization/__init__.py` mentions a different `TokenBudget` from `operations/optimization`.

## Connected to

- **owns / owned-by:** none.
- **looks-like-but-is-not:** [Budget](Budget.md) (USD and time, with token counters per phase in `phase_tokens`). This manager is not connected to it.

## If you change this

- **Hits:** the root `token_budget.py` duplicate and its test.
- **Does not hit:** any spend cap. Changing it cannot change what a run is allowed to spend.

## Surfaces

| Surface | Role |
|---|---|
| `tests/unit/test_hunt_t10_cost_optimization.py` | reads |

## See

- Source: `orchestrator/infrastructure/token_budget.py:136`; duplicate `orchestrator/token_budget.py`
