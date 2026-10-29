---
type: object
cluster: money
universe: ghost
status: verified
verified: 2026-10-22 at 3b5f746d (source unchanged since d74829e6)
entity: orchestrator/cost.py
---

# CostForecaster (with ForecastReport and RiskLevel)

A static pre-flight estimator that projects total cost and time for a task list and labels the risk against a budget.

## Why this shape

It is a stateless static method that takes a `CostPredictor`, so it adds no state to the engine (`cost.py:580-594`). Risk is the ratio of estimated total to `budget.max_usd`: 0.8 or more is HIGH, 0.5 or more is MEDIUM, else LOW (`cost.py:652-665`).

## Shape

- `forecast(tasks, profiles, predictor, budget=None) -> ForecastReport` (`cost.py:588-594`).
- Per task it uses the cheapest model, then generation cost, review cost and an `_EVAL_OVERHEAD` of 0.15 (`cost.py:56`, `:627-642`).
- `ForecastReport` is frozen with `will_exceed_budget` (`cost.py:108-128`); `RiskLevel` has LOW, MEDIUM, HIGH (`cost.py:95-100`).

Citations: `orchestrator/cost.py:95-128`, `:580-672`

## Why ghost

A grep for `CostForecaster` and `.forecast(` across `orchestrator/` and `tests/` found no call site outside `cost.py`. The only other hits are the `costing/core.py` shim docstring and the `tests/unit/test_hunt_t1_money.py` docstring. A dynamic call would not show up. Do not rely on it being exercised.

## Connected to

- **owns / owned-by:** none found.
- **joins:** takes [CostPredictor](CostPredictor.md), `Task` ([Task](../work/Task.md)) and `Budget` ([Budget](Budget.md)).
- **looks-like-but-is-not:** the estimate passed to `BudgetHierarchy.can_afford_job` is a plain number supplied by the caller, not a `ForecastReport`.

## If you change this

- **Hits:** nothing found.
- **Does not hit:** any charge or reservation path.

## Surfaces

| Surface | Role |
|---|---|
| none found | none |

## See

- Source: `orchestrator/cost.py:580`
