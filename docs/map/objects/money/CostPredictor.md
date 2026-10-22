---
type: object
cluster: money
universe: live
status: verified
verified: 2026-10-22 at 3b5f746d (source unchanged since d74829e6)
entity: orchestrator/cost.py
---

# CostPredictor

Predicts the cost of one call for a (model, task type) pair from a moving average of past costs, falling back to a static estimate.

## Why this shape

An exponential moving average per pair adapts to real spend without storing history. With no data it falls back to `COST_TABLE` and typical token counts, so `predict` never has to fail (`cost.py:499-557`).

## Shape

- `__init__(alpha=0.1)`; `alpha` must be in (0, 1] (`cost.py:513-524`).
- `record` ignores zero or negative costs, so cache hits do not drag the average down (`cost.py:526-534`).
- `predict` returns the average if present, else `_static_estimate` (`cost.py:536-557`).
- `cheapest_model` skips candidates that predict zero (`cost.py:559-572`).
- The averages live in `_ema` in memory only. Nothing saves them (`cost.py:524`).

Citations: `orchestrator/cost.py:499-572`

## Connected to

- **owns:** the in-memory `_ema` map.
- **owned-by:** optional argument to `Orchestrator` (`engine.py:425`), `ConstraintPlanner` (`planner.py:181-188`), and `budget_enforcer.py:65`.
- **joins:** `planner.py:485` uses a learned value when one exists.
- **looks-like-but-is-not:** `engine_core/container.py:41` imports `CostPredictor` from `..cost_predictor`. No `orchestrator/cost_predictor` module exists, so that import is stale (I did not read the block header around it).

## If you change this

- **Hits:** `planner.py`, `budget_enforcer.py`, `engine.py`; [CostForecaster](CostForecaster.md) calls `predict` and `cheapest_model`.
- **Does not hit:** `Budget`, `BudgetHierarchy`. It only estimates; it never charges.
- **Restart loses learning:** the averages start empty each process.

## Surfaces

| Surface | Role |
|---|---|
| `planner.py` | reads |
| `application/budget_enforcer.py` | reads (holds it) |
| whoever calls `record` | writes. Callers of `record` were not traced. |

## See

- Source: `orchestrator/cost.py:499`
