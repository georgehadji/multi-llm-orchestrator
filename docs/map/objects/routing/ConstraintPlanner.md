---
type: object
cluster: routing
universe: leftover
status: verified
verified: 2026-10-22 at 26fbe258 (source unchanged since d74829e6)
entity: orchestrator/planner.py
---

# ConstraintPlanner

Picks a model for a task by filtering on health, policy, capability, budget and rate limits, then scoring the survivors. Built by the container, but its selection methods are only called from tests.

## Why this shape

Selection is split into filters plus a pluggable optimization backend (`GreedyBackend` by default) so a different strategy can be swapped in, with an optional shadow backend for comparison (`planner.py:175-210`). The static `FALLBACK_CHAIN` is meant to be replaced at runtime by `replan()` (`planner.py:346-349`).

## Shape

- Constructor: `profiles`, `policy_engine`, `api_health`, optional `backend`, `cost_predictor`, `shadow_backend` (`planner.py:175-190`).
- `select_model` (`:216`), `select_reviewer` (cross-provider first, `:270`), `replan` (`:335`).
- Five filters in `_apply_filters`: API health, policy, capability for the task type, budget, rate limit (`:378-475`).
- `_estimate_typical_cost` uses `CostPredictor` when it has learned data, else token-count pricing (`:477-497`).
- `_score` is quality × trust ÷ (estimated cost + ε) (`:499-511`).

Citations: `orchestrator/planner.py:165-511`

## Connected to

- **owns:** the shared `RateLimitTracker` (`planner.py:190`).
- **owned-by:** the container builds it without a `cost_predictor` or backend (`engine_core/container.py:611-613`); it is held as `planner` (`:220`).
- **joins:** `engine.py:806` and `:837` read `self._c.planner._profiles`; the profiles are `ModelProfile` from `policy.py:77`.
- **looks-like-but-is-not:** `application/fallback_handler.py` has its own `select_model` (`:136`) and `select_reviewer` (`:169`) that read `ROUTING_TABLE` directly (`:103-106`), and `task_executor.py:179` calls that one.

## If you change this

- **Hits:** `tests/unit/test_planner.py` and `tests/integration/test_acr_parity.py:61-62` (the only callers of its selection methods that I found); `container.py`; `engine.py`'s use of `_profiles`.
- **Does not hit:** the model actually chosen at run time. `FallbackHandler` does that, so editing the planner's filters or scoring does not change production routing.
- **Predictor not wired:** the container builds the planner without a `CostPredictor` (`container.py:611-613`), so `_estimate_typical_cost` uses the static estimate by default.
- **Caveat:** a grep for `select_model(`, `select_reviewer` and `.replan(` found no production callers. A call through a dynamic name would not show.

## Surfaces

| Surface | Role |
|---|---|
| `engine.py` | reads `_profiles` |
| tests | calls the selection methods |

## See

- Source: `orchestrator/planner.py:165`; [CostPredictor](../money/CostPredictor.md)
