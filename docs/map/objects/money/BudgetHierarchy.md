---
type: object
cluster: money
universe: live
status: verified
verified: 2026-10-22 at 3b5f746d (source unchanged since d74829e6)
entity: orchestrator/cost.py
---

# BudgetHierarchy

Cross-run spending caps at three levels: org, team and job. The tightest cap wins.

## Why this shape

Per-run `Budget` cannot stop spend across many runs, so this class keeps running totals, optionally in SQLite, and reserves estimated cost up front so concurrent jobs see each other's pending spend (`cost.py:136-155`, `:254-266`).

## Shape

- Constructor: `org_max_usd`, `team_budgets`, `job_budgets`, optional `db_path` (`cost.py:157-163`).
- Spent: `_org_spent`, `_team_spent`, `_job_spent`, loaded from the DB when `db_path` is set (`cost.py:169-171`).
- Reservations are in memory only and never persisted: `_reserved_usd`, `_team_reserved`, `_reservations`, `_anon_reservations` (`cost.py:173-187`).
- Flow: `can_afford_job` (check and reserve) → `charge_job` (settle) or `release_reservation` (abort) (`cost.py:198`, `:409`, `:431`).
- Synchronous, no lock. Its docstring says it does not replace the per-run `Budget` (`cost.py:146-149`).

Citations: `orchestrator/cost.py:136-491`

## Connected to

- **owns:** nothing.
- **owned-by:** passed into `Orchestrator(budget_hierarchy=...)` (`engine.py:424`, `:440`, `:573`) and held by the container (`engine_core/container.py:1100`).
- **joins:** `application/budget_enforcer.py:382` (pre-flight, raises `BudgetExceededError`) and `:393` (charge); `application/project_runner.py:492` (release on failure).
- **looks-like-but-is-not:** [Budget](Budget.md). `orchestrator/costing/core.py` is a re-export shim of this module (it once was a diverged fork; see `docs/hunts/t1-money/inventory.md`).

## If you change this

- **Hits:** `budget_enforcer.py`, `project_runner.py`, `engine.py`, `container.py`; the `budget_hierarchy` SQLite table (`cost.py:301`).
- **Does not hit:** `Budget`, `ProjectState` (it is not saved there).
- **Side effect in a query:** `can_afford_job` reserves when it returns `True` (`cost.py:254-266`). A caller that never calls `charge_job` or `release_reservation` leaks the reservation for the life of the process.
- **Job cap ignores reservations on check:** the job check uses `_job_spent` only (`cost.py:241-243`), while `remaining("job")` subtracts the job's reservation (`cost.py:468`). Org and team checks include reservations.
- **Not built by default:** no production code constructs it; only docstrings (`cost.py:26`, `:149`) show construction. It exists only if the caller passes one.

## Surfaces

| Surface | Role |
|---|---|
| `budget_enforcer.py`, `project_runner.py` | reads and writes |
| `safety/guardrails.py:121`, `application/unattended_guard.py:94` | mention it as the hard limit (text) |

## See

- Source: `orchestrator/cost.py:136`
