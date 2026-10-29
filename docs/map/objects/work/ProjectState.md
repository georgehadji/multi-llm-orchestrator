---
type: object
cluster: work
universe: live
status: verified
verified: 2026-10-22 at 80161d2c (source unchanged since d74829e6)
entity: orchestrator/models.py
---

# ProjectState

The whole run in one serializable object: description, criteria, budget, tasks, results and status. It is what gets saved and resumed.

## Why this shape

Resume needs everything in one blob. All eight fields are written and read back by `_state_to_dict` / `_state_from_dict` (`infrastructure/state.py:202-225`), so unlike `Task` and `TaskResult` nothing is dropped.

## Shape

- `project_description`, `success_criteria`, `budget: Budget | None` (`models.py:1195-1199`).
- `tasks: dict[str, Task]`, `results: dict[str, TaskResult]` (`models.py:1201-1203`).
- `api_health: dict[str, bool]`, `status = PARTIAL_SUCCESS`, `execution_order: list[str]` (`models.py:1205-1209`).
- `Budget` is imported into `models.py` from `budget.py` (`models.py:1054`), so `models.Budget` works but the class lives in `budget.py:36`.

Citations: `orchestrator/models.py:1191-1209`

## Connected to

- **owns:** [Task](Task.md), [TaskResult](TaskResult.md), a `Budget`, a `ProjectStatus` ([WorkEnums](WorkEnums.md)).
- **owned-by:** returned by `Orchestrator.run_project` (`engine.py:955`) and `run_job` (`engine.py:1080`).
- **joins:** stored by `StateManager.save_project` / `save_checkpoint` / `load_project` (`infrastructure/state.py:349`, `:378`, `:423`).
- **looks-like-but-is-not:** `BudgetHierarchy` (`cost.py:136`) is cross-run and is not stored here.

## If you change this

- **Hits:** `_state_to_dict` / `_state_from_dict`; `infrastructure/state_migration.py`; `engine_core/state_coordinator.py`, `engine_core/pipeline_runner.py`, `application/resumption_service.py`; about 32 files reference it.
- **Does not hit:** `Verdict`, and `BudgetHierarchy`.
- **Old checkpoints:** a new required field breaks `_state_from_dict` for saved blobs; give it a default and read it with `d.get`.

## Surfaces

| Surface | Role |
|---|---|
| `engine.py`, `application/project_runner.py` | writes |
| `infrastructure/state.py` | reads and writes |
| `api_server.py`, `dashboard_core/core.py`, writers | reads |

## See

- Source: `orchestrator/models.py:1191`
