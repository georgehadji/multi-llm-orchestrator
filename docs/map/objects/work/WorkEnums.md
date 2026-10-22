---
type: object
cluster: work
universe: live
status: verified
verified: 2026-10-22 at 80161d2c (source unchanged since d74829e6)
entity: orchestrator/models.py
---

# TaskType, TaskStatus, ProjectStatus

The three enums that classify a task, a task's state, and a whole run's outcome. One card for the group; the index lists them on one line.

## Why this shape

All are `str, Enum`, so values serialize as plain strings in checkpoints (`state.py:88`, `:210`).

## Shape

- `TaskType` (13): `code_generation`, `code_review`, `complex_reasoning`, `creative_writing`, `data_extraction`, `summarization`, `evaluation`, `image_generation`, `video_generation`, `mechanism_research`, `modify_file`, `delete_file`, `install_dependency` (`models.py:65-91`). Lowercase values.
- `TaskStatus` (5): `pending`, `running`, `completed`, `failed`, `degraded` (`models.py:838-848`). Lowercase values.
- `ProjectStatus` (6): `SUCCESS`, `PARTIAL_SUCCESS`, `COMPLETED_DEGRADED`, `BUDGET_EXHAUSTED`, `TIMEOUT`, `SYSTEM_FAILURE` (`models.py:823-835`). Uppercase values.

Citations: `orchestrator/models.py:65`, `:823`, `:838`

## Connected to

- **owns / owned-by:** `Task.type`, `Task.status`, `TaskResult.status`, `ProjectState.status`.
- **looks-like-but-is-not:** `TaskStatus` in `engine_core/a2a_protocol.py:601` is a different enum. `DesignVariant`, `DesignScope`, `Genre` in `models.py` are not work-cluster types.

## If you change this

- **Hits:**
  - `TaskType`: about 109 files reference it. Routing tables key on it, and `TaskType(d["type"])` in `state.py:101` raises on an unknown stored value.
  - `TaskStatus`: `application/executor.py:45` treats `COMPLETED` and `DEGRADED` as success; `analysis/progress.py:89` maps statuses to symbols.
  - `ProjectStatus`: `state.py:223` restores it from the stored string.
- **Does not hit:** the a2a `TaskStatus`. Removing or renaming a member breaks saved checkpoints that contain the old value.

## Surfaces

| Surface | Role |
|---|---|
| executors, engine | writes |
| `infrastructure/state.py`, progress and dashboard code | reads |

## See

- Source: `orchestrator/models.py`
