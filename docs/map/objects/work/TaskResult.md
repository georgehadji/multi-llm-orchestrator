---
type: object
cluster: work
universe: live
status: verified
verified: 2026-10-22 at 80161d2c (source unchanged since d74829e6)
entity: orchestrator/models.py
---

# TaskResult

What one task produced: output text, score, which model ran it, cost, status and attempt history.

## Why this shape

One record per task so a run can be saved and resumed. `success` is derived, not stored (`models.py:1169-1173`).

## Shape

- Core: `task_id`, `output`, `score`, `model_used: Model`, `reviewer_model`, `tokens_used`, `iterations`, `cost_usd`, `status=COMPLETED`, `critique` (`models.py:1127-1145`).
- Quality: `deterministic_check_passed`, `degraded_fallback_count`, `attempt_history: list[AttemptRecord]`, `preflight_result`, `preflight_passed` (`models.py:1147-1155`).
- TDD artifacts: `test_files`, `tests_passed`, `tests_total`; plus `metadata`, `task_type` (`models.py:1157-1167`).
- `success` is true only when status is `COMPLETED` and `score > 0.5`. The docstring says "score > 0" but the code uses 0.5; the code wins (`models.py:1171-1173`).

Citations: `orchestrator/models.py:1124-1173`

## Connected to

- **owns:** a list of [AttemptRecord](AttemptRecord.md).
- **owned-by:** [ProjectState](ProjectState.md) (`results: dict[str, TaskResult]`, `models.py:1203`).
- **joins:** [Task](Task.md) by id.
- **looks-like-but-is-not:** `TaskResult` in `engine_core/a2a_protocol.py:671`, and `AgentTaskResult` in `agents/`.

## If you change this

- **Hits:** `_result_to_dict` / `_result_from_dict` (`infrastructure/state.py:149`, `:168`); executors that build results (`application/executor.py`, `application/task_executor.py:327`); output writers and dashboards that read results.
- **Does not hit:** `Task`. Importing the wrong `TaskResult` (the a2a one) compiles but is a different type.
- **Silent loss today:** `_result_to_dict` does not write `preflight_result`, `preflight_passed`, `test_files`, `tests_passed`, `tests_total` or `metadata` (`state.py:149-165`). After resume they fall back to dataclass defaults.

## Surfaces

| Surface | Role |
|---|---|
| `application/task_executor.py` | writes |
| `infrastructure/state.py` | reads and writes |
| about 51 files reference the name | reads (grep count, includes the a2a type) |

## See

- Source: `orchestrator/models.py:1124`
