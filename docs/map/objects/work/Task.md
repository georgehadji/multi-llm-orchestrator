---
type: object
cluster: work
universe: live
status: verified
verified: 2026-10-22 at 80161d2c (source unchanged since d74829e6)
entity: orchestrator/models.py
---

# Task

One unit of work the engine sends to an LLM: a prompt plus its type, dependencies, thresholds and retry limits.

## Why this shape

A plain mutable dataclass with no behaviour, because `models.py` must stay pure data (rule 2). Type-specific defaults (thresholds, iterations, token limits) are set by `TaskFactory.create()`, not here, so the dataclass defaults are only fallbacks (`models.py:1117-1121`).

## Shape

- Required: `id`, `type: TaskType`, `prompt` (`models.py:1079-1083`).
- Run control: `dependencies`, `acceptance_threshold=0.85`, `max_iterations=3`, `max_output_tokens=1500`, `status=PENDING`, `hard_validators` (`models.py:1087-1097`).
- App Builder fields: `target_path`, `module_name`, `tech_context`, `target_language` (`models.py:1101-1107`).
- Routing and mode fields: `preferred_model`, `revision_context`, `mode` (`""` means STANDARD), `design_variant` (`models.py:1109-1115`).

Citations: `orchestrator/models.py:1077-1121`

## Connected to

- **owns:** nothing. It points at `TaskType` and `TaskStatus` ([WorkEnums](WorkEnums.md)).
- **owned-by:** [ProjectState](ProjectState.md) (`tasks: dict[str, Task]`, `models.py:1201`).
- **joins:** [TaskResult](TaskResult.md) by `task.id == result.task_id`.
- **looks-like-but-is-not:** the agent-layer `task` objects in `agents/` (they return `AgentTaskResult`, e.g. `agents/developer.py:47`).

## If you change this

- **Hits:**
  - `TaskFactory.create` (`domain/task_factory.py:65`; root `task_factory.py` is a shim).
  - Persistence: `_task_to_dict` / `_task_from_dict` (`infrastructure/state.py:77`, `:97`).
  - Every stage and executor that reads task fields.
- **Does not hit:** `TaskResult` shape. A new `Task` field is not saved on resume unless you add it to the two state functions.
- **Silent loss today:** `target_language`, `preferred_model`, `revision_context`, `mode` and `design_variant` are not in `_task_to_dict` (`state.py:79-94`), so a resumed task loses them. This describes what the code writes, not intent.
- **Default mismatch:** `max_output_tokens` is 1500 on `Task` (`models.py:1093`) but the restore default is 4096 (`state.py:114`). Only matters for old checkpoints missing the key.

## Surfaces

| Surface | Role |
|---|---|
| `domain/task_factory.py` | writes (builds tasks) |
| `infrastructure/state.py` | reads and writes (checkpoints) |
| `engine.py` `_execute_task` (`engine.py:1274`) | reads |

## See

- Source: `orchestrator/models.py:1077`
