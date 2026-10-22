# Map — how to walk it

**Job:** let an agent answer "what is X" and "what else moves if I change X" without reading source.

## Inputs
The code under `orchestrator/`. Verify a claim against source, not against another doc. If a comment or doc and the code disagree, the code wins and the card says so.

## Universes
| Universe | Meaning | Rule |
|---|---|---|
| live | In force | Implement and cite against these |
| leftover | Still present, not the main path | Touch only if that path is in scope |
| ghost | Named or filed, not wired | Do not implement against these |

Known leftovers: root shims `orchestrator/state.py` and `orchestrator/api_clients.py` (re-export from `infrastructure/`); several 250–400 byte files in `orchestrator/engine_core/` (not yet read). Ghosts: none confirmed yet.

## Name collisions
- `AuthenticationError` is defined in both `domain/exceptions.py` and `infrastructure/llm_client.py`.
- `Budget` (per run, `budget.py`) is not `BudgetHierarchy` (cross-run, `cost.py`). Both can be active.
- `domain/ports.py` and root `ports.py` are different files.

## Doc drift found while mapping
Older docs (`docs/DEV_GUIDE.md`, `docs/CI_AND_CONTRACTS.md`) say `Budget` lives in `models.py` and `engine.py` is about 5,100 lines. Current source: `Budget` is defined in `budget.py:36` and re-exported from `models.py:1054`; `engine.py` is about 1,400 lines. They also say 5 import contracts; `.importlinter` has at least 6.

## Findings from the work cluster (slice 2)
- `JobSpec` and `ProjectSpec` in `models.py` are not the live inputs. The live `JobSpec` is `policy.py:227`; the live chat-built `ProjectSpec` is `application/conversation_agent.py:40`.
- `Task` and `TaskResult` lose fields on resume (see their cards). `ProjectState` does not.
- More collisions: `TaskStatus` and `TaskResult` also exist in `engine_core/a2a_protocol.py`; `Verdict` also exists in `generators/wf100/report.py`.

## Findings from the money cluster (slice 2)
- `Budget.charge` skips per-phase accounting for phases not in its five keys; `"critique"` and `"verbalized_sampling"` charges only raise the total (`budget.py:128`).
- `BudgetHierarchy.can_afford_job` reserves as a side effect; a caller must settle or release. Nothing in production code builds a `BudgetHierarchy` unless the caller passes one.
- Ghosts: `CostForecaster` (no call site) and `TokenBudgetManager` (no live caller). `orchestrator/costing/core.py` and root `token_budget.py` are leftover copies or shims.
- `engine_core/container.py:41` imports `CostPredictor` from a `cost_predictor` module that does not exist.

## Output
Cards under `objects/<cluster>/`. Status values: `stub`, `verified` (needs date, commit and citations), `stale`.

## Human check
A card marked `verified` must cite `path:line` and carry a date and commit. Slice N does not start until a person has read slice N-1.
