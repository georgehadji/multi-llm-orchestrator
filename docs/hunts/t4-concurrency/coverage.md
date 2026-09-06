# T4 — Concurrency & Resource Lifecycle — Phase 8 Coverage & Residual-Risk Statement

## Scope actually audited
`orchestrator/engine.py` (all `asyncio.*` primitive sites read),
`orchestrator/engine_core/pipeline_runner.py` (full file, the live
multi-task execution engine), `orchestrator/engine_core/a2a_protocol.py`'s
task-dispatch methods, `orchestrator/engine_core/health.py`'s background
monitoring loop, `orchestrator/engine_core/stages/context_enricher.py` and
`orchestrator/application/skill_manager.py`'s fire-and-forget task patterns,
`orchestrator/application/dependency_resolver.py` (to check how a missing
`results` entry is actually consumed).

## Gates (this tier's fixed tree)
```
black --line-length=100 --check --fast <changed files>          PASS
ruff check <changed files>                                        PASS
lint-imports                                                       PASS (5/5 KEPT)
python scripts/check_root_module_freeze.py                        PASS (256/256)
python scripts/check_test_markers.py                               PASS
mypy orchestrator/domain/ .../application/ .../container.py       PASS (58 files, 0 issues)
bandit -lll -r orchestrator/engine_core/pipeline_runner.py          PASS (0 issues)
python -m pytest tests/unit/test_hunt_t4_concurrency.py -m unit    PASS (3/3)
```

## RED→GREEN verification
Verified via `git stash push --keep-index` on the one fixed source file,
full T4 suite re-run, fix restored via `git stash pop`. 2 of 3 tests failed
against the pre-fix tree for the exact predicted reason (`AssertionError:
't1' in {}` and `KeyError: 'bad'` — the failed task's entry genuinely
absent from `results`). The third (`test_c1_successful_task_still_recorded_
normally`) correctly passed in both states — it pins that the success path
is unchanged by this fix, not a behavior the fix altered. All 3 pass on the
fixed tree.

## Verdict
- **VERIFIED DEFECT fixed:** 1 — `PipelineRunner.execute_all()` silently
  dropped failed tasks from `ProjectState.results` instead of recording
  them as `FAILED`.
- **CLEARED (innocent):** 5 fire-and-forget `asyncio.create_task()` sites
  checked for the GC-mid-flight leak pattern, all found to already use the
  correct held-reference pattern (see inventory.md's Phase 0 delta).
- **Residual, not fixed (dead code):** `A2ACoordinator.distribute_task()`
  has a genuine task-leak shape (orphaned tasks on early exception/
  cancellation, no `try`/`finally`) but has zero live callers — the class
  actually wired in under the `A2AManager` name is a different class,
  `A2AQueueManager`. `[REQUIRES HUMAN REVIEW]` rather than fixed blind:
  whether `A2ACoordinator` is meant to be revived, replaced, or deleted is
  an architecture question this tier shouldn't guess at, matching the
  disposition of similar dead-but-real findings in T1 (C3) and T3.
- One theoretical edge case noted, not pursued: `engine.py::
  run_project_streaming()`'s background task is only `await`ed after its
  `async for event in subscription` loop completes — if a caller abandons
  the async generator early (breaks out of iteration, or is garbage
  collected) without exhausting it, that `await` never runs and the
  background `_run()` task could continue detached. No live caller doing
  this was found or ruled out; `[UNK]`.

## Clean claim this tier is permitted to make, and no more
Within the scope listed above — the live `PipelineRunner.execute_all()`
concurrency engine and the concurrency-primitive sites enumerated in Phase
0 — no VERIFIED defect remains unfixed. This does not claim
`engine_core/`'s other ~9000 lines were exhaustively swept for every one
of the plan's 205-site estimate; candidates were generated from the
highest-confidence live entry point (the actual task-execution engine)
plus every `create_task`/`ensure_future` site, not a line-by-line review
of every file under `engine_core/`.

## What this tier does NOT claim
- It does not claim `A2ACoordinator` is safe to leave as-is — only that
  fixing dead code speculatively (with no caller to validate the fix
  against) is lower-value than flagging it for a real decision.
- It does not claim every one of the plan's ~205 concurrency-primitive
  sites was individually inspected — this tier's candidates came from a
  targeted survey (the live execution engine + every task-creation site),
  not an exhaustive line-by-line pass. T6 (error-path sweep) and any
  future revisit of `engine_core/`'s remaining files may still find more.
- It does not claim `engine_core/container.py`'s or other files' internal
  locking (beyond the sites explicitly checked) is race-free — those
  weren't independently re-derived from first principles this tier.

## `hunt_iterations` / `fix_revisions`
`hunt_iterations`: 1/3 used. `fix_revisions`: 1/1 used — the fix was
correct on first pass, confirmed via the retroactive RED→GREEN stash test
above.
