# T4 — Concurrency & Resource Lifecycle — Phase 4 Inventory

Per `docs/DEFECT_HUNT_PLAN.md` §3/§4. Budget: 14 candidates. Spent: 3.

## Phase 0 delta

- `orchestrator/engine.py` is 1264 lines — much smaller than its "God
  Mediator" reputation suggests; most of the 205 concurrency-primitive
  sites live in `orchestrator/engine_core/` (9184 lines across many
  files), not `engine.py` itself.
- The actual multi-task concurrent execution engine is
  `orchestrator/engine_core/pipeline_runner.py::PipelineRunner.execute_all()`
  (level-based `asyncio.gather` + `asyncio.Semaphore` + `asyncio.Lock`) —
  this is what `ProjectRunnerCallables.execute_all` (established in T1) is
  bound to in the live pipeline.
- Checked every `asyncio.create_task`/`ensure_future` site in
  `engine_core/`+`application/`+`engine.py` (7 sites) for the classic
  "task garbage-collected mid-flight" leak (CPython's own documented
  gotcha: a `create_task()` result must be referenced somewhere or the
  event loop may drop it). Five were already correct, using the
  recommended pattern (store in a set, `add_done_callback` to discard):
  `engine_core/stages/context_enricher.py`, `application/skill_manager.py`,
  `engine_core/health.py` (stored as `self._check_task`, properly
  cancelled+awaited in `stop_monitoring()`), and `engine.py:959`'s
  streaming-generator task (properly `await`ed after the event loop, even
  if early generator abandonment remains a theoretical edge case not
  pursued further this tier).
- `engine_core/a2a_protocol.py::A2ACoordinator.distribute_task()` has a
  real task-leak shape (creates N tasks upfront, awaits them sequentially
  in a loop with no `try`/`finally`; an early exception or cancellation
  mid-loop leaves the remaining tasks orphaned, running detached with no
  error handling) — but `A2ACoordinator`/`distribute_task`/
  `get_a2a_manager()` (which resolves to it) have **zero callers anywhere**
  in the repository (confirmed by grep). The class actually wired in via
  `A2AManager` is an *alias for a different class*, `A2AQueueManager` (line
  829: `A2AManager = A2AQueueManager`), not `A2ACoordinator`. Recorded as
  a residual finding, not fixed — see coverage.md.

## Candidates

### C1 — VERIFIED DEFECT — a failed task vanished from `ProjectState.results` instead of being recorded as failed
- **Property violated:** threat 3 (silent wrong result) via a concurrency
  mechanism (class 3) — "an orchestrator that reports success on work it
  did not do... ranked above crashes: a crash is visible" (§1.1). Here the
  failure is neither a crash nor a success — it's an absence.
- **Location:** `orchestrator/engine_core/pipeline_runner.py::
  PipelineRunner.execute_all().run_one()` (pre-fix).
- **Finding:** `run_one()` awaited `execute_task_fn(task)` with no
  surrounding `try`/`except`. When it raised, the exception propagated out
  of the coroutine to `asyncio.gather(..., return_exceptions=True)`, which
  captured it as that task's "result" — but `results[tid] = res` (the line
  that writes into the shared dict) is only reached on the success path,
  *before* the exception point. The post-`gather` loop only logged the
  failure (`logger.error(...)`); it never wrote anything to `results`.
  Net effect: a failed task's ID is simply **absent** from
  `ProjectState.results`, indistinguishable from "never attempted."
- **Reachability:** live — this is the actual task-execution loop bound
  to `ProjectRunnerCallables.execute_all` in production (confirmed in T1's
  investigation of `project_runner.py`).
- **Innocence attempt:** checked whether a missing entry causes a crash
  downstream (it would be a worse bug if so) — `application/
  dependency_resolver.py::get_dependency_context()` already defends
  against a missing `results[dep_id]` (`if dep_id not in results or not
  results[dep_id].success: ... continue`), so nothing crashes. But
  graceful degradation elsewhere doesn't make the missing record correct
  — anyone inspecting `ProjectState.results` after a run (dashboards,
  `docs/hunts`-style forensics, a human debugging a partial run) sees no
  trace that the task was ever attempted, only that it's missing, with no
  way to distinguish "failed" from "not reached yet" from `results` alone.
- **Fix:** `run_one()` now catches the exception around the
  `execute_task_fn` call and builds a `TaskResult(status=FAILED,
  critique=str(exc), ...)` via a new `_build_failure_result()` helper —
  deliberately mirroring the **already-established** convention in
  `application/task_executor.py::_build_failure_result()` (same
  placeholder-model choice, `Model.GPT_4O_MINI`, for a task that never
  got far enough to select a real one), rather than inventing a new
  shape. The lock-protected write to `results[tid]` now always happens,
  success or failure. The post-`gather` `return_exceptions=True` backstop
  is kept as-is (genuine defense-in-depth for a failure outside the
  execute_task_fn boundary, e.g. in the lock/semaphore machinery itself),
  not removed as redundant.
- **Tests:** `test_c1_failed_task_gets_recorded_as_failed_not_dropped`,
  `test_c1_successful_task_still_recorded_normally`,
  `test_c1_mixed_level_records_both_success_and_failure` (two concurrent
  tasks in the same level, one fails, one succeeds — both must appear in
  `results`).

## Gates (Phase 8, run against the fixed tree)

See `coverage.md`.
