# T11 — Infrastructure / Engine_Core / Domain / Events Core — Phase 4 Inventory

Per `docs/DEFECT_HUNT_PLAN.md` §3/§4/§7.5 and `docs/hunts/BACKEND_REMAINDER_WAVES_PLAN.md`'s
wave definition. Budget: 118 files surveyed (the largest wave to date), 3 fixed.

## Phase 0/1/2/3 — survey method

A background agent read all 118 files in the wave's declared scope, at a
tiered depth: four specific pre-identified leads investigated in full
depth first, a second tier of architecturally-central files (domain
ports/contracts, the DI container's neighbors, the execution pipeline,
the event-bus packages) read for correctness, and the remainder given a
purpose+reachability pass with anything suspicious flagged rather than
forced to uniform depth. This matches the plan's own guidance that a
118-file wave does not need — and the budget does not permit — the same
per-file depth as a 19-48-file wave.

## Candidates fixed

### C1 — VERIFIED DEFECT — `config/costs.json` silently priced `qwen/qwen3.6-flash` at $0.00 on every real call

- **Property violated:** class 1 (money/budget escape) — a real, billable
  API call computing and reporting exactly $0.00 cost.
- **Location:** `infrastructure/llm_client.py:386-390`, inside
  `UnifiedClient.call()`'s real (non-cached, non-batch) response branch —
  confirmed to be the hot path: nearly every OpenRouter call returns a
  native `ChatCompletion`, not a wrapped `APIResponse`, so this branch runs
  for essentially all real traffic. `cost_rates = self._cost_service.
  get_cost(model_enum)` calls `domain/services/config_services.py::
  CostService.get_cost()`, which does
  `costs.get(model.value, {"input": 0.0, "output": 0.0})` against
  `config/costs.json` (read via `infrastructure/adapters/config_adapter.py::
  JsonConfigAdapter`) — a silent zero-default for any missing key.
- **Finding, and an important correction to the initial survey's severity
  claim:** the survey's first pass reported **5** models missing a
  `costs.json` entry: `qwen/qwen3.6-flash`, `anthropic/claude-opus-4`,
  `claude-opus-4.1`, `claude-sonnet-4`, `internal/nano-banana-2`.
  Independently re-verified each before fixing (per this hunt's standing
  "trust but verify" discipline for background-agent findings) and found
  the severity claim needed correction: `models.py` lines 373-375 show
  `CLAUDE_OPUS_4_0`/`CLAUDE_OPUS_4_1`/`CLAUDE_SONNET_4_0` are **commented
  out** under an explicit "Deprecated Models (for reference)" heading —
  confirmed via direct enum iteration
  (`for m in Model: assert m.value not in {...}`) that none of these three
  strings correspond to any live `Model` enum member. `model_enum` can
  never actually equal these values, so `CostService.get_cost()` can never
  be called with them — these 3 of the 5 were not a live risk at all.
  `internal/nano-banana-2` *is* a live enum member (`NANO_BANANA_2`), but
  its own `models.py::COST_TABLE`/`CONTEXT_WINDOWS` entries carry the
  comments `# Example cost` / `# Example context size` — a
  self-documented placeholder, not researched real-world pricing.
  `qwen/qwen3.6-flash` is the one genuine, currently-real, actively-used
  model affected: `models.py::COST_TABLE` already has correct pricing
  (`$0.1875`/`$1.125` per 1M tokens, verified against the repo's own
  OpenRouter snapshot earlier this same hunt), but `config/costs.json` —
  the file the live cost-computation path actually reads — never had an
  entry for it at all.
- **Reachability:** live, on every real (non-cached) call to
  `qwen/qwen3.6-flash` — a model this repo's `ROUTING_TABLE`/fallback
  chains actively route to.
- **Innocence attempt:** none needed for `qwen/qwen3.6-flash` — a plain
  missing config entry with no mitigating default. `CLAUDE_OPUS_4_0`/
  `4_1`/`SONNET_4_0` are CLEARED (unreachable, not live enum values, see
  above).
- **Fix:** added `"qwen/qwen3.6-flash": {"input": 0.1875, "output":
  1.125}` to `config/costs.json`, matching `models.py::COST_TABLE`'s
  already-correct values. Also added
  `"internal/nano-banana-2": {"input": 0.01, "output": 0.01}` for
  consistency with `models.py`'s own placeholder entry (cheap to add,
  removes an inconsistency between two files both admittedly holding
  example/placeholder numbers for this one internal-only model) — did
  **not** research real pricing for it, since its own source comments
  self-identify it as a non-production example.
- **Test:** `test_c1_qwen_3_6_flash_has_nonzero_cost_entry` — calls the
  real `CostService(JsonConfigAdapter()).get_cost(Model.QWEN_3_6_FLASH)`
  and asserts non-zero input/output cost (pre-fix: `{"input": 0.0,
  "output": 0.0}`).
- **Discovered while fixing, recorded but NOT part of this fix:** running
  the repo's own config-drift diagnostic
  (`.claude/skills/orchestrator-diagnostics-and-tooling/scripts/
  check_config_drift.py`) after this fix confirmed the 2 new entries
  resolved the exact gap found, and surfaced two much larger, separate,
  pre-existing conditions: (a) `routing.json` has a large number of
  task-type keys and model-value entries that don't match live
  `TaskType`/`Model` enum values and are silently dropped by whatever
  reads it — dozens of entries across many models; (b) 7 `Model` values
  (all `:free`-suffix OpenRouter variants) have no `costs.json` entry —
  but for these the $0.0 default is very plausibly *correct* (a genuinely
  free-tier model's real cost is $0.00), not a bug, so these were **not**
  added. Neither (a) nor (b) was part of this survey's own findings or
  this tier's declared scope; (a) in particular looks large enough to
  warrant its own dedicated investigation rather than a fold-in here.
  `[REQUIRES HUMAN REVIEW]` / candidate for a future tier.

### C2 — VERIFIED DEFECT — `engine_core/project_planner.py::get_execution_levels()` silently dropped circularly-dependent tasks with no diagnostic

- **Property violated:** class 3 (silent wrong/incomplete result) — the
  live task-scheduling method returning fewer tasks than were given it,
  with nothing anywhere naming which tasks were lost or why.
- **Location:** `get_execution_levels()`'s Kahn's-algorithm loop.
- **Finding:** the `while ready:` loop stops the moment `ready` is empty;
  if a cycle exists, the cyclic tasks' in-degree never reaches 0 and they
  simply never appear in any level, with no completeness check
  afterward. This method is called directly by
  `engine_core/pipeline_runner.py::PipelineRunner.execute_all()` — the
  real live execution loop; only tasks appearing in the returned levels
  are ever passed to `execute_task_fn`.
- **Reachability:** live. **Genuine partial mitigation found and
  honestly reported** (not claimed as a full innocence defense):
  `engine_core/state_coordinator.py::determine_final_status()` computes
  `all_tasks_executed = len(state.results) == len(state.tasks)`, so a
  cycle correctly downgrades the run to `PARTIAL_SUCCESS` rather than
  falsely claiming full `SUCCESS` — but no log, error, or result field
  anywhere names *which* tasks were dropped or that a cycle was the
  cause, leaving an operator with an unexplained partial run. The sibling
  `application/dependency_resolver.py::topological_sort()` already does
  this correctly (`raise ValueError(f"Circular dependency detected
  involving tasks: {missing}")`) but a separate, confirmed-broken wiring
  bug in `container.py:536-548` (an import of a class,
  `engine_deps.py::_DepResolver`, that does not exist anywhere in that
  file) means the real resolver is never actually injected — a no-op stub
  is used instead, so this file's own local fallback is the *only* path
  that ever actually runs, cycle or not.
- **Innocence attempt:** none for the missing diagnostic itself — the
  partial-success fallback (above) is real but does not name the cause,
  which is the actual defect being fixed here.
- **Fix:** after the loop, compares the set of scheduled task IDs against
  the full input task set; if any are missing, logs an `ERROR` naming
  exactly which task IDs were dropped due to a circular dependency. This
  is deliberately conservative: it does **not** raise (which would be a
  behavior change for two live callers, `pipeline_runner.py` and
  `engine.py`, whose exception-handling around this call was not
  independently re-verified as safe to change) and does **not** change
  the returned `levels` value at all — `StateCoordinator`'s existing
  `PARTIAL_SUCCESS` fallback behavior is preserved exactly, now simply
  accompanied by an actionable diagnostic instead of silence.
- **Not fixed here — `[REQUIRES HUMAN REVIEW]`:** the broken
  `container.py:536-548` `_DepResolver` import (wiring a real,
  cycle-raising resolver into the live pipeline is an architecture
  decision — this hunt has consistently declined to make that call
  unilaterally, e.g. T1's `cost_optimization`/`BudgetEnforcer`, T8's
  `configure_logging`, T9's plugin isolation subsystem).
- **Test:** `test_c2_circular_dependency_is_logged_not_silent` — a real
  trigger: three tasks, two of them (`a`, `b`) forming a genuine cycle,
  one (`c`) acyclic; asserts `c` still schedules normally (the fix does
  not touch the good path) and an `ERROR` log names both `a` and `b`
  (pre-fix: no log at all, `caplog.records == []`).

### C3 — VERIFIED DEFECT — `engine_core/pipeline_executor.py`'s retry loop treated its own VS tail-escape signal as terminal

- **Property violated:** a feature that configures a retry, then never
  actually gets to run it.
- **Location:** the `while True: ... if ctx.abort_reason not in
  ("retry_for_quality", "ara_retry"): break` loop.
- **Finding:** `engine_core/stages/self_consistency.py`'s own VS
  tail-escape branch (verbalized-sampling diversity escape when a task is
  stuck below its quality threshold) sets `ctx.task.revision_context` to
  a real, purpose-built prompt, then sets `ctx.abort_reason =
  "vs_retry_escape"` specifically to request another pipeline pass with
  that revision context — but the executor's loop only recognized two
  other string values as "keep retrying," so `"vs_retry_escape"` fell
  through to `break`, discarding the escape attempt immediately and
  returning whatever score the task had *before* the escape was even
  configured.
- **Reachability:** live whenever `flags.vs_retry_escape` is enabled
  (default `False`, but `application/search_strategy_tuner.py` can
  autonomously flip it on when a run's `repetition_rate > 0.3` — exactly
  the "stuck, need a diversity escape" condition this feature exists for).
- **Innocence attempt:** none — a plain missing-value-in-tuple bug, no
  ambiguity about intent (the stage's own code and log message
  ("VS tail-escape engaged") make the intended retry unambiguous).
- **Fix:** added `"vs_retry_escape"` to the loop's recognized
  `abort_reason` tuple. Verified the stage's own `ctx.reset_for_retry()`
  call (made just before setting `abort_reason`) is idempotent
  (`should_abort = False; abort_reason = ""`), so the loop's second call
  to the same method on the retry path is harmless, and confirmed
  `self_consistency.py` already bounds retries via its own `max_attempts`
  check before ever reaching this branch — no infinite-loop risk
  introduced.
- **Test:** `test_c3_pipeline_executor_retries_on_vs_retry_escape` — a
  real trigger: a fake pipeline whose `.run()` returns `abort_reason =
  "vs_retry_escape"` on the first call and a normal completion on the
  second; asserts the pipeline is actually called twice and the final
  status is `COMPLETED` (pre-fix: called once, loop broke immediately on
  the escape signal).

## Candidates surveyed, not fixed — cleared or `[REQUIRES HUMAN REVIEW]`

The full 118-file, tiered-depth disposition table is preserved in this
session's transcript; the items below are the ones with real, independent
significance for a future reader (someone continuing this wave or citing
its coverage):

- **`infrastructure/streaming_resilient.py::CircuitBreaker`** — an
  independently-verified real defect (an unsynchronized `OPEN→HALF_OPEN`
  transition with no lock and no probe-in-flight guard, letting up to
  `half_open_max_calls` — default 3 — requests through during HALF_OPEN
  instead of exactly one; the same defect class T5 already fixed in the
  canonical `circuit_breaker.py`). **Not fixed**: `ResilientStreamingPipeline`
  (its only constructor) has zero importers anywhere in the repo —
  confirmed fully dead. `[REQUIRES HUMAN REVIEW]` if this module is ever
  wired in; recorded here so a later tier doesn't need to re-discover it.
- **`unified_events/core.py::UnifiedEventBus`** — constructed live in
  `container.py` and wired into `application/project_runner.py`'s real
  `.publish()` calls, but `.start()` is never called anywhere in the
  repo — confirmed by exhaustive grep. Consequences, all independently
  plausible given the code read: event persistence and the
  `ProjectStateProjection`/`MetricsProjection` read-models never update
  (any `get_project_state()`/`get_metrics()` caller gets permanently
  stale/empty data), and — more concerning — `self._event_queue`
  (unbounded `asyncio.Queue`) grows for the life of the process every
  time `publish()` is called with nothing ever draining it, a genuine
  resource-growth shape in a long-running host (dashboard/API server).
  `events/async_event_store.py::AsyncEventStore` — self-described in its
  own docstring as "CRITICAL FIX: Replaces synchronous EventStore in
  unified_events/core.py" — was apparently built to address part of this
  and also has zero live callers. **Not fixed**: calling `.start()` would
  activate a large amount of previously-dormant machinery (persistence,
  projections, subscriber notification) that has not been exercised or
  validated as free of its own bugs — a materially bigger behavior change
  than this tier's three fixes, and the specific question of "should the
  queue be bounded even without full `.start()` wiring" is a design
  choice, not an obvious bug fix. `[REQUIRES HUMAN REVIEW]`.
- **`engine_core/dep_resolver.py` vs `engine_core/dependency_resolver.py`**
  — investigated as a possible duplicate pair per the wave's assigned
  leads; turned out **not** to be one. `dep_resolver.py::DependencyResolver`
  (third-party import scanning for generated projects) is live
  (`appbuilder/builder.py`, `app_builder.py`). `dependency_resolver.py` is
  a dead, zero-importer shim to the unrelated, actually-live
  `application/dependency_resolver.py::DependencyResolver` (task-DAG
  resolver) — `engine_core/__init__.py` imports the real one directly,
  bypassing this shim entirely. Confirmed harmless (nothing imports the
  dead shim) but a confusing orphan next to a live file of a near-identical
  name. Not fixed — deleting it or renaming for clarity is a judgment call
  with no behavioral stakes either way; recorded as `[UNK]`/low-priority.
- **`domain/model_registry.py`'s stale `QWEN_3_6_FLASH` `COST_TABLE`/
  `MODEL_MAX_TOKENS` entries** — confirmed stale (still `{"input": 0.12,
  "output": 0.50}`/`32768` after the id/comment were updated elsewhere) and
  confirmed dead (zero callers of `ModelRegistry.get_cost`/`get_max_tokens`
  anywhere). Not fixed this tier — genuinely lower priority than C1 (which
  is live) and not part of the 3-fix budget spent; `[UNK]`, a one-line fix
  if anyone picks it up later.
- **`engine_core/escalation.py` + `engine_core/evaluation.py`** — both
  self-documented DEPRECATED (raise `DeprecationWarning` on construction;
  the module docstring says new code must not import from it), both dead
  (zero live callers), and both internally broken (`UnifiedClient` has no
  `.acomplete()` method anywhere in the repo — every call would raise
  `AttributeError`, silently caught and replaced with a fabricated
  `EvaluationResult(score=0.5, ...)`, the same "fake neutral score" shape
  T6 already flagged live elsewhere). Not fixed — the module already
  declares itself obsolete; repairing a self-declared-deprecated module's
  internals serves no live purpose.
- **Large confirmed-dead cluster in `infrastructure/`** — `cache_optimizer.py`,
  `caching.py` (which also defines a second, unrelated `DiskCache` — a
  naming collision with the canonical, live `infrastructure/cache.py::
  DiskCache`), `secure_cache.py`, `semantic_cache.py`, `database_manager.py`
  (despite its docstring's claim to centralize "all 9 SQLite databases" —
  every live DB-backed module manages its own connection independently),
  `monitoring.py`, `infrastructure/sandboxes/docker_sandbox.py` (reconfirmed
  a bare stub, no `exec()` method, doesn't satisfy `SandboxPort`), and
  `infrastructure/sandboxes/subprocess_sandbox.py` (a correct, complete
  implementation — env-scrubbing, timeout/kill handling — with zero live
  callers). All individually verified dead (zero live construction sites),
  none fixed — consistent with this repo's own documented "unwired
  parallel infrastructure" pattern rather than new risk.
- **`domain/ports.py`, `domain/exceptions.py`, `domain/task_factory.py`,
  `domain/services/config_services.py`, `domain/verification.py`,
  `events/core.py`, `events/hooks.py`** — read for correctness given their
  architectural centrality; no defects found beyond `config_services.py`'s
  role in C1 (the `CostService` class itself is correct; the bug is the
  missing config data it reads) and one cosmetic, self-masked note:
  `domain/exceptions.py::TimeoutError` shadows the builtin name within
  that module, but `ApplicationError.is_retriable()`'s use of it is
  harmless today because the builtin `TimeoutError` (and its 3.11+ alias
  `asyncio.TimeoutError`) is itself an `OSError` subclass, already in the
  same isinstance tuple. CLEARED, flagged only as a landmine for the next
  edit to that line.

## Gates (Phase 8, run against the fixed tree)

See `coverage.md`.
