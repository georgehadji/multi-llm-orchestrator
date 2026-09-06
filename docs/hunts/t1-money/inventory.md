# T1 — Money — Phase 4 Inventory

Per `docs/DEFECT_HUNT_PLAN.md` §3/§4. Scope: `budget.py`, `cost.py`, `costing/` (pulled in —
see Phase 0 delta), `application/budget_enforcer.py`, `engine_core/budget_enforcer.py`,
`application/task_executor.py` + `task_handlers.py` (pulled in), `application/project_runner.py`
(pulled in), `cost_optimization/` (surveyed, not deep-mined — see coverage.md). Budget: 14
candidates. Spent: 6.
Tree at start of tier: `1f7ebaa` (branch `claude/llm-orchestrator-website-factory-luk61w`).

## Phase 0 delta

- **Census re-verify (§4 Step 1):** the plan's own file list for T1 (`cost_tracker.py`,
  `cost_analytics.py` as flat modules) is stale. Both are now 11-line deprecated re-export
  shims pointing at a package the plan never lists: `orchestrator/costing/`. Scope is corrected
  to include `orchestrator/costing/{core,tracker,analytics}.py`.
- Pulled into scope beyond the plan's literal list, same rule T0 used for C2: `costing/core.py`
  (C1 is a direct, concrete instance of a dual-implementation money bug), `application/
  task_executor.py` + `task_handlers.py` (C3/C4), `application/project_runner.py` (C5) — each
  reached by following a real import/call chain from a plan-listed file, not by drifting scope.
- Taxonomy weight for this tier: class 8 (contract/dependency) and class 3 (concurrency)
  dominate the findings, matching §1.2's HIGH weighting for both.
- `hunt_iterations` used: 1. `fix_revisions` used: 1 (all four fixes landed first pass).
- Reachability method used throughout: static import grep from the four real entry points
  (`cli_dispatch.py`, `engine.py`, `engine_core/container.py`, `application/project_runner.py`),
  cross-checked against live Python introspection (`inspect.signature`, `dir()`) rather than
  reading source alone wherever a claim was safe to run directly — e.g. C4's `client.call()`
  existence was verified by importing `UnifiedClient` and inspecting it directly after a grep
  false-negative, rather than trusting the grep miss.

## Candidates

### C1 — VERIFIED DEFECT — two independent, diverged `BudgetHierarchy` implementations
- **Property violated:** class 8 (contract/dependency) — a class name must resolve to one
  behavior, not two.
- **Location:** `orchestrator/costing/core.py` (pre-fix, 505 lines) vs `orchestrator/cost.py`
  (617 lines).
- **Finding:** `costing/core.py` was a hand-copied fork of `cost.py`'s `BudgetHierarchy` /
  `CostPredictor` / `CostForecaster` / `ForecastReport` / `RiskLevel`. A byte-level diff showed
  the two were identical (including matching `BUG-004 FIX` comments and the TOCTOU-reservation
  pattern) except for one divergence: `cost.py`'s `BudgetHierarchy.__init__` later gained a
  `db_path` parameter plus `_init_db`/`_load_from_db`/`_save_to_db`, called from `charge_job()`
  and `reset_spend()` — SQLite persistence so cross-run caps survive process restarts. This was
  never backported to `costing/core.py`.
- **Reachability:** `costing/__init__.py:3` does `from .core import *` unconditionally, so
  `orchestrator.costing.BudgetHierarchy` (and `orchestrator.costing.core.BudgetHierarchy`)
  publicly resolve to the stale, non-persisted class. Grepped every live import of
  `BudgetHierarchy` across `orchestrator/`: `engine_core/container.py:40`, `engine.py:364`,
  `application/budget_enforcer.py:19`, `planner.py:44` — **all four import from `orchestrator.
  cost`, never from `orchestrator.costing`**. Both existing hierarchy-persistence tests
  (`tests/test_budget_hierarchy_persistence.py`, `tests/unit/test_bug_scan.py`) also import
  exclusively from `orchestrator.cost`. So today nothing constructs the stale copy — but nothing
  stops a future `from orchestrator.costing import BudgetHierarchy` (a very natural typo/
  autocomplete given `costing.tracker`/`costing.analytics` genuinely are the canonical location
  for their respective classes — see next point) from silently losing cross-run persistence.
- **Innocence attempt:** checked whether `costing/core.py` is itself marked deprecated the way
  its siblings are — it is not. `cost_tracker.py`/`cost_analytics.py` (flat, deprecated) point
  *forward* to `costing.tracker`/`costing.analytics` (canonical) with a `DeprecationWarning`.
  `costing/core.py` has no such marking in either direction, and — check run against
  `REMEDIATION_PLAN.md`'s own §P4.3 "Resolve 2 remaining duplicate pairs" — this pair is not
  one of the two it names (`architecture_rules.py`/`safety/architecture_rules.py` and
  `multi_platform_generator.py`/`generators/multi_platform_generator.py`). The repo's own
  duplicate-pair census is itself short by at least one pair. No innocence available.
- **Related, not separately fixed:** `orchestrator/services/executor.py` vs `orchestrator/
  application/executor.py` (`ExecutorService`, the class the live container actually
  constructs) are a *fourth* undocumented duplicate pair — diffed directly, found NO
  behavioral divergence (docstring/comment/whitespace only, functionally identical). Recorded
  as a hygiene observation, not a fix, since there is no behavior to reconcile.
- **Fix:** `costing/core.py` rewritten as a re-export shim (`from orchestrator.cost import *`),
  matching the pattern of the sibling `engine_core/budget_enforcer.py` shim (silent, no
  `DeprecationWarning` — a warning here would fire on every legitimate `costing.tracker`/
  `costing.analytics` import too, since `costing/__init__.py` eagerly wildcard-imports all
  three submodules). `orchestrator.costing.BudgetHierarchy` now *is*
  `orchestrator.cost.BudgetHierarchy` (same object), permanently closing the divergence.
- **Tests:** `test_c1_costing_core_budget_hierarchy_is_the_canonical_class`,
  `test_c1_costing_package_budget_hierarchy_is_canonical`,
  `test_c1_costing_core_budget_hierarchy_supports_persistence`.

### C2 — VERIFIED DEFECT — `BudgetEnforcer.record_cost()`: lock bypass + call to a nonexistent method
- **Property violated:** class 3 (concurrency — TOCTOU) and class 8 (contract) simultaneously.
- **Location:** `orchestrator/application/budget_enforcer.py:210-235` (pre-fix).
- **Finding, part A (concurrency):** `record_cost()` did `self.budget.spent_usd += cost_usd`
  directly — a synchronous, unlocked read-modify-write on the exact field `Budget.charge()` /
  `commit_reservation()` / `release_reservation()` protect with `asyncio.Lock` specifically to
  prevent concurrent tasks from racing on it (`budget.py`'s own `BUG-001 FIX`/`FIX-001a`
  comments name this race directly: "prevent race conditions when multiple concurrent tasks
  check budget simultaneously"). `record_cost()` bypassed that lock entirely.
- **Finding, part B (contract):** the same method then called
  `self.budget_hierarchy.record_cost(task_id, cost_usd)` — silenced with
  `# type: ignore[attr-defined]`. Read both `BudgetHierarchy` implementations (`cost.py` and,
  pre-fix, `costing/core.py`) in full: **neither has a `record_cost` method.** The real method
  is `charge_job(job_id, team, amount)`, which `record_cost`'s own parameters (`task_id`,
  `cost_usd`, `phase`) cannot supply — no `team` is available at all. Calling this line with a
  truthy `budget_hierarchy` is a guaranteed `AttributeError`.
- **Same defect shape recurring:** `tests/test_bug_regression.py` documents a *prior* bug in
  this exact class — BUG-004, `record_time()` called nonexistent `Budget.elapsed_time`, fixed
  by deleting the method, regression-tested via `assert not hasattr(BudgetEnforcer,
  "record_time")`. `record_cost()` is the same defect shape (a method calling a nonexistent
  attribute on a collaborator) recurring in a sibling method the existing regression suite does
  not cover.
- **Reachability:** grepped the whole repo for `.record_cost(` — the only call sites are this
  method's own definition and an unrelated `pc.record_cost(...)` in
  `tests/test_new_modules.py:1242` (five positional args, a different method on a different
  object entirely). **Zero live callers of `BudgetEnforcer.record_cost()`.** Dead code today —
  see C3 for why (`BudgetEnforcer` itself is never instantiated in production) — but public API
  on an exported class, one wiring change away from firing.
- **Innocence attempt:** checked whether `job_id`/`team` could be derived some other way inside
  `record_cost` (e.g. from `self` state) — `BudgetEnforcer` stores neither; they are per-call
  parameters on `enforce_hierarchy_job` (the correct, already-used static method — see
  `project_runner.py:481,497`). No innocence available for the `AttributeError`; the lock bypass
  has no innocence either since `Budget.charge()` already exists and is exactly the lock-safe
  primitive this method should have used.
- **Fix:** `record_cost()` made `async def`; delegates to `await self.budget.charge(cost_usd,
  phase or "generation")` (lock-protected) instead of the raw `+=`. The `budget_hierarchy` call
  is removed — cross-run hierarchy charging is `enforce_hierarchy_job`'s job (it has the
  `job_id`/`team` this method never did), not this per-call method's. The enforcer's own
  (separate) `phase_spent` bookkeeping is preserved unchanged.
- **Tests:** `test_c2_budget_hierarchy_has_no_record_cost_method` (pins the fact that
  motivated the fix), `test_c2_record_cost_charges_budget_via_lock_protected_charge`,
  `test_c2_record_cost_does_not_crash_with_a_hierarchy_present`,
  `test_c2_record_cost_still_updates_enforcers_own_phase_tracking`.

### C3 — VERIFIED FINDING, `[REQUIRES HUMAN REVIEW]` — the whole `BudgetEnforcer`/typed-dispatch subsystem is never instantiated in production
- **Finding:** `orchestrator/engine_core/container.py:747` hardcodes `budget_enforcer = None`
  (mirroring `health_tracker = None` / `resumption_service = None` a few lines above — but
  unlike `health_tracker`, which the container's own comment says "the engine rebuilds ... with
  the real bridges later," nothing rebuilds `budget_enforcer`). This `None` is threaded through
  `ServiceContainer.budget_enforcer: Optional[BudgetEnforcer] = None` into `TaskExecutor`'s
  constructor, whose own parameter is typed **non-Optional** (`budget_enforcer: BudgetEnforcer`)
  — a type contract falsified at runtime on every construction.
  Repo-wide grep for the literal constructor call `BudgetEnforcer(` found **zero matches
  anywhere in `orchestrator/`** — only in `tests/test_bug_regression.py:171`. `engine.py` only
  ever calls `BudgetEnforcer`'s two `@staticmethod`s (`check_phase_cap` at line 1023,
  `should_exit_early` at line 1149) directly on the class — never on an instance.
- **A second, independent confirmation of the same shape:** `orchestrator/application/
  task_executor.py`'s `TaskExecutor` — the ONLY class that consumes `budget_enforcer` as an
  instance and the only caller of `orchestrator/task_handlers.py`'s typed-handler dispatch — is
  **itself never constructed anywhere in `orchestrator/`** (grepped `TaskExecutor(`: the only
  match in the whole repository is `tests/unit/test_task_executor.py:89`, its own unit test).
  The container's `executor` field is actually populated by a *different* class,
  `ExecutorService` (`container.py:665`, defined in `orchestrator/services/executor.py`), which
  correctly reads real cost off `result.task_result.cost_usd` — the live pipeline's task
  execution does not go through `TaskExecutor`/`task_handlers.py` at all today. Both `TaskExecutor`
  and `task_handlers.get_handler` are nonetheless exported from `orchestrator/application/
  __init__.py.__all__` and `orchestrator/engine_core/__init__.py.__all__`, fully docstringed,
  with `task_handlers.py`'s own inline comment calling itself "Phase 3: typed dispatch" (i.e.
  the *preferred*, not experimental, path) — a live discoverability hazard even though it is
  not a live money-loss path today.
- **Related, same theme:** `Budget`'s own concurrency-safety primitives — `reserve()`,
  `commit_reservation()`, `release_reservation()`, and `can_afford()` — built explicitly to fix
  a documented historical race (BUG-001/FIX-001a, the same one C2 references) are **also never
  called anywhere in `orchestrator/`** (grepped `.reserve(`, `.commit_reservation(`,
  `.can_afford(`: zero production call sites). The one live pre-iteration budget gate found
  (`engine_core/state_coordinator.py:29-30`, `determine_final_status`) reads
  `budget.remaining_usd`/`budget.time_remaining()` directly and correctly (and is itself
  resume-safe, using `Budget.original_start_time` via `elapsed_seconds` — no defect found there).
  So the specific concurrency-safety guarantee §1.2 weights HIGH for this tier ("check-then-act
  on budget is a money bug") is not actually protecting any live code path; it simply isn't
  exercised by anything, safe or not.
- **Why not fixed here:** actually instantiating `BudgetEnforcer` in the composition root (or
  wiring `TaskExecutor` in as the live executor) is a live-behavior change — previously-silent
  enforcement (phase hard-halt at 2x, budget threshold warnings) would start firing, and
  swapping the live task-execution path to go through `TaskExecutor`/`task_handlers.py` instead
  of `ExecutorService` is an architecture decision with a much larger blast radius than a
  bounded tier should make unilaterally — exactly the class of change T0's C4 (CI Python
  matrix) was flagged rather than actioned for.
- **Disposition:** residual, `[REQUIRES HUMAN REVIEW]`. Recommendation: either (a) decide
  `TaskExecutor`/`task_handlers.py` is the intended future path, fix C4 (done — see below) and
  finish wiring it in, deleting `ExecutorService`, or (b) decide `ExecutorService` is canonical
  and delete/clearly-mark-experimental the `TaskExecutor`/`task_handlers.py` branch so it stops
  looking production-ready. Either is legitimate; leaving both live-looking and only one
  actually live is not.

### C4 — VERIFIED DEFECT (in the C3-dead subsystem) — typed handlers discard real LLM cost
- **Property violated:** class 8 (contract) / threat 1 (money loss) — a `TaskResult` must
  report what the call it wraps actually cost.
- **Location:** `orchestrator/task_handlers.py` — `_BaseHandler._call_llm()` (pre-fix) and all
  four registered handlers: `CodeGenerationHandler`, `CodeReviewHandler`, `EvaluationHandler`,
  `ReasoningHandler` (confirmed exhaustive via grep for `@register(TaskType.` — exactly these
  four exist).
- **Finding:** `_call_llm()` called `client.call(...)` — confirmed via `inspect.signature` to
  be a real method (`UnifiedClient.call`, defined in `orchestrator/infrastructure/llm_client.py:
  263`, returning `APIResponse` with `.text`/`.input_tokens`/`.output_tokens`/`.cost_usd`, per
  `llm_client.py:34-44`) — and returned only `response.text`, discarding the rest. Every one of
  the four handlers then built its `TaskResult` with **hardcoded** `cost_usd=0.0,
  tokens_used={"input": 0, "output": 0}` regardless of what the (real, billable) call actually
  cost, and accepted a `budget: Budget` parameter that was never referenced in any handler body.
  `api_clients.py` (`UnifiedClient`'s re-export point) has no auto-charging side channel either
  (grepped for `CostTracker`/`.track_usage(`/telemetry-based auto-charge: none) — so if this
  path executes, the real spend reaches no cost-tracking mechanism at all: not `TaskResult`, not
  `Budget`, not `BudgetHierarchy`, not `CostTracker`/`CostAnalytics`.
- **Reachability:** per C3, `TaskExecutor`/`task_handlers.py` is not constructed by the live
  pipeline today, so this is not currently draining a production budget. It is, however, a
  100%-reproducible defect in code that is fully built, exported, and one wiring change away
  from being the live path — and per `task_executor.py`'s own control flow, this typed-handler
  branch is tried **first**, before the (correctly cost-tracking) fallback path, for any task
  type with a registered handler — so wiring `TaskExecutor` in today, as-is, would make this
  the *default* path for CODE_GEN/CODE_REVIEW/EVALUATE/REASONING tasks, not a rare corner case.
- **Innocence attempt:** checked whether a downstream caller re-derives cost from something
  other than `TaskResult.cost_usd` for this path — no; the one live executor (`ExecutorService`)
  that does this correctly reads `result.task_result.cost_usd`, which is exactly the field these
  handlers zeroed out. No innocence available.
- **Fix:** `_call_llm()` now returns the full `APIResponse`. Each handler uses
  `response.text`/`.cost_usd`/`.input_tokens`/`.output_tokens` to build an accurate
  `TaskResult`, and calls a new `_BaseHandler._charge()` helper (`await budget.charge(cost_usd,
  phase)` via `getattr(budget, "charge", None)`) so a real `Budget` gets charged when one is
  given — defensively tolerant of `None` (or any other object without `.charge`), since C3
  means the currently-plumbed value cannot be trusted to be a real `Budget`. Phase mapping:
  CODE_GEN/REASONING → `"generation"`, CODE_REVIEW → `"cross_review"`, EVALUATE → `"evaluation"`
  (matching `BUDGET_PARTITIONS`' existing phase names).
- **Tests:** `test_c4_call_llm_returns_full_response_not_bare_text`,
  `test_c4_handler_reports_real_cost_and_tokens_not_hardcoded_zero` (parametrized across all
  four handler classes), `test_c4_handler_charges_a_real_budget_when_given_one`,
  `test_c4_handler_tolerates_budget_none`.

### C5 — VERIFIED DEFECT — `run_job()` clamps cross-run overspend at `max_usd`
- **Property violated:** threat 1 (money loss) — cross-run `BudgetHierarchy` totals must equal
  true spend.
- **Location:** `orchestrator/application/project_runner.py::run_job()` (pre-fix).
- **Finding:** the settlement call computed `actual_spend = self._budget.max_usd -
  self._budget.remaining_usd`. `Budget.remaining_usd` (`budget.py:79-82`) is
  `max(0.0, max_usd - spent_usd - _reserved_usd)` — it **floors at zero**. `Budget.charge()`
  (`budget.py:124-129`) enforces **no ceiling** — `self.spent_usd += amount` unconditionally, so
  `spent_usd` can and does exceed `max_usd` (the only ceiling enforcement found anywhere,
  `state_coordinator.py`'s `budget_exhausted` check, stops *starting new* iterations after the
  fact; it does not retroactively cap an in-flight overspend). Whenever `spent_usd > max_usd`,
  `remaining_usd` clamps to `0.0`, so `actual_spend = max_usd - 0 = max_usd` — silently
  truncating the true overspend before it ever reaches `BudgetHierarchy.charge_job()`.
  (`Budget._reserved_usd` was confirmed always `0.0` in production per C3's finding that
  `reserve()`/`commit_reservation()` have zero callers, so this reduces exactly to `actual_spend
  = min(spent_usd, max_usd)` today — a real, live formula, not a hypothetical one.)
- **Reachability:** `run_job()` is `ProjectRunner`'s documented policy-driven entry point
  (`project_runner.py:463`'s own docstring: "BudgetEnforcer → run_project → charge → flush
  telemetry"), gated only on `self._budget_hierarchy is not None` — live whenever a caller
  supplies a `BudgetHierarchy`, exactly the scenario `BudgetHierarchy` exists for (cross-run org/
  team/job caps).
- **Innocence attempt:** checked whether overspending `max_usd` is itself impossible (i.e.
  whether this formula is dead because `spent_usd` never exceeds `max_usd`) — it is not: no
  ceiling check exists on `Budget.charge()` itself, and the exhaustion check is
  post-hoc/next-iteration, not atomic with the charge that pushes spend over. No innocence
  available.
- **Fix:** one-line change — `actual_spend = self._budget.spent_usd` (the true total, never
  clamped). `Budget.spent_usd` is exactly the value `run_job()`'s own docstring means by
  "charge" — no reason to derive it indirectly through `remaining_usd`'s clamped arithmetic.
- **Tests:** `test_c5_project_runner_no_longer_derives_spend_from_remaining_usd` (source
  no-regression guard), `test_c5_run_job_charges_hierarchy_with_true_overspend_not_clamped_value`
  (real trigger through the actual `ProjectRunner.run_job()` call path with a real
  `BudgetHierarchy`, not a mocked formula).

### C6 — VERIFIED FINDING, `[REQUIRES HUMAN REVIEW]` — the `cost_optimization/` package's savings machinery is built but never wired in
- **Finding** (established via a dedicated reachability-mapping pass, cross-checked with direct
  Python introspection): `orchestrator/cost_optimization/`'s Tier 1-3 classes — `BatchClient`,
  `PromptCacher`, `ModelCascader`, `SpeculativeGenerator`, `StreamingValidator`, `TokenBudget`,
  `AdaptiveTemperatureController`, `DependencyContextInjector`, `EvalDatasetBuilder` — are
  imported into both `engine.py` (module level, behind a feature flag that defaults **on**) and
  `engine_core/container.py`'s `ServiceContainer` fields, but **never instantiated** in either
  file (every one of these names appears only in its own import statement in both files;
  `ServiceContainer.build()`'s actual construction call omits every one of the corresponding
  fields, leaving them at their `None` default). `cost_optimization/
  cost_optimization_integration.py`'s `Tier1OptimizationMixin` is confirmed to have **zero
  importers anywhere in the repository** — its own trailing docstring
  (`INTEGRATION_INSTRUCTIONS`) documents the integration steps that were never performed. The
  one genuinely live usage of the package is narrow: TDD-profile configuration
  (`get_optimization_config`/`update_config`/`get_tdd_profile`) via `cli_dispatch.py` and
  `testing/first_generator.py` — none of the cost-reducing mechanisms themselves.
  Separately, of the package's 13 submodules, 6 (`speculative_gen.py`, `batch_client.py`,
  `cost_optimization_integration.py`, `model_cascading.py`, `token_budget.py`,
  `streaming_validator.py`) maintain their own independent, hardcoded per-model price tables,
  disconnected from `Budget`/`BudgetHierarchy`/the canonical `COST_TABLE` — one
  (`streaming_validator.py`) explicitly labels its own table "deprecated" in its own comments.
- **Why this is a T1 (money) finding, not just dead code:** every one of these mechanisms
  exists specifically to *reduce* real spend (batching, caching, cheaper-model cascading,
  speculative generation). None of them being wired in means the system has been paying full,
  unoptimized price on every call this entire time — a standing, silent excess-spend condition,
  the mirror image of C4/C5's overspend-via-bug findings.
- **Why not fixed here:** wiring in a whole cost-optimization subsystem changes cost, latency,
  and output quality simultaneously across every run — an architecture/product decision, not a
  contained bug fix, and squarely the kind of change T0's C4 and this tier's C3 were also
  correctly not actioned unilaterally for.
- **Disposition:** residual, `[REQUIRES HUMAN REVIEW]`. Not fixed; no tests added (there is no
  regression to pin — the recommendation is to build, not to preserve current behavior).

## Out-of-tier finding handed off, not adopted into T1

While tracing `cost_optimization/docker_sandbox.py`'s importers for C6, a confirmed-broken
relative import was found in `orchestrator/safety/code_executor.py:183`
(`from ...cost_optimization.docker_sandbox import DockerSandbox`, 3 dots — resolves beyond the
top-level package, verified via `importlib.util.resolve_name` to raise `ImportError`
unconditionally) sitting behind `require_sandbox: bool = True`, the class's own default. A
comment directly above shows the previous, correct 2-dot import. This is a real, cleanly
diagnosed defect, but it is a Class 5/6 execution-safety import bug with nothing to do with
cost/budget semantics — squarely T7's territory ("execution & filesystem surface"), not T1's.
No live caller of `orchestrator/safety/code_executor.py` was found either, so it is dormant like
several of this tier's findings, just not a money one. Queued separately rather than folded into
this tier's fix set — see the spawned task.

## Gates (Phase 8, run against the fixed tree)

See `coverage.md`.
