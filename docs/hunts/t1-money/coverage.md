# T1 — Money — Phase 8 Coverage & Residual-Risk Statement

## Scope actually audited
`orchestrator/budget.py`, `orchestrator/cost.py`, `orchestrator/costing/{__init__,core,tracker,
analytics}.py`, `orchestrator/application/budget_enforcer.py`, `orchestrator/engine_core/
budget_enforcer.py`, `orchestrator/application/task_executor.py`, `orchestrator/task_handlers.py`,
`orchestrator/application/project_runner.py`, `orchestrator/engine_core/container.py` (the
`budget_enforcer`/`executor` wiring only), `orchestrator/services/executor.py` +
`orchestrator/application/executor.py` (diffed for divergence, none found). `orchestrator/
cost_optimization/` (13 submodules) was surveyed for reachability and money-touch (see C6) but
not deep-mined for its own internal candidates beyond that — see "What this tier does NOT
claim" below.

## Gates (this tier's fixed tree)
```
black --line-length=100 --check --fast orchestrator/costing/core.py \
    orchestrator/application/budget_enforcer.py orchestrator/application/project_runner.py \
    orchestrator/task_handlers.py tests/unit/test_hunt_t1_money.py                           PASS
ruff check <same files>                                                                      PASS
lint-imports                                                                                  PASS (5/5 contracts KEPT, 824 files/1386 deps analyzed)
python scripts/check_root_module_freeze.py                                                   PASS (256/256, no additions)
python scripts/check_test_markers.py                                                         PASS
mypy orchestrator/domain/ orchestrator/application/ orchestrator/engine_core/container.py \
     --ignore-missing-imports --no-strict-optional --python-version=3.12 --follow-imports=silent  PASS (58 files, 0 issues)
bandit -lll -r <changed files>                                                                PASS (0 issues, 1146 lines scanned)
python -m pytest tests/unit/test_hunt_t1_money.py -m unit                                     PASS (16/16)
python -m pytest tests/ -q -m "unit or integration"                                           see below
```

## RED→GREEN verification (Phase 3a / CLAUDE.md TDD rule, done explicitly, not narratively)
All four fixes were implemented before the regression suite was written in this session, so
RED was verified retroactively but for real: `git stash push --keep-index` on the four fixed
source files (test file kept), full suite re-run, fixes restored via `git stash pop`.
- **15 of 16 tests failed against the pre-fix tree**, each for the exact predicted reason
  (wrong class identity, `TypeError: unexpected keyword argument 'db_path'`, `TypeError: object
  NoneType can't be used in 'await' expression`, `AttributeError: 'BudgetHierarchy' object has
  no attribute 'record_cost'`, `'str' object has no attribute 'text'`, `cost_usd` asserting
  `0.0` instead of the fixture's non-zero value, the clamped-`$10` vs true-`$12` mismatch, and
  the source-guard regex still matching the old formula).
- **1 test correctly passed in both states** (`test_c2_budget_hierarchy_has_no_record_cost_method`)
  — it pins a fact about `BudgetHierarchy` itself (no `record_cost` method, before or after),
  not a behavior the fix changed; documented as such in the test's own docstring.
- All 16 pass against the fixed tree (see gates above).

## Verdict
- **VERIFIED DEFECTs fixed:** 4 — C1 (dual `BudgetHierarchy` implementations diverged on
  persistence), C2 (`BudgetEnforcer.record_cost()` bypassed `Budget`'s lock and called a
  nonexistent `BudgetHierarchy` method), C4 (typed task handlers discarded real LLM cost/tokens
  and never charged the budget they were given), C5 (`run_job()` silently clamped cross-run
  overspend at `max_usd` before charging `BudgetHierarchy`).
- **CLEARED (innocent):** 0 this tier — every candidate generated was evidentially confirmed;
  none were withdrawn on innocence. (`services/executor.py` vs `application/executor.py` was
  investigated and found to have no behavioral divergence — recorded as a hygiene observation
  under C1, not counted as its own cleared candidate since it was never raised as a defect
  candidate in the first place.)
- **Residual, not fixed, explicitly flagged:** 2 — C3 (`BudgetEnforcer`/`TaskExecutor`/
  `task_handlers.py`'s typed dispatch is fully built but never instantiated in the live
  pipeline; `Budget`'s own reserve/commit/release concurrency-safety pattern is likewise
  unused), `[REQUIRES HUMAN REVIEW]` because reconciling it either way is an architecture
  decision, not a contained fix; C6 (the `cost_optimization/` package's cost-reduction
  machinery — batching, caching, model cascading, speculative generation — is imported
  everywhere it should be used and instantiated nowhere), `[REQUIRES HUMAN REVIEW]` for the
  same reason.
- **Out-of-tier, handed off:** 1 — a confirmed-broken 3-dot relative import in
  `orchestrator/safety/code_executor.py:183` (found while tracing C6's `docker_sandbox.py`
  importers), queued as a separate task rather than folded into this tier (it is an execution/
  filesystem-safety import bug, T7's territory, not a money defect, and this tier's fix budget
  and test suite are scoped to money candidates only).
- §6.3 (the dual-budget conflation warning) was watched for throughout and not triggered: every
  candidate here is a defect in how `Budget` and `BudgetHierarchy` each independently fail (or
  interact through a documented seam like `run_job()`'s settlement), not a case of treating them
  as one mechanism.

## Clean claim this tier is permitted to make, and no more
Within the scope listed above — `Budget`, `BudgetHierarchy` (both implementations),
`BudgetEnforcer`, the typed-handler cost-reporting path, and `ProjectRunner.run_job()`'s
cross-run settlement — no VERIFIED defect remains unfixed. This says nothing about the other
~90 T2–T8 candidates, the un-deep-mined internals of `cost_optimization/`'s 13 submodules
(beyond the reachability/money-touch survey in C6), or the ~85% of the backend this program
will never reach (§8 of the plan).

## What this tier does NOT claim
- It does not claim the `cost_optimization/` package's individual submodules are free of
  defects — C6 establishes the package is *unreachable as a whole* from the live pipeline
  (with a narrow TDD-config exception), which means a defect inside e.g. `model_cascading.py`'s
  own cost math was not chased, because it cannot currently fire. If C6's wiring question is
  resolved in favor of turning the package on, its internals need their own pass first.
- It does not claim C3's or C6's architecture questions have a right answer — only that leaving
  fully-built, publicly-exported, production-shaped code permanently unreached is a maintenance
  and discoverability hazard regardless of which way the eventual decision goes.
- It does not claim the docker_sandbox relative-import bug is fixed, or that it is the only such
  import-path defect in the repo — only that it was found, verified, and correctly routed to a
  future tier (T7) rather than absorbed into this one's scope.
- It does not claim every one of the ~977 broad-except sites or ~205 concurrency primitives
  touching this tier's files were individually audited — T1's Phase 2 candidates were generated
  from the money-accounting classes' own contracts (does a documented invariant hold, does a
  call site match its callee's real interface), not from an exhaustive line-by-line sweep of
  every file in scope. T4/T6 own the general concurrency/error-path sweep.

## `hunt_iterations` / `fix_revisions`
`hunt_iterations`: 1/3 used. `fix_revisions`: 1/1 used (all four fixes correct on first pass,
verified by the retroactive RED→GREEN stash test above; no rework triggered
`[REQUIRES HUMAN REVIEW]` on any of the four fixed candidates).
