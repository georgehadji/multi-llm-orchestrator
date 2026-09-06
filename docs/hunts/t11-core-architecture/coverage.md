# T11 — Infrastructure / Engine_Core / Domain / Events Core — Phase 8 Coverage & Residual-Risk Statement

## Scope actually audited

118 files (`domain/`'s 12, `engine_core/`'s 47, `events/`+`unified_events/`'s
10, `infrastructure/`'s 51 excluding already-touched `state.py`/`llm_client.py`)
— the largest single wave in this hunt to date. A background agent read all
118 at a deliberately tiered depth (four pre-identified leads in full
depth, architecturally-central files read for correctness, the remainder
purpose+reachability only), per the plan's own guidance that a wave this
size does not need uniform per-file depth. Every fix below was
independently re-verified from source before being applied — in
particular, C1's severity claim was materially corrected during
verification (3 of the survey's original 5 "missing cost entries" turned
out to reference commented-out, non-live enum members; see inventory.md).

## Gates (this tier's fixed tree)

```
black --line-length=100 --check --fast <3 changed/new Python files>     PASS
ruff check <3 changed/new Python files>                                   PASS
python3 -c "json.load(...)" on config/costs.json                          PASS (valid JSON,
                                                                              146 entries, +2)
lint-imports                                                                PASS (5/5 KEPT,
                                                                              824 files)
python scripts/check_root_module_freeze.py                                 PASS (256/256)
python scripts/check_test_markers.py                                        PASS
mypy orchestrator/domain/ .../application/ .../container.py                PASS — isolated
                                                                              diff against the
                                                                              pre-fix tree is
                                                                              empty (zero output
                                                                              difference)
bandit -lll -r <2 changed Python source files>                              PASS (0 issues at
                                                                              any severity)
python -m pytest tests/unit/test_hunt_t11_core_architecture.py             PASS (3/3)
python -m pytest tests/unit/test_pipeline_executor.py                      PASS (9/9 existing,
                                                                              zero regressions)
python -m pytest tests/ -k "project_planner or pipeline_runner or
                             self_consistency"                              PASS (10/10, zero
                                                                              regressions)
python -m pytest tests/ -q -m "unit or integration"                        2530 passed (+3 over
                                                                              T10's 2527), 2
                                                                              pre-registered
                                                                              environmental
                                                                              failures unchanged,
                                                                              21 skipped — zero
                                                                              regressions
```

## RED→GREEN verification

Verified via `git stash push --keep-index` on the 3 fixed files
(`config/costs.json`, `engine_core/project_planner.py`,
`engine_core/pipeline_executor.py` — new test file stays present,
staged), full T11 test file re-run against the pre-fix tree, fix restored
via `git stash pop`. All 3 tests failed against the pre-fix tree for the
exact predicted reason:
- C1: `cost["input"] > 0.0` failed — real value `{"input": 0.0, "output":
  0.0}`.
- C2: `caplog.records == []` — no log at all naming the dropped cyclic
  tasks (the acyclic task's own correct scheduling was unaffected either
  way, confirming the assertion isolates the right behavior).
- C3: the fake pipeline's `.run()` was called exactly once, not twice —
  the loop broke immediately on the `"vs_retry_escape"` signal instead of
  retrying.

All 3 pass on the fixed tree.

## Verdict

- **VERIFIED DEFECT fixed:** 3 — C1 (a real, actively-routed model
  silently costing $0.00 per call — the highest-severity live defect
  found in this hunt to date, in the money/budget-escape threat class),
  C2 (a circular task dependency silently dropping tasks from execution
  with zero diagnostic, in the live scheduling path every project run
  goes through), C3 (a self-defeating retry-escape feature that
  configures a retry and then never runs it).
- **Residual, surveyed but not fixed — `[REQUIRES HUMAN REVIEW]`:** an
  unsynchronized circuit-breaker HALF_OPEN race in
  `infrastructure/streaming_resilient.py` (same defect class as T5's
  fix, but confirmed fully dead); `unified_events/core.py`'s
  `UnifiedEventBus` never having `.start()` called (dead projections,
  unbounded queue growth in long-running hosts — a materially larger
  wiring decision than this tier's three fixes); a broken
  `container.py:536-548` import that defeats real cycle-detection
  wiring for C2's own scheduling path; `domain/model_registry.py`'s
  stale `QWEN_3_6_FLASH` entries (dead, lower priority than the live C1).
- **Cleared (innocent):** `engine_core/adaptive_router.py` (a correct,
  non-diverged shim to the already-T8-fixed root module);
  `engine_core/dep_resolver.py` vs `dependency_resolver.py` (not a
  duplicate pair — genuinely different, unrelated responsibilities; the
  `dependency_resolver.py` shim is dead but harmless); `domain/
  exceptions.py`'s self-masked `TimeoutError` shadow; a large confirmed-dead
  cluster of parallel `infrastructure/` cache/monitoring/sandbox
  implementations.
- **Discovered but explicitly out of this tier's declared scope, not
  folded in:** a large `routing.json` config-drift condition (many
  task-type/model keys silently dropped due to enum mismatches),
  surfaced only as a side effect of running the repo's own diagnostic
  tool while verifying C1 — large enough to warrant its own dedicated
  investigation rather than a fold-in here.

## Clean claim this tier is permitted to make, and no more

Within the 118 files in this wave's declared scope: every file received
at least a purpose+reachability pass, with the four pre-identified leads
and the architecturally-central Tier B files read for correctness. This
does **not** claim uniform deep-audit coverage of all 118 files — the
Tier C files (the majority) received purpose+reachability only, per the
plan's own explicit acknowledgment that a wave this size cannot receive
T0-T7-depth treatment on every file within budget. It does **not** claim
`routing.json`'s drift (discovered incidentally) is addressed.

## What this tier does NOT claim

- It does not claim `qwen/qwen3.6-flash` is the only model ever affected
  by a costs.json gap — only that it was the one genuinely live,
  currently-real gap found; the `routing.json` drift discovered
  incidentally suggests further config-consistency issues may exist
  elsewhere, unexamined.
- It does not claim `container.py`'s broken `_DepResolver` import is
  fixed — C2's fix makes the *symptom* (silent task-drop) visible without
  touching the *cause* (the real cycle-detecting resolver never being
  wired in), which remains a deliberate architecture decision left to a
  human.
- It does not claim `UnifiedEventBus`/`AsyncEventStore` are safe to wire
  in as-is — only that their current dormancy was confirmed and its
  consequences (stale projections, unbounded queue growth) were traced,
  not fixed.
- It does not claim the 4 stage files not individually re-verified beyond
  "confirmed live via discovery mechanism" (`stages/constitution_gate.py`,
  `critique.py`, `design_critique.py`, `persuasion_defense.py`,
  `preflight.py`, `validate.py`, `generate.py`, `evaluate.py`) are free of
  their own internal defects — they were read at a lighter depth than the
  three files where fixes landed.

## `hunt_iterations` / `fix_revisions`

`hunt_iterations`: 1/3 used (Phase 1/3 survey delegated to one background
agent run, tiered internally rather than requiring a second pass).
`fix_revisions`: 1/1 per fixed candidate — all three fixes were correct on
first pass, confirmed via the retroactive RED→GREEN stash test above. One
correction was made to the *survey's own severity claim* (not a fix
revision) during independent re-verification of C1 before implementing it.
