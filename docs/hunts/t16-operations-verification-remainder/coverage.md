# T16 — operations/ remainder, verification/, policy*, telemetry/logging misc — Phase 8 Coverage & Residual-Risk Statement

## Scope actually audited

75 files per the plan's estimate, plus `project_mgmt/` (6) and `workspace/` (4) pulled in as
the same duplicate-pair shape this hunt targets (see Phase 0 in inventory.md) — 85 files
total. A background agent gave full correctness-depth to the two mandated leads
(SecretsFilter chain, policy-system claim) and to `policy_engine.py`/`policy.py`/
`policy_dsl.py`/all 6 `verification/` files (all mandated), plus every file this tier found a
real defect in. I independently re-verified every claim this tier acted on: re-read the
cited source lines directly rather than trusting the report's line numbers, re-ran every
claimed duplicate-pair diff myself, re-confirmed caller counts via my own repo-wide greps,
and — for the highest-regression-risk fix (the `services/` shim conversion, C5) — ran all 8
pre-existing, independently-owned test files against the new shims before treating the fix
as safe, rather than relying on a "cosmetic-only diff" characterization alone.

## Gates (this tier's fixed tree)

```
black --line-length=100 --check --fast <12 changed files>                PASS (12/12
                                                                             unchanged)
ruff check <12 changed files>                                             PASS
lint-imports                                                              PASS (5/5 KEPT,
                                                                             824 files)
python scripts/check_root_module_freeze.py                               PASS (256/256)
python scripts/check_test_markers.py                                     PASS
mypy orchestrator/domain/ .../application/ .../container.py              PASS — isolated
                                                                             diff shows only
                                                                             REMOVED errors
                                                                             (1571 → 1554),
                                                                             zero new ones;
                                                                             the 17 removed
                                                                             errors are the
                                                                             exact call_model/
                                                                             bad-attribute/
                                                                             untyped-duplicate
                                                                             errors this
                                                                             tier's fixes
                                                                             eliminated
bandit -lll -r <12 changed files>                                         PASS ("Test
                                                                             results: No
                                                                             issues
                                                                             identified" —
                                                                             the reported
                                                                             list, which is
                                                                             what this gate
                                                                             tracks per this
                                                                             hunt's established
                                                                             convention)
python -m pytest tests/unit/test_hunt_t16_operations_verification_remainder.py  PASS (7/7)
python -m pytest <8 services/-dependent test files>                       PASS (73/73,
                                                                             pre-existing
                                                                             tests, run
                                                                             directly against
                                                                             the new shims —
                                                                             not part of the
                                                                             7 new tests)
python -m pytest tests/ -q -m "unit or integration"                       2572 passed (+7
                                                                             over T15's 2565),
                                                                             2 pre-registered
                                                                             environmental
                                                                             failures
                                                                             unchanged
                                                                             (test_openrouter_
                                                                             model_audit.py,
                                                                             network-blocked
                                                                             in this
                                                                             sandbox), 20
                                                                             skipped, 157
                                                                             deselected —
                                                                             zero regressions
                                                                             (236s)
```

## RED→GREEN verification

Verified via `git stash push` on the 12 fixed source files (new test file staged separately,
stays present), full T16 test file re-run against the pre-fix tree, fix restored via
`git stash pop`. All 7 tests failed against the pre-fix tree for the exact predicted reason:
- C1: raw `sk-...` secret appeared unmasked in captured log output.
- C2: the mock client's `.call` was awaited 0 times (pre-fix code calls the nonexistent
  `.call_model`, which — being a plain `AsyncMock`'s auto-created child attribute — silently
  returns further mock objects instead of raising outright, then fails downstream when
  `re.search` is given a non-string `content`, caught by the same broad `except Exception`;
  confirmed via the logged "Failed to generate test: expected string or bytes-like object,
  got 'AsyncMock'" message, the real original failure shape).
- C3, C4, C5: identity assertions failed (`orchestrator.project_mgmt.analyzer.ProjectAnalyzer`
  / `operations.hitl_workflow.HITLWorkflow` / `services.executor.ExecutorService` were
  distinct objects from their root/`application/` counterparts pre-fix).
- C6: `10.0 == 8.0` failed — confirming the stale hardcoded fallback was in effect.
- C7: `SystemExit: 1` raised on bare import — confirming the pre-fix crash-on-import,
  reproducing the exact original bug (the RED run also regenerated the stray
  `orchestrator/self_test_log.txt` artifact in the source tree — direct empirical proof of
  the source-tree-write half of the bug — deleted after confirming RED, before restoring the
  fix).

## Verdict

- **VERIFIED DEFECT fixed:** 7 (12 files touched — C4 and C5 are each one defect *shape*
  fixed across 4 and 3 files respectively) — C1 (SecretsFilter never installed on any real
  logging entry point across the CLI, API server, dashboard, and MCP server processes — the
  highest real-world-impact fix this tier), C2 (`testing/validator.py`'s test generator
  silently reporting false-positive "generated and validated" tests via a nonexistent client
  method), C3 (`project_mgmt/analyzer.py` stale duplicate, missing the `ArchitectureScorer`
  delegation and two whole methods root gained), C4 (four `operations/` duplicate pairs,
  two with real functional divergence — an asyncio task-reference fix and an SSRF guard —
  the dead copies now inherit), C5 (three `services/` duplicate pairs silently shadowing
  their own package's canonical re-export, affecting 8 pre-existing test files — now
  verified to test the real production classes with zero regressions), C6
  (`crosscutting/config.py`'s always-failing re-export with a stale, wrong fallback value),
  C7 (`operations/quick_self_test.py`'s import-time crash and source-tree pollution).
- **Residual, surveyed but not fixed — `[REQUIRES HUMAN REVIEW]`:** the policy system
  (`policy_engine.py::enforce()`/`.check()`, `ConstraintPlanner.select_model()` et al.) has
  **zero live enforcement on any entry point**, contradicting `engine.py::run_job()`'s own
  docstring — an architectural/product decision about where to gate, not a mechanical fix;
  `operations/autonomy_config.py`'s Multi-Mode Selector is fully built and completely inert,
  bypassed by an unrelated ad-hoc mapping in `cli_dispatch.py` — fixing this means rewriting
  four live CLI-flag-handling blocks, a behavioral change requiring a decision about which
  mapping should win, not a single-file patch.
- **Cleared (innocent):** `orchestrator/verification.py`'s permanent shadowing by the
  `verification/` package — same shape as T2's already-recorded `gateway.py`/`agents.py`
  instances, left as a documented hygiene landmine per that established precedent, not
  modified; `orchestrator/logging.py`'s dead structlog fork (zero callers, a fix here
  wouldn't reach anything); `ShellTool`'s shell-execution surface (by design, zero live
  construction); `services/completion_judge.py`/`autonomy_costs.py` (dead code, no
  duplicate to reconcile); `crosscutting/config.py`'s config-loading silent excepts and 3
  sibling `operations/` files' identical pattern (the survey itself distinguished these from
  the T6/T8/T9/T13 false-clean-scan pattern and declined to elevate them — respected here,
  not relitigated); `tools/`/`skills/skills.py` (fully built, unused, not broken).
- **Discovered but explicitly out of this tier's fix scope, documented as residual:**
  `memory_tier.py`'s shared silent-file-skip bug in `_touch_memory`/`delete_project_memories`
  — T14 already evaluated and explicitly declined to elevate this in an earlier, already-
  closed tier; this tier fixed the newly-found duplicate-pair hygiene issue around it (the
  `operations/` copy is now a shim) without reopening T14's considered disposition of the
  underlying bug itself.

## Clean claim this tier is permitted to make, and no more

Within this wave's 85-file scope: every file received at least a purpose+reachability pass;
the two mandated deep-dive leads, all 6 `verification/` files, all 3 `policy*` files, and
every file this tier changed received a correctness-level read with (for every fix) empirical
re-verification from source, not just the survey's own characterization. This does **not**
claim uniform deep-audit coverage of all 85 files — per the survey's own honest accounting,
roughly 19 of `operations/`'s files, `testing/first_generator.py`'s 571-line body, and five
shim-confirmed-but-uninspected `project_mgmt/`/`workspace/` canonical files were confirmed
only at the reachability/shim-structure level, not read line-by-line.

## What this tier does NOT claim

- It does not claim the policy system is fixed — only that its complete non-enforcement is
  confirmed, traced to specific line-level root causes, and that closing it requires a
  product decision this tier did not make.
- It does not claim `operations/quick_self_test.py`'s self-test is functional — only that it
  no longer crashes or pollutes the source tree merely by being imported; its target
  (`dashboard_mission_control`) remains a retired module with no confirmed drop-in
  replacement.
- It does not claim `services/completion_judge.py`/`autonomy_costs.py` or the ~19 unread
  `operations/` files are defect-free — only that they were not found to contain the specific
  defect shapes this hunt has been checking for, at the depth this tier's time budget allowed.

## `hunt_iterations` / `fix_revisions`

`hunt_iterations`: 1/3 used. `fix_revisions`: 0 — all 12 fixes correct on first pass,
RED→GREEN verified on the first attempt for all 7 test cases. The `services/` shim
conversion (C5) received extra empirical scrutiny beyond the standard RED→GREEN cycle (all
73 pre-existing tests in the 8 affected files run directly against the new shims) given its
above-average regression blast radius; no survey severity claim required correction this
tier.

## Programme close

This is the eighth and final wave of the T9-T16 continuation plan. As stated in
`docs/hunts/BACKEND_REMAINDER_WAVES_PLAN.md` from the outset, the 8-wave decomposition was a
data-backed, estimate-based prioritization of the ~845 files left with no individual
disposition after T0-T8 — not a mathematically-verified 100% line-by-line partition. A
closing summary of the full T0-T16 programme (total VERIFIED DEFECTs fixed, recurring bug
patterns, accumulated `[REQUIRES HUMAN REVIEW]` items, and this same scope caveat) follows
separately in `docs/hunts/INVENTORY.md` and the final report to the user.
