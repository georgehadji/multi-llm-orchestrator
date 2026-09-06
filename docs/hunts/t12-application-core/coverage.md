# T12 — application/ orchestration core — Phase 8 Coverage & Residual-Risk Statement

## Scope actually audited

93 files (`application/`'s 42 minus 2 already fixed in T1, `planning/`'s 3, `routing/`'s 2,
`reasoning/`'s 6, `agents/`+`agents.py`'s 16, `supervisor/`'s 7, `delegation/`'s 3, `meta/`'s
7, `nash/`'s 6). A background agent read all 93 at tiered depth (four pre-identified leads
plus two subsystem-reachability questions in full depth, architecturally-central files at
correctness depth, the remainder at purpose+reachability), backed by an automated
`importlib.import_module` sweep across every file — the mechanism that actually surfaced C2
and C3 below (a human/LLM read of import statements in isolation would very plausibly have
missed the wrong-dot-count bugs; an executable import check does not). Every fix was
independently re-verified from source and, where practical, empirically re-executed (real
imports, real construction calls with a dummy API key, real pre-existing test suites)
before being applied — see inventory.md's "Method"/Phase 1-3 section and each C-item's own
verification notes.

## Gates (this tier's fixed tree)

```
black --line-length=100 --check --fast <8 changed/new files>            PASS (1 file needed
                                                                            `black` applied —
                                                                            the new test file;
                                                                            re-verified tests
                                                                            still pass after)
ruff check <8 changed/new files>                                          PASS
lint-imports                                                              PASS (5/5 KEPT,
                                                                             824 files)
python scripts/check_root_module_freeze.py                               PASS (256/256)
python scripts/check_test_markers.py                                     PASS
mypy orchestrator/domain/ .../application/ .../container.py              PASS — isolated
                                                                             diff against the
                                                                             pre-fix tree is
                                                                             empty after
                                                                             sorting both
                                                                             outputs (raw diff
                                                                             showed only
                                                                             mypy's own
                                                                             non-deterministic
                                                                             error-ordering,
                                                                             confirmed by
                                                                             identical line
                                                                             counts, 1879 both
                                                                             sides, and a
                                                                             zero-byte sorted
                                                                             diff)
bandit -lll -r <7 changed Python source files>                            PASS (0 issues at
                                                                             high severity; 3
                                                                             pre-existing
                                                                             low-severity
                                                                             B110 findings on
                                                                             lines this tier
                                                                             did not touch)
python -m pytest tests/unit/test_hunt_t12_application_core.py            PASS (5/5)
python -m pytest tests/unit/test_container_acr_wiring.py
             tests/unit/test_container_stage_discovery.py
             tests/unit/test_engine_run_project.py
             tests/unit/test_hunt_t5_resilience.py                       PASS (26/26 existing,
                                                                             zero regressions —
                                                                             includes
                                                                             ServiceContainer's
                                                                             real, non-mocked
                                                                             build() path)
python -m pytest tests/ -q -m "unit or integration"                      2535 passed (+5 over
                                                                             T11's 2530), 2
                                                                             pre-registered
                                                                             environmental
                                                                             failures unchanged
                                                                             (test_openrouter_
                                                                             model_audit.py,
                                                                             blocked outbound
                                                                             network access to
                                                                             live OpenRouter
                                                                             endpoints — same
                                                                             failure signature
                                                                             as every prior
                                                                             tier), 21 skipped,
                                                                             157 deselected —
                                                                             zero regressions
```

## RED→GREEN verification

Verified via `git stash push --keep-index` on all 7 fixed source files (new test file stays
present, staged), full T12 test file re-run against the pre-fix tree, fix restored via `git
stash pop`. All 5 tests failed against the pre-fix tree for the exact predicted reason:

- C1: `charge_fn` never awaited (`Expected mock to have been awaited once. Awaited 0 times.`).
- C2: `ModuleNotFoundError: No module named 'orchestrator.engine_core.ara_pipelines'` —
  the exact original error.
- C3: `ModuleNotFoundError: No module named 'orchestrator.routing.selector'` — the exact
  original error.
- C4: `assert ViaShim is Canonical` failed — the two classes were genuinely different
  objects pre-fix.
- C5: `warm_prompt_cache` mock never awaited, with the captured log showing the exact
  original error: `Cache warming failed (non-critical): No module named
  'orchestrator.operations.cache_warmup'`.

All 5 pass on the fixed tree, both before and after the `black` reformat of the test file.

## Verdict

- **VERIFIED DEFECT fixed:** 5 — C1 (the Instructor fast-path decomposition path made a
  real, billable API call with zero budget tracking — a money/budget-escape defect on the
  common case for real project descriptions), C2 (3 files / 7 wrong-depth relative imports
  that made a fully-built, enabled-by-default, ~5,000-line reasoning-pipeline subsystem
  completely unreachable, silently, via bare `except ImportError` with no logging), C3 (a
  broken import that has been present since routing/__init__.py's introduction, breaking
  both the routing/ package and, transitively, model_routing.py — the file CLAUDE.md's own
  architecture table names for LLM routing), C4 (a diverged duplicate pair where the
  subpackage copy still carried an AttributeError-prone status check its root canonical
  twin had already fixed), C5 (a broken import, corrected together with a factual correction
  to T10's prior "live hot-path" characterization — confirmed via git history to be fully
  dead code that has never had a caller).
- **Residual, surveyed but not fixed — `[REQUIRES HUMAN REVIEW]`:** `HumanInTheLoop`'s
  fail-closed gate never reaching `ProjectRunner` (and, as a direct consequence,
  `UnattendedGuard`'s checkpoint detection always reporting no checkpoint present) — a
  3-line wiring fix that very likely only ever makes the gate more accurate, but touches
  fail-closed safety-gate behavior this tier did not independently verify as risk-free;
  `orchestrator/agents.py` permanently shadowed by the `orchestrator/agents/` package
  (currently inert, needs a rename/restructuring decision, not a mechanical fix);
  `application/task_executor.py`'s missing `await` on an async dependency-context call
  (confirmed real, but inside code with zero live callers); `delegation/batch_runner.py`'s
  dead-code docstring-contract violation.
- **Cleared (innocent):** the `executor.py`/`task_executor.py` cluster (genuinely different
  responsibilities, not a duplicate pair); `application/dependency_resolver.py` (a third,
  genuinely distinct, if currently-dead, implementation — not a duplicate of either
  `engine_core/dep_resolver.py` or `engine_core/dependency_resolver.py`); the cross-tier
  `engine_core/budget_enforcer.py` vs `application/budget_enforcer.py` comparison (the
  former is a plain shim, inherits T1's fix automatically, zero divergence risk); `nash/`
  and `meta/` as packages (both genuinely live and reachable, not orphaned/experimental,
  though each contains some internally-unreferenced code); `supervisor/`'s CLI registration
  being dead (an alternate live HTTP-route path exists); `planning/decomposer.py`'s
  self-shadowing import (currently harmless, dead code).
- **Discovered but explicitly out of this tier's declared scope, not folded in:**
  `commands/nash.py`'s unconditional crash on `nash backup` (missing `nash_backup.py`,
  never existed) — handed to T15 (commands/ is its declared scope); `cli.py`'s stale
  docstring naming a nonexistent dispatcher module — handed to T15 (cli* is its declared
  scope); a broad structural pattern (roughly a third of `application/` plus most of
  `agents/`/`planning/` are real, tested-in-isolation, but production-unwired) — recorded
  for visibility, not assigned to any specific future wave.

## Clean claim this tier is permitted to make, and no more

Within the 93 files in this wave's declared scope: every file received at least an automated
import-success check plus a purpose+reachability pass; the four pre-identified leads, the two
subsystem-reachability questions, and the files touched by the five fixes received a
correctness-level read. This does **not** claim uniform deep-audit coverage of all 93 files —
most of `agents/`'s individual role classes, `reasoning/ara_pipelines.py`'s 22 individual
pipeline implementations (4,288 lines, class/def inventory only), and most of `nash/`'s and
`meta/`'s internals beyond the specific files diffed for C4 were not read start-to-end.

## What this tier does NOT claim

- It does not claim C1's charged amount is the *real* cost of the Instructor API call — only
  that it is a reasonable, consistently-derived estimate, materially better than the previous
  unconditional $0.00. A follow-up that threads real token usage out of
  `structured_outputs.py` (outside any tier's current scope) remains open.
- It does not claim the ARA reasoning-pipeline subsystem (C2) is correct or safe to route
  real production tasks through now that it's reachable — only that it is now importable and
  constructible. Its 22 individual reasoning-method implementations were not
  correctness-audited in this tier (class/def inventory only).
- It does not claim `HumanInTheLoop`/`UnattendedGuard` (residual, above) are currently unsafe
  — only that the checkpoint-detection wiring gap was traced and its likely-safe fix
  identified, not applied.
- It does not claim `agents/`'s or `planning/`'s "mostly unwired" pattern (see inventory.md)
  is itself a defect requiring a fix — it is reported as a structural observation for whoever
  next works in this area, not a verdict that this code should be deleted or wired in.

## `hunt_iterations` / `fix_revisions`

`hunt_iterations`: 1/3 used (Phase 1-3 survey delegated to one background agent run).
`fix_revisions`: 4/5 fixes (C1, C3, C4, C5) correct on first pass. C2 needed one revision
*during verification, before any test was written*: an empirical import-success check after
the first edit (`method_selector.py:17` only) immediately surfaced three more instances of
the identical wrong-dot-count bug in the same file (lines 18, 21, 593), fixed in the same
pass. No fix required a second revision after RED→GREEN verification began.
