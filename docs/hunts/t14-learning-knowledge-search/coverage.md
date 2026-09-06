# T14 — learning/knowledge/search — Phase 8 Coverage & Residual-Risk Statement

## Scope actually audited

71 files (`learning/` 14, `knowledge/` 6, `nexus_search/` 21, `pattern_learner/` 5,
`context_mgmt/` 7, root `memory_tier.py` 1, `ingest/` 2, `analysis/` 15). A background agent
surveyed all 71, running an exhaustive direct-`import` test across every module (the
mechanism that surfaced C1-C4) plus full-depth reads of four pre-identified leads, and
purpose+reachability depth elsewhere, per the plan's own framing of this wave as auxiliary /
not on the critical execution path. Every finding was independently re-verified from source —
diffs re-run directly, imports re-executed, every claimed-dead class re-confirmed via a fresh
repo-wide grep for its name — before any fix was applied.

## Gates (this tier's fixed tree)

```
black --line-length=100 --check --fast <6 changed files>                 PASS
ruff check <6 changed files>                                              PASS
lint-imports                                                              PASS (5/5 KEPT,
                                                                             824 files)
python scripts/check_root_module_freeze.py                               PASS (256/256)
python scripts/check_test_markers.py                                     PASS
mypy orchestrator/domain/ .../application/ .../container.py              PASS — isolated
                                                                             diff empty
bandit -lll -r <6 changed Python source files>                            PASS (0 issues at
                                                                             high severity;
                                                                             1 medium/low-
                                                                             confidence B608
                                                                             heuristic hit on
                                                                             the now-validated
                                                                             f-string itself —
                                                                             a known bandit
                                                                             false-positive
                                                                             class for
                                                                             pre-validated
                                                                             identifiers,
                                                                             already annotated
                                                                             `# noqa: S608`
                                                                             matching the
                                                                             source it was
                                                                             ported from)
python -m pytest tests/unit/test_hunt_t14_learning_knowledge_search.py   PASS (9/9)
python -m pytest tests/unit/test_knowledge_rerank.py
             tests/test_optimizations.py tests/test_e2e_full_suite.py
             tests/test_bug_regression.py                                PASS (86/86
                                                                             existing, zero
                                                                             regressions)
python -m pytest tests/ -q -m "unit or integration"                      2558 passed (+9
                                                                             over T13's 2549),
                                                                             2 pre-registered
                                                                             environmental
                                                                             failures
                                                                             unchanged, 20
                                                                             skipped, 157
                                                                             deselected — zero
                                                                             regressions
```

The full run completed (142s) confirming zero regressions, as the targeted checks below
already indicated: 9 new tests accounted for the entire pass-count increase, with no change
to the skip/deselect counts.

## RED→GREEN verification

Verified via `git stash push --keep-index` on the 6 fixed source files (new test file stays
present, staged), full T14 test file re-run against the pre-fix tree, fix restored via `git
stash pop`. 8 of 9 tests failed against the pre-fix tree for the exact predicted reason:
- C1: `ImportError: cannot import name 'KnowledgeEntry'` — the exact original error.
- C2: `ImportError: cannot import name 'ModelPerformanceGraph'` — the exact original error.
- C3, C4: shim-identity assertions failed — the two classes were genuinely different objects
  pre-fix (unshimmed duplicates).
- C5 (×3 injection/allowlist tests): `Failed: DID NOT RAISE ValueError` — confirming the
  vulnerability was real and unmitigated pre-fix.
- C5 (shim identity): failed — `analysis/performance.py` was still an independent duplicate
  pre-fix.

The 1 no-regression test (a normal query still builds correctly) passed on both trees, as
expected.

## Verdict

- **VERIFIED DEFECT fixed:** 5 — C1, C2 (two `knowledge/` shims importing names that don't
  exist in their canonical target — the third and fourth instances of this hunt's broken-shim
  pattern this month), C3, C4 (two unshimmed duplicate files whose stripped imports caused
  `NameError` the moment specific methods actually ran — C4 additionally carried a
  confidently-wrong comment claiming a real, existing module "does not exist"), C5 (a
  SQL-injection-shaped divergence: the dead copy of an unshimmed duplicate had already been
  hardened, the fix never backported to the live copy — now applied at the canonical source,
  with the now-redundant dead copy converted to a shim).
- **Residual, surveyed but not fixed:** 7 further duplicate pairs confirmed clean (zero
  current divergence) but still unshimmed — a standing structural risk given this tier alone
  produced 3 confirmed divergences of exactly this shape, not urgent enough to force into this
  tier's budget; a latent split-brain singleton (`leaderboard.py` root vs. `analysis/`,
  currently inert since only one side is ever populated); several lower-priority informational
  findings (an unreachable duplicate `except` block, a soft silent-skip in `memory_tier.py`'s
  best-effort file scan, 4 of 7 `nexus_search/optimization/` modules unwired, an unwired
  generated-app RLS-policy generator matching T13's already-documented pattern).
- **Cleared (innocent):** `learning/log_config.py` (a deliberate, non-forked, minimal
  logger-namespacing helper — records still propagate to root's secrets-masking filter via
  Python's logging hierarchy); `nexus_search/`'s three internal 5-way basename collisions (all
  genuinely disjoint subsystems); `nexus_search/` as a whole (confirmed live, reachable from
  `engine.py`, the ARA reasoning pipeline, and its own CLI — soft-optional by design, not
  orphaned); `agents/metrics.py` and `meta/performance.py` (genuinely different purposes
  despite sharing a basename with this tier's fixed files).

## Clean claim this tier is permitted to make, and no more

Within the 71 files in this wave's declared scope: every file received at minimum a direct
import-success check and a purpose+reachability pass; the four pre-identified leads and the
five fixed files received a correctness-level read. This does **not** claim uniform
deep-audit coverage of all 71 files — most of `nexus_search/agents/`, `pattern_learner/`, and
`context_mgmt/`'s individual modules were read at header/reachability depth only.

## What this tier does NOT claim

- It does not claim the full test suite's aggregate pass count at the moment of commit —
  only that every narrower check available at commit time passed, and that the full run's
  result carries forward into T15's cumulative tally.
- It does not claim the 7 residual unshimmed-but-currently-clean duplicate pairs are safe to
  leave indefinitely — only that they show no divergence today.
- It does not claim `nexus_search/optimization/`'s 4 unwired modules are dead weight worth
  removing — only that they are currently unreachable from the package's own live code path.

## `hunt_iterations` / `fix_revisions`

`hunt_iterations`: 1/3 used. `fix_revisions`: all 5 fixes correct on first pass, RED→GREEN
verified on the first attempt. No survey severity claim required correction this tier.
