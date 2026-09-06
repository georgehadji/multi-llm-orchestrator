# T13 — generators/design/quality — Phase 8 Coverage & Residual-Risk Statement

## Scope actually audited

122 files (`generators/` 34, `appbuilder/` 5, `codebase/` 8, `design/` 33, `scaffold/` 10,
`output/` 3, `quality/` 19, plus 3 root `code_*.py` files named by the plan). A background
agent surveyed all 122, giving full depth to the pre-identified `website_validator.py` lead
and a systematic root-vs-subpackage duplicate-pair sweep (this wave's file set had an
unusually large number of such pairs), and purpose+reachability depth to the remainder, per
the plan's own framing of this wave as lower architectural blast radius than T9-T12. Every
finding was independently re-verified from source before acting — most significantly, the
survey's "highest impact this wave" claim about `design/component_registry.py` was
downgraded to `[REQUIRES HUMAN REVIEW]` after direct inspection of
`generators/website_generator.py` surfaced a pre-existing, explicitly-documented workaround
comment the survey's report did not mention (see inventory.md).

## Gates (this tier's fixed tree)

```
black --line-length=100 --check --fast <3 changed/new files>              PASS
ruff check <3 changed/new files>                                          PASS
lint-imports                                                              PASS (5/5 KEPT,
                                                                             824 files)
python scripts/check_root_module_freeze.py                               PASS (256/256)
python scripts/check_test_markers.py                                     PASS
mypy orchestrator/domain/ .../application/ .../container.py              PASS — isolated
                                                                             diff empty
                                                                             (sorted; T13's
                                                                             fixed files
                                                                             aren't reached
                                                                             by the core-path
                                                                             invocation)
bandit -lll -r <2 changed Python source files>                            PASS (0 issues at
                                                                             high severity)
python -m pytest tests/unit/test_hunt_t13_generators_design_quality.py    PASS (5/5)
python -m pytest tests/ -k "quality_control or quality_gate"              PASS (27/27
                                                                             existing +
                                                                             new, zero
                                                                             regressions)
python -m pytest tests/ -q -m "unit or integration"                      2549 passed (+14
                                                                             over the prior
                                                                             commit's 2535 —
                                                                             13 new hunt
                                                                             tests across
                                                                             T13 and the
                                                                             out-of-band
                                                                             routing.json
                                                                             fix, +1 test
                                                                             newly un-skipped
                                                                             by the supplied
                                                                             OpenRouter
                                                                             snapshot), 2
                                                                             pre-registered
                                                                             environmental
                                                                             failures
                                                                             unchanged, 20
                                                                             skipped (was 21
                                                                             — the snapshot-
                                                                             based audit test
                                                                             now runs for
                                                                             real), 157
                                                                             deselected —
                                                                             zero regressions
```

## RED→GREEN verification

Verified via `git stash push --keep-index` on the 2 fixed source files (new test file stays
present, staged), full T13 test file re-run against the pre-fix tree, fix restored via `git
stash pop`. Both defect-proving tests failed against the pre-fix tree for the exact predicted
reason:
- C1: `passed=True` with message `"No security issues found"` — the exact original
  false-clean result, even with an unreadable file present.
- C2: `assert ViaShim is Canonical` failed — the two classes were genuinely different objects
  pre-fix (an independent, unshimmed duplicate).

The 3 other tests (no-regression, real-secret-detection, no-circular-import) passed on both
trees, as expected — they don't depend on the fix.

## Verdict

- **VERIFIED DEFECT fixed:** 2 — C1 (`quality_control.py`'s security scan silently reported
  false-clean on unreadable files — the third instance of this hunt's Pattern 4, after T8's
  `website_validator.py` and T9's `generated_output_scanner.py`), C2 (an unshimmed duplicate
  of C1's file carrying the identical bug, converted to a shim).
- **Residual, surveyed but not fixed — `[REQUIRES HUMAN REVIEW]`:** `design/component_registry.py`'s
  broken import to two classes (`ComponentSource`, `ComponentSpec`) that have never existed
  anywhere in this repository's history — but this is a previously-known, already-mitigated
  gap (a documented `# FIXED: ... component_registry has broken dependencies` workaround is
  already in place in `generators/website_generator.py`), not a silent regression; completing
  it means designing a compatibility-scoring algorithm from scratch, a product decision, not
  a mechanical fix. Also residual: `output/organizer.py` missing two pipeline steps present in
  the live root file (dead, no live importers); `docker_generator.py`'s root/subpackage
  divergence (hardcoded default DB credentials vs. hardened generation — both copies dead
  today, security-adjacent); `design/frontend_security.py`'s complete, unused CSP/CSRF
  generation library; a dead legacy import chain (root `website_generator.py` →
  nonexistent `component_registry.py` → `cli_website.py`, fully unreachable, delete-vs-repair
  is a scope decision); a same-named-class naming footgun between two independent
  `CodebaseAnalyzer` implementations (`[UNK]`, not confirmed as an active mix-up).
- **Cleared (innocent):** `generators/website_validator.py` (already fixed by T8, reconfirmed
  intact, flows through a root shim converted by an earlier non-hunt commit); 21 confirmed
  clean shims and depth-adjusted duplicate pairs; 10+ confirmed genuinely-different-purpose
  name collisions (not duplicates) including a 3-way `assembler.py` collision and a 4-way
  `decomposer.py` collision.
- **Discovered but explicitly out of this tier's declared scope, not folded in:** none new
  this tier beyond what's already recorded as residual above — everything found fell within
  the 122-file declared scope or its immediate, necessary cross-references (e.g. root
  `website_generator.py`, which the `design/` finding's trigger required reading).

## Clean claim this tier is permitted to make, and no more

Within the 122 files in this wave's declared scope: every file received at least a
purpose+reachability pass; the pre-identified `website_validator.py` lead, the duplicate-pair
sweep, and the two fixed files received a correctness-level read. This does **not** claim
uniform deep-audit coverage of all 122 files — the majority of `generators/wf100/`'s files
(one, `checks.py` at 2251 lines, was header-only), most of `design/`'s Hallmark-skill
subsystem, and most of `scaffold/`'s template files were read at header/reachability depth
only.

## What this tier does NOT claim

- It does not claim `design/component_registry.py`'s underlying feature gap is acceptable to
  leave as-is indefinitely — only that completing it is a design decision outside this hunt's
  mandate, and that the existing fallback is honest (generic components, not corrupted ones)
  rather than dangerous.
- It does not claim the `docker_generator.py` and `output/organizer.py` divergences are safe
  to ignore forever — only that both are confirmed fully dead today, so no live behavior is
  currently affected.
- It does not claim the `[UNK]` naming-collision findings (`CodebaseAnalyzer` ×2, several
  `themes.py`/`routing.py` pairs not fully diffed) are free of defects — only that they were
  not chased to a definite verdict within this tier's budget.

## `hunt_iterations` / `fix_revisions`

`hunt_iterations`: 1/3 used (Phase 1-3 survey delegated to one background agent run).
`fix_revisions`: both fixes (C1, C2) correct on first pass, RED→GREEN verified on the first
attempt. One correction was made to the *survey's own severity classification* (component_registry.py,
downgraded from VERIFIED DEFECT to REQUIRES HUMAN REVIEW) during independent verification —
not a fix revision.
