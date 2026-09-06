# T17 — Duplicate-pair convergence sweep — Phase 8 Coverage & Residual-Risk Statement

## Scope actually audited

**All 65** both-sides-define root/subpackage pairs in `orchestrator/`, enumerated
by AST rather than sampled. This is the first wave in the T0–T17 programme whose
Phase-1 surface is **exhaustive over its pattern** — the coverage claim below is
therefore total in a way no prior wave's could be, but total over a *pattern*,
not over the code.

Every pair received four mechanical signals (divergence size, importer counts,
package-`__init__` exposure, and real subprocess importability of both sides).
Every pair with a non-trivial diff and shared definitions had its diff read in
full. Six pairs were converged; each convergence was preconditioned on a
mechanical name-superset proof and confirmed afterward by object identity.

## Gates

```
black --line-length=100 --check --fast <8 changed files>          PASS (1 file
                                                                      reformatted,
                                                                      re-verified)
ruff check <8 changed files>                                      PASS
lint-imports                                                      PASS (5/5 KEPT)
python scripts/check_root_module_freeze.py                        PASS (256/256)
python scripts/check_duplicate_pairs.py            (new this wave) PASS (59 pairs
                                                                      == baseline)
python scripts/check_test_markers.py                              PASS
mypy orchestrator/domain/ .../application/ .../container.py       PASS — net −17
                                                                      errors
                                                                      (1554 → 1537),
                                                                      186 → 184 files
bandit -lll -r <changed files>                                    PASS (no issues
                                                                      identified)
python -m pytest tests/unit/test_hunt_t17_duplicate_pairs.py      PASS (8/8)
python -m pytest tests/ -q -m "unit or integration"               2580 passed
                                                                      (+8 over T16's
                                                                      2572), 2
                                                                      pre-registered
                                                                      environmental
                                                                      failures
                                                                      unchanged, 20
                                                                      skipped, 157
                                                                      deselected —
                                                                      zero regressions
```

### The mypy delta, stated precisely

The headline is −17 errors, but the composition matters more than the total:

- **26** error lines on `design/design_system.py` disappeared (it is now a shim
  with nothing to typecheck). Same for the other converged forks.
- **47** error lines appeared on root `design_system.py` — a file this wave did
  **not modify**. Before the fix mypy reported **zero** lines for it, because
  nothing in the checked scope reached it; `design/component_registry.py`
  resolved `DesignSystem` to the fork instead. Post-fix that import resolves to
  root, so mypy now follows into a file whose pre-existing type looseness was
  previously invisible. These are not new defects and not defects this wave
  introduced.
- Two pre-fix errors encoded **C4 itself** — `"DesignSystem" has no attribute
  "tone"` and `has no attribute "to_prompt_context"` on `component_registry.py`
  — and both are gone post-fix. Independent, tool-generated corroboration of the
  finding and of the fix.
- `ComponentSource` / `ComponentSpec` `attr-defined` errors persist identically
  on both sides of the fix, consistent with C2's finding that those names exist
  nowhere in the repository.

## RED→GREEN verification

`git stash push` on the six converged files (test file and gate script staged,
left in place), full T17 test file re-run against the pre-fix tree, `git stash
pop`. **All 8 tests failed pre-fix, each for the exact predicted reason:**

- C1: `ModuleNotFoundError: No module named 'orchestrator.integrations.api_builder'`
  — the original error verbatim.
- C3 (×4, parametrized): distinct objects, e.g. `<class
  'orchestrator.design.component_library.ComponentLibrary'> is not <class
  'orchestrator.component_library.ComponentLibrary'>`.
- C4: the package-level `DesignSystem` was the stale fork.
- Both gate tests: the pre-fix tree carries 65 pairs against a 59-pair baseline,
  so the gate correctly reported 6 unrecorded duplicates and exited 1.

## Verdict

- **VERIFIED DEFECT fixed:** 2 — C1 (a byte-identical duplicate that could not
  load), C4 (a stale fork exposed through its package namespace, missing the
  exact attributes the website generator formats into output; latent rather than
  live, since all real consumers import root).
- **VERIFIED DEFECT recorded, not fixed:** 1 — C2 (root `website_generator.py`
  and therefore `cli_website.py` unimportable). The causal fix requires
  `ComponentSource`, which does not exist anywhere in the repository; a
  path-only change would mask the symptom. `[REQUIRES HUMAN REVIEW]`,
  consistent with T13's disposition of the same subsystem.
- **Hygiene converged:** 4 byte-identical pairs (C3), removing ~3,500 lines of
  duplicated source that could diverge silently.
- **FALSE (innocent):** 1 — C5, a stale model id in a dict nothing reads.
  Recorded to prevent re-raising.
- **UNKNOWN:** 1 — C6, `ide_backend/log_config.py`'s separate unfiltered logger
  hierarchy. No executable trigger is possible without `fastapi`; routed to T21
  rather than promoted on reasoning.
- **Triaged, not converged:** 57 — overwhelmingly import-depth-only differences.
  Frozen in the gate baseline.

## Clean claim this wave is permitted to make, and no more

**Within the duplicate-pair pattern:** every both-sides-define root/subpackage
pair in `orchestrator/` at this commit was enumerated and triaged; 6 were
converged; the remaining 59 are frozen by a gate that fails on any new one. That
claim is total over this pattern.

It says **nothing** about defects inside those files beyond divergence between
the two copies. A pair whose two copies agree perfectly is "resolved" by this
wave's definition while both copies may still share the same bug — indeed C1's
two copies were byte-identical *and* one of them could not load.

## What this wave does NOT claim

- It does not claim the 57 unconverged pairs are equivalent — only that their
  observed differences are import-depth and comments, which is weaker than a
  proof of semantic equivalence.
- It does not claim `cli_website.py` or root `website_generator.py` now work;
  C2 is recorded, not fixed.
- It does not claim the `design/` component-registry subsystem is repaired.
  `ComponentSource` remains undefined and the subsystem remains
  `[REQUIRES HUMAN REVIEW]`.
- It does not claim anything about `ide_backend/`, which could not be exercised.

## `hunt_iterations` / `fix_revisions`

`hunt_iterations`: 1/3. `fix_revisions`: 0 on shipped orchestrator code; 2 on the
new gate script, both caught by its own tests before commit. One Phase-3
hypothesis (that converging `design_system` would repair `component_registry`)
was falsified and discarded before any fix was written.

## Next wave

**T18 — silent-failure sweep at scale.** 923 broad-except sites, of which 77
have an immediate `pass`/`continue` and no logging; T6 ranked only 18.
