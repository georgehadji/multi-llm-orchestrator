# T7 — Execution & Filesystem Surface — Phase 8 Coverage & Residual-Risk Statement

## Scope actually audited

`orchestrator/app_verifier.py` + `orchestrator/appbuilder/verifier.py`
(full read, byte-diffed), `orchestrator/appbuilder/builder.py` +
`orchestrator/appbuilder/__init__.py` (read, to trace the package-export
resolution order that turns the diff into a live-surface bug). The
remaining 52 of the 54 subprocess/eval/exec-touching files enumerated in
Phase 0 were **not** read or candidate-generated against this tier — see
residual note below.

## Gates (this tier's fixed tree)
```
black --line-length=100 --check --fast <changed files>          PASS
ruff check <changed files>                                        PASS
lint-imports                                                       PASS (5/5 KEPT)
python scripts/check_root_module_freeze.py                        PASS (256/256)
python scripts/check_test_markers.py                               PASS
mypy orchestrator/domain/ .../application/ .../container.py       PASS (58 files, 0 issues)
bandit -lll -r orchestrator/appbuilder/verifier.py                  PASS (0 issues)
python -m pytest tests/unit/test_hunt_t7_execution.py             PASS (3/3)
python -m pytest tests/ -k "appbuilder or app_verifier or app_builder"
                                                                    PASS (2/2 — the new tests;
                                                                     zero pre-existing tests
                                                                     covered this area at all)
```

## RED→GREEN verification
Verified via `git stash push --keep-index` on the one fixed source file
(test file stays present/staged), full T7 test file re-run, fix restored
via `git stash pop`. All 3 tests failed against the pre-fix tree for the
exact predicted reason: two failed on class identity
(`orchestrator.appbuilder.AppVerifier`/`orchestrator.appbuilder.verifier.
AppVerifier` not `is` the canonical `orchestrator.app_verifier.
AppVerifier`), and the third — the concrete real trigger — failed with the
recorded pip-install subprocess argument being the literal relative string
`"generated_app/requirements.txt"` rather than an absolute path. All 3
pass on the fixed tree.

Note: an earlier draft of the third test used `tmp_path / "generated_app"`
as `output_dir`, which is already absolute (pytest's `tmp_path` fixture
returns an absolute path), so it passed even against the pre-fix buggy
class and would not have caught a regression. Caught by literally running
the RED check and seeing it pass when it should fail; corrected to
`monkeypatch.chdir(tmp_path)` + a genuinely relative `Path("generated_app")`
before re-verifying RED→GREEN.

## Verdict
- **VERIFIED DEFECT fixed:** 1 — `appbuilder/verifier.py` had silently
  diverged from the canonical, `AppBuilder`-trusted `app_verifier.py`,
  losing an absolute-path fix for a `cwd`-relative subprocess argument,
  and — because of `appbuilder/__init__.py`'s import order — leaking the
  buggy class through the package's own public `AppVerifier` name even
  though no internal caller was affected.
- **Reachability caveat:** the specific landmine (a caller doing
  `from orchestrator.appbuilder import AppVerifier`) has zero live callers
  today — confirmed by grep. The fix is still correct and warranted: the
  module is unconditionally imported by `appbuilder/__init__.py`, the
  public name it exposes was silently wrong, and the established
  duplicate-pair remediation (shim to canonical) is the same fix regardless
  of current caller count, consistent with how T4/T5 handled similarly
  dormant-but-real divergences.
- **Residual, not independently investigated this tier:** 52 of the 54
  files identified in the Phase 0 subprocess/eval/exec census
  (`testing/first_generator.py`, `operations/deployment_service.py`,
  `generators/website_generator.py`, `infrastructure/verification_checks.py`,
  `ide_backend/ide_orchestrator_server.py`, `cost_optimization/github_push.py`,
  `safety/secure_execution.py`, `safety/sandbox.py`, `preview_server.py`,
  `safety/dependency_scanner.py`, `quality_control.py`,
  `nexus_search/server_manager.py`, and 40 more) were enumerated by match
  count only — none were read, diffed, or candidate-generated against.
  `[UNK]` whether any carry their own command-injection, path-traversal,
  or unsafe-`eval`/`exec` defects; this tier makes no claim about them.

## Clean claim this tier is permitted to make, and no more
Within the scope listed above — the `app_verifier.py`/`appbuilder/
verifier.py` duplicate pair and the package-export path that exposes it —
no VERIFIED defect remains unfixed. This does **not** claim any of the
other 52 subprocess/eval/exec-touching files were audited, nor that
`safety/sandbox.py` or `safety/secure_execution.py` — the two files whose
names most directly suggest security-relevant subprocess isolation — were
reviewed for injection or sandbox-escape defects this tier.

## What this tier does NOT claim
- It does not claim any file besides `app_verifier.py`/`appbuilder/
  verifier.py` was examined for subprocess argument construction safety,
  shell injection, or unsafe `eval`/`exec` use.
- It does not claim `AppBuilder`'s broader build pipeline (beyond the one
  `AppVerifier` construction line) was reviewed.
- It does not claim the dormant package-export landmine was the only
  consequence of this divergence — only that it is the one confirmed via
  direct `is`-identity checks and grep for callers.

## `hunt_iterations` / `fix_revisions`
`hunt_iterations`: 1/3 used. `fix_revisions`: 2/2 — the shim fix itself was
correct on first pass; the *test* needed one revision (see RED→GREEN note
above: an absolute-by-construction `tmp_path` input made the third test a
false-negative-prone check until corrected to a genuinely relative path).
