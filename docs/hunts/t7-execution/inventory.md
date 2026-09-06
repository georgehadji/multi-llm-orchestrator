# T7 — Execution & Filesystem Surface — Phase 4 Inventory

Per `docs/DEFECT_HUNT_PLAN.md` §3/§4. Budget: 10 candidates. Spent: 1.

## Phase 0/1 — surface census & reachability triage

`grep -rln "subprocess\.\|eval(\|exec(" orchestrator/ --include='*.py'`
(excluding `__pycache__`) → **54 files** touch process spawning or dynamic
code execution. Ranked by match count, top of the list:

| File | matches |
|---|---|
| `testing/first_generator.py` | 14 |
| `operations/deployment_service.py` | 12 |
| `generators/website_generator.py` | 12 |
| `infrastructure/verification_checks.py` | 11 |
| `ide_backend/ide_orchestrator_server.py` | 11 |
| `cost_optimization/github_push.py` | 11 |
| `appbuilder/verifier.py` | 11 |
| `app_verifier.py` | 11 |
| `safety/secure_execution.py` | 9 |
| `safety/sandbox.py` | 9 |
| `preview_server.py` | 9 |
| `safety/dependency_scanner.py` | 8 |
| `quality_control.py` | 8 |
| `quality/quality_control.py` | 8 (checked clean in T2) |
| `nexus_search/server_manager.py` | 8 |

`app_verifier.py` / `appbuilder/verifier.py` matched the by-now-familiar
root-vs-subpackage duplicate-pair naming shape already found diverged in
T1/T2/T3/T5 — pursued first given that strong prior hit rate.

## Candidates

### C1 — VERIFIED DEFECT — `appbuilder/verifier.py` diverged from canonical `app_verifier.py`, dropping an absolute-path fix for a `cwd`-relative subprocess call, and leaking the buggy class through the package's own public name

- **Property violated:** a subprocess spawned with `cwd=output_dir` must
  receive absolute paths for any argument built from a path outside that
  `cwd` — a relative path gets resolved a second time against `cwd` and
  fails to open.
- **Location:** `orchestrator/appbuilder/verifier.py` (pre-fix, 274 lines)
  vs canonical `orchestrator/app_verifier.py` (277 lines).
- **Finding:**
  `diff orchestrator/app_verifier.py orchestrator/appbuilder/verifier.py`
  showed the *entire* divergence was 3 removed lines (an explanatory
  comment) + 1 changed line: `app_verifier.py`'s
  `verify_local()` builds its pip-install subprocess call with
  `str(req_file.resolve())`; `appbuilder/verifier.py` still used the
  unresolved `str(req_file)`, with the comment explaining why:
  `verify_local()` runs `subprocess.run(..., cwd=output_dir)`, so a
  `requirements.txt` path built relative to repo-root would be resolved
  against `output_dir` a second time and double-nest, failing to open.
- **Reachability, part A (the class construction path):**
  `orchestrator/appbuilder/builder.py::AppBuilder` (the only class that
  constructs an `AppVerifier` in a live pipeline) imports `AppVerifier`
  directly from `orchestrator.app_verifier` (the canonical, fixed module),
  **not** from its sibling `.verifier` — so `AppBuilder`'s own internal use
  was never affected by this specific divergence.
- **Reachability, part B (the real live landmine — package-level export
  identity):** `orchestrator/appbuilder/__init__.py` does
  `from .builder import *` **then** `from .verifier import *`. Because
  `builder.py` has no `__all__` and re-exports its own imported
  `AppVerifier` name, the later wildcard import from the *local*
  `.verifier` module **overwrites** it in the `appbuilder` package
  namespace. Verified directly:
  `orchestrator.appbuilder.AppVerifier is orchestrator.appbuilder.verifier.AppVerifier`
  → `True` (the buggy local class), while
  `orchestrator.appbuilder.AppVerifier is orchestrator.app_verifier.AppVerifier`
  → `False` (pre-fix). Any code doing
  `from orchestrator.appbuilder import AppVerifier` — the natural way to
  consume a package's own public surface — silently gets the buggy,
  relative-path class instead of the one `AppBuilder` itself trusts.
  Grepped for such a caller (`from orchestrator.appbuilder import` /
  `from ..appbuilder import` / `from .appbuilder import`) across
  `orchestrator/` and `tests/` — **zero found**, so this specific landmine
  is currently dormant (no live caller uses the package-level name today),
  but it is a real, exposed public-API divergence rather than genuinely
  dead code — the module itself (`appbuilder/verifier.py`) is imported by
  `appbuilder/__init__.py` on every `import orchestrator.appbuilder`.
- **Innocence attempt:** none — same "two independently-fixed-then-
  diverged copies" pattern as every prior tier's root-cause-1 finding
  (T1 `cost.py`, T2 `secrets_generator.py`/`codebase_writer.py`, T3
  `checkpoints.py`, T5 `circuit_breaker.py`), here manifesting as a
  package `__init__.py` re-export ordering hazard rather than a
  same-named-but-uncoordinated-import hazard.
- **Fix:** `orchestrator/appbuilder/verifier.py` rewritten as a re-export
  shim of the canonical `orchestrator.app_verifier` (mirroring the shim
  style already used for `operations/circuit_breaker.py`,
  `secrets_generator.py`, etc.) — this both closes the package-export
  landmine (`orchestrator.appbuilder.AppVerifier` now resolves to the
  fixed class) and removes the dead, diverged duplicate implementation.
- **Tests:** `test_c1_appbuilder_verifier_module_is_canonical`,
  `test_c1_appbuilder_package_exposes_canonical_appverifier` (the actual
  live-surface bug — asserts `orchestrator.appbuilder.AppVerifier is
  orchestrator.app_verifier.AppVerifier`), and
  `test_c1_pip_install_uses_absolute_requirements_path` (real trigger: a
  genuinely relative `output_dir` Path — via `monkeypatch.chdir` +
  `Path("generated_app")`, not a `tmp_path`-derived absolute path which
  would have trivially passed either way — fed into `verify_local()` with
  `subprocess.run` patched to record its argv; asserts the recorded
  `pip install -r <path>` argument is absolute).

## Gates (Phase 8, run against the fixed tree)

See `coverage.md`.
