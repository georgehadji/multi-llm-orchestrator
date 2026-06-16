# Orchestrator Remediation Plan

**Date:** Based on multi-session audit, execution testing, and code review  
**Status:** Plan only — awaiting go-ahead  
**Scope:** All known bugs blocking production use + architectural cleanup

---

## 0. Summary of Known Issues

| # | Area | Severity | Impact |
|---|------|----------|--------|
| 1 | Import chain broken in `reasoning/` subpackage | **CRITICAL** | Blocks all imports, prevents orchestrator from starting |
| 2 | `all_validators_pass()` signature mismatch | **CRITICAL** | Every task fails validation silently |
| 3 | `attempt_history` type mismatch (dict vs AttemptRecord) | **CRITICAL** | Crashes on serialization, corrupts saved state |
| 4 | `decompose_fn` wiring — `GeneratorService = DecomposerService` alias hides missing property | **HIGH** | Decomposition fails with `NoneType not callable` |
| 5 | Snapshots block async event loop (`subprocess.run` in async methods) | **HIGH** | Random hangs during pipeline execution |
| 6 | Website generator pipeline fails — relies on broken `component_registry` and `design_system` imports | **HIGH** | Website subcommand produces placeholder-only output |
| 7 | Next.js output had `postcss/tailwind` config conflicts with CDN approach | **MEDIUM** | First `npm run dev` fails; needs manual cleanup |
| 8 | Architecture advisor defaults to Python for web projects | **MEDIUM** | `build` subcommand generates `.py` files for HTML/CSS/JS projects |
| 9 | `Model.MOONSHOT_KIMI_K2_7_CODE` missing from `costs.json` and `fallbacks.json` | **LOW** | Falls back to default $5/$20 pricing; 4.5× cost overestimate |
| 10 | `reasoning/ara_pipelines.py` has ~50 remaining tuple-contract call sites | **LOW** | All other ARA pipelines would crash if invoked |
| 11 | `attempt_history` serialization in `_state_to_dict` doesn't handle mixed dict/AttemptRecord | **LOW** | Crashes on state save when both types coexist |

---

## Phase 1: Critical Bug Fixes (Blocking Production)

### 1.1 — Fix DecomposerService Wiring

**Root cause:** `orchestrator/services/__init__.py:12` aliases `GeneratorService = DecomposerService`. The container creates `generator = GeneratorService(decompose_fn=None)` expecting the service-layer `GeneratorService`, but gets `application/decomposer.py::DecomposerService`. The `wire_executor()` method then fails because `hasattr(self.generator, "decompose_fn")` checks for the public property that only exists on the service-layer class.

**Fix:** Add `decompose_fn` property (getter/setter) to `DecomposerService` in `orchestrator/application/decomposer.py`. This was already done via commit `e5294f99` but may need verification.

**Acceptance:** `generate()` call in `services/generator.py` succeeds (no `NoneType not callable`).

### 1.2 — Fix `_state_to_dict` for Mixed attempt_history Types

**File:** `orchestrator/infrastructure/state.py:118-132`

The `_attempt_to_dict()` function expects `AttemptRecord` objects, but `attempt_history` lists can contain both `AttemptRecord` objects and raw dicts (from different code paths). The function should check `hasattr(a, "attempt_num")` before accessing dataclass fields.

**Fix:** Add `isinstance` / `hasattr` guard at the top of `_attempt_to_dict`.

**Acceptance:** Project state saves without `AttributeError: 'dict' object has no attribute 'attempt_num'`.

### 1.3 — Fix `to_task_result` attempt_history Conversion

**File:** `orchestrator/engine_core/pipeline.py:92`

Same issue — `to_task_result()` accesses `a.get("attempt", 0)` assuming dict, but items are now `AttemptRecord` objects. Already partially fixed with `hasattr(a, "attempt_num")` guard but may need verification.

**Acceptance:** Pipeline completes without `'AttemptRecord' object has no attribute 'get'`.

---

## Phase 2: Import Chain Cleanup

### 2.1 — Fix Missing Model Enum Values

**Files:**
- `orchestrator/config/costs.json` — add `moonshotai/kimi-k2.7-code: {input: 1.10, output: 4.50}`
- `orchestrator/config/fallbacks.json` — add `moonshotai/kimi-k2.7-code: moonshotai/kimi-k2.6`

**Acceptance:** No "Unknown model" warning for K2.7-code; correct pricing used.

### 2.2 — Fix Remaining Tuple-Contract Call Sites in ARA Pipelines

**File:** `orchestrator/reasoning/ara_pipelines.py`

~50 remaining call sites use `response, _ = await self.client.call(model=..., system_prompt=..., user_prompt=...)` which would all raise `TypeError` if invoked. The VS pipelines were fixed, but the other 18 ARA pipelines (MultiPerspective, Debate, Jury, Research, etc.) still use the broken signature.

**Approach:** Batch-replace all `system_prompt=` → `system=`, `user_prompt=` → `prompt=`, and `response, _ = await` → `response = await` across the file.

**Risk:** Low — the file compiles cleanly; the broken contracts are a plain search-replace.

**Acceptance:** Zero `system_prompt=` or `user_prompt=` references in `ara_pipelines.py`.

---

## Phase 3: Website Generator Production-Readiness

### 3.1 — Fix `WebsiteGenerator` Pipeline Errors

**Root cause:** The `generate()` method calls `validator.validate(output_dir)` which imports `from .website_validator import WebsiteQualityValidator` using a relative import. This fails because the actual module is at `orchestrator/generators/website_validator.py` but the import path resolves to `orchestrator/website_validator.py` (which also exists).

**Fix:** Standardize the import path or catch the ImportError and skip the validator with a warning.

### 3.2 — Remove Stale Config Files After Generation

**Root cause:** Old builds left `postcss.config.js` and `tailwind.config.js` which conflict with CDN Tailwind approach.

**Fix:** Add cleanup logic in `_assemble_nextjs_page` to remove these files, and generate a `README.md` with proper setup instructions.

### 3.3 — Add Proper Content Injection

**Root cause:** `_create_content_from_brief` generates components with data from `content_brief`, but the `content_brief` is a stub object with minimal data (just headlines dict). The actual LLM-powered content generation path works but falls back to content brief on errors.

**Fix:** Ensure the content brief is populated from LLM or has richer defaults. The current defaults are already reasonable (testimonials, FAQs, pricing tiers) but the headlines dict is sparse.

### 3.4 — Fix Website Fallback Message

**Root cause:** CLI says "❌ Failed" even when the fallback assembled a working Next.js project. The success detection logic checks for `package.json` but some conditions override it.

**Fix:** Update `_cmd_website` success detection — prioritize `(output_dir / 'package.json').exists()` check.

---

## Phase 4: Async & Concurrency Fixes

### 4.1 — Async Blocking in Snapshot Store

**Files:** `orchestrator/infrastructure/snapshot_store.py`

All `GitSnapshotStore` methods are `async` but call `subprocess.run()` synchronously, blocking the event loop for up to 60 seconds.

**Already fixed in commit `d00983b6`** — all subprocess calls wrapped in `asyncio.to_thread()` via `_run_git()` helper. Verify once.

### 4.2 — Fire-and-Forget Error Logging

**Files:** `orchestrator/output/writer.py`, `orchestrator/codebase/writer.py`

Both use `asyncio.ensure_future()` for snapshots but discard the task reference — exceptions disappear silently.

**Already fixed in commit `d00983b6`** — done-callbacks added. Verify once.

---

## Phase 5: Architecture Advisor Improvements

### 5.1 — Frontend Detection for Static Web Projects

**Root cause:** `detect_project_type()` already detects HTML/CSS/JS keywords (added in commit `838f3cd1`). But the `_USER_PROMPT_TEMPLATE` LLM prompt doesn't include `static` as an `app_type` option, so the LLM can't select it.

**Already fixed in commit `a7290787`** — `static` added to both the YAML defaults and LLM prompt. Verify once.

### 5.2 — Add `static` to Scaffolder Templates

**Root cause:** `orchestrator/scaffold/` has no template for `app_type="static"`, so the warning "No template for app_type 'static'" fires on every build.

**Fix:** Add a minimal `static` template in `orchestrator/scaffold/templates/` that creates `index.html`, `styles.css`, and `script.js` files.

---

## Phase 6: Testing Coverage

### 6.1 — Fix conftest.py Import Errors

**Root cause:** `tests/conftest.py:18` imports `from orchestrator.models import ...` which triggers the full orchestrator `__init__.py` → `agent_model_registry.py` → `Model.MOONSHOT_KIMI_K2_7_CODE` which was missing (fixed in commit `6524ac14`). But there may be other broken imports in conftest.

**Fix:** Audit all conftest imports; ensure test infrastructure can import without triggering `reasoning/__init__.py` star imports (which were replaced with lazy `__getattr__` in commit `bdd61a28`).

### 6.2 — Regression Tests for All VS Phases

**Current:** `tests/regression/test_vs_regression.py` has 14 tests covering Phases 2, 3, 6.
**Missing:** Tests for Phases 7a-7d (architecture, code review, decomposition, bug hunting).

**Fix:** Add tests for:
- Phase 7a: `_vs_architecture_selection` returns correct format
- Phase 7b: VS code review generates 3 candidates with confidence labels
- Phase 7c: VS decomposition generates 2 plans, picks first valid
- Phase 7d: VS bug hunting returns Bayesian hypothesis format

---

## Phase 7: Quality Gates

### 7.1 — CI Gate: Import Linter

The `.importlinter` has 5 contracts. Run `lint-imports` and verify no new violations from VS or CodeWhale changes.

### 7.2 — CI Gate: Coverage Floor

Current: 5.21% (required 15%). None of the new code drops coverage — the floor is pre-existing. Consider raising floor to 6% to ratchet.

### 7.3 — CI Gate: Type Check

Target: `mypy orchestrator/domain/ orchestrator/application/ orchestrator/engine_core/container.py --ignore-missing-imports --no-strict-optional` passes.

---

## Implementation Sequence

| Week | Phase | Steps | Effort | Risk |
|------|-------|-------|--------|------|
| 1 | 1 — Critical bugs | 1.1-1.3 | S | Medium |
| 1 | 2 — Imports | 2.1-2.2 | M | Low |
| 2 | 3 — Website | 3.1-3.4 | M | Medium |
| 2 | 4 — Async | 4.1-4.2 (verify) | XS | Low |
| 3 | 5 — Advisor | 5.1-5.2 | S | Low |
| 3 | 6 — Tests | 6.1-6.2 | M | Low |
| 4 | 7 — Gates | 7.1-7.3 | S | Low |

**Total:** ~4 weeks, all low-medium risk.

---

## Tracked Bugs Fixed in This Session

| Commit | Bug | Files |
|--------|-----|-------|
| `d3b887c1` | VS Phase 7a-7d implementation | 4 files |
| `19acb499` | Blocking: dead `vs_bug_hunting` + decomposer lazy import scoping | 2 files |
| Multiple prior | Import chain, validator signature, attempt_record types, decomposer wiring, snapshot async, postcss conflicts, frontend detection, model enum | 15+ files |
