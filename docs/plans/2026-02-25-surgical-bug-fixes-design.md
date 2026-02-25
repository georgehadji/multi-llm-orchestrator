# Surgical Bug Fixes — Design

**Date:** 2026-02-25
**Branch:** claude/strange-edison
**Scope:** Fix 31 failing tests with minimal code changes (4 files, ≤6 lines)

---

## Problem

31 tests fail across 5 root causes introduced in recent feature work
(auto-resume detection, project enhancer, architecture advisor).

## Root Causes & Fixes

### Fix 1 — `set` not JSON-serializable
- **File:** `orchestrator/state.py:403`
- **Cause:** `_extract_keywords()` returns `set`; `json.dumps()` cannot serialize sets
- **Fix:** `json.dumps(list(keywords))`
- **Tests fixed:** 12 (`test_engine_e2e` × 9, `test_state_migration` × 2, `test_streaming` × 1)

### Fix 2 — `_recency_factor` receives `datetime`, expects `float`
- **File:** `orchestrator/cli.py:494`
- **Cause:** `now = datetime.utcnow()` and `updated_dt = datetime.utcfromtimestamp(...)` are
  `datetime` objects; `_recency_factor` signature is `(float, float)`. Subtracting two
  datetimes yields `timedelta`, which cannot be compared with `<= 30`.
- **Fix:** `_recency_factor(updated_dt.timestamp(), reference_time=now.timestamp())`
- **Tests fixed:** 3 (`test_cli_resume` × 3)

### Fix 3 — Wrong model name in test
- **File:** `tests/test_policy_dsl.py:175`
- **Cause:** Test uses `"moonshot-v1"` but `Model.KIMI_K2_5.value == "kimi-k2.5"`.
  The DSL silently skips unknown model names → `blocked_models` is `None`.
- **Fix:** Replace `"moonshot-v1"` with `"kimi-k2.5"` in test fixture
- **Tests fixed:** 1

### Fix 4 — Test expectation contradicts intentional design
- **File:** `tests/test_policy_governance.py:146`
- **Cause:** Test asserts MONITOR overrides HARD ("most permissive wins").
  The policy engine deliberately implements "most restrictive wins" for security compliance
  (GDPR rationale documented in `policy_engine.py:154–157`).
- **Fix:** Flip assertion to `result.passed is False`; update test name and docstring
  to reflect actual designed behavior
- **Tests fixed:** 1

## Non-Goals

- No refactoring of `_extract_keywords` return type
- No consolidation of duplicate implementations
- No performance changes
- No new features

## Verification

Run `python -m pytest tests/ --tb=no -q` → expect 0 failures (608+ passing).
