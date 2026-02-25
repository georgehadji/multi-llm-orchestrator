# Surgical Bug Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix 31 failing tests with 4 minimal code changes across 4 files.

**Architecture:** Each fix targets the exact root cause identified in the design doc. No refactoring, no new abstractions. Tests are verified before and after each change.

**Tech Stack:** Python 3.12, pytest, asyncio

---

### Task 1: Fix `set` not JSON-serializable in `state.py`

**Files:**
- Modify: `orchestrator/state.py:403`

**Step 1: Verify the test currently fails**

Run:
```bash
python -m pytest tests/test_state_migration.py::TestExtractAndStoreKeywords::test_extract_and_store_keywords_with_valid_description -v
```
Expected: FAIL with `TypeError: Object of type set is not JSON serializable`

**Step 2: Apply the fix**

In `orchestrator/state.py`, line 403, change:
```python
    return json.dumps(keywords)
```
to:
```python
    return json.dumps(list(keywords))
```

**Step 3: Run the targeted tests**

Run:
```bash
python -m pytest tests/test_state_migration.py tests/test_engine_e2e.py tests/test_streaming.py -v --tb=short
```
Expected: All previously failing tests in these files now PASS.

**Step 4: Commit**

```bash
git add orchestrator/state.py
git commit -m "fix: convert keywords set to list before json.dumps in state.py"
```

---

### Task 2: Fix `_recency_factor` receiving `datetime` instead of `float` in `cli.py`

**Files:**
- Modify: `orchestrator/cli.py:494`

**Step 1: Verify the test currently fails**

Run:
```bash
python -m pytest tests/test_cli_resume.py::TestCheckResume::test_single_match_user_says_no -v
```
Expected: FAIL with `TypeError: '<=' not supported between instances of 'datetime.timedelta' and 'int'`

**Step 2: Apply the fix**

In `orchestrator/cli.py`, line 494, change:
```python
        recency = _recency_factor(updated_dt, reference_time=now)
```
to:
```python
        recency = _recency_factor(updated_dt.timestamp(), reference_time=now.timestamp())
```

**Step 3: Run the targeted tests**

Run:
```bash
python -m pytest tests/test_cli_resume.py -v --tb=short
```
Expected: All 3 previously failing `TestCheckResume` tests now PASS.

**Step 4: Commit**

```bash
git add orchestrator/cli.py
git commit -m "fix: pass unix timestamps to _recency_factor instead of datetime objects"
```

---

### Task 3: Fix wrong model name in `test_policy_dsl.py`

**Files:**
- Modify: `tests/test_policy_dsl.py:175` (fixture around line 163–175)

**Step 1: Verify the test currently fails**

Run:
```bash
python -m pytest tests/test_policy_dsl.py::test_blocked_models_parsed -v
```
Expected: FAIL with `AssertionError: assert None is not None` and log warning `Unknown model 'moonshot-v1'`

**Step 2: Confirm the correct model value**

Run:
```bash
python -c "from orchestrator.models import Model; print(Model.KIMI_K2_5.value)"
```
Expected output: `kimi-k2.5`

**Step 3: Apply the fix**

In `tests/test_policy_dsl.py`, in the `test_blocked_models_parsed` function, change:
```python
            "blocked_models": ["moonshot-v1"]  # Model.KIMI_K2_5.value
```
to:
```python
            "blocked_models": ["kimi-k2.5"]  # Model.KIMI_K2_5.value
```

**Step 4: Run the targeted test**

Run:
```bash
python -m pytest tests/test_policy_dsl.py::test_blocked_models_parsed -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_policy_dsl.py
git commit -m "fix: use correct kimi-k2.5 model value in test_blocked_models_parsed"
```

---

### Task 4: Fix test assertion that contradicts intentional policy engine design

**Files:**
- Modify: `tests/test_policy_governance.py` (around line 130–150)

**Step 1: Verify the test currently fails**

Run:
```bash
python -m pytest tests/test_policy_governance.py::TestEnforcementMode::test_most_permissive_mode_wins_across_policies -v
```
Expected: FAIL with `assert False is True`

**Step 2: Understand the intentional design**

Read `orchestrator/policy_engine.py` lines 153–157. The comment explicitly states:
> "HARD (0) > SOFT (1) > MONITOR (2) — stricter policies always win. Rationale: if ANY policy is HARD, all violations must block the model. A permissive MONITOR policy from one rule must never override a HARD compliance rule from another (e.g. GDPR region constraint)."

The test was written with the opposite assumption. The implementation is correct; the test is wrong.

**Step 3: Apply the fix**

In `tests/test_policy_governance.py`, update the test to reflect the actual designed behavior:

Change the method name from:
```python
    def test_most_permissive_mode_wins_across_policies(self):
```
to:
```python
    def test_most_restrictive_mode_wins_across_policies(self):
```

Change the docstring from:
```python
        """When multiple policies have different modes, most permissive wins."""
```
to:
```python
        """When multiple policies have different modes, most restrictive wins (HARD beats MONITOR)."""
```

Change the comment and assertion from:
```python
        # MONITOR is more permissive than HARD → effective mode = MONITOR
        result = engine.check(Model.GPT_4O, profile, [hard_policy, monitor_policy])
        assert result.passed is True   # MONITOR overrides HARD
```
to:
```python
        # HARD is more restrictive than MONITOR → effective mode = HARD
        result = engine.check(Model.GPT_4O, profile, [hard_policy, monitor_policy])
        assert result.passed is False  # HARD overrides MONITOR (most restrictive wins)
```

**Step 4: Run the targeted test**

Run:
```bash
python -m pytest tests/test_policy_governance.py::TestEnforcementMode::test_most_restrictive_mode_wins_across_policies -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_policy_governance.py
git commit -m "fix: correct test to match intentional most-restrictive-wins policy design"
```

---

### Task 5: Full verification

**Step 1: Run the complete test suite**

Run:
```bash
python -m pytest tests/ --tb=no -q
```
Expected: 0 failures (639+ passing, 2 skipped)

**Step 2: If anything still fails**

Run with full tracebacks on the failures only:
```bash
python -m pytest tests/ --tb=short -q 2>&1 | grep -A 10 "FAILED\|ERROR"
```

**Step 3: Final commit (if verification required any small adjustments)**

```bash
git add -A
git commit -m "fix: verify all tests passing after surgical fixes"
```
