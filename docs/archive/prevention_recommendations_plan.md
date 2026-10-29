# Prevention Recommendation Implementation Plan

**Date:** 2026-07-22  
**Source:** Precision Defect Auditor V3 — Master Report  
**Scope:** 3 prevention recommendations + 1 outstanding Finding from Implementation Audit Report  
**Branch:** `fix/prevention-recommendations`

---

## 1. Summary

The V3 static audit and the sibling V7 defect hunt identified one real defect (D1 — subprocess leak on timeout, already fixed in `10403400`) and no surviving static defects after innocence filtering. The master report recommended three prevention measures and one outstanding audit finding remains partially addressed. This plan implements all four items.

---

## 2. Work Items

### 2.1 REC-1: Add targeted bandit scan to CI

**Current state:**
- `bandit[toml]>=1.7.0` is already a dev dependency (`pyproject.toml:58`)
- Bandit config exists at `pyproject.toml:370-372`:
  ```toml
  [tool.bandit]
  exclude_dirs = ["tests", "docs"]
  skips = ["B101"]  # Skip assert_used warnings in non-test code
  ```
- The config excludes `tests/` which is correct, but there's no targeted scan for the verification module
- The current CI at `.github/workflows/ci.yml` already runs bandit across the repo

**Gap:** The verification module uses `exec()` (`B102` — exec_used) and `subprocess` (`B602` — subprocess_popen_with_shell_equals_true, `B603` — subprocess_without_shell_equals_true). These are legitimate uses but should be explicitly reviewed and suppressed inline rather than silently passing.

**Implementation tasks:**

1. Add `# nosec` comments with justification on the security-relevant lines in `verification_checks.py`:
   - `_make_build_check._check` line ~135: `exec(code, ns)` → `exec(code, ns)  # nosec B102 — isolated namespace, no caller access`
   - `_run_command` line ~42: `asyncio.create_subprocess_exec(...)` → already uses list args (no shell injection), add explanatory comment

2. Create a CI script `scripts/bandit_verification.sh` (or `.ps1` for Windows) that runs:
   ```
   bandit -r orchestrator/infrastructure/verification_checks.py orchestrator/application/verification_gate.py --severity-level medium
   ```

3. Integrate into `.github/workflows/ci.yml` as a separate step after ruff.

4. Document the `exec()` usage in the module docstring as an intentional design choice (isolated namespace, no caller scope access, only on LLM-generated artifacts already validated by syntax check).

**Files affected:**
- `orchestrator/infrastructure/verification_checks.py` — add `# nosec` annotations
- `scripts/bandit_verification.sh` (new) — targeted scan script
- `.github/workflows/ci.yml` — add bandit step

**Test:** Run bandit before and after — before should flag exec/subprocess; after should show clean with justified suppressions.

---

### 2.2 REC-2: Enforce `proc.kill()` pattern in subprocess code

**Current state:**
- The D1 defect was a missing `proc.kill()` in the `asyncio.TimeoutError` handler
- No existing ruff rule or flake8 plugin catches this pattern
- The codebase has no custom lint rules
- `ruff` already has `B` (flake8-bugbear) rules in the select set

**Approach:** The simplest effective approach is a **CI guard script** — not a custom lint rule. A Python script that scans for `asyncio.create_subprocess_exec` calls and verifies the associated `except asyncio.TimeoutError` block contains `proc.kill()` (or `proc.terminate()`).

Rationale: Writing a custom ruff rule requires Rust + ruff plugin infrastructure. A grep-level check is 80% effective at 2% of the cost. Since the codebase has only ONE module with subprocess calls (`verification_checks.py`), a targeted check is sufficient.

**Implementation tasks:**

1. Create `scripts/check_subprocess_cleanup.py`:
   - Parse `orchestrator/infrastructure/verification_checks.py` AST
   - Find every `asyncio.create_subprocess_exec` assignment (e.g., `proc = await asyncio.create_subprocess_exec(...)`)
   - For each, verify that in the same function, every `except asyncio.TimeoutError` block either:
     - Contains `proc.kill()` or `proc.terminate()`, OR
     - Is followed by `await proc.wait()` in a finally block
   - Report any violations as errors

2. Add to `.github/workflows/ci.yml` under lint step.

3. Ensure the script is also runnable locally: `python scripts/check_subprocess_cleanup.py`.

**Files affected:**
- `scripts/check_subprocess_cleanup.py` (new)
- `.github/workflows/ci.yml` — add step

**Test:** The check should pass on current code (after D1 fix). Temporarily remove the `proc.kill()` lines and verify the check fails.

---

### 2.3 REC-3: Pin `VerificationPolicy.checks` type as `Mapping`

**Current state** (`orchestrator/domain/verification.py:88`):
```python
checks: dict[str, CheckOutcome] = field(default_factory=dict)
```
The docstring at line 80 already says "Mapping of check name → requirement level." The `dict` annotation doesn't signal immutability intent even though the class is `@dataclass(frozen=True)`.

**Fix:** Change the type annotation to `Mapping[str, CheckOutcome]`:
```python
from collections.abc import Mapping

checks: Mapping[str, CheckOutcome] = field(default_factory=dict)
```

The runtime behavior is identical (it's still a `dict` at runtime). The `Mapping` type is a read-only view that signals to type checkers and readers that mutation is not intended.

**Implementation tasks:**

1. Add `from collections.abc import Mapping` to imports.
2. Change `checks: dict[str, CheckOutcome]` → `checks: Mapping[str, CheckOutcome]`.
3. Verify all existing code passes mypy (the `Mapping` type is covariant and `dict` satisfies it at assignment).
4. Verify `required_checks` and `mandatory_checks` properties (which iterate `.items()`) still work (they do — `Mapping` supports `.items()`).
5. Update `test_immutable` test to explicitly verify that type checker treats `checks` as read-only (add a `# type: ignore` comment explaining the runtime-vs-static distinction).

**Files affected:**
- `orchestrator/domain/verification.py` — 2-line change
- `tests/unit/test_domain_verification.py` — update immutability test

**Risk:** LOW. `Mapping` is a supertype of `dict` for read operations. No callers iterate or mutate `checks` directly (they use `get_requirement()`, `required_checks`, etc.).

---

### 2.4 FINDING-2 (Audit Report): VerificationPolicy unused at runtime

**Current state:** The audit report (commit `b9c10ca9`, Finding #2) noted that `verification_policy` is created in `container.py:634` and stored on `ServiceContainer`, but was not passed to `VerificationGate(policy=...)`.

**Status:** This was addressed in commit `352f5c1a` (line `verification_policy` is now created BEFORE the gate and passed as `VerificationGate(checks=gate_checks, policy=verification_policy)`).

**Verification:** Re-read the container code to confirm.

**Recommendation:** Add a unit test that verifies `VerificationGate.run()` produces NOT_RUN receipts when policy requires checks with no registered adapters. This test was already written in `tests/regression/test_wbs1_verification_regression.py::TestPolicyBehavior::test_policy_creates_not_run_receipts`.

**Status of Finding #2: CLOSED — the fix was applied, the test exists, and the test passes (verified in 91/91 test suite).**

---

## 3. Implementation Order

| Order | Item | Effort | Risk | Files |
|---|---|---|---|---|
| 1 | REC-3: Pin `Mapping` type | 5 min | LOW | 2 files |
| 2 | REC-1: Add bandit annotations + CI | 15 min | LOW | 3 files |
| 3 | REC-2: Subprocess cleanup guard script | 30 min | LOW | 2 files |
| 4 | FINDING-2: Verify closed + test | 5 min | NONE | 0 files (already done) |

---

## 4. Architecture Compliance

| Concern | Assessment |
|---|---|
| New domain imports | `collections.abc.Mapping` — stdlib, no new dependency. Domain-pure. |
| New infrastructure scripts | `scripts/` directory — separate from `orchestrator/`, no import boundary impact. |
| CI changes | Additive only — new steps, no modification of existing gates. |
| No new root-level modules | Scripts go in `scripts/`, not `orchestrator/`. Compliant. |
| Import contracts | Unchanged — all 5 contracts remain satisfied. |

---

## 5. CI Impact

After implementation, the CI pipeline order becomes:

```
1. black --check
2. ruff check
3. lint-imports
4. python scripts/check_subprocess_cleanup.py   ← NEW (REC-2)
5. bandit -r orchestrator/infrastructure/verification_checks.py ...  ← ENHANCED (REC-1)
6. mypy (domain + application)
7. pytest (not slow, not requires_api)
8. contract tests
```

---

## 6. Acceptance Criteria

- [ ] `REC-1`: Bandit scan of verification module passes with 0 medium+ severity findings. All necessary `# nosec` annotations are justified in comments.
- [ ] `REC-2`: `check_subprocess_cleanup.py` passes on current code. Removing `proc.kill()` causes it to fail. Script is documented and runnable locally.
- [ ] `REC-3`: `VerificationPolicy.checks` is typed as `Mapping`. All 91 existing tests pass. mypy on `domain/verification.py` passes.
- [ ] `FINDING-2`: Confirmed closed — container wires policy to gate, test verifies NOT_RUN behavior.
- [ ] All existing CI gates unchanged (black, ruff, lint-imports, pytest all pass).
