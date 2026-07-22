# Architecture Score Improvement Plan — 8/10 → 9/10+

**Target:** WBS-1 Verification Subsystem  
**Current Score:** 8/10 (ARCH-AUDIT-V2, commit `773dafd8`)  
**Target Score:** 9/10+  
**Principle:** Fix the two scoring deductions, resolve all MEDIUM+ risks, and implement one long-term improvement. No erosion of the existing architecture.

---

## 1. Score Gap Analysis

| Factor | Current | Target | Required Action |
|---|---|---|---|
| Layer separation | ✅ Clean | ✅ Maintain | No change needed |
| Pattern consistency | ✅ Consistent | ✅ Maintain | No change needed |
| Observability | ⚠️ Gate disabled silently | ✅ Metric emitted | PR-2: Gate init metric |
| Testability | ✅ 91 tests | ✅ Maintain | No regression |
| Scalability | ⚠️ 30s mypy bottleneck | ✅ Optional/cached | PR-5: Tiered check timing |
| Schema contracts | ❌ Unversioned dict | ✅ Typed/versioned | PR-1: DeterministicRecord |
| Abstraction economy | ❌ Duplicate classes | ✅ Single adapter | PR-6: Consolidate adapters |
| Failure resilience | ⚠️ No kill+wait timeout | ✅ Timeout-guarded | PR-3: Kill+wait timeout |

**Deduction sources:**
- **−1 point:** `CritiqueReport.deterministic` — untyped dict with no schema contract (MEDIUM risk).
- **−1 point:** `VerificationCheck` / `VerificationCheckAdapter` — duplicate abstraction (LOW severity, but architectural noise).

---

## 2. Work Items

### PR-1: Replace `deterministic: Optional[dict]` with a typed `DeterministicResult`

**Severity:** MEDIUM → fixes the −1 point deduction and risk #3.  
**Effort:** 30 minutes  
**Files:** `orchestrator/operations/feedback.py`, `orchestrator/application/evaluator.py`

**Design:**
Replace the untyped `Optional[dict]` on `CritiqueReport.deterministic` with a typed `DeterministicResult` frozen dataclass in `orchestrator/domain/verification.py` (domain-pure). This:

1. Defines the schema contract: `passed: bool`, `checks: dict[str, bool]`, `reasons: dict[str, str]`, `artifact_hash: str | None`, `failure_summary: dict[str, str]`, `status_summary: str`
2. Includes a `version: int = 1` field for forward-compatibility.
3. Provides `to_dict()` / `from_dict()` that include the version key.
4. `from_dict()` validates the version and raises `ValueError` on mismatch — no silent corruption.

**Concrete steps:**

1. Add `DeterministicResult` frozen dataclass to `orchestrator/domain/verification.py`:
   ```python
   @dataclass(frozen=True)
   class DeterministicResult:
       """Serialized gate result carried through CritiqueReport.
       
       Versioned for forward-compatibility across serialization boundaries.
       """
       passed: bool
       checks: dict[str, bool] = field(default_factory=dict)
       reasons: dict[str, str] = field(default_factory=dict)
       artifact_hash: str | None = None
       failure_summary: dict[str, str] = field(default_factory=dict)
       status_summary: str = "(no checks)"
       version: int = 1
       
       def to_dict(self) -> dict:
           return {"version": self.version, "passed": self.passed, ...}
       
       @classmethod
       def from_dict(cls, data: dict) -> "DeterministicResult":
           if data.get("version", 1) != 1:
               raise ValueError(f"Unsupported version: {data.get('version')}")
           return cls(**{k: data.get(k) for k in cls.__dataclass_fields__ if k != "version"})
   ```

2. Change `CritiqueReport.deterministic` from `Optional[dict]` to `DeterministicResult | None`.

3. Update `EvaluatorService._evaluate_inner()` to construct `DeterministicResult` from `GateResult` instead of building a raw dict.

4. Update `test_deterministic_field_roundtrip` and related tests in `tests/regression/test_wbs1_verification_regression.py`.

5. Verify all 91 tests pass, black + ruff clean.

**Acceptance criteria:**
- `DeterministicResult` is a frozen dataclass with `version: int = 1`.
- `from_dict()` rejects unknown versions.
- Roundtrip `to_dict()` → `from_dict()` is identity.
- All 91 tests pass without modification to test assertions beyond the type change.
- No `Optional[dict]` remains on the `deterministic` field.

---

### PR-2: Emit telemetry metric on gate initialization

**Severity:** MEDIUM → fixes risk #2.  
**Effort:** 15 minutes  
**Files:** `orchestrator/engine_core/container.py`

**Design:** Use the existing `TelemetryCollector` (already available as `self.telemetry` in the container) to emit a counter metric when the gate is wired or fails.

**Concrete steps:**

1. In `container.py`, after successful gate wiring (line 647), add:
   ```python
   if hasattr(container, 'telemetry') and container.telemetry is not None:
       container.telemetry.record_call(
           model=Model("verification_gate"),  # type: ignore
           latency_ms=0,
           cost_usd=0,
           success=True,
           quality_score=float(len(gate_checks)),
       )
   ```

2. In the `except ImportError` block (line 658), add:
   ```python
   logger.warning("WBS-1: VerificationGate FAILED to initialize — all verification disabled")
   if hasattr(container, 'telemetry') and container.telemetry is not None:
       container.telemetry.record_call(
           model=Model("verification_gate"),  # type: ignore
           latency_ms=0,
           cost_usd=0,
           success=False,
       )
   ```

3. Alternatively, add a dedicated `health: dict` field to `ServiceContainer` that records gate status:
   ```python
   container.health["verification_gate"] = {"active": True, "checks": len(gate_checks)}
   ```
   This is simpler and avoids coupling to TelemetryCollector internals.

**Acceptance criteria:**
- Gate wired → health metric shows `active=True, checks=N`.
- Import fails → health metric shows `active=False`.
- Metric is inspectable from any code holding a `ServiceContainer` reference.
- No new imports in `container.py` beyond the already-available `logger`.

---

### PR-3: Add timeout to kill+wait cleanup in `_run_command`

**Severity:** MEDIUM → fixes risk #1.  
**Effort:** 10 minutes  
**Files:** `orchestrator/infrastructure/verification_checks.py`

**Design:** The current code at verification_checks.py:58-62 does `proc.kill()` (bare, no await) then `await proc.wait()`. If the process ignores SIGKILL (Windows edge case), `wait()` blocks indefinitely. Wrap both in a 5-second timeout.

**Concrete steps:**

1. Refactor lines 56-62 to:
   ```python
   except asyncio.TimeoutError:
       logger.warning("Command timed out after %ss: %s", timeout, " ".join(command))
       try:
           proc.kill()
       except Exception:
           pass
       try:
           await asyncio.wait_for(proc.wait(), timeout=5.0)
       except asyncio.TimeoutError:
           logger.error("Subprocess %d refused to die after kill", proc.pid)
       return -1, "", f"TIMEOUT after {timeout}s"
   ```

2. Update `scripts/check_subprocess_cleanup.py` to verify the new pattern (kill + timed wait). Currently the guard checks for `kill()` and `wait()` independently; it must now check that `wait()` is wrapped in `asyncio.wait_for`.

**Acceptance criteria:**
- Kill+wait cleanup has a hard 5-second ceiling.
- If process survives kill, a second `TimeoutError` is logged and the function returns -1 anyway.
- `scripts/check_subprocess_cleanup.py` detects the new pattern (passes on correct code, fails if wait_for removed).
- All 91 tests pass.

---

### PR-4: Add `gate_active` property to `ServiceContainer`

**Severity:** LOW → observability gap.  
**Effort:** 5 minutes  
**Files:** `orchestrator/engine_core/container.py`

**Design:** A simple read-only property on `ServiceContainer` that reports whether the verification gate is wired:

```python
@property
def gate_active(self) -> bool:
    return self.verification_gate is not None
```

**Concrete steps:**

1. Add the property to `ServiceContainer`.
2. No test changes needed — this is a convenience accessor.

**Acceptance criteria:**
- `container.gate_active` returns `True` when gate is wired, `False` otherwise.
- Used in logging/telemetry in PR-2.

---

### PR-5: Make mypy check timeout configurable and independent of the gate loop

**Severity:** LOW → fixes risk #5.  
**Effort:** 20 minutes  
**Files:** `orchestrator/infrastructure/verification_checks.py`, `orchestrator/engine_core/container.py`

**Design:** The 30-second default for `_make_type_check` is hardcoded. Make it configurable via environment variable `ORCH_MYPY_TIMEOUT` (default 30). This allows production tuning without code changes.

**Concrete steps:**

1. In `_make_type_check`, read the timeout from env:
   ```python
   import os
   mypy_timeout = float(os.getenv("ORCH_MYPY_TIMEOUT", "30.0"))
   ```

2. Validate the env value (must be > 0 and ≤ 120).

3. Document in the module docstring and in `.env.example`.

**Acceptance criteria:**
- `ORCH_MYPY_TIMEOUT=10` cuts mypy check to 10 seconds.
- Invalid values are logged and default to 30.
- All 91 tests pass (tests don't depend on the env var).

---

### PR-6: Consolidate `VerificationCheck` and `VerificationCheckAdapter`

**Severity:** LOW → fixes the −1 point deduction.  
**Effort:** 45 minutes  
**Files:** `orchestrator/application/verification_gate.py`, `orchestrator/infrastructure/verification_checks.py`, `orchestrator/engine_core/container.py`

**Design:** The two classes (`VerificationCheck` at verification_gate.py:35-42 and `VerificationCheckAdapter` at verification_checks.py:189-203) differ only by layer: one is an application dataclass, the other is an infrastructure class. Both hold `name`, a callable `run`/`_run`, and optional `command`. The infrastructure class is the concrete implementation; the application dataclass is the protocol-level wrapper.

Approach: Keep `VerificationCheck` as the canonical application-layer type. Make `VerificationCheckAdapter` a factory that produces `VerificationCheck` instances directly:

```python
# verification_checks.py
class VerificationCheckAdapter:
    @staticmethod
    def make(name: str, run: CheckFn, command: str | None = None) -> VerificationCheck:
        """Create an application-layer VerificationCheck from infrastructure code."""
        from orchestrator.application.verification_gate import VerificationCheck
        return VerificationCheck(name=name, run=run, command=command)
```

This eliminates the adapter-as-separate-class and makes it a thin factory, reducing cognitive overhead. The `container.py` wiring already wraps adapters into `VerificationCheck` objects — with this change, the wrapping is consolidated into the factory method.

**Alternatively (lighter touch):** Keep both classes but make `VerificationCheckAdapter` inherit from or delegate to `VerificationCheck`. This avoids any behavioral change while reducing conceptual duplication.

**Recommendation:** Use the factory approach — simplest, fewest changes.

**Concrete steps:**

1. Replace `VerificationCheckAdapter.__init__` / `__call__` with a `make()` static method.
2. Update all 5 factory functions in verification_checks.py to call `VerificationCheckAdapter.make(...)`.
3. Update `default_checks()` to return `list[VerificationCheck]` instead of `list[VerificationCheckAdapter]`.
4. Simplify `container.py` gate wiring — `default_checks()` already returns `VerificationCheck` objects, eliminating the wrapping list comprehension.
5. Update `tests/unit/test_verification_checks.py` to import `VerificationCheck` instead of `VerificationCheckAdapter`.
6. Remove the `VerificationCheckAdapter` class → just keep the static factory function.

**Acceptance criteria:**
- Only one class represents "a named check" in the codebase (VerificationCheck).
- All 91 tests pass with updated imports.
- container.py gate wiring is simplified (no per-item wrapping needed).
- No regression in the subprocess cleanup guard script.

---

## 3. Implementation Order

| Order | Item | Fixes Deduction | Fixes Risk | Effort | Depends On |
|---|---|---|---|---|---|
| 1 | PR-3: Kill+wait timeout | — | #1 (MEDIUM) | 10 min | — |
| 2 | PR-1: DeterministicResult type | −1 point | #3 (MEDIUM) | 30 min | — |
| 3 | PR-2: Gate init metric | — | #2 (MEDIUM) | 15 min | PR-4 (gate_active property) |
| 4 | PR-4: gate_active property | — | #4 (LOW) | 5 min | — |
| 5 | PR-6: Consolidate adapters | −1 point | — | 45 min | PR-1 (imports DeterministicResult) |
| 6 | PR-5: Configurable mypy timeout | — | #5 (LOW) | 20 min | — |

Total effort: ~2 hours.

---

## 4. Score Projection

| Factor | Before | After |
|---|---|---|
| Layer separation | 9/10 | 10/10 — deterministic schema contract bridges the gap |
| Pattern consistency | 8/10 | 10/10 — single canonical check abstraction |
| Observability | 6/10 | 8/10 — gate init metric + gate_active property |
| Testability | 9/10 | 9/10 — maintained |
| Scalability | 7/10 | 8/10 — mypy timeout configurable, tiered timeout |
| **Projected Score** | **8/10** | **9/10** |

Conservative estimate: 9/10 after PR-1 + PR-3 + PR-6. Could reach 9.5/10 with PR-2 + PR-4 + PR-5.

The ceiling at 9/10 is because the subsystem is fundamentally a synchronous-async gate that runs checks sequentially — true 10/10 would require parallelization, caching, and distributed execution, which are not justified for the current scale (5 checks, single-task evaluation).

---

## 5. Architecture Compliance

| Concern | Assessment |
|---|---|
| Domain purity | PR-1 adds DeterministicResult to domain layer — pure frozen dataclass, stdlib imports only. No regression. |
| Application no-infra | PR-1, PR-3, PR-5 touch infrastructure files only. PR-6 consolidates but the factory approach keeps infra imports in infra. |
| Import contracts | All 5 lint-imports contracts remain satisfied. No new boundary crossings. |
| No engine.py business logic | All changes are in domain, application existing files, infrastructure existing files, and container composition root. |
| Backward compat | `DeterministicResult.from_dict()` accepts old unversioned dicts (version default = 1). `to_dict()` includes version. Old consumers unaware of version are unaffected. |

---

## 6. Acceptance Criteria (Summary)

- [ ] `DeterministicResult` is a typed, versioned frozen dataclass (PR-1).
- [ ] `proc.wait()` has a 5-second ceiling, `check_subprocess_cleanup.py` verifies it (PR-3).
- [ ] Gate init emits an observable telemetry metric (PR-2).
- [ ] `container.gate_active` property exposes gate state (PR-4).
- [ ] Only `VerificationCheck` class exists in the codebase (PR-6).
- [ ] `ORCH_MYPY_TIMEOUT` env var controls type check duration (PR-5).
- [ ] All 91 tests pass. black + ruff + lint-imports + guard script pass.
- [ ] No new import boundary violations.
- [ ] Re-run ARCH-AUDIT-V2: projected score ≥ 9/10.

---

## 7. Rollback Plan

Each PR is independently revertible:
- **PR-1:** Revert `DeterministicResult` class, restore `Optional[dict]`. Old data with `version` key is forward-compatible (consumers ignore unknown keys).
- **PR-2:** Revert telemetry call. No data impact.
- **PR-3:** Revert to bare `await proc.wait()`. Risk: re-introduces unbounded wait.
- **PR-4:** Delete property. No impact.
- **PR-5:** Remove env var read, hardcode 30. No impact.
- **PR-6:** Restore `VerificationCheckAdapter` class. Container wiring is simplified — must restore wrapping list comprehension.
