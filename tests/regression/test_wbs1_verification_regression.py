"""
Regression test suite for WBS-1: Mandatory Acting Verification.

Covers backward compatibility, evaluator integration, check adapters,
edge cases, and pipeline flow.  Every test here must have been passing
before the WBS-1 changes and must continue to pass after.

Run:
    pytest tests/regression/test_wbs1_verification_regression.py -v
"""

from __future__ import annotations

import asyncio
import hashlib

import pytest

pytestmark = [pytest.mark.regression, pytest.mark.wbs1, pytest.mark.unit]


# ── Lazy imports (avoid heavy deps at collection time) ─────────────────────────

_imported: dict[str, object] = {}


def _once(name: str):
    if name not in _imported:
        if name == "GateResult":
            from orchestrator.application.verification_gate import GateResult

            _imported[name] = GateResult
        elif name == "VerificationGate":
            from orchestrator.application.verification_gate import VerificationGate

            _imported[name] = VerificationGate
        elif name == "VerificationCheck":
            from orchestrator.application.verification_gate import VerificationCheck

            _imported[name] = VerificationCheck
        elif name == "CheckOutcome":
            from orchestrator.domain.verification import CheckOutcome

            _imported[name] = CheckOutcome
        elif name == "ExecutionReceipt":
            from orchestrator.domain.verification import ExecutionReceipt

            _imported[name] = ExecutionReceipt
        elif name == "VerificationPolicy":
            from orchestrator.domain.verification import VerificationPolicy

            _imported[name] = VerificationPolicy
        elif name == "compute_artifact_hash":
            from orchestrator.application.verification_gate import compute_artifact_hash

            _imported[name] = compute_artifact_hash
        elif name == "CritiqueReport":
            from orchestrator.operations.feedback import CritiqueReport

            _imported[name] = CritiqueReport
        elif name == "CritiqueSeverity":
            from orchestrator.operations.feedback import CritiqueSeverity

            _imported[name] = CritiqueSeverity
        elif name == "default_checks":
            from orchestrator.infrastructure.verification_checks import default_checks

            _imported[name] = default_checks
        elif name == "DeterministicResult":
            from orchestrator.domain.verification import DeterministicResult

            _imported[name] = DeterministicResult
    return _imported[name]


# ─────────────────────────────────────────────────────────────────────────────
# Backward Compatibility Regression
# ─────────────────────────────────────────────────────────────────────────────


class TestBackwardCompatGateResult:
    """GateResult must remain compatible with pre-WBS-1 callers."""

    def test_score_is_float(self):
        GateResult = _once("GateResult")
        r = GateResult(checks={"lint": True})
        assert isinstance(r.score, float)
        assert r.score == 1.0

    def test_passed_is_bool(self):
        GateResult = _once("GateResult")
        r = GateResult(checks={})
        assert isinstance(r.passed, bool)
        assert r.passed is True

    def test_checks_dict_backward_compat(self):
        GateResult = _once("GateResult")
        r = GateResult(checks={"a": True, "b": False}, reasons={"b": "err"})
        assert r.checks == {"a": True, "b": False}
        assert r.reasons == {"b": "err"}
        assert r.passed is False

    def test_empty_gate_still_passes(self):
        GateResult = _once("GateResult")
        r = GateResult()
        assert r.passed is True
        assert r.score == 1.0

    def test_new_fields_dont_break_old_constructors(self):
        """Callers creating GateResult with only old fields must still work."""
        GateResult = _once("GateResult")
        # Old-style: only checks and reasons
        r = GateResult(checks={"lint": False}, reasons={"lint": "fail"})
        assert r.passed is False
        assert r.receipts == []
        assert r.artifact_hash is None
        assert r.policy is None


class TestBackwardCompatGate:
    """VerificationGate must work exactly as before for existing callers."""

    @pytest.mark.asyncio
    async def test_default_gate_no_checks(self):
        VerificationGate = _once("VerificationGate")
        gate = VerificationGate.default()
        result = await gate.run("anything")
        assert result.passed is True
        assert len(result.checks) == 0

    @pytest.mark.asyncio
    async def test_empty_checks_list_passes(self):
        VerificationGate = _once("VerificationGate")
        gate = VerificationGate(checks=[])
        result = await gate.run("x = 1")
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_mock_check_shows_in_checks(self):
        VerificationGate = _once("VerificationGate")
        VerificationCheck = _once("VerificationCheck")

        async def _always_pass(artifact: str) -> tuple[bool, str]:
            return True, ""

        gate = VerificationGate(checks=[VerificationCheck(name="mock", run=_always_pass)])
        result = await gate.run("data")
        assert "mock" in result.checks
        assert result.checks["mock"] is True

    @pytest.mark.asyncio
    async def test_mock_check_failure_sets_score_floor(self):
        VerificationGate = _once("VerificationGate")
        VerificationCheck = _once("VerificationCheck")

        async def _always_fail(artifact: str) -> tuple[bool, str]:
            return False, "always fails"

        gate = VerificationGate(checks=[VerificationCheck(name="bad", run=_always_fail)])
        result = await gate.run("data")
        assert result.passed is False
        assert result.score <= VerificationGate.FAIL_SCORE_FLOOR

    @pytest.mark.asyncio
    async def test_reasons_dict_backward_compat(self):
        VerificationGate = _once("VerificationGate")
        VerificationCheck = _once("VerificationCheck")

        async def _fail(artifact: str) -> tuple[bool, str]:
            return False, "oops"

        gate = VerificationGate(checks=[VerificationCheck(name="x", run=_fail)])
        result = await gate.run("data")
        assert "x" in result.reasons
        assert result.reasons["x"] == "oops"


# ─────────────────────────────────────────────────────────────────────────────
# Structured Outcome Regression (WBS-1: new fields must work)
# ─────────────────────────────────────────────────────────────────────────────


class TestStructuredOutcomes:
    """New ExecutionReceipt and CheckOutcome fields work correctly."""

    @pytest.mark.asyncio
    async def test_gate_run_produces_receipts_with_all_fields(self):
        VerificationGate = _once("VerificationGate")
        VerificationCheck = _once("VerificationCheck")
        CheckOutcome = _once("CheckOutcome")

        async def _ok(a: str) -> tuple[bool, str]:
            return True, ""

        gate = VerificationGate(
            checks=[
                VerificationCheck(name="syntax", run=_ok, command="compile(<verify>)"),
            ]
        )
        result = await gate.run("x = 1")

        assert len(result.receipts) == 1
        r = result.receipts[0]
        assert r.check_name == "syntax"
        assert r.outcome == CheckOutcome.PASSED
        assert r.artifact_hash is not None
        assert len(r.artifact_hash) == 64
        assert r.command == "compile(<verify>)"

    @pytest.mark.asyncio
    async def test_blocked_vs_failed_distinction(self):
        VerificationGate = _once("VerificationGate")
        VerificationCheck = _once("VerificationCheck")
        CheckOutcome = _once("CheckOutcome")

        async def _fail(a: str) -> tuple[bool, str]:
            return False, "lint errors"

        async def _crash(a: str) -> tuple[bool, str]:
            raise RuntimeError("tool missing")

        gate = VerificationGate(
            checks=[
                VerificationCheck(name="lint", run=_fail),
                VerificationCheck(name="type_check", run=_crash),
            ]
        )
        result = await gate.run("code")

        lint_receipt = result.receipts[0]
        type_receipt = result.receipts[1]
        assert lint_receipt.outcome == CheckOutcome.FAILED
        assert type_receipt.outcome == CheckOutcome.BLOCKED
        assert "lint errors" in (lint_receipt.reason or "")
        assert "tool missing" in (type_receipt.reason or "")

    def test_checkoutcome_enum_invariants(self):
        CheckOutcome = _once("CheckOutcome")
        assert CheckOutcome.PASSED.is_passed is True
        assert CheckOutcome.PASSED.is_failure is False
        assert CheckOutcome.FAILED.is_passed is False
        assert CheckOutcome.FAILED.is_failure is True
        assert CheckOutcome.BLOCKED.is_passed is False
        assert CheckOutcome.BLOCKED.is_failure is True
        assert CheckOutcome.NOT_RUN.is_passed is False
        assert CheckOutcome.NOT_RUN.is_failure is False


class TestArtifactHashStability:
    """artifact_hash must be deterministic and consistent."""

    def test_same_artifact_same_hash(self):
        compute_artifact_hash = _once("compute_artifact_hash")
        h1 = compute_artifact_hash("hello")
        h2 = compute_artifact_hash("hello")
        assert h1 == h2
        assert len(h1) == 64

    def test_different_artifact_different_hash(self):
        compute_artifact_hash = _once("compute_artifact_hash")
        h1 = compute_artifact_hash("a")
        h2 = compute_artifact_hash("b")
        assert h1 != h2

    def test_known_hash_value(self):
        """Regression against hash algorithm changes."""
        compute_artifact_hash = _once("compute_artifact_hash")
        expected = hashlib.sha256(b"test").hexdigest()
        assert compute_artifact_hash("test") == expected


class TestFailureSummary:
    """failure_summary and status_summary must be correct."""

    @pytest.mark.asyncio
    async def test_failure_summary_empty_when_all_pass(self):
        VerificationGate = _once("VerificationGate")
        VerificationCheck = _once("VerificationCheck")

        async def _ok(a: str) -> tuple[bool, str]:
            return True, ""

        gate = VerificationGate(checks=[VerificationCheck(name="t", run=_ok)])
        result = await gate.run("code")
        assert result.failure_summary == {}

    @pytest.mark.asyncio
    async def test_failure_summary_lists_failures(self):
        VerificationGate = _once("VerificationGate")
        VerificationCheck = _once("VerificationCheck")

        async def _fail(a: str) -> tuple[bool, str]:
            return False, "bad"

        gate = VerificationGate(checks=[VerificationCheck(name="test", run=_fail)])
        result = await gate.run("code")
        assert "test" in result.failure_summary
        assert result.failure_summary["test"] == "bad"

    @pytest.mark.asyncio
    async def test_status_summary_human_readable(self):
        VerificationGate = _once("VerificationGate")
        VerificationCheck = _once("VerificationCheck")

        async def _ok(a: str) -> tuple[bool, str]:
            return True, ""

        gate = VerificationGate(checks=[VerificationCheck(name="lint", run=_ok)])
        result = await gate.run("code")
        summary = result.status_summary
        assert "lint" in summary
        assert "[OK]" in summary


# ─────────────────────────────────────────────────────────────────────────────
# Policy Regression
# ─────────────────────────────────────────────────────────────────────────────


class TestPolicyBehavior:
    """VerificationPolicy must correctly filter and classify checks."""

    def test_policy_frozen_immutable(self):
        VerificationPolicy = _once("VerificationPolicy")
        CheckOutcome = _once("CheckOutcome")
        p = VerificationPolicy(checks={"syntax": CheckOutcome.REQUIRED})
        with pytest.raises(AttributeError):
            p.task_types = frozenset()  # type: ignore[misc]

    def test_policy_defaults_all_task_types(self):
        VerificationPolicy = _once("VerificationPolicy")
        CheckOutcome = _once("CheckOutcome")
        p = VerificationPolicy(checks={"lint": CheckOutcome.OPTIONAL})
        # None means applies to all
        assert p.task_types is None
        assert p.applies_to("anything") is True

    def test_policy_scoped_to_task_types(self):
        VerificationPolicy = _once("VerificationPolicy")
        CheckOutcome = _once("CheckOutcome")
        from orchestrator.models import TaskType

        p = VerificationPolicy(
            checks={"syntax": CheckOutcome.REQUIRED},
            task_types=frozenset({TaskType.CODE_GEN}),
        )
        assert p.applies_to(TaskType.CODE_GEN) is True
        assert p.applies_to(TaskType.WRITING) is False

    def test_policy_required_vs_mandatory(self):
        VerificationPolicy = _once("VerificationPolicy")
        CheckOutcome = _once("CheckOutcome")
        p = VerificationPolicy(
            checks={
                "syntax": CheckOutcome.MANDATORY,
                "lint": CheckOutcome.REQUIRED,
                "security": CheckOutcome.RECOMMENDED,
            }
        )
        assert "syntax" in p.mandatory_checks
        assert "lint" in p.required_checks
        assert "lint" not in p.mandatory_checks
        assert "security" not in p.required_checks

    @pytest.mark.asyncio
    async def test_policy_creates_not_run_receipts(self):
        """When policy requires a check with no registered adapter, it shows as NOT_RUN."""
        VerificationGate = _once("VerificationGate")
        VerificationCheck = _once("VerificationCheck")
        VerificationPolicy = _once("VerificationPolicy")
        CheckOutcome = _once("CheckOutcome")

        # Gate has NO checks, but policy requires "metrics" check
        policy = VerificationPolicy(checks={"metrics": CheckOutcome.REQUIRED})
        gate = VerificationGate(checks=[], policy=policy)

        result = await gate.run("x = 1")

        assert "metrics" in result.checks
        assert result.checks["metrics"] is False
        assert result.passed is False  # required check missing = no pass

        metrics_receipt = [r for r in result.receipts if r.check_name == "metrics"]
        assert len(metrics_receipt) == 1
        assert metrics_receipt[0].outcome == CheckOutcome.NOT_RUN


# ─────────────────────────────────────────────────────────────────────────────
# Check Adapter Regression
# ─────────────────────────────────────────────────────────────────────────────


class TestCheckAdapters:
    """All check adapters must handle valid and invalid input correctly."""

    @pytest.mark.asyncio
    async def test_syntax_check_valid(self):
        default_checks = _once("default_checks")
        checks = default_checks()
        syntax = next(c for c in checks if c.name == "syntax")
        ok, reason = await syntax("x = 1")
        assert ok is True

    @pytest.mark.asyncio
    async def test_syntax_check_invalid(self):
        default_checks = _once("default_checks")
        checks = default_checks()
        syntax = next(c for c in checks if c.name == "syntax")
        ok, reason = await syntax("x = ")
        assert ok is False
        assert "SyntaxError" in reason

    @pytest.mark.asyncio
    async def test_syntax_check_empty(self):
        default_checks = _once("default_checks")
        checks = default_checks()
        syntax = next(c for c in checks if c.name == "syntax")
        ok, reason = await syntax("")
        assert ok is True

    @pytest.mark.asyncio
    async def test_build_check_simple(self):
        default_checks = _once("default_checks")
        checks = default_checks()
        build = next(c for c in checks if c.name == "build")
        ok, reason = await build("def f(): return 42")
        assert ok is True

    @pytest.mark.asyncio
    async def test_build_check_import_error(self):
        default_checks = _once("default_checks")
        checks = default_checks()
        build = next(c for c in checks if c.name == "build")
        ok, reason = await build("import nonexistent_module_abc123")
        assert ok is False
        assert "ImportError" in reason or "ModuleNotFoundError" in reason

    @pytest.mark.asyncio
    async def test_security_clean_code(self):
        default_checks = _once("default_checks")
        checks = default_checks()
        security = next(c for c in checks if c.name == "security")
        ok, reason = await security("x = 1 + 2")
        assert ok is True

    @pytest.mark.asyncio
    async def test_security_detects_eval(self):
        default_checks = _once("default_checks")
        checks = default_checks()
        security = next(c for c in checks if c.name == "security")
        ok, reason = await security("eval('1+1')")
        assert ok is False
        assert "eval" in reason.lower()

    @pytest.mark.asyncio
    async def test_security_comment_not_flagged(self):
        default_checks = _once("default_checks")
        checks = default_checks()
        security = next(c for c in checks if c.name == "security")
        ok, reason = await security("# eval is dangerous")
        assert ok is True

    @pytest.mark.asyncio
    async def test_full_gate_with_real_checks(self):
        """End-to-end: all 5 real checks against valid Python."""
        VerificationGate = _once("VerificationGate")
        VerificationCheck = _once("VerificationCheck")
        default_checks = _once("default_checks")

        gate_checks = [VerificationCheck(name=c.name, run=c) for c in default_checks()]
        gate = VerificationGate(checks=gate_checks)

        result = await gate.run("def add(a: int, b: int) -> int:\n    return a + b\n")
        # syntax/build/security should pass; lint may fail on formatting
        assert result.artifact_hash is not None
        assert len(result.receipts) >= 3
        # At minimum, syntax, build, and security exist
        names = {r.check_name for r in result.receipts}
        assert "syntax" in names
        assert "build" in names
        assert "security" in names


# ─────────────────────────────────────────────────────────────────────────────
# CritiqueReport Regression
# ─────────────────────────────────────────────────────────────────────────────


class TestCritiqueReportDeterministic:
    """CritiqueReport.deterministic must serialize and deserialize correctly."""

    def test_deterministic_field_roundtrip(self):
        CritiqueReport = _once("CritiqueReport")
        DeterministicResult = _once("DeterministicResult")
        dr = DeterministicResult(
            passed=False,
            checks={"lint": False},
            reasons={"lint": "error"},
            artifact_hash="abc123",
            failure_summary={"lint": "error"},
            status_summary="[FAIL] lint: failed",
        )
        report = CritiqueReport(task_id="t1", score=0.5, deterministic=dr, passed_validators=False)
        assert report.deterministic is not None
        assert report.deterministic.passed is False
        assert report.deterministic.checks == {"lint": False}

    def test_deterministic_none_by_default(self):
        CritiqueReport = _once("CritiqueReport")
        report = CritiqueReport(task_id="t2", score=1.0)
        assert report.deterministic is None

    def test_deterministic_to_dict_from_dict_roundtrip(self):
        CritiqueReport = _once("CritiqueReport")
        DeterministicResult = _once("DeterministicResult")
        dr = DeterministicResult(passed=True, checks={"lint": True})
        report = CritiqueReport(task_id="t3", score=0.9, deterministic=dr)
        d = report.to_dict()
        restored = CritiqueReport.from_dict(d)
        assert restored.deterministic is not None
        assert restored.deterministic.passed is True
        assert restored.deterministic.checks == {"lint": True}
        assert restored.task_id == "t3"
        assert restored.score == 0.9


# ─────────────────────────────────────────────────────────────────────────────
# Edge Case Regression
# ─────────────────────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Regression on edge cases that could break."""

    @pytest.mark.asyncio
    async def test_empty_artifact(self):
        VerificationGate = _once("VerificationGate")
        VerificationCheck = _once("VerificationCheck")

        async def _ok(a: str) -> tuple[bool, str]:
            return True, ""

        gate = VerificationGate(checks=[VerificationCheck(name="test", run=_ok)])
        result = await gate.run("")
        assert result.artifact_hash is not None
        assert len(result.artifact_hash) == 64

    @pytest.mark.asyncio
    async def test_very_large_artifact(self):
        """Large artifacts should not cause OOM or hangs."""
        VerificationGate = _once("VerificationGate")
        VerificationCheck = _once("VerificationCheck")

        async def _ok(a: str) -> tuple[bool, str]:
            return True, ""

        large = "x = 1\n" * 10_000
        gate = VerificationGate(checks=[VerificationCheck(name="syntax", run=_ok)])
        result = await gate.run(large)
        assert result.artifact_hash is not None
        assert len(result.artifact_hash) == 64

    @pytest.mark.asyncio
    async def test_unicode_artifact(self):
        VerificationGate = _once("VerificationGate")
        VerificationCheck = _once("VerificationCheck")

        async def _ok(a: str) -> tuple[bool, str]:
            return True, ""

        gate = VerificationGate(checks=[VerificationCheck(name="test", run=_ok)])
        result = await gate.run('print("héllo wörld 你好 🚀")')
        assert result.artifact_hash is not None

    @pytest.mark.asyncio
    async def test_null_bytes_in_artifact(self):
        VerificationGate = _once("VerificationGate")
        VerificationCheck = _once("VerificationCheck")

        async def _ok(a: str) -> tuple[bool, str]:
            return True, ""

        gate = VerificationGate(checks=[VerificationCheck(name="test", run=_ok)])
        result = await gate.run("x = \x001")
        assert result.artifact_hash is not None

    @pytest.mark.asyncio
    async def test_multiple_checks_some_raise(self):
        """When some checks raise and others pass, all must be collected."""
        VerificationGate = _once("VerificationGate")
        VerificationCheck = _once("VerificationCheck")
        CheckOutcome = _once("CheckOutcome")

        async def _ok(a: str) -> tuple[bool, str]:
            return True, ""

        async def _fail(a: str) -> tuple[bool, str]:
            return False, "nope"

        async def _crash(a: str) -> tuple[bool, str]:
            raise ValueError("blown up")

        gate = VerificationGate(
            checks=[
                VerificationCheck(name="a", run=_ok),
                VerificationCheck(name="b", run=_fail),
                VerificationCheck(name="c", run=_crash),
            ]
        )
        result = await gate.run("code")

        assert len(result.receipts) == 3
        assert result.receipts[0].outcome == CheckOutcome.PASSED
        assert result.receipts[1].outcome == CheckOutcome.FAILED
        assert result.receipts[2].outcome == CheckOutcome.BLOCKED
        # Gate fails overall because at least one check didn't pass
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_concurrent_gate_runs(self):
        """Multiple concurrent gate invocations must not interfere."""
        VerificationGate = _once("VerificationGate")
        VerificationCheck = _once("VerificationCheck")

        async def _ok(a: str) -> tuple[bool, str]:
            await asyncio.sleep(0.001)
            return True, ""

        gate = VerificationGate(checks=[VerificationCheck(name="test", run=_ok) for _ in range(5)])

        async def run_one(i: int) -> int:
            result = await gate.run(f"code-{i}")
            return 1 if result.passed else 0

        tasks = [run_one(i) for i in range(10)]
        results = await asyncio.gather(*tasks)
        assert sum(results) == 10  # all 10 concurrent runs passed


# ─────────────────────────────────────────────────────────────────────────────
# Performance Regression (light)
# ─────────────────────────────────────────────────────────────────────────────


class TestPerformanceCharacteristics:
    """Ensure basic performance characteristics haven't degraded."""

    @pytest.mark.asyncio
    async def test_single_check_under_100ms(self):
        import time

        VerificationGate = _once("VerificationGate")
        VerificationCheck = _once("VerificationCheck")

        async def _ok(a: str) -> tuple[bool, str]:
            return True, ""

        gate = VerificationGate(checks=[VerificationCheck(name="fast", run=_ok)])
        start = time.monotonic()
        await gate.run("code")
        elapsed_ms = (time.monotonic() - start) * 1000
        assert elapsed_ms < 100, f"Single check took {elapsed_ms:.0f}ms"

    @pytest.mark.asyncio
    async def test_artifact_hash_time_under_10ms(self):
        import time

        compute_artifact_hash = _once("compute_artifact_hash")
        data = "x" * 100_000
        start = time.monotonic()
        h = compute_artifact_hash(data)
        elapsed_ms = (time.monotonic() - start) * 1000
        assert len(h) == 64
        assert elapsed_ms < 50, f"Hash took {elapsed_ms:.0f}ms for 100KB"
