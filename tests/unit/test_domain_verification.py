"""
Tests for WBS-1: Domain verification types.
Mandatory acting verification — structured status, policy, enhanced gate result.

RED first — these tests should fail before the domain types exist.
"""

from __future__ import annotations

import hashlib

import pytest

pytestmark = pytest.mark.unit

from orchestrator.domain.verification import (
    CheckOutcome,
    ExecutionReceipt,
    VerificationPolicy,
)
from orchestrator.application.verification_gate import (
    GateResult,
    VerificationCheck,
    VerificationGate,
)

# ── CheckOutcome ──────────────────────────────────────────────────────────────


class TestCheckOutcome:
    """WBS-1: distinguish not_run, passed, failed, blocked."""

    def test_enum_values(self):
        assert CheckOutcome.NOT_RUN.value == "not_run"
        assert CheckOutcome.PASSED.value == "passed"
        assert CheckOutcome.FAILED.value == "failed"
        assert CheckOutcome.BLOCKED.value == "blocked"

    def test_is_passed_only_for_passed(self):
        assert CheckOutcome.PASSED.is_passed is True
        assert CheckOutcome.NOT_RUN.is_passed is False
        assert CheckOutcome.FAILED.is_passed is False
        assert CheckOutcome.BLOCKED.is_passed is False

    def test_is_failure_for_failed_and_blocked(self):
        assert CheckOutcome.FAILED.is_failure is True
        assert CheckOutcome.BLOCKED.is_failure is True
        assert CheckOutcome.PASSED.is_failure is False
        assert CheckOutcome.NOT_RUN.is_failure is False


# ── ExecutionReceipt ──────────────────────────────────────────────────────────


class TestExecutionReceipt:
    """WBS-1: artifact hashes and command receipts."""

    def test_minimal_receipt(self):
        receipt = ExecutionReceipt(check_name="lint", outcome=CheckOutcome.PASSED)
        assert receipt.check_name == "lint"
        assert receipt.outcome == CheckOutcome.PASSED
        assert receipt.reason is None
        assert receipt.duration_ms is None
        assert receipt.command is None
        assert receipt.artifact_hash is None

    def test_full_receipt(self):
        receipt = ExecutionReceipt(
            check_name="tests",
            outcome=CheckOutcome.FAILED,
            reason="3 test failures",
            duration_ms=1500.0,
            command="pytest tests/",
            artifact_hash="abc123",
        )
        assert receipt.check_name == "tests"
        assert receipt.outcome == CheckOutcome.FAILED
        assert receipt.reason == "3 test failures"
        assert receipt.duration_ms == 1500.0
        assert receipt.command == "pytest tests/"
        assert receipt.artifact_hash == "abc123"

    def test_blocked_receipt(self):
        """Blocked means the check could not execute (e.g. tool missing)."""
        receipt = ExecutionReceipt(
            check_name="type_check",
            outcome=CheckOutcome.BLOCKED,
            reason="mypy not installed",
        )
        assert receipt.outcome == CheckOutcome.BLOCKED
        assert receipt.is_blocked is True
        assert receipt.is_failure is True

    def test_not_run_receipt(self):
        receipt = ExecutionReceipt(
            check_name="security",
            outcome=CheckOutcome.NOT_RUN,
            reason="skipped by policy",
        )
        assert receipt.outcome == CheckOutcome.NOT_RUN
        assert receipt.is_blocked is False
        assert receipt.is_failure is False


# ── VerificationPolicy ────────────────────────────────────────────────────────


class TestVerificationPolicy:
    """WBS-1: policy per task type and artifact type."""

    def test_minimal_policy(self):
        """A policy with no checks still has valid defaults."""
        policy = VerificationPolicy(checks={})
        assert policy.checks == {}
        assert policy.get_requirement("nonexistent") is None

    def test_policy_with_checks(self):
        policy = VerificationPolicy(
            checks={
                "syntax": CheckOutcome.REQUIRED,
                "lint": CheckOutcome.RECOMMENDED,
                "tests": CheckOutcome.REQUIRED,
                "security": CheckOutcome.OPTIONAL,
            }
        )
        assert policy.get_requirement("syntax") == CheckOutcome.REQUIRED
        assert policy.get_requirement("lint") == CheckOutcome.RECOMMENDED
        assert policy.get_requirement("tests") == CheckOutcome.REQUIRED
        assert policy.get_requirement("security") == CheckOutcome.OPTIONAL

    def test_required_checks_returns_required(self):
        policy = VerificationPolicy(
            checks={
                "syntax": CheckOutcome.REQUIRED,
                "lint": CheckOutcome.RECOMMENDED,
                "security": CheckOutcome.OPTIONAL,
            }
        )
        required = policy.required_checks
        assert "syntax" in required
        assert "lint" not in required
        assert "security" not in required

    def test_mandatory_checks_fail_closed(self):
        """MANDATORY checks must run AND pass — fail closed if can't execute."""
        policy = VerificationPolicy(
            checks={
                "syntax": CheckOutcome.MANDATORY,
                "lint": CheckOutcome.REQUIRED,
            }
        )
        mandatory = policy.mandatory_checks
        assert "syntax" in mandatory
        assert "lint" not in mandatory

    def test_immutable(self):
        """Policy must be immutable (frozen dataclass)."""
        policy = VerificationPolicy(checks={"syntax": CheckOutcome.REQUIRED})
        with pytest.raises(AttributeError):
            policy.task_types = frozenset()  # type: ignore[misc]

    def test_task_type_filter(self):
        """Policy can be scoped to specific task types."""
        from orchestrator.models import TaskType

        policy = VerificationPolicy(
            checks={"syntax": CheckOutcome.REQUIRED},
            task_types=frozenset({TaskType.CODE_GEN, TaskType.CODE_REVIEW}),
        )
        assert TaskType.CODE_GEN in policy.task_types
        assert TaskType.WRITING not in policy.task_types

    def test_default_task_types_all(self):
        """When task_types is None, policy applies to all tasks."""
        policy = VerificationPolicy(checks={"syntax": CheckOutcome.REQUIRED})
        assert policy.task_types is None  # None = applies to all


# ── Enhanced GateResult ───────────────────────────────────────────────────────


class TestEnhancedGateResult:
    """WBS-1: artifact hashes, execution receipts, backward compat."""

    def test_backward_compat_passed_property(self):
        """Legacy callers still get .passed from check booleans."""
        result = GateResult(
            checks={"lint": True, "tests": True},
            receipts=[
                ExecutionReceipt("lint", CheckOutcome.PASSED),
                ExecutionReceipt("tests", CheckOutcome.PASSED),
            ],
        )
        assert result.passed is True

    def test_backward_compat_failed_property(self):
        result = GateResult(
            checks={"lint": True, "tests": False},
            receipts=[
                ExecutionReceipt("lint", CheckOutcome.PASSED),
                ExecutionReceipt("tests", CheckOutcome.FAILED, reason="failed"),
            ],
        )
        assert result.passed is False

    def test_empty_checks_passes(self):
        result = GateResult()
        assert result.passed is True

    def test_artifact_hash(self):
        h = hashlib.sha256(b"artifact content").hexdigest()
        result = GateResult(artifact_hash=h)
        assert result.artifact_hash == h

    def test_receipts_structured(self):
        result = GateResult(
            checks={"tests": False},
            reasons={"tests": "failed"},
            receipts=[
                ExecutionReceipt("tests", CheckOutcome.FAILED, reason="failed"),
            ],
        )
        assert len(result.receipts) == 1
        assert result.receipts[0].check_name == "tests"
        assert result.receipts[0].outcome == CheckOutcome.FAILED

    def test_failure_summary(self):
        """Summary groups failed and blocked checks."""
        result = GateResult(
            checks={"lint": True, "tests": False, "type_check": False},
            reasons={"tests": "failed", "type_check": "mypy errors"},
            receipts=[
                ExecutionReceipt("lint", CheckOutcome.PASSED),
                ExecutionReceipt("tests", CheckOutcome.FAILED, reason="failed"),
                ExecutionReceipt("type_check", CheckOutcome.FAILED, reason="mypy errors"),
            ],
        )
        summary = result.failure_summary
        assert "tests" in summary
        assert "type_check" in summary
        assert "lint" not in summary

    def test_score_floor_when_failed(self):
        result = GateResult(
            checks={"tests": False},
            score=VerificationGate.FAIL_SCORE_FLOOR,
        )
        assert result.score == VerificationGate.FAIL_SCORE_FLOOR


# ── VerificationGate with enhanced features ───────────────────────────────────


class TestVerificationGateEnhanced:
    """WBS-1: gate produces receipts and handles policy."""

    @pytest.mark.asyncio
    async def test_run_produces_receipts(self):
        """Gate.run() should produce ExecutionReceipts."""

        async def _pass(artifact: str) -> tuple[bool, str]:
            return True, ""

        gate = VerificationGate(
            checks=[
                VerificationCheck(name="lint", run=_pass),
            ]
        )
        result = await gate.run("artifact")
        assert len(result.receipts) == 1
        assert result.receipts[0].check_name == "lint"
        assert result.receipts[0].outcome == CheckOutcome.PASSED

    @pytest.mark.asyncio
    async def test_blocked_check_in_receipts(self):
        """A check that raises is recorded as BLOCKED, not FAILED."""

        async def _explodes(artifact: str) -> tuple[bool, str]:
            raise RuntimeError("tool not found")

        gate = VerificationGate(checks=[VerificationCheck(name="type_check", run=_explodes)])
        result = await gate.run("artifact")
        assert len(result.receipts) == 1
        assert result.receipts[0].outcome == CheckOutcome.BLOCKED
        assert "tool not found" in (result.receipts[0].reason or "")

    @pytest.mark.asyncio
    async def test_failed_check_in_receipts(self):
        async def _fail(artifact: str) -> tuple[bool, str]:
            return False, "syntax error"

        gate = VerificationGate(checks=[VerificationCheck(name="syntax", run=_fail)])
        result = await gate.run("artifact")
        assert result.receipts[0].outcome == CheckOutcome.FAILED
        assert result.receipts[0].reason == "syntax error"

    def test_score_floor_constant_below_acceptance_threshold(self):
        assert VerificationGate.FAIL_SCORE_FLOOR < 0.3

    def test_default_gate_empty_by_default(self):
        """default() returns a gate with no checks, preserving safe default."""
        gate = VerificationGate.default()
        assert len(gate._checks) == 0
