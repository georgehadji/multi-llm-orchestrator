"""
Tests for ENH-1: VerificationGate — deterministic test/lint floor for the evaluator.

RED first.
"""

from __future__ import annotations

import pytest

from orchestrator.application.verification_gate import (
    GateResult,
    VerificationGate,
    VerificationCheck,
)

# ── GateResult ────────────────────────────────────────────────────────────────


class TestGateResult:
    def test_passed_when_all_checks_pass(self):
        result = GateResult(checks={"lint": True, "tests": True})
        assert result.passed is True

    def test_failed_when_any_check_fails(self):
        result = GateResult(checks={"lint": True, "tests": False})
        assert result.passed is False

    def test_empty_checks_passes(self):
        result = GateResult(checks={})
        assert result.passed is True

    def test_reasons_lists_failures(self):
        result = GateResult(checks={"lint": True, "tests": False}, reasons={"tests": "3 failures"})
        assert "tests" in result.reasons
        assert result.reasons["tests"] == "3 failures"


# ── VerificationGate ──────────────────────────────────────────────────────────


class TestVerificationGateWithMockChecks:
    def _gate_with(self, results: dict[str, bool]) -> VerificationGate:
        """Build a gate whose checks return fixed results."""
        checks = []
        for name, passes in results.items():
            reason = "" if passes else f"{name} failed"

            async def _check(artifact: str, _name=name, _passes=passes, _reason=reason):
                return _passes, _reason

            checks.append(VerificationCheck(name=name, run=_check))
        return VerificationGate(checks=checks)

    @pytest.mark.asyncio
    async def test_all_pass_returns_passed_true(self):
        gate = self._gate_with({"lint": True, "tests": True})
        result = await gate.run("some artifact")
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_one_fail_returns_passed_false(self):
        gate = self._gate_with({"lint": True, "tests": False})
        result = await gate.run("some artifact")
        assert result.passed is False
        assert "tests" in result.reasons

    @pytest.mark.asyncio
    async def test_all_fail_collects_all_reasons(self):
        gate = self._gate_with({"lint": False, "tests": False})
        result = await gate.run("some artifact")
        assert result.passed is False
        assert "lint" in result.reasons
        assert "tests" in result.reasons

    @pytest.mark.asyncio
    async def test_empty_gate_passes(self):
        gate = VerificationGate(checks=[])
        result = await gate.run("artifact")
        assert result.passed is True


# ── Score floor when gate fails ───────────────────────────────────────────────


class TestScoreFloor:
    def test_floor_constant_is_below_acceptance_threshold(self):
        assert VerificationGate.FAIL_SCORE_FLOOR < 0.3

    @pytest.mark.asyncio
    async def test_failed_gate_score_is_at_floor(self):
        async def _fail(artifact: str):
            return False, "tests exploded"

        gate = VerificationGate(checks=[VerificationCheck(name="tests", run=_fail)])
        result = await gate.run("artifact")
        assert result.passed is False
        assert result.score <= VerificationGate.FAIL_SCORE_FLOOR


# ── Nodding-loop regression ───────────────────────────────────────────────────


class TestNodingLoopRegression:
    """Core ENH-1 fixture: a failing artifact must not pass the gate."""

    @pytest.mark.asyncio
    async def test_artifact_with_failing_tests_blocked(self):
        """Simulate an artifact whose unit tests fail — gate must veto."""
        failing_artifact = "def add(a, b): return a - b  # wrong implementation"

        async def _fake_test_runner(artifact: str):
            # In real gate, this runs pytest; here we simulate the outcome
            return False, "AssertionError: assert add(1,2) == 3"

        gate = VerificationGate(checks=[VerificationCheck(name="tests", run=_fake_test_runner)])
        result = await gate.run(failing_artifact)
        assert result.passed is False
        assert result.score <= VerificationGate.FAIL_SCORE_FLOOR
