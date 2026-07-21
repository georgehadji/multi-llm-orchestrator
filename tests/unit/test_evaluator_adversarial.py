"""
Tests for ENH-1: adversarial evaluator stance + VerificationGate integration.

Verifies:
- Adversarial system prompt replaces trusting one
- Gate veto caps score at FAIL_SCORE_FLOOR regardless of LLM
- Gate pass allows LLM score through
- passed_validators field reflects gate outcome
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.unit

from orchestrator.application.evaluator import EvaluatorService
from orchestrator.application.verification_gate import (
    GateResult,
    VerificationCheck,
    VerificationGate,
)
from orchestrator.operations.feedback import CritiqueReport


def _mock_task(task_id: str = "t1") -> MagicMock:
    task = MagicMock()
    task.id = task_id
    task.prompt = "Write a function"
    task.acceptance_threshold = 0.7
    task.type = MagicMock()
    task.type.value = "code"
    return task


def _evaluator(gate: VerificationGate | None = None) -> EvaluatorService:
    client = MagicMock()
    budget = AsyncMock()
    budget.charge = AsyncMock()
    get_models = MagicMock(return_value=[])  # no LLM models → fast fallback
    return EvaluatorService(
        client=client,
        budget=budget,
        get_models_fn=get_models,
        verification_gate=gate,
    )


# ── Adversarial prompt ────────────────────────────────────────────────────────


class TestAdversarialPrompt:
    def test_system_prompt_contains_assume_broken(self):
        assert "BROKEN" in EvaluatorService._SYSTEM_PROMPT
        assert "Do NOT praise" in EvaluatorService._SYSTEM_PROMPT

    def test_system_prompt_does_not_contain_trusting_phrase(self):
        assert "precise evaluator" not in EvaluatorService._SYSTEM_PROMPT
        assert "Score exactly" not in EvaluatorService._SYSTEM_PROMPT


# ── Gate veto (nodding-loop regression) ──────────────────────────────────────


class TestGateVeto:
    @pytest.mark.asyncio
    async def test_failing_gate_caps_score_at_floor(self):
        async def _fail(artifact: str):
            return False, "test suite: 5 failures"

        gate = VerificationGate(checks=[VerificationCheck(name="tests", run=_fail)])
        ev = _evaluator(gate=gate)
        report = await ev.evaluate(_mock_task(), "broken code artifact")

        assert report.score <= VerificationGate.FAIL_SCORE_FLOOR
        assert report.passed_validators is False

    @pytest.mark.asyncio
    async def test_failing_gate_bypasses_llm(self):
        """LLM must not be called when gate fails — no spend wasted."""

        async def _fail(artifact: str):
            return False, "lint: 12 errors"

        gate = VerificationGate(checks=[VerificationCheck(name="lint", run=_fail)])
        ev = _evaluator(gate=gate)
        ev._client.call = AsyncMock()

        await ev.evaluate(_mock_task(), "artifact")
        ev._client.call.assert_not_called()

    @pytest.mark.asyncio
    async def test_passing_gate_allows_llm_path(self):
        """Gate pass → LLM evaluation attempted (mocked models empty → fallback 0.5)."""

        async def _pass(artifact: str):
            return True, ""

        gate = VerificationGate(checks=[VerificationCheck(name="lint", run=_pass)])
        ev = _evaluator(gate=gate)
        report = await ev.evaluate(_mock_task(), "good artifact")

        # No LLM models wired → 0.5 fallback, but passed_validators=True
        assert report.score == 0.5
        assert report.passed_validators is True

    @pytest.mark.asyncio
    async def test_no_gate_wired_skips_check(self):
        """No gate → no veto, no error; LLM path proceeds normally."""
        ev = _evaluator(gate=None)
        report = await ev.evaluate(_mock_task(), "artifact")
        assert isinstance(report, CritiqueReport)
        assert report.score == 0.5  # no models → fallback


# ── Nodding-loop core fixture ─────────────────────────────────────────────────


class TestNodingLoopFixture:
    @pytest.mark.asyncio
    async def test_broken_code_cannot_pass_gate(self):
        """Key regression: code with failing tests must score at floor, not complete."""
        broken_code = "def add(a, b): return a - b"

        async def _simulated_test_runner(artifact: str):
            # Simulates running pytest on the artifact's project
            return False, "AssertionError: add(2,3) returned -1, expected 5"

        gate = VerificationGate(
            checks=[VerificationCheck(name="pytest", run=_simulated_test_runner)]
        )
        ev = _evaluator(gate=gate)
        report = await ev.evaluate(_mock_task(), broken_code)

        assert report.passed_validators is False
        assert report.score <= VerificationGate.FAIL_SCORE_FLOOR
        # Score must be below any standard acceptance_threshold (0.7)
        assert report.score < 0.7
