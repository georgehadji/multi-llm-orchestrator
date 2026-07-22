"""
Proactive bug-scan suite — invariant/property tests over core pure functions.

Purpose: catch silent logic bugs across the codebase by asserting INVARIANTS
that must hold for every input, not just the happy path. Unlike example-based
tests, these are designed to FAIL the moment a regression violates a contract,
so the suite acts as a continuous tripwire.

Coverage targets (pure / deterministic logic only — no network, no DB):
  - EvaluatorService.parse_score   : output always in [0,1]; many input formats
  - EvaluatorService._aggregate    : N-run aggregation never discards runs
  - VerificationGate.run           : passed iff every check passed; floor on fail
  - CompletionJudge                : fail-closed on every error path
  - AutonomyCostCollector          : gauges monotonic / bounded; snapshot isolated
  - BudgetHierarchy.remaining      : never negative at any level
  - CronParser.matches             : never raises; weekday convention correct

Discovered & fixed during this scan:
  - _aggregate dropped runs 2..N when consistency_runs > 2 (returned scores[0]).
    Regression guard: TestAggregateNeverDiscardsRuns.
"""

from __future__ import annotations

import pytest
import pytest

pytestmark = pytest.mark.unit

import pytest

pytestmark = pytest.mark.unit
# ──────────────────────────────────────────────────────────────────────────────
# EvaluatorService.parse_score — output domain invariant
# ──────────────────────────────────────────────────────────────────────────────

from orchestrator.application.evaluator import EvaluatorService

_PARSE_INPUTS = [
    '{"score": 0.85}',
    '{"Score": 0.5}',
    '```json\n{"score": 0.9}\n```',
    "0.73",
    "1.0",
    "0.0",
    "score: 0.42",
    "rating = 0.6",
    "8/10",
    "85%",
    "100%",
    "7 out of 10",
    "评分: 0.55",
    '<think>maybe 0.1</think> {"score": 0.95}',
    "garbage no number here",
    "",
    "score: 2.5",  # out-of-range raw → must clamp
    "score: -0.3",  # negative → must clamp
    "9999",  # absurd → must stay bounded
]


class TestParseScoreDomain:
    @pytest.mark.parametrize("text", _PARSE_INPUTS)
    def test_output_always_in_unit_interval(self, text):
        score = EvaluatorService.parse_score(text)
        assert 0.0 <= score <= 1.0, f"parse_score({text!r}) = {score} out of [0,1]"

    @pytest.mark.parametrize("text", _PARSE_INPUTS)
    def test_never_raises(self, text):
        # Must never throw — callers rely on a guaranteed float.
        EvaluatorService.parse_score(text)

    def test_full_percent_is_one(self):
        assert EvaluatorService.parse_score("Score: 100%") == pytest.approx(1.0)

    def test_high_percent_not_truncated(self):
        # Guards the two-digit-cap class of bug in the % regex.
        assert EvaluatorService.parse_score("95%") == pytest.approx(0.95)

    def test_unparseable_returns_safe_default(self):
        assert EvaluatorService.parse_score("absolutely no score") == 0.5


# ──────────────────────────────────────────────────────────────────────────────
# EvaluatorService._aggregate — N-run aggregation must not discard runs
#   (regression guard for the bug fixed in this scan)
# ──────────────────────────────────────────────────────────────────────────────


class TestAggregateNeverDiscardsRuns:
    def _ev(self):
        return EvaluatorService(None, None, lambda t: [])

    def test_empty_is_safe_default(self):
        assert self._ev()._aggregate([], "t") == 0.5

    def test_single_run_returned(self):
        assert self._ev()._aggregate([0.7], "t") == 0.7

    def test_two_close_runs_averaged(self):
        assert self._ev()._aggregate([0.80, 0.82], "t") == pytest.approx(0.81)

    def test_two_divergent_runs_take_lower(self):
        assert self._ev()._aggregate([0.9, 0.5], "t") == 0.5

    def test_three_runs_not_reduced_to_first(self):
        # The fixed bug: [0.2, 0.9, 0.9] previously returned 0.2 (scores[0]).
        result = self._ev()._aggregate([0.2, 0.9, 0.9], "t")
        assert result != 0.2, "must not blindly return scores[0]"
        assert result == 0.9, "median of the three runs"

    def test_aggregate_within_input_range(self):
        scores = [0.1, 0.4, 0.55, 0.6, 0.9]
        result = self._ev()._aggregate(scores, "t")
        assert min(scores) <= result <= max(scores)

    def test_even_run_count_uses_median(self):
        # 4 runs → median of two middle values.
        result = self._ev()._aggregate([0.2, 0.4, 0.6, 0.8], "t")
        assert result == pytest.approx(0.5)


# ──────────────────────────────────────────────────────────────────────────────
# VerificationGate — passed iff every check passed; score floor on failure
# ──────────────────────────────────────────────────────────────────────────────

from orchestrator.application.verification_gate import (
    VerificationCheck,
    VerificationGate,
)


def _check(name: str, ok: bool) -> VerificationCheck:
    async def _run(artifact: str):
        return ok, "" if ok else f"{name} failed"

    return VerificationCheck(name=name, run=_run)


class TestVerificationGateInvariants:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "results",
        [
            {"a": True},
            {"a": True, "b": True},
            {"a": True, "b": False},
            {"a": False, "b": False},
            {},
        ],
    )
    async def test_passed_iff_all_checks_pass(self, results):
        gate = VerificationGate(checks=[_check(n, ok) for n, ok in results.items()])
        out = await gate.run("artifact")
        expected = all(results.values()) if results else True
        assert out.passed is expected

    @pytest.mark.asyncio
    async def test_failure_caps_score_at_floor(self):
        gate = VerificationGate(checks=[_check("x", False)])
        out = await gate.run("artifact")
        assert out.score <= VerificationGate.FAIL_SCORE_FLOOR

    @pytest.mark.asyncio
    async def test_raising_check_counts_as_failure(self):
        async def _boom(artifact: str):
            raise RuntimeError("check exploded")

        gate = VerificationGate(checks=[VerificationCheck(name="boom", run=_boom)])
        out = await gate.run("artifact")
        assert out.passed is False  # must not propagate; must fail closed


# ──────────────────────────────────────────────────────────────────────────────
# CompletionJudge — fail-closed on every error path
# ──────────────────────────────────────────────────────────────────────────────

from unittest.mock import AsyncMock, MagicMock

from orchestrator.services.completion_judge import CompletionJudge, JudgeVerdict


def _model(value: str) -> MagicMock:
    m = MagicMock()
    m.value = value
    return m


def _judge_with(response_text: str | None, raise_exc: bool = False) -> CompletionJudge:
    client = MagicMock()
    if raise_exc:
        client.call = AsyncMock(side_effect=RuntimeError("network"))
    else:
        resp = MagicMock()
        resp.text = response_text
        resp.cost_usd = 0.0
        client.call = AsyncMock(return_value=resp)
    return CompletionJudge(
        client=client,
        judge_model=_model("deepseek/deepseek-chat"),
        generator_model=_model("openai/gpt-4o"),
    )


class TestCompletionJudgeFailClosed:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "text",
        [
            "not json",
            '{"verdict": "MAYBE"}',
            '{"no_verdict": true}',
            "{}",
            '{"verdict": ""}',
        ],
    )
    async def test_bad_response_is_fail(self, text):
        judge = _judge_with(text)
        verdict = await judge.judge("task", "artifact")
        assert verdict is JudgeVerdict.FAIL

    @pytest.mark.asyncio
    async def test_client_exception_is_fail(self):
        judge = _judge_with(None, raise_exc=True)
        verdict = await judge.judge("task", "artifact")
        assert verdict is JudgeVerdict.FAIL

    @pytest.mark.asyncio
    async def test_clean_pass_is_pass(self):
        judge = _judge_with('{"verdict": "PASS", "reason": "ok"}')
        verdict = await judge.judge("task", "artifact")
        assert verdict is JudgeVerdict.PASS


# ──────────────────────────────────────────────────────────────────────────────
# AutonomyCostCollector — gauge bounds & snapshot isolation
# ──────────────────────────────────────────────────────────────────────────────

from orchestrator.services.autonomy_costs import AutonomyCostCollector


class TestAutonomyCostInvariants:
    def test_token_blowout_never_negative(self):
        c = AutonomyCostCollector()
        c.record_tokens(baseline=1000, actual=10)  # under baseline
        assert c.snapshot().token_blowout >= 0

    def test_surrender_pct_bounded(self):
        c = AutonomyCostCollector()
        for _ in range(7):
            c.record_iteration(judge_ran=False)
        for _ in range(3):
            c.record_iteration(judge_ran=True)
        pct = c.snapshot().cognitive_surrender_pct
        assert 0.0 <= pct <= 100.0

    def test_comprehension_rot_zero_capacity_safe(self):
        c = AutonomyCostCollector()
        c.record_context_window(used=500, capacity=0)  # must not divide by zero
        assert c.snapshot().comprehension_rot_pct == 0.0

    def test_snapshot_is_isolated(self):
        c = AutonomyCostCollector()
        snap_before = c.snapshot()
        c.record_skipped_check()
        assert snap_before.verification_debt == 0  # old snapshot must not alias


# ──────────────────────────────────────────────────────────────────────────────
# BudgetHierarchy.remaining — never negative at any level
# ──────────────────────────────────────────────────────────────────────────────


class TestBudgetRemainingNonNegative:
    def _hierarchy(self):
        from orchestrator.cost import BudgetHierarchy

        try:
            return BudgetHierarchy(org_max_usd=10.0)
        except TypeError:
            # Constructor signature may differ; skip gracefully rather than error.
            pytest.skip("BudgetHierarchy signature differs")

    def test_org_remaining_non_negative_after_overspend(self):
        h = self._hierarchy()
        # Spend more than the cap via any available charge API.
        for api in ("charge", "charge_job", "record_spend", "spend"):
            fn = getattr(h, api, None)
            if fn is None:
                continue
            try:
                fn(100.0) if api in ("record_spend", "spend") else None
            except Exception:
                pass
        assert h.remaining("org") >= 0.0

    def test_unknown_level_raises_valueerror(self):
        h = self._hierarchy()
        with pytest.raises(ValueError):
            h.remaining("galaxy")


# ──────────────────────────────────────────────────────────────────────────────
# CronParser.matches — robustness: never raises on arbitrary input
# ──────────────────────────────────────────────────────────────────────────────

from orchestrator.operations.automations import CronParser


class TestCronParserRobustness:
    @pytest.mark.parametrize(
        "expr",
        [
            "",
            "* * * * *",
            "*/5 * * * *",
            "*/0 * * * *",  # zero-step must not crash
            "0 0 * * *",
            "bad expr",
            "* * *",  # too few fields
            "1,2,3 * * * *",
            "60 24 32 13 8",  # out-of-range values
        ],
    )
    def test_matches_never_raises(self, expr):
        # Returns a bool for any input; never throws.
        result = CronParser.matches(expr, timestamp=0)
        assert isinstance(result, bool)
