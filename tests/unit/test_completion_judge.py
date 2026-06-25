"""
Tests for ENH-2: CompletionJudge — maker-checker stop condition.

Loop Engineering §VI: "Generator/Evaluator must be different models."

Verifies:
- Judge model is always different from generator model
- Judge returns PASS / FAIL verdict (not a score)
- PASS allows loop to exit; FAIL forces another iteration
- No-judge-available → loop exits (conservative fallback)
- Judge uses cheapest available model tier
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.services.completion_judge import (
    CompletionJudge,
    JudgeVerdict,
    SameModelError,
)


def _mock_client(response_text: str = '{"verdict": "PASS", "reason": "looks good"}') -> MagicMock:
    client = MagicMock()
    response = MagicMock()
    response.text = response_text
    response.cost_usd = 0.001
    client.call = AsyncMock(return_value=response)
    return client


def _mock_model(value: str = "deepseek/deepseek-chat") -> MagicMock:
    m = MagicMock()
    m.value = value
    return m


# ── JudgeVerdict ─────────────────────────────────────────────────────────────

class TestJudgeVerdict:
    def test_pass_is_truthy(self):
        assert JudgeVerdict.PASS

    def test_fail_is_falsy(self):
        assert not JudgeVerdict.FAIL

    def test_pass_str(self):
        assert str(JudgeVerdict.PASS) == "PASS"

    def test_fail_str(self):
        assert str(JudgeVerdict.FAIL) == "FAIL"


# ── SameModelError ────────────────────────────────────────────────────────────

class TestSameModelError:
    def test_is_runtime_error(self):
        assert issubclass(SameModelError, RuntimeError)


# ── CompletionJudge construction ──────────────────────────────────────────────

class TestCompletionJudgeConstruction:
    def test_raises_if_judge_equals_generator(self):
        model = _mock_model("openai/gpt-4o")
        with pytest.raises(SameModelError):
            CompletionJudge(
                client=_mock_client(),
                judge_model=model,
                generator_model=model,
            )

    def test_raises_if_judge_value_equals_generator_value(self):
        with pytest.raises(SameModelError):
            CompletionJudge(
                client=_mock_client(),
                judge_model=_mock_model("openai/gpt-4o"),
                generator_model=_mock_model("openai/gpt-4o"),
            )

    def test_different_models_accepted(self):
        judge = CompletionJudge(
            client=_mock_client(),
            judge_model=_mock_model("deepseek/deepseek-chat"),
            generator_model=_mock_model("openai/gpt-4o"),
        )
        assert judge is not None


# ── judge() method ────────────────────────────────────────────────────────────

class TestJudgeMethod:
    def _judge(self, response_text: str = '{"verdict": "PASS", "reason": "all checks pass"}'):
        return CompletionJudge(
            client=_mock_client(response_text),
            judge_model=_mock_model("deepseek/deepseek-chat"),
            generator_model=_mock_model("openai/gpt-4o"),
        )

    @pytest.mark.asyncio
    async def test_pass_response_returns_pass(self):
        judge = self._judge('{"verdict": "PASS", "reason": "looks correct"}')
        result = await judge.judge(task_prompt="Write add()", artifact="def add(a,b): return a+b")
        assert result is JudgeVerdict.PASS

    @pytest.mark.asyncio
    async def test_fail_response_returns_fail(self):
        judge = self._judge('{"verdict": "FAIL", "reason": "wrong logic"}')
        result = await judge.judge(task_prompt="Write add()", artifact="def add(a,b): return a-b")
        assert result is JudgeVerdict.FAIL

    @pytest.mark.asyncio
    async def test_malformed_json_returns_fail(self):
        """Unparseable response → conservative FAIL (don't approve broken output)."""
        judge = self._judge("not json at all")
        result = await judge.judge(task_prompt="Write add()", artifact="def add(a,b): return a+b")
        assert result is JudgeVerdict.FAIL

    @pytest.mark.asyncio
    async def test_client_exception_returns_fail(self):
        """LLM call failure → conservative FAIL."""
        client = MagicMock()
        client.call = AsyncMock(side_effect=RuntimeError("timeout"))
        judge = CompletionJudge(
            client=client,
            judge_model=_mock_model("deepseek/deepseek-chat"),
            generator_model=_mock_model("openai/gpt-4o"),
        )
        result = await judge.judge(task_prompt="Write add()", artifact="def add(a,b): return a+b")
        assert result is JudgeVerdict.FAIL

    @pytest.mark.asyncio
    async def test_judge_calls_correct_model(self):
        """Judge must send to judge_model, not generator_model."""
        judge_model = _mock_model("deepseek/deepseek-chat")
        gen_model = _mock_model("openai/gpt-4o")
        client = _mock_client()

        judge = CompletionJudge(
            client=client,
            judge_model=judge_model,
            generator_model=gen_model,
        )
        await judge.judge(task_prompt="Write add()", artifact="def add(a,b): return a+b")
        call_args = client.call.call_args
        assert call_args[0][0] is judge_model or call_args.kwargs.get("model") is judge_model

    @pytest.mark.asyncio
    async def test_unknown_verdict_string_returns_fail(self):
        """Unknown verdict value → conservative FAIL."""
        judge = self._judge('{"verdict": "MAYBE", "reason": "unclear"}')
        result = await judge.judge(task_prompt="Write add()", artifact="artifact")
        assert result is JudgeVerdict.FAIL


# ── CompletionJudge.from_models factory ──────────────────────────────────────

class TestFromModelsFactory:
    def test_returns_none_when_no_judge_models(self):
        """No cheap models available → factory returns None (caller skips judge)."""
        result = CompletionJudge.from_models(
            client=_mock_client(),
            judge_candidates=[],
            generator_model=_mock_model("openai/gpt-4o"),
        )
        assert result is None

    def test_returns_none_when_all_candidates_equal_generator(self):
        gen = _mock_model("openai/gpt-4o")
        result = CompletionJudge.from_models(
            client=_mock_client(),
            judge_candidates=[gen],
            generator_model=gen,
        )
        assert result is None

    def test_picks_first_non_generator_candidate(self):
        gen = _mock_model("openai/gpt-4o")
        cheap = _mock_model("deepseek/deepseek-chat")
        judge_obj = CompletionJudge.from_models(
            client=_mock_client(),
            judge_candidates=[gen, cheap],
            generator_model=gen,
        )
        assert judge_obj is not None
        assert judge_obj._judge_model is cheap
