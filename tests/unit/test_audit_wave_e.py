"""Wave E of the 2026-10-29 audit remediation plan: routing and the quality loop."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.models import Model

pytestmark = pytest.mark.unit


def _client():
    from orchestrator.infrastructure.llm_client import UnifiedClient

    with patch.object(UnifiedClient, "_init_clients", lambda self: None):
        client = UnifiedClient(cache=MagicMock(get=AsyncMock(return_value=None), put=AsyncMock()))
    client._get_client_for_model = AsyncMock(return_value=object())
    return client


def _ok_response():
    from types import SimpleNamespace

    return SimpleNamespace(text="ok", input_tokens=1, output_tokens=1, cost_usd=0.0, latency_ms=0.0)


# ── E1: one breaker per model, cancellation is not a failure ────────────────


class TestPerModelBreaker:
    @pytest.mark.asyncio
    async def test_failures_on_one_model_do_not_block_another(self) -> None:
        from orchestrator.circuit_breaker import CircuitBreakerOpen

        client = _client()

        async def dispatch(*, model_id, **_):
            if model_id == Model.GPT_4O_MINI.value:
                raise RuntimeError("provider down")
            return _ok_response()

        client._dispatch = dispatch
        for _ in range(5):
            with pytest.raises(RuntimeError):
                await client.call(Model.GPT_4O_MINI, "p", bypass_cache=True)
        with pytest.raises(CircuitBreakerOpen):
            await client.call(Model.GPT_4O_MINI, "p", bypass_cache=True)

        resp = await client.call(Model.GPT_4O, "p", bypass_cache=True)
        assert resp.text == "ok"

    @pytest.mark.asyncio
    async def test_cancellation_is_not_counted_as_failure(self) -> None:
        from orchestrator.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(name="t", failure_threshold=1, reset_timeout=60.0)
        with pytest.raises(asyncio.CancelledError):
            async with cb.context():
                raise asyncio.CancelledError
        assert cb.total_failures == 0
        assert cb.is_closed


# ── E3: retries= reaches the SDK ────────────────────────────────────────────


class TestRetriesReachSdk:
    @pytest.mark.asyncio
    async def test_two_transient_failures_then_success_with_retries_2(self, monkeypatch) -> None:
        from types import SimpleNamespace

        import httpx
        from openai import AsyncOpenAI

        monkeypatch.setattr("openai._base_client.AsyncAPIClient._calculate_retry_timeout",
                            lambda *a, **k: 0.0)
        hits = []

        def handler(request: httpx.Request) -> httpx.Response:
            hits.append(1)
            if len(hits) <= 2:
                return httpx.Response(503, json={"error": {"message": "busy"}})
            return httpx.Response(
                200,
                json={
                    "id": "c1",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "m",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": "done"},
                        }
                    ],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
                },
            )

        sdk = AsyncOpenAI(
            api_key="test",
            base_url="http://fake.local/v1",
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
        client = _client()
        client._get_client_for_model = AsyncMock(return_value=SimpleNamespace(client=sdk))

        resp = await client.call(Model.GPT_4O, "p", retries=2, bypass_cache=True)

        assert resp.text == "done"
        assert len(hits) == 3


# ── E4: the retry gate uses the task's own acceptance threshold ─────────────


class TestRetryThreshold:
    @pytest.mark.asyncio
    async def test_score_between_default_and_task_threshold_retries(self) -> None:
        from orchestrator.engine_core.pipeline import PipelineContext
        from orchestrator.engine_core.stages import SelfConsistencyStage
        from orchestrator.models import Task, TaskType

        task = Task(id="t1", type=TaskType.CODE_GEN, prompt="p", acceptance_threshold=0.85)
        ctx = PipelineContext(task=task)
        ctx.score = 0.8
        ctx.model = Model.GPT_4O_MINI
        result = await SelfConsistencyStage(**SelfConsistencyStage.build_kwargs()).process(ctx)
        assert result.should_abort is True
        assert result.abort_reason == "retry_for_quality"


# ── E5: the executor keeps the best attempt, and records each score ────────


class TestKeepBestAttempt:
    @pytest.mark.asyncio
    async def test_better_first_attempt_wins(self) -> None:
        from orchestrator.engine_core.pipeline_executor import PipelineExecutor
        from orchestrator.models import AttemptRecord, Task, TaskType

        scores = iter([(0.8, "good", Model.GPT_4O), (0.6, "worse", Model.GPT_4O_MINI)])

        async def run(ctx):
            ctx.score, ctx.output, ctx.model = next(scores)
            if ctx.attempt == 0:
                ctx.attempt_history.append(
                    AttemptRecord(
                        attempt_num=0,
                        model_used=ctx.model.value,
                        output_snippet="good",
                        failure_reason="below",
                        score=ctx.score,
                    )
                )
                ctx.attempt = 1
                ctx.should_abort = True
                ctx.abort_reason = "retry_for_quality"
            return ctx

        executor = PipelineExecutor(
            pipeline=MagicMock(run=AsyncMock(side_effect=run)),
            selector=MagicMock(select=MagicMock(return_value=None)),
            background_tasks=set(),
        )
        executor._build_skill_prefix = AsyncMock(return_value="")
        executor._enrich_with_visual_context = AsyncMock(side_effect=lambda t: t)
        executor._check_anti_slop = MagicMock()
        executor._record_trajectory = AsyncMock()

        task = Task(id="t1", type=TaskType.CODE_GEN, prompt="p", acceptance_threshold=0.9)
        result = await executor.execute(task)

        assert result.score == 0.8
        assert result.output == "good"
        assert result.model_used == Model.GPT_4O
        assert result.attempt_history[0]["score"] == 0.8
