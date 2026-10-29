"""Wave E of the 2026-10-29 audit remediation plan: routing and the quality loop."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.budget import Budget
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

        monkeypatch.setattr(
            "openai._base_client.AsyncAPIClient._calculate_retry_timeout", lambda *a, **k: 0.0
        )
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


# ── E6: ARA pipelines import, and a fallback never reads as a pass ─────────


class TestAraPipelines:
    def test_every_relative_import_resolves(self) -> None:
        import ast
        import importlib
        import importlib.util
        from pathlib import Path

        import orchestrator.reasoning.ara_pipelines as mod

        tree = ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level:
                name = importlib.util.resolve_name(
                    "." * node.level + (node.module or ""), mod.__package__
                )
                target = importlib.import_module(name)
                for alias in node.names:
                    assert hasattr(target, alias.name), f"{name}.{alias.name}"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "module", ["orchestrator.ara_integration", "orchestrator.reasoning.ara_integration"]
    )
    async def test_fallback_scores_zero(self, module) -> None:
        import importlib

        from orchestrator.models import Task, TaskType

        cls = importlib.import_module(module).ARAPipelineIntegration
        ara = cls.__new__(cls)
        ara.client = MagicMock(call=AsyncMock(return_value=MagicMock(text="out")))
        ara._get_available_models = lambda _t: [Model.GPT_4O_MINI]
        result = await ara._fallback_execute(Task(id="t", type=TaskType.CODE_GEN, prompt="p"), "")
        assert result.score == 0.0


# ── E7: real token counts; critique runs when VS review has no sampler ─────


class TestGenerateTokensAndCritique:
    @pytest.mark.asyncio
    async def test_generate_records_response_tokens(self, monkeypatch) -> None:
        from orchestrator.api_clients import APIResponse
        from orchestrator.crosscutting.config import flags
        from orchestrator.engine_core.pipeline import PipelineContext
        from orchestrator.engine_core.stages.generate import GenerateStage
        from orchestrator.models import Task, TaskType

        monkeypatch.setattr(flags, "vs_generate", False, raising=False)
        resp = APIResponse(text="x", model=Model.GPT_4O, input_tokens=12, output_tokens=7)
        stage = GenerateStage(
            client=MagicMock(call=AsyncMock(return_value=resp)),
            budget=Budget(max_usd=10.0),
            selector=MagicMock(),
        )
        task = Task(id="t", type=TaskType.CODE_GEN, prompt="p", preferred_model=Model.GPT_4O)
        ctx = await stage.process(PipelineContext(task=task, tokens_used={"input": 0, "output": 0}))
        assert ctx.tokens_used == {"input": 12, "output": 7}

    @pytest.mark.asyncio
    async def test_vs_review_flag_without_sampler_still_critiques(self, monkeypatch) -> None:
        from orchestrator.crosscutting.config import flags
        from orchestrator.engine_core.pipeline import PipelineContext
        from orchestrator.engine_core.stages.critique import CritiqueStage
        from orchestrator.models import Task, TaskType

        monkeypatch.setattr(flags, "vs_code_review", True, raising=False)
        client = MagicMock(call=AsyncMock(return_value=MagicMock(text="needs tests", cost_usd=0.0)))
        stage = CritiqueStage(client=client, get_reviewer_fn=lambda *a, **k: Model.GPT_4O)
        task = Task(id="t", type=TaskType.CODE_GEN, prompt="p")
        ctx = PipelineContext(task=task, model=Model.GPT_4O_MINI)
        ctx.output = "def f(): pass"
        ctx = await stage.process(ctx)
        assert ctx.critique == "needs tests"


# ── E2: one fallback hop on a provider error ────────────────────────────────


def _generate_stage(call, record_failure_fn=None):
    from orchestrator.engine_core.stages.generate import GenerateStage

    return GenerateStage(
        client=MagicMock(call=AsyncMock(side_effect=call)),
        budget=Budget(max_usd=10.0),
        selector=MagicMock(),
        record_failure_fn=record_failure_fn,
    )


class TestGenerateFallback:
    @pytest.mark.asyncio
    async def test_primary_error_falls_back_once_and_records_failure(self, monkeypatch) -> None:
        from orchestrator.api_clients import APIResponse
        from orchestrator.crosscutting.config import flags
        from orchestrator.engine_core.pipeline import PipelineContext
        from orchestrator.models import FALLBACK_CHAIN, Task, TaskType

        monkeypatch.setattr(flags, "vs_generate", False, raising=False)
        fallback = FALLBACK_CHAIN[Model.GPT_4O]

        async def call(*, model, **_):
            if model == Model.GPT_4O:
                raise RuntimeError("provider down")
            return APIResponse(text="from fallback", model=model)

        record = AsyncMock()
        task = Task(id="t", type=TaskType.CODE_GEN, prompt="p", preferred_model=Model.GPT_4O)
        ctx = await _generate_stage(call, record).process(
            PipelineContext(task=task, tokens_used={"input": 0, "output": 0})
        )

        assert ctx.output == "from fallback"
        assert ctx.model == fallback
        assert not ctx.should_abort
        assert record.await_args.args[0] == Model.GPT_4O

    @pytest.mark.asyncio
    async def test_second_failure_is_a_named_abort(self, monkeypatch) -> None:
        from orchestrator.crosscutting.config import flags
        from orchestrator.engine_core.pipeline import PipelineContext
        from orchestrator.models import Task, TaskType

        monkeypatch.setattr(flags, "vs_generate", False, raising=False)

        async def call(**_):
            raise RuntimeError("everything down")

        record = AsyncMock()
        task = Task(id="t", type=TaskType.CODE_GEN, prompt="p", preferred_model=Model.GPT_4O)
        ctx = await _generate_stage(call, record).process(
            PipelineContext(task=task, tokens_used={"input": 0, "output": 0})
        )

        assert ctx.should_abort is True
        assert ctx.abort_reason == "provider_error"
        assert record.await_count == 2

    def test_container_late_binds_record_failure(self) -> None:
        from orchestrator.engine_core.container import ServiceContainer
        from orchestrator.engine_core.pipeline import TaskPipeline

        stage = _generate_stage(lambda **_: None)
        container = ServiceContainer.__new__(ServiceContainer)
        container.pipeline_executor = None
        container.pipeline = TaskPipeline([stage])
        fn = AsyncMock()
        container.wire_pipeline_executor(record_failure_fn=fn)
        assert stage._record_failure_fn is fn
