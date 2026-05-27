"""Unit tests for observability injection into services (Phase 6)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.models import Model, Task, TaskResult, TaskStatus, TaskType
from orchestrator.services.executor import ExecutorResult, ExecutorService
from orchestrator.services.evaluator import EvaluatorService
from orchestrator.services.generator import GeneratorResult, GeneratorService
from orchestrator.tracing import InMemoryExporter, Tracer

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_tracer() -> tuple[Tracer, InMemoryExporter]:
    tracer = Tracer(enabled=True)
    exporter = InMemoryExporter()
    tracer.configure_exporter(exporter)
    return tracer, exporter


def _task(task_id: str = "t1") -> Task:
    return Task(
        id=task_id,
        type=TaskType.CODE_GEN,
        prompt="Write hello world",
        context="",
        dependencies=[],
    )


def _ok_result(task_id: str = "t1") -> TaskResult:
    return TaskResult(
        task_id=task_id,
        output="print('hello')",
        score=0.9,
        model_used=Model.GPT_4O_MINI,
        status=TaskStatus.COMPLETED,
        task_type=TaskType.CODE_GEN.value,
        critique="",
        iterations=1,
        cost_usd=0.001,
        tokens_used={"input": 10, "output": 5},
    )


# ─────────────────────────────────────────────────────────────────────────────
# ExecutorService observability
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_executor_emits_trace_span():
    tracer, exporter = _make_tracer()
    execute_fn = AsyncMock(return_value=_ok_result())
    svc = ExecutorService(execute_fn=execute_fn, tracer=tracer)

    result = await svc.execute(_task())
    assert result.succeeded is True

    await tracer.flush()
    spans = exporter.get_spans()
    assert len(spans) == 1
    assert spans[0].name == "executor.task"
    assert spans[0].attributes.get("task_id") == "t1"
    assert spans[0].status == "OK"


@pytest.mark.asyncio
async def test_executor_trace_on_error():
    tracer, exporter = _make_tracer()
    execute_fn = AsyncMock(side_effect=RuntimeError("boom"))
    svc = ExecutorService(execute_fn=execute_fn, tracer=tracer)

    result = await svc.execute(_task())
    assert result.error is not None

    await tracer.flush()
    spans = exporter.get_spans()
    assert spans[0].status == "ERROR"
    assert any(
        "boom" in str(e.get("attributes", {}).get("exception.message", "")) for e in spans[0].events
    )


@pytest.mark.asyncio
async def test_executor_telemetry_recorded():
    telemetry = MagicMock()
    execute_fn = AsyncMock(return_value=_ok_result())
    svc = ExecutorService(execute_fn=execute_fn, telemetry=telemetry)

    await svc.execute(_task())

    telemetry.record_call.assert_called_once()
    call_kwargs = telemetry.record_call.call_args.kwargs
    assert call_kwargs["model"] == Model.GPT_4O_MINI
    assert call_kwargs["success"] is True
    assert call_kwargs["cost_usd"] == 0.001


@pytest.mark.asyncio
async def test_executor_no_tracer_no_telemetry():
    """Backward compat: services work when tracer/telemetry are None."""
    execute_fn = AsyncMock(return_value=_ok_result())
    svc = ExecutorService(execute_fn=execute_fn)
    result = await svc.execute(_task())
    assert result.succeeded is True


# ─────────────────────────────────────────────────────────────────────────────
# GeneratorService observability
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_generator_emits_trace_span():
    tracer, exporter = _make_tracer()
    decompose_fn = AsyncMock(return_value={"t1": _task()})
    svc = GeneratorService(decompose_fn=decompose_fn, tracer=tracer)

    result = await svc.decompose("Build API", "fast")
    assert result.succeeded is True

    await tracer.flush()
    spans = exporter.get_spans()
    assert len(spans) == 1
    assert spans[0].name == "generator.decompose"
    assert spans[0].status == "OK"


@pytest.mark.asyncio
async def test_generator_trace_on_error():
    tracer, exporter = _make_tracer()
    decompose_fn = AsyncMock(side_effect=RuntimeError("boom"))
    svc = GeneratorService(decompose_fn=decompose_fn, tracer=tracer)

    result = await svc.decompose("Build API", "fast")
    assert result.error is not None

    await tracer.flush()
    spans = exporter.get_spans()
    assert spans[0].status == "ERROR"


@pytest.mark.asyncio
async def test_generator_no_tracer():
    decompose_fn = AsyncMock(return_value={"t1": _task()})
    svc = GeneratorService(decompose_fn=decompose_fn)
    result = await svc.decompose("Build API", "fast")
    assert result.succeeded is True


# ─────────────────────────────────────────────────────────────────────────────
# EvaluatorService observability
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_evaluator_emits_trace_span():
    tracer, exporter = _make_tracer()
    client = MagicMock()
    client.call = AsyncMock(
        return_value=MagicMock(
            text='{"score": 0.85, "reasoning": "good"}',
            cost_usd=0.001,
            input_tokens=10,
            output_tokens=5,
        )
    )
    budget = MagicMock()
    budget.charge = AsyncMock()
    svc = EvaluatorService(
        client=client,
        budget=budget,
        get_models_fn=lambda _tt: [Model.GPT_4O_MINI],
        tracer=tracer,
    )

    score = await svc.evaluate(_task(), "output")
    assert 0.0 <= score.score <= 1.0

    await tracer.flush()
    spans = exporter.get_spans()
    assert len(spans) == 1
    assert spans[0].name == "evaluator.evaluate"
    assert spans[0].attributes.get("task_id") == "t1"
    assert spans[0].status == "OK"


@pytest.mark.asyncio
async def test_evaluator_telemetry_recorded():
    telemetry = MagicMock()
    client = MagicMock()
    client.call = AsyncMock(
        return_value=MagicMock(
            text='{"score": 0.85, "reasoning": "good"}',
            cost_usd=0.001,
            input_tokens=10,
            output_tokens=5,
        )
    )
    budget = MagicMock()
    budget.charge = AsyncMock()
    svc = EvaluatorService(
        client=client,
        budget=budget,
        get_models_fn=lambda _tt: [Model.GPT_4O_MINI],
        telemetry=telemetry,
    )

    await svc.evaluate(_task(), "output")

    telemetry.record_call.assert_called_once()
    call_kwargs = telemetry.record_call.call_args.kwargs
    assert call_kwargs["model"] == Model.GPT_4O_MINI
    assert call_kwargs["success"] is True
    assert call_kwargs["quality_score"] == 0.85


@pytest.mark.asyncio
async def test_evaluator_no_tracer_no_telemetry():
    client = MagicMock()
    client.call = AsyncMock(
        return_value=MagicMock(
            text='{"score": 0.85}',
            cost_usd=0.001,
        )
    )
    budget = MagicMock()
    budget.charge = AsyncMock()
    svc = EvaluatorService(
        client=client,
        budget=budget,
        get_models_fn=lambda _tt: [Model.GPT_4O_MINI],
    )
    score = await svc.evaluate(_task(), "output")
    assert 0.0 <= score.score <= 1.0
