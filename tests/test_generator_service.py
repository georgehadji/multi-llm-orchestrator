"""Unit tests for orchestrator.services.generator.GeneratorService."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from orchestrator.exceptions import OrchestratorError
from orchestrator.models import Task, TaskType
from orchestrator.services.generator import GeneratorResult, GeneratorService


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _task(task_id: str) -> Task:
    return Task(
        id=task_id,
        type=TaskType.CODE_GEN,
        prompt="Do something",
        context="",
        dependencies=[],
    )


def _tasks(*ids: str) -> dict[str, Task]:
    return {tid: _task(tid) for tid in ids}


# ─────────────────────────────────────────────────────────────────────────────
# GeneratorResult properties
# ─────────────────────────────────────────────────────────────────────────────


def test_generator_result_succeeded_with_tasks():
    r = GeneratorResult(tasks=_tasks("t1", "t2"), wall_time_ms=100.0)
    assert r.succeeded is True
    assert r.task_count == 2


def test_generator_result_not_succeeded_when_empty():
    r = GeneratorResult(tasks={}, wall_time_ms=10.0)
    assert r.succeeded is False


def test_generator_result_not_succeeded_with_error():
    r = GeneratorResult(
        tasks=_tasks("t1"),
        wall_time_ms=10.0,
        error=RuntimeError("fail"),
    )
    assert r.succeeded is False


# ─────────────────────────────────────────────────────────────────────────────
# Happy path
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_decompose_returns_tasks_on_success():
    fn = AsyncMock(return_value=_tasks("t1", "t2", "t3"))
    svc = GeneratorService(decompose_fn=fn)

    result = await svc.decompose("Build an API", "All endpoints tested")

    assert result.succeeded
    assert result.task_count == 3
    assert result.error is None
    assert result.wall_time_ms >= 0
    fn.assert_called_once_with("Build an API", "All endpoints tested")


@pytest.mark.asyncio
async def test_decompose_forwards_kwargs():
    fn = AsyncMock(return_value=_tasks("t1"))
    svc = GeneratorService(decompose_fn=fn)

    await svc.decompose("proj", "crit", app_profile="some_profile")

    fn.assert_called_once_with("proj", "crit", app_profile="some_profile")


# ─────────────────────────────────────────────────────────────────────────────
# Error normalization — never raises
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_decompose_normalizes_generic_exception():
    async def boom(*_a, **_kw):
        raise ValueError("unexpected")

    svc = GeneratorService(decompose_fn=boom)
    result = await svc.decompose("proj", "crit")

    assert not result.succeeded
    assert isinstance(result.error, OrchestratorError)
    assert result.tasks == {}


@pytest.mark.asyncio
async def test_decompose_passes_orchestrator_error_through():
    async def boom(*_a, **_kw):
        raise OrchestratorError("known failure")

    svc = GeneratorService(decompose_fn=boom)
    result = await svc.decompose("proj", "crit")

    assert isinstance(result.error, OrchestratorError)
    assert "known failure" in str(result.error)


@pytest.mark.asyncio
async def test_decompose_normalizes_timeout():
    async def slow(*_a, **_kw):
        await asyncio.sleep(10)
        return _tasks("t1")

    svc = GeneratorService(decompose_fn=slow, decompose_timeout=0.05)
    result = await svc.decompose("proj", "crit")

    assert not result.succeeded
    assert isinstance(result.error, OrchestratorError)
    assert "timed out" in str(result.error).lower()


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_metrics_accumulate_on_success():
    fn = AsyncMock(return_value=_tasks("t1", "t2"))
    svc = GeneratorService(decompose_fn=fn)

    await svc.decompose("p", "c")
    await svc.decompose("p2", "c2")

    m = svc.metrics_snapshot()
    assert m["total_calls"] == 2
    assert m["total_succeeded"] == 2
    assert m["total_tasks_generated"] == 4  # 2 tasks × 2 calls
    assert m["total_failed"] == 0


@pytest.mark.asyncio
async def test_metrics_accumulate_on_failure():
    async def boom(*_a, **_kw):
        raise RuntimeError("fail")

    svc = GeneratorService(decompose_fn=boom)
    await svc.decompose("p", "c")

    m = svc.metrics_snapshot()
    assert m["total_calls"] == 1
    assert m["total_failed"] == 1
    assert m["total_tasks_generated"] == 0


@pytest.mark.asyncio
async def test_metrics_avg_wall_ms():
    fn = AsyncMock(return_value=_tasks("t1"))
    svc = GeneratorService(decompose_fn=fn)
    await svc.decompose("p", "c")

    m = svc.metrics_snapshot()
    assert m["avg_wall_ms"] >= 0
