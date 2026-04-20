"""Unit tests for orchestrator.services.executor.ExecutorService."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.exceptions import TaskError, TaskTimeoutError
from orchestrator.models import Model, Task, TaskResult, TaskStatus, TaskType
from orchestrator.services.executor import ExecutorResult, ExecutorService


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


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


def _failed_result(task_id: str = "t1") -> TaskResult:
    return TaskResult(
        task_id=task_id,
        output="",
        score=0.0,
        model_used=Model.GPT_4O_MINI,
        status=TaskStatus.FAILED,
        task_type=TaskType.CODE_GEN.value,
        critique="failed",
        iterations=0,
        cost_usd=0.0,
        tokens_used={"input": 0, "output": 0},
    )


# ─────────────────────────────────────────────────────────────────────────────
# ExecutorResult properties
# ─────────────────────────────────────────────────────────────────────────────


def test_executor_result_succeeded_when_completed():
    r = ExecutorResult(task_result=_ok_result(), wall_time_ms=100.0)
    assert r.succeeded is True


def test_executor_result_not_succeeded_when_failed():
    r = ExecutorResult(task_result=_failed_result(), wall_time_ms=50.0)
    assert r.succeeded is False


def test_executor_result_not_succeeded_when_error_set():
    r = ExecutorResult(
        task_result=_failed_result(),
        wall_time_ms=50.0,
        error=RuntimeError("boom"),
    )
    assert r.succeeded is False


# ─────────────────────────────────────────────────────────────────────────────
# Happy path
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_execute_returns_result_on_success():
    fn = AsyncMock(return_value=_ok_result())
    svc = ExecutorService(execute_fn=fn)

    result = await svc.execute(_task())

    assert result.succeeded
    assert result.task_result.status == TaskStatus.COMPLETED
    assert result.wall_time_ms >= 0
    assert result.error is None
    fn.assert_called_once()


@pytest.mark.asyncio
async def test_execute_measures_wall_time():
    async def slow_fn(_task):
        await asyncio.sleep(0.05)
        return _ok_result()

    svc = ExecutorService(execute_fn=slow_fn)
    result = await svc.execute(_task())
    assert result.wall_time_ms >= 40  # at least 40ms


# ─────────────────────────────────────────────────────────────────────────────
# Error normalization — never raises
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_execute_normalizes_generic_exception():
    async def boom(_task):
        raise ValueError("unexpected boom")

    svc = ExecutorService(execute_fn=boom)
    result = await svc.execute(_task())

    assert not result.succeeded
    assert result.error is not None
    assert isinstance(result.error, TaskError)
    assert result.task_result.status == TaskStatus.FAILED


@pytest.mark.asyncio
async def test_execute_normalizes_task_error():
    async def boom(_task):
        raise TaskError("explicit task failure")

    svc = ExecutorService(execute_fn=boom)
    result = await svc.execute(_task())

    assert isinstance(result.error, TaskError)
    assert result.task_result.status == TaskStatus.FAILED


@pytest.mark.asyncio
async def test_execute_normalizes_timeout():
    async def slow(_task):
        await asyncio.sleep(10)
        return _ok_result()

    svc = ExecutorService(execute_fn=slow, task_timeout=0.05)
    result = await svc.execute(_task())

    assert isinstance(result.error, TaskTimeoutError)
    assert result.task_result.status == TaskStatus.FAILED


# ─────────────────────────────────────────────────────────────────────────────
# Metrics accumulation
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_metrics_increment_on_success():
    fn = AsyncMock(return_value=_ok_result())
    svc = ExecutorService(execute_fn=fn)

    await svc.execute(_task("t1"))
    await svc.execute(_task("t2"))

    m = svc.metrics_snapshot()
    assert m["total_submitted"] == 2
    assert m["total_completed"] == 2
    assert m["total_failed"] == 0


@pytest.mark.asyncio
async def test_metrics_increment_on_failure():
    async def boom(_task):
        raise RuntimeError("fail")

    svc = ExecutorService(execute_fn=boom)
    await svc.execute(_task("t1"))

    m = svc.metrics_snapshot()
    assert m["total_submitted"] == 1
    assert m["total_failed"] == 1
    assert m["total_completed"] == 0


@pytest.mark.asyncio
async def test_metrics_avg_wall_ms_computed():
    fn = AsyncMock(return_value=_ok_result())
    svc = ExecutorService(execute_fn=fn)

    await svc.execute(_task("t1"))
    await svc.execute(_task("t2"))

    m = svc.metrics_snapshot()
    assert m["avg_wall_ms"] >= 0


# ─────────────────────────────────────────────────────────────────────────────
# Concurrent safety
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_concurrent_executions_all_recorded():
    call_count = 0

    async def counted(_task):
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.01)
        return _ok_result(_task.id)

    svc = ExecutorService(execute_fn=counted)
    tasks = [_task(f"t{i}") for i in range(10)]
    results = await asyncio.gather(*[svc.execute(t) for t in tasks])

    assert all(r.succeeded for r in results)
    assert svc.metrics.total_submitted == 10
    assert svc.metrics.total_completed == 10
    assert call_count == 10
