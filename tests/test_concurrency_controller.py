"""Unit tests for orchestrator.concurrency_controller.TaskConcurrencyGuard."""

from __future__ import annotations

import asyncio

import pytest

from orchestrator.concurrency_controller import TaskConcurrencyGuard


# ─────────────────────────────────────────────────────────────────────────────
# Basic functionality
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_guard_acquires_and_releases():
    guard = TaskConcurrencyGuard(name="test", max_concurrent=2)
    async with guard:
        assert guard.active_count == 1
    assert guard.active_count == 0


@pytest.mark.asyncio
async def test_guard_tracks_total_acquired():
    guard = TaskConcurrencyGuard(name="test", max_concurrent=2)
    async with guard:
        pass
    async with guard:
        pass
    assert guard._total_acquired == 2


@pytest.mark.asyncio
async def test_guard_stats_structure():
    guard = TaskConcurrencyGuard(name="myguard", max_concurrent=3)
    async with guard:
        s = guard.stats()
        assert s["name"] == "myguard"
        assert s["max_concurrent"] == 3
        assert s["active"] == 1
        assert "total_acquired" in s
        assert "avg_wait_ms" in s


# ─────────────────────────────────────────────────────────────────────────────
# Serialization
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_guard_serializes_correctly():
    guard = TaskConcurrencyGuard(name="ser", max_concurrent=1)
    async with guard:
        pass
    s = guard.stats()
    assert s["total_acquired"] == 1
    assert s["avg_wait_ms"] >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Concurrency serialization (max_concurrent=1)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_guard_serializes_tasks():
    """With max_concurrent=1, tasks must run one at a time."""
    guard = TaskConcurrencyGuard(name="serial", max_concurrent=1)
    order: list[str] = []

    async def task(name: str) -> None:
        async with guard:
            order.append(f"start:{name}")
            await asyncio.sleep(0)
            order.append(f"end:{name}")

    await asyncio.gather(task("A"), task("B"))

    # Each task must fully complete before the next starts
    assert order.index("end:A") < order.index("start:B") or order.index("end:B") < order.index("start:A")


@pytest.mark.asyncio
async def test_guard_allows_parallel_with_higher_concurrency():
    """With max_concurrent=2, two tasks can overlap."""
    guard = TaskConcurrencyGuard(name="parallel", max_concurrent=2)
    started: list[str] = []
    barrier = asyncio.Event()

    async def task(name: str) -> None:
        async with guard:
            started.append(name)
            await barrier.wait()

    t1 = asyncio.create_task(task("A"))
    t2 = asyncio.create_task(task("B"))
    # Give both tasks a chance to acquire the guard
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert len(started) == 2  # both running concurrently
    barrier.set()
    await asyncio.gather(t1, t2)


# ─────────────────────────────────────────────────────────────────────────────
# Exception safety
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_guard_releases_on_exception():
    guard = TaskConcurrencyGuard(name="exc", max_concurrent=1)
    with pytest.raises(RuntimeError):
        async with guard:
            raise RuntimeError("boom")
    assert guard.active_count == 0

    # Guard is still usable after exception
    async with guard:
        assert guard.active_count == 1
    assert guard.active_count == 0


# ─────────────────────────────────────────────────────────────────────────────
# Integration: ExecutorService + guard
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_executor_service_uses_guard():
    """ExecutorService with guard=TaskConcurrencyGuard should still execute tasks."""
    from unittest.mock import AsyncMock

    from orchestrator.models import Model, Task, TaskResult, TaskStatus, TaskType
    from orchestrator.services.executor import ExecutorService

    guard = TaskConcurrencyGuard(name="exec-guard", max_concurrent=1)

    task = Task(
        id="t1",
        type=TaskType.CODE_GEN,
        prompt="hello",
        context="",
        dependencies=[],
    )
    mock_result = TaskResult(
        task_id="t1",
        output="done",
        score=0.9,
        model_used=Model.GPT_4O_MINI,
        status=TaskStatus.COMPLETED,
        task_type=TaskType.CODE_GEN.value,
        critique="",
        iterations=1,
        cost_usd=0.001,
        tokens_used={"input": 10, "output": 5},
    )
    execute_fn = AsyncMock(return_value=mock_result)
    svc = ExecutorService(execute_fn=execute_fn, guard=guard)

    result = await svc.execute(task)

    assert result.succeeded
    assert guard._total_acquired == 1
