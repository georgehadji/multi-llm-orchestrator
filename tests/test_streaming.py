# tests/test_streaming.py
import asyncio
from orchestrator.streaming import (
    TaskStarted, TaskProgressUpdate, TaskCompleted, TaskFailed,
    ProjectStarted, ProjectCompleted, BudgetWarning, ProjectEventBus,
)
from orchestrator.models import Model, TaskStatus


def test_task_started_fields():
    ev = TaskStarted(task_id="t1", task_type="code_generation", model="deepseek-chat")
    assert ev.task_id == "t1"
    assert ev.task_type == "code_generation"
    assert ev.model == "deepseek-chat"


def test_task_progress_update_fields():
    ev = TaskProgressUpdate(task_id="t1", iteration=2, score=0.82, best_score=0.90)
    assert ev.iteration == 2
    assert ev.score == 0.82
    assert ev.best_score == 0.90


def test_task_completed_fields():
    ev = TaskCompleted(task_id="t1", score=0.95, status=TaskStatus.COMPLETED,
                       model="deepseek-chat", cost_usd=0.002, iterations=2)
    assert ev.score == 0.95
    assert ev.status == TaskStatus.COMPLETED


def test_task_failed_fields():
    ev = TaskFailed(task_id="t1", reason="timeout after 120s", model="deepseek-chat")
    assert ev.reason == "timeout after 120s"


def test_project_event_bus_subscribe_and_publish():
    bus = ProjectEventBus()
    received = []

    async def run():
        sub = bus.subscribe()
        await bus.publish(TaskStarted("t1", "code_generation", "deepseek-chat"))
        await bus.publish(TaskCompleted("t1", 0.9, TaskStatus.COMPLETED, "deepseek-chat", 0.001, 1))
        await bus.close()
        async for ev in sub:
            received.append(ev)

    asyncio.run(run())
    assert len(received) == 2
    assert isinstance(received[0], TaskStarted)
    assert isinstance(received[1], TaskCompleted)


def test_event_bus_multiple_subscribers():
    bus = ProjectEventBus()
    recv_a, recv_b = [], []

    async def run():
        sub_a = bus.subscribe()
        sub_b = bus.subscribe()
        await bus.publish(TaskStarted("t1", "code_generation", "deepseek-chat"))
        await bus.close()
        async for ev in sub_a:
            recv_a.append(ev)
        async for ev in sub_b:
            recv_b.append(ev)

    asyncio.run(run())
    assert len(recv_a) == 1
    assert len(recv_b) == 1
