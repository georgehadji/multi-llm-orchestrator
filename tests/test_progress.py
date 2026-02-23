# tests/test_progress.py
from orchestrator.progress import ProgressRenderer
from orchestrator.streaming import (
    ProjectStarted, TaskStarted, TaskProgressUpdate,
    TaskCompleted, TaskFailed, ProjectCompleted,
)
from orchestrator.models import TaskStatus


def test_renderer_tracks_task_counts():
    r = ProgressRenderer(quiet=True)
    r.handle(ProjectStarted("proj-1", total_tasks=3, budget_usd=5.0))
    r.handle(TaskStarted("task_001", "code_generation", "deepseek-chat"))
    r.handle(TaskCompleted("task_001", 0.92, TaskStatus.COMPLETED, "deepseek-chat", 0.001, 2))
    assert r.completed == 1
    assert r.total == 3


def test_renderer_tracks_failures():
    r = ProgressRenderer(quiet=True)
    r.handle(ProjectStarted("proj-1", total_tasks=2, budget_usd=5.0))
    r.handle(TaskFailed("task_001", "timeout", "deepseek-chat"))
    assert r.failed == 1


def test_renderer_summary_string():
    r = ProgressRenderer(quiet=True)
    r.handle(ProjectStarted("proj-1", total_tasks=2, budget_usd=5.0))
    r.handle(TaskCompleted("task_001", 0.9, TaskStatus.COMPLETED, "deepseek-chat", 0.001, 1))
    summary = r.summary()
    assert "1" in summary and "2" in summary   # 1 completed / 2 total


def test_renderer_multiple_completions():
    r = ProgressRenderer(quiet=True)
    r.handle(ProjectStarted("proj-1", total_tasks=3, budget_usd=5.0))
    r.handle(TaskCompleted("t1", 0.9, TaskStatus.COMPLETED, "deepseek-chat", 0.001, 1))
    r.handle(TaskCompleted("t2", 0.85, TaskStatus.DEGRADED, "deepseek-chat", 0.001, 1))
    r.handle(TaskFailed("t3", "timeout", "deepseek-chat"))
    assert r.completed == 2
    assert r.failed == 1
    assert r.total == 3


def test_renderer_handles_all_event_types_without_error():
    """ProgressRenderer should not raise on any event type."""
    from orchestrator.streaming import BudgetWarning
    r = ProgressRenderer(quiet=True)
    r.handle(ProjectStarted("proj-1", 5, 10.0))
    r.handle(TaskStarted("t1", "code_generation", "deepseek-chat"))
    r.handle(TaskProgressUpdate("t1", 1, 0.7, 0.7))
    r.handle(TaskCompleted("t1", 0.9, TaskStatus.COMPLETED, "deepseek-chat", 0.001, 1))
    r.handle(BudgetWarning("generation", 3.0, 5.0, 0.6))
    r.handle(ProjectCompleted("proj-1", "SUCCESS", 3.5, 120.0, 4, 1))
    # No assertion needed — just verify no exception
