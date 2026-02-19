"""
Tests for App Builder extensions to the Task model (Task 1).
"""
from orchestrator.models import Task, TaskType


def test_task_has_target_path_default_empty():
    """Task.target_path must default to empty string."""
    t = Task(id="t1", type=TaskType.CODE_GEN, prompt="write code")
    assert t.target_path == ""


def test_task_has_module_name_default_empty():
    """Task.module_name must default to empty string."""
    t = Task(id="t1", type=TaskType.CODE_GEN, prompt="write code")
    assert t.module_name == ""


def test_task_target_path_can_be_set():
    """Both fields can be set at construction time."""
    t = Task(
        id="t1",
        type=TaskType.CODE_GEN,
        prompt="p",
        target_path="src/routes/auth.py",
        module_name="src.routes.auth",
    )
    assert t.target_path == "src/routes/auth.py"
    assert t.module_name == "src.routes.auth"
