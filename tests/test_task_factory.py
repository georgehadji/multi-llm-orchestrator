"""Unit tests for orchestrator.task_factory — TDD RED phase."""

from __future__ import annotations

import pytest

from orchestrator.task_factory import TaskFactory
from orchestrator.models import Task, TaskType

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make(task_type: TaskType = TaskType.CODE_GEN, **kwargs) -> Task:
    return TaskFactory.create(id="t1", task_type=task_type, prompt="do it", **kwargs)


# ── Return type ───────────────────────────────────────────────────────────────


class TestTaskFactoryCreate:
    def test_returns_task_instance(self):
        task = _make()
        assert isinstance(task, Task)

    def test_id_preserved(self):
        task = TaskFactory.create(id="my-id", task_type=TaskType.CODE_GEN, prompt="p")
        assert task.id == "my-id"

    def test_prompt_preserved(self):
        task = TaskFactory.create(id="t", task_type=TaskType.CODE_GEN, prompt="Build X")
        assert task.prompt == "Build X"

    def test_type_preserved(self):
        task = _make(task_type=TaskType.EVALUATE)
        assert task.type == TaskType.EVALUATE


# ── Defaults for each task type ───────────────────────────────────────────────


class TestCodeGenDefaults:
    def test_acceptance_threshold(self):
        task = _make(TaskType.CODE_GEN)
        assert task.acceptance_threshold == pytest.approx(0.85)

    def test_max_iterations(self):
        task = _make(TaskType.CODE_GEN)
        assert task.max_iterations == 3

    def test_max_output_tokens(self):
        task = _make(TaskType.CODE_GEN)
        assert task.max_output_tokens == 8192


class TestCodeReviewDefaults:
    def test_acceptance_threshold(self):
        task = _make(TaskType.CODE_REVIEW)
        assert task.acceptance_threshold == pytest.approx(0.75)

    def test_max_output_tokens(self):
        task = _make(TaskType.CODE_REVIEW)
        assert task.max_output_tokens == 4096


class TestReasoningDefaults:
    def test_acceptance_threshold(self):
        task = _make(TaskType.REASONING)
        assert task.acceptance_threshold == pytest.approx(0.90)

    def test_max_iterations(self):
        task = _make(TaskType.REASONING)
        assert task.max_iterations == 3


class TestEvaluateDefaults:
    def test_acceptance_threshold(self):
        task = _make(TaskType.EVALUATE)
        assert task.acceptance_threshold == pytest.approx(0.80)

    def test_max_output_tokens(self):
        task = _make(TaskType.EVALUATE)
        assert task.max_output_tokens == 2048


# ── Optional kwargs ───────────────────────────────────────────────────────────


class TestOptionalFields:
    def test_dependencies_default_empty(self):
        task = _make()
        assert task.dependencies == []

    def test_dependencies_passed(self):
        task = _make(dependencies=["t0"])
        assert task.dependencies == ["t0"]

    def test_hard_validators_default_empty(self):
        task = _make()
        assert task.hard_validators == []

    def test_hard_validators_passed(self):
        task = _make(hard_validators=["ruff", "python_syntax"])
        assert task.hard_validators == ["ruff", "python_syntax"]

    def test_target_path_default_empty(self):
        task = _make()
        assert task.target_path == ""

    def test_target_path_passed(self):
        task = _make(target_path="src/api.py")
        assert task.target_path == "src/api.py"

    def test_context_default_empty(self):
        task = _make()
        assert task.context == ""

    def test_context_passed(self):
        task = _make(context="FastAPI project")
        assert task.context == "FastAPI project"

    def test_tech_context_passed(self):
        task = _make(tech_context="Python 3.12")
        assert task.tech_context == "Python 3.12"

    def test_module_name_passed(self):
        task = _make(module_name="src.api")
        assert task.module_name == "src.api"
