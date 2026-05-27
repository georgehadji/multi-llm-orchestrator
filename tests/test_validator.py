"""
Tests for the TaskValidator module (extracted from engine.py Phase 3).
==========================================================================
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.engine_core.validator import TaskValidator
from orchestrator.models import Model, Task, TaskType


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.call = AsyncMock()
    return client


@pytest.fixture
def mock_budget():
    budget = MagicMock()
    budget.charge = AsyncMock()
    return budget


@pytest.fixture
def validator(mock_client, mock_budget):
    return TaskValidator(client=mock_client, budget=mock_budget)


class TestSyntaxValidation:
    """Test _validate_syntax_streaming and _validate_syntax_batch."""

    VALID_PYTHON = "def hello():\\n    return 'world'\\n"
    INVALID_PYTHON = "def hello(:\\n    return 'world'\\n"

    def test_streaming_valid(self, validator):
        assert validator.validate_syntax_streaming("print('hello')") is True

    def test_streaming_unbalanced_brackets(self, validator):
        assert validator.validate_syntax_streaming("print('hello'") is False

    def test_syntax_batch_valid(self, validator):
        assert validator.validate_syntax_batch("""x = 1
y = x + 2""") is True

    def test_syntax_batch_invalid(self, validator):
        assert validator.validate_syntax_batch("x = ") is False

    def test_syntax_batch_empty(self, validator):
        assert validator.validate_syntax_batch("") is False

    def test_syntax_batch_with_fences(self, validator):
        fenced = "```python\nx = 1\n```"
        assert validator.validate_syntax_batch(fenced) is True


class TestFilterValidators:
    """Test filter_validators_for_task."""

    def make_task(self, hard_validators=None):
        return Task(
            id="test_task",
            type=TaskType.CODE_GEN,
            prompt="Write code",
            hard_validators=hard_validators or ["python_syntax", "pytest"],
        )

    def test_python_task_preserves_validators(self, validator):
        task = self.make_task()
        result = validator.filter_validators_for_task(task, "import os\\nprint('hi')")
        assert "python_syntax" in result

    def test_non_python_task_filters(self, validator):
        task = Task(
            id="test_task",
            type=TaskType.WRITING,
            prompt="Write HTML",
            hard_validators=["python_syntax", "pytest", "json_schema"],
        )
        result = validator.filter_validators_for_task(task, "<html></html>")
        assert "python_syntax" not in result
        assert "json_schema" in result
