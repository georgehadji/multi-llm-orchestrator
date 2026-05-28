"""
Unit tests for orchestrator.output.code_cleaner.clean_code_output

P3-7 of REFACTORING_PLAN_V7.md — extracted from engine.py as a pure function.
"""

import pytest

from orchestrator.output.code_cleaner import clean_code_output
from orchestrator.models import TaskType


def test_non_code_gen_returns_unchanged():
    """Non-CODE_GEN tasks pass through untouched."""
    text = "  Some text with `markdown` and\n\n```python\ncode\n```  "
    assert clean_code_output(text, TaskType.WRITING) == text
    assert clean_code_output(text, TaskType.SUMMARIZE) == text


def test_removes_backtick_fences():
    text = "```python\ndef foo():\n    pass\n```"
    result = clean_code_output(text, TaskType.CODE_GEN)
    assert "```" not in result
    assert "def foo():" in result


def test_removes_fenced_block_with_no_language():
    text = "```\nsome code\n```"
    result = clean_code_output(text, TaskType.CODE_GEN)
    assert "```" not in result
    assert "some code" in result


def test_removes_placeholder_comments_cpp_style():
    text = "int x = 1;\n// Add content here\nint y = 2;\n"
    result = clean_code_output(text, TaskType.CODE_GEN)
    assert "Add content" not in result
    assert "int x = 1;" in result
    assert "int y = 2;" in result


def test_removes_todo_comments():
    """C++ style TODO is stripped; Python # style is not (original behaviour)."""
    text = "int x = 0;\n// TODO: implement this\nint y = 0;\n"
    result = clean_code_output(text, TaskType.CODE_GEN)
    assert "TODO:" not in result
    assert "int x = 0;" in result


def test_collapses_multiple_blank_lines():
    text = "line1\n\n\n\nline2"
    result = clean_code_output(text, TaskType.CODE_GEN)
    assert "\n\n\n" not in result
    assert "line1" in result
    assert "line2" in result


def test_strips_leading_trailing_whitespace():
    text = "  \n\ndef foo(): pass\n\n  "
    result = clean_code_output(text, TaskType.CODE_GEN)
    assert result == result.strip()


def test_empty_input_returns_empty():
    assert clean_code_output("", TaskType.CODE_GEN) == ""


def test_pure_code_unchanged_except_strip():
    code = "def foo():\n    return 42"
    result = clean_code_output(code, TaskType.CODE_GEN)
    assert result == code
