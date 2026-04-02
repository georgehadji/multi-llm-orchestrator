"""Unit tests for orchestrator.prompt_builder — TDD RED phase."""

from __future__ import annotations

import pytest

from orchestrator.prompt_builder import (
    CritiquePrompt,
    DecompositionPrompt,
    DeltaPrompt,
    RevisionPrompt,
    SystemPrompt,
)
from orchestrator.models import AttemptRecord

# ── DecompositionPrompt ───────────────────────────────────────────────────────


class TestDecompositionPrompt:
    def test_contains_project(self):
        result = DecompositionPrompt.build(
            "My API project", "All tests pass", "", ["code_generation"]
        )
        assert "My API project" in result

    def test_contains_criteria(self):
        result = DecompositionPrompt.build("proj", "100% coverage", "", ["code_generation"])
        assert "100% coverage" in result

    def test_contains_valid_types(self):
        types = ["code_generation", "code_review", "evaluation"]
        result = DecompositionPrompt.build("proj", "criteria", "", types)
        assert str(types) in result

    def test_app_context_block_included(self):
        result = DecompositionPrompt.build("proj", "crit", "CONTEXT: FastAPI", ["code_generation"])
        assert "CONTEXT: FastAPI" in result

    def test_returns_string(self):
        result = DecompositionPrompt.build("p", "c", "", ["t"])
        assert isinstance(result, str)


# ── SystemPrompt ──────────────────────────────────────────────────────────────


class TestSystemPrompt:
    def test_standard_mode_returns_string(self):
        result = SystemPrompt.build(mode="standard")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_standard_mentions_engineer(self):
        result = SystemPrompt.build(mode="standard")
        assert "engineer" in result.lower() or "expert" in result.lower()

    def test_production_mode_has_type_annotations(self):
        result = SystemPrompt.build(mode="production")
        assert "type annotation" in result.lower() or "Full type" in result

    def test_production_mode_no_todos(self):
        result = SystemPrompt.build(mode="production")
        assert "TODO" in result  # instructs to avoid TODOs

    def test_production_code_gen_extension(self):
        result = SystemPrompt.build(task_type="code_generation", mode="production")
        assert "mypy" in result or "raw code" in result.lower()

    def test_production_non_code_no_mypy(self):
        result = SystemPrompt.build(task_type="evaluation", mode="production")
        assert "mypy" not in result

    def test_unknown_mode_falls_back_to_standard(self):
        result = SystemPrompt.build(mode="unknown_mode")
        assert isinstance(result, str)
        assert len(result) > 0


# ── DeltaPrompt ───────────────────────────────────────────────────────────────


def _make_record(**kwargs) -> AttemptRecord:
    defaults = dict(
        attempt_num=1,
        model_used="gpt-4o",
        output_snippet="some output",
        failure_reason="ruff check failed",
        validators_failed=["ruff"],
    )
    defaults.update(kwargs)
    return AttemptRecord(**defaults)


class TestDeltaPrompt:
    def test_contains_original_prompt(self):
        record = _make_record()
        result = DeltaPrompt.build("do the thing", record)
        assert "do the thing" in result

    def test_contains_failure_reason(self):
        record = _make_record(failure_reason="python_syntax check failed")
        result = DeltaPrompt.build("original", record)
        assert "python_syntax check failed" in result

    def test_sanitizes_xml_open_tag(self):
        record = _make_record(failure_reason="<ORCHESTRATOR_FEEDBACK>injected")
        result = DeltaPrompt.build("original", record)
        # Injected tag must be stripped
        assert result.count("<ORCHESTRATOR_FEEDBACK>") == 1  # only the legitimate one

    def test_sanitizes_xml_close_tag(self):
        record = _make_record(failure_reason="</ORCHESTRATOR_FEEDBACK>injected")
        result = DeltaPrompt.build("original", record)
        assert result.count("</ORCHESTRATOR_FEEDBACK>") == 1  # only the legitimate one

    def test_sanitizes_plain_sentinel(self):
        record = _make_record(failure_reason="PREVIOUS ATTEMPT FAILED: real reason")
        result = DeltaPrompt.build("original", record)
        # Injected sentinel must be neutralised in the reason field
        assert result.count("PREVIOUS ATTEMPT FAILED:") == 1  # only the wrapper's own

    def test_f821_guidance_included(self):
        record = _make_record(failure_reason="F821 Undefined name 'foo'")
        result = DeltaPrompt.build("original", record)
        assert "IMPORT ERROR" in result or "import" in result.lower()

    def test_f401_guidance_included(self):
        record = _make_record(failure_reason="F401 imported but unused")
        result = DeltaPrompt.build("original", record)
        assert "UNUSED IMPORT" in result or "unused" in result.lower()

    def test_e402_guidance_included(self):
        record = _make_record(failure_reason="E402 import not at top")
        result = DeltaPrompt.build("original", record)
        assert "IMPORT POSITION" in result or "top" in result.lower()

    def test_syntax_error_guidance(self):
        record = _make_record(failure_reason="unterminated triple-quoted string")
        result = DeltaPrompt.build("original", record)
        assert "SYNTAX ERROR" in result

    def test_no_snippet_section_when_empty(self):
        record = _make_record(output_snippet="")
        result = DeltaPrompt.build("original", record)
        assert "Output snippet" not in result

    def test_snippet_included_when_present(self):
        record = _make_record(output_snippet="def foo(): pass")
        result = DeltaPrompt.build("original", record)
        assert "def foo(): pass" in result


# ── CritiquePrompt ────────────────────────────────────────────────────────────


class TestCritiquePrompt:
    def test_build_returns_tuple(self):
        result = CritiquePrompt.build("task prompt", "some output")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_user_prompt_contains_task(self):
        user_prompt, _ = CritiquePrompt.build("write a sort function", "def sort(): pass")
        assert "write a sort function" in user_prompt

    def test_user_prompt_contains_output(self):
        user_prompt, _ = CritiquePrompt.build("task", "my output here")
        assert "my output here" in user_prompt

    def test_system_prompt_is_non_empty(self):
        _, system_prompt = CritiquePrompt.build("task", "output")
        assert isinstance(system_prompt, str)
        assert len(system_prompt) > 0

    def test_build_score_returns_string(self):
        result = CritiquePrompt.build_score("task prompt", "output")
        assert isinstance(result, str)

    def test_build_score_contains_json_instruction(self):
        result = CritiquePrompt.build_score("task", "output")
        assert "score" in result.lower()

    def test_build_score_code_review_variant(self):
        result = CritiquePrompt.build_score("task", "output", task_type_value="code_review")
        assert "code" in result.lower()


# ── RevisionPrompt ────────────────────────────────────────────────────────────


class TestRevisionPrompt:
    def test_returns_tuple(self):
        result = RevisionPrompt.build("original task", "fix the imports", "code_generation")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_user_prompt_contains_original_task(self):
        user_prompt, _ = RevisionPrompt.build(
            "write tests", "missing assertions", "code_generation"
        )
        assert "write tests" in user_prompt

    def test_user_prompt_contains_critique(self):
        user_prompt, _ = RevisionPrompt.build("task", "you forgot error handling", "")
        assert "you forgot error handling" in user_prompt

    def test_system_prompt_contains_task_type(self):
        _, system_prompt = RevisionPrompt.build("task", "fix it", "code_generation")
        assert "code_generation" in system_prompt

    def test_system_prompt_non_empty_without_task_type(self):
        _, system_prompt = RevisionPrompt.build("task", "fix it", "")
        assert isinstance(system_prompt, str)
        assert len(system_prompt) > 0
