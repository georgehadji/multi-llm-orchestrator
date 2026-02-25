"""
Tests — Improvement 8: Failure History + Delta-Prompt Retries
=============================================================
Covers:
  - AttemptRecord dataclass construction and field defaults
  - _build_delta_prompt: content, structure, validators_str formatting
  - attempt_history populated in TaskResult after failed iterations
  - TASK_RETRY_WITH_HISTORY hook fires on failure
  - State serialization round-trip preserves attempt_history
  - summary.json includes attempt_history for each task
"""
from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from orchestrator.api_clients import APIResponse
from orchestrator.engine import Orchestrator
from orchestrator.hooks import EventType
from orchestrator.models import (
    AttemptRecord, Budget, Model, ProjectState, ProjectStatus,
    Task, TaskStatus, TaskType, build_default_profiles,
)
from orchestrator.output_writer import write_output_dir


# ─── helpers ─────────────────────────────────────────────────────────────────

def _api_response(text: str, model: Model = Model.GEMINI_FLASH) -> APIResponse:
    resp = APIResponse(text=text, input_tokens=50, output_tokens=100, model=model)
    resp.latency_ms = 50.0
    return resp


def _score_json(score: float, reasoning: str = "ok") -> str:
    return json.dumps({"score": score, "reasoning": reasoning})


def _decomp_json(tasks: list[dict]) -> str:
    return json.dumps(tasks)


def _run(coro):
    return asyncio.run(coro)


def _make_orch() -> Orchestrator:
    orch = Orchestrator(budget=Budget(max_usd=5.0, max_time_seconds=300), max_concurrency=1)
    for m in Model:
        orch.api_health[m] = True
    return orch


_SINGLE_TASK = [
    {
        "id": "t1",
        "type": "code_generation",
        "prompt": "write code",
        "acceptance_threshold": 0.8,
        "max_iterations": 3,
        "dependencies": [],
        "priority": 0,
    }
]


# ─── AttemptRecord dataclass ─────────────────────────────────────────────────

class TestAttemptRecord:
    def test_required_fields(self):
        r = AttemptRecord(
            attempt_num=1,
            model_used="gpt-4o",
            output_snippet="hello",
            failure_reason="bad score",
        )
        assert r.attempt_num == 1
        assert r.model_used == "gpt-4o"
        assert r.output_snippet == "hello"
        assert r.failure_reason == "bad score"

    def test_validators_failed_defaults_empty(self):
        r = AttemptRecord(attempt_num=1, model_used="x", output_snippet="x", failure_reason="x")
        assert r.validators_failed == []

    def test_validators_failed_stored(self):
        r = AttemptRecord(
            attempt_num=2,
            model_used="gpt-4o",
            output_snippet="x",
            failure_reason="fail",
            validators_failed=["python_syntax", "json_valid"],
        )
        assert r.validators_failed == ["python_syntax", "json_valid"]


# ─── _build_delta_prompt ─────────────────────────────────────────────────────

class TestBuildDeltaPrompt:
    def _orch(self) -> Orchestrator:
        return _make_orch()

    def test_original_prompt_preserved(self):
        orch = self._orch()
        record = AttemptRecord(attempt_num=1, model_used="m", output_snippet="x",
                               failure_reason="bad score")
        result = orch._build_delta_prompt("Do something cool", record)
        assert result.startswith("Do something cool")

    def test_failure_section_included(self):
        orch = self._orch()
        record = AttemptRecord(attempt_num=1, model_used="gpt-4o", output_snippet="x",
                               failure_reason="Score 0.400 below threshold 0.7")
        result = orch._build_delta_prompt("original", record)
        assert "PREVIOUS ATTEMPT FAILED" in result
        assert "gpt-4o" in result
        assert "Score 0.400 below threshold 0.7" in result

    def test_no_validators_shows_none(self):
        orch = self._orch()
        record = AttemptRecord(attempt_num=1, model_used="m", output_snippet="x",
                               failure_reason="fail", validators_failed=[])
        result = orch._build_delta_prompt("original", record)
        assert "Validators failed: none" in result

    def test_validators_listed(self):
        orch = self._orch()
        record = AttemptRecord(attempt_num=1, model_used="m", output_snippet="x",
                               failure_reason="fail",
                               validators_failed=["python_syntax", "json_valid"])
        result = orch._build_delta_prompt("original", record)
        assert "python_syntax" in result
        assert "json_valid" in result

    def test_correction_instruction_included(self):
        orch = self._orch()
        record = AttemptRecord(attempt_num=1, model_used="m", output_snippet="x",
                               failure_reason="missing docstring")
        result = orch._build_delta_prompt("original", record)
        assert "Please correct specifically" in result
        assert "missing docstring" in result


# ─── Attempt history in TaskResult via engine ─────────────────────────────────

class TestAttemptHistoryInEngine:
    def test_failed_attempts_recorded(self):
        """When a task fails twice then succeeds, attempt_history has 2 entries."""
        orch = _make_orch()
        scores = [0.3, 0.4, 0.9]
        score_iter = iter(scores)

        async def fake_call(model, prompt, **kwargs):
            system = kwargs.get("system", "").lower()
            if "decomposition" in system:
                return _api_response(_decomp_json(_SINGLE_TASK), model)
            if "evaluator" in system or "score" in prompt.lower():
                return _api_response(_score_json(next(score_iter, 0.9)), model)
            return _api_response("def foo(): pass", model)

        async def _do_run():
            with patch.object(orch.client, "call", side_effect=fake_call):
                return await orch.run_project("project", "criteria")

        state = _run(_do_run())
        result = state.results.get("t1")
        assert result is not None
        # At least one failed iteration should have been recorded
        assert len(result.attempt_history) >= 1
        assert result.attempt_history[0].attempt_num == 1

    def test_no_history_on_first_success(self):
        """When a task succeeds on the first iteration, attempt_history is empty."""
        orch = _make_orch()

        async def fake_call(model, prompt, **kwargs):
            system = kwargs.get("system", "").lower()
            if "decomposition" in system:
                return _api_response(_decomp_json(_SINGLE_TASK), model)
            if "evaluator" in system or "score" in prompt.lower():
                return _api_response(_score_json(0.95), model)
            return _api_response("def foo():\n    return 42", model)

        async def _do_run():
            with patch.object(orch.client, "call", side_effect=fake_call):
                return await orch.run_project("project", "criteria")

        state = _run(_do_run())
        result = state.results.get("t1")
        assert result is not None
        assert result.attempt_history == []

    def test_failure_reason_contains_score(self):
        """The failure reason in AttemptRecord mentions the actual score."""
        orch = _make_orch()
        scores = [0.45, 0.9]
        score_iter = iter(scores)

        async def fake_call(model, prompt, **kwargs):
            system = kwargs.get("system", "").lower()
            if "decomposition" in system:
                return _api_response(_decomp_json(_SINGLE_TASK), model)
            if "evaluator" in system or "score" in prompt.lower():
                return _api_response(_score_json(next(score_iter, 0.9)), model)
            return _api_response("stub code", model)

        async def _do_run():
            with patch.object(orch.client, "call", side_effect=fake_call):
                return await orch.run_project("project", "criteria")

        state = _run(_do_run())
        result = state.results.get("t1")
        assert result is not None
        assert len(result.attempt_history) >= 1
        record = result.attempt_history[0]
        assert "0.45" in record.failure_reason or "0.450" in record.failure_reason

    def test_delta_prompt_injected_on_retry(self):
        """On retry, the prompt passed to the model contains PREVIOUS ATTEMPT FAILED."""
        orch = _make_orch()
        scores = [0.3, 0.9]
        score_iter = iter(scores)
        received_prompts: list[str] = []

        async def fake_call(model, prompt, **kwargs):
            received_prompts.append(prompt)
            system = kwargs.get("system", "").lower()
            if "decomposition" in system:
                return _api_response(_decomp_json(_SINGLE_TASK), model)
            if "evaluator" in system or "score" in prompt.lower():
                return _api_response(_score_json(next(score_iter, 0.9)), model)
            return _api_response("some code", model)

        async def _do_run():
            with patch.object(orch.client, "call", side_effect=fake_call):
                return await orch.run_project("project", "criteria")

        _run(_do_run())

        # Find generate prompts that contain the task prompt text
        generate_prompts = [p for p in received_prompts if "write code" in p]
        retry_prompts = [p for p in generate_prompts if "PREVIOUS ATTEMPT FAILED" in p]
        assert len(retry_prompts) >= 1


# ─── TASK_RETRY_WITH_HISTORY hook ────────────────────────────────────────────

class TestRetryWithHistoryHook:
    def test_hook_fires_on_failure(self):
        """TASK_RETRY_WITH_HISTORY fires once per failed iteration."""
        orch = _make_orch()
        fired_events: list[dict] = []
        orch.add_hook(
            EventType.TASK_RETRY_WITH_HISTORY,
            lambda **kw: fired_events.append(kw),
        )

        scores = [0.3, 0.9]
        score_iter = iter(scores)

        async def fake_call(model, prompt, **kwargs):
            system = kwargs.get("system", "").lower()
            if "decomposition" in system:
                return _api_response(_decomp_json(_SINGLE_TASK), model)
            if "evaluator" in system or "score" in prompt.lower():
                return _api_response(_score_json(next(score_iter, 0.9)), model)
            return _api_response("some code", model)

        async def _do_run():
            with patch.object(orch.client, "call", side_effect=fake_call):
                return await orch.run_project("project", "criteria")

        _run(_do_run())

        assert len(fired_events) == 1
        assert fired_events[0]["task_id"] == "t1"
        assert fired_events[0]["attempt_num"] == 1
        assert isinstance(fired_events[0]["record"], AttemptRecord)


# ─── State serialization round-trip ──────────────────────────────────────────

class TestAttemptHistorySerialization:
    def test_attempt_history_round_trips_through_state(self):
        """AttemptRecord survives StateManager serialize → deserialize."""
        from orchestrator.state import _result_to_dict, _result_from_dict
        from orchestrator.models import TaskResult

        record = AttemptRecord(
            attempt_num=1,
            model_used="gpt-4o",
            output_snippet="def foo(): pass",
            failure_reason="Score 0.400 below threshold 0.7",
            validators_failed=["python_syntax"],
        )
        result = TaskResult(
            task_id="t1",
            output="final output",
            score=0.9,
            model_used=Model.GPT_4O,
            status=TaskStatus.COMPLETED,
            attempt_history=[record],
        )

        data = _result_to_dict(result)
        restored = _result_from_dict(data)

        assert len(restored.attempt_history) == 1
        r = restored.attempt_history[0]
        assert r.attempt_num == 1
        assert r.model_used == "gpt-4o"
        assert r.output_snippet == "def foo(): pass"
        assert r.failure_reason == "Score 0.400 below threshold 0.7"
        assert r.validators_failed == ["python_syntax"]

    def test_empty_attempt_history_round_trips(self):
        from orchestrator.state import _result_to_dict, _result_from_dict
        from orchestrator.models import TaskResult

        result = TaskResult(
            task_id="t1",
            output="ok",
            score=1.0,
            model_used=Model.GPT_4O,
            status=TaskStatus.COMPLETED,
        )
        data = _result_to_dict(result)
        restored = _result_from_dict(data)
        assert restored.attempt_history == []


# ─── summary.json includes attempt_history ───────────────────────────────────

class TestAttemptHistoryInSummaryJson:
    def test_attempt_history_written_to_summary(self):
        """write_output_dir includes attempt_history for each task."""
        from orchestrator.models import ProjectState, ProjectStatus, TaskResult, TaskStatus

        record = AttemptRecord(
            attempt_num=1,
            model_used="gpt-4o",
            output_snippet="stub",
            failure_reason="bad score",
            validators_failed=["python_syntax"],
        )
        task = Task(
            id="t1",
            type=TaskType.CODE_GEN,
            prompt="write something",
            acceptance_threshold=0.7,
            max_iterations=3,
        )
        result = TaskResult(
            task_id="t1",
            output="final code",
            score=0.9,
            model_used=Model.GPT_4O,
            status=TaskStatus.COMPLETED,
            attempt_history=[record],
        )
        state = ProjectState(
            project_description="test",
            success_criteria="pass",
            status=ProjectStatus.SUCCESS,
            budget=Budget(max_usd=1.0, max_time_seconds=60),
            tasks={"t1": task},
            results={"t1": result},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            write_output_dir(state, tmpdir, project_id="test_proj")
            summary = json.loads((Path(tmpdir) / "summary.json").read_text())

        task_entry = summary["tasks"][0]
        assert task_entry["task_id"] == "t1"
        assert len(task_entry["attempt_history"]) == 1
        ah = task_entry["attempt_history"][0]
        assert ah["attempt_num"] == 1
        assert ah["model_used"] == "gpt-4o"
        assert ah["failure_reason"] == "bad score"
        assert ah["validators_failed"] == ["python_syntax"]
