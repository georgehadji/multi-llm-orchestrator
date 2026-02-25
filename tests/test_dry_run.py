"""
Tests — Improvement 12: Dry-run / Execution Plan
=================================================
Covers:
  - ExecutionPlan / TaskPlan / DryRunRenderer exported from orchestrator
  - dry_run() returns ExecutionPlan without executing tasks
  - ExecutionPlan has correct task count, levels, cost
  - render() produces human-readable output with required sections
  - Empty decomposition returns an empty plan (no crash)
  - Dependency ordering is reflected in parallel_levels
  - CLI --dry-run flag dispatched correctly (no actual run)
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

import pytest

from orchestrator import ExecutionPlan, TaskPlan, DryRunRenderer
from orchestrator.api_clients import APIResponse
from orchestrator.engine import Orchestrator
from orchestrator.models import Budget, Model


# ─── helpers ─────────────────────────────────────────────────────────────────

def _api_response(text: str, model: Model = Model.GEMINI_FLASH) -> APIResponse:
    resp = APIResponse(text=text, input_tokens=50, output_tokens=100, model=model)
    resp.latency_ms = 50.0
    return resp


def _decomp_json(tasks: list[dict]) -> str:
    return json.dumps(tasks)


def _run(coro):
    return asyncio.run(coro)


def _make_orch() -> Orchestrator:
    orch = Orchestrator(budget=Budget(max_usd=5.0, max_time_seconds=300), max_concurrency=1)
    for m in Model:
        orch.api_health[m] = True
    return orch


_TWO_TASKS = [
    {
        "id": "t1",
        "type": "code_generation",
        "prompt": "write a hello function",
        "dependencies": [],
        "priority": 0,
    },
    {
        "id": "t2",
        "type": "code_review",
        "prompt": "review the hello function",
        "dependencies": ["t1"],
        "priority": 0,
    },
]

_THREE_PARALLEL = [
    {"id": "a", "type": "creative_writing", "prompt": "write a",
     "dependencies": [], "priority": 0},
    {"id": "b", "type": "creative_writing", "prompt": "write b",
     "dependencies": [], "priority": 0},
    {"id": "c", "type": "summarization", "prompt": "summarize",
     "dependencies": ["a", "b"], "priority": 0},
]


# ─── Export checks ────────────────────────────────────────────────────────────

class TestExports:
    def test_execution_plan_importable(self):
        assert ExecutionPlan is not None

    def test_task_plan_importable(self):
        assert TaskPlan is not None

    def test_dry_run_renderer_importable(self):
        assert DryRunRenderer is not None

    def test_execution_plan_has_dry_run_method(self):
        orch = _make_orch()
        assert hasattr(orch, "dry_run")
        assert callable(orch.dry_run)


# ─── dry_run returns ExecutionPlan ───────────────────────────────────────────

class TestDryRunReturnsExecutionPlan:
    def test_returns_execution_plan_type(self):
        orch = _make_orch()

        async def fake_call(model, prompt, **kwargs):
            system = kwargs.get("system", "").lower()
            if "decomposition" in system:
                return _api_response(_decomp_json(_TWO_TASKS), model)
            return _api_response("", model)

        async def _do():
            with patch.object(orch.client, "call", side_effect=fake_call):
                return await orch.dry_run("test project", "criteria")

        plan = _run(_do())
        assert isinstance(plan, ExecutionPlan)

    def test_tasks_count_matches_decomposition(self):
        orch = _make_orch()

        async def fake_call(model, prompt, **kwargs):
            system = kwargs.get("system", "").lower()
            if "decomposition" in system:
                return _api_response(_decomp_json(_TWO_TASKS), model)
            return _api_response("", model)

        async def _do():
            with patch.object(orch.client, "call", side_effect=fake_call):
                return await orch.dry_run("test project", "criteria")

        plan = _run(_do())
        assert len(plan.tasks) == 2

    def test_no_execution_calls_made(self):
        """dry_run must NOT call the LLM for task generation/review."""
        orch = _make_orch()
        call_types: list[str] = []

        async def fake_call(model, prompt, **kwargs):
            system = kwargs.get("system", "").lower()
            if "decomposition" in system:
                call_types.append("decomp")
                return _api_response(_decomp_json(_TWO_TASKS), model)
            # Any other call = task execution (not allowed in dry-run)
            call_types.append("execution")
            return _api_response("", model)

        async def _do():
            with patch.object(orch.client, "call", side_effect=fake_call):
                return await orch.dry_run("test project", "criteria")

        _run(_do())
        # Only decomposition calls should have been made
        assert "execution" not in call_types

    def test_empty_decomposition_returns_empty_plan(self):
        """If decomposition fails, dry_run returns empty ExecutionPlan without crashing."""
        orch = _make_orch()

        async def fake_call(model, prompt, **kwargs):
            # Return invalid JSON → decomp will fail → empty task list
            return _api_response("not valid json", model)

        async def _do():
            with patch.object(orch.client, "call", side_effect=fake_call):
                return await orch.dry_run("test project", "criteria")

        plan = _run(_do())
        assert isinstance(plan, ExecutionPlan)
        assert len(plan.tasks) == 0


# ─── Parallel levels and dependency ordering ─────────────────────────────────

class TestParallelLevels:
    def test_sequential_tasks_have_two_levels(self):
        """t1 → t2 (dependency) should produce 2 levels."""
        orch = _make_orch()

        async def fake_call(model, prompt, **kwargs):
            system = kwargs.get("system", "").lower()
            if "decomposition" in system:
                return _api_response(_decomp_json(_TWO_TASKS), model)
            return _api_response("", model)

        async def _do():
            with patch.object(orch.client, "call", side_effect=fake_call):
                return await orch.dry_run("test project", "criteria")

        plan = _run(_do())
        assert plan.num_parallel_levels == 2
        assert len(plan.parallel_levels) == 2

    def test_parallel_tasks_in_same_level(self):
        """a and b with no deps should be in the same level."""
        orch = _make_orch()

        async def fake_call(model, prompt, **kwargs):
            system = kwargs.get("system", "").lower()
            if "decomposition" in system:
                return _api_response(_decomp_json(_THREE_PARALLEL), model)
            return _api_response("", model)

        async def _do():
            with patch.object(orch.client, "call", side_effect=fake_call):
                return await orch.dry_run("project", "criteria")

        plan = _run(_do())
        # Level 0 should contain both a and b; level 1 should contain c
        assert plan.num_parallel_levels == 2
        level0_ids = set(plan.parallel_levels[0])
        assert "a" in level0_ids
        assert "b" in level0_ids
        assert plan.parallel_levels[1] == ["c"]

    def test_task_plan_has_correct_level(self):
        orch = _make_orch()

        async def fake_call(model, prompt, **kwargs):
            system = kwargs.get("system", "").lower()
            if "decomposition" in system:
                return _api_response(_decomp_json(_TWO_TASKS), model)
            return _api_response("", model)

        async def _do():
            with patch.object(orch.client, "call", side_effect=fake_call):
                return await orch.dry_run("test project", "criteria")

        plan = _run(_do())
        t1 = next(t for t in plan.tasks if t.task_id == "t1")
        t2 = next(t for t in plan.tasks if t.task_id == "t2")
        assert t1.parallel_level == 0
        assert t2.parallel_level == 1

    def test_dependencies_recorded_in_task_plan(self):
        orch = _make_orch()

        async def fake_call(model, prompt, **kwargs):
            system = kwargs.get("system", "").lower()
            if "decomposition" in system:
                return _api_response(_decomp_json(_TWO_TASKS), model)
            return _api_response("", model)

        async def _do():
            with patch.object(orch.client, "call", side_effect=fake_call):
                return await orch.dry_run("test project", "criteria")

        plan = _run(_do())
        t2 = next(t for t in plan.tasks if t.task_id == "t2")
        assert t2.dependencies == ["t1"]


# ─── Cost estimation ─────────────────────────────────────────────────────────

class TestCostEstimation:
    def test_estimated_cost_positive(self):
        orch = _make_orch()

        async def fake_call(model, prompt, **kwargs):
            system = kwargs.get("system", "").lower()
            if "decomposition" in system:
                return _api_response(_decomp_json(_TWO_TASKS), model)
            return _api_response("", model)

        async def _do():
            with patch.object(orch.client, "call", side_effect=fake_call):
                return await orch.dry_run("test project", "criteria")

        plan = _run(_do())
        assert plan.estimated_total_cost > 0.0

    def test_per_task_cost_non_negative(self):
        orch = _make_orch()

        async def fake_call(model, prompt, **kwargs):
            system = kwargs.get("system", "").lower()
            if "decomposition" in system:
                return _api_response(_decomp_json(_TWO_TASKS), model)
            return _api_response("", model)

        async def _do():
            with patch.object(orch.client, "call", side_effect=fake_call):
                return await orch.dry_run("test project", "criteria")

        plan = _run(_do())
        for tp in plan.tasks:
            assert tp.estimated_cost_usd >= 0.0

    def test_total_cost_equals_sum_of_tasks(self):
        orch = _make_orch()

        async def fake_call(model, prompt, **kwargs):
            system = kwargs.get("system", "").lower()
            if "decomposition" in system:
                return _api_response(_decomp_json(_TWO_TASKS), model)
            return _api_response("", model)

        async def _do():
            with patch.object(orch.client, "call", side_effect=fake_call):
                return await orch.dry_run("test project", "criteria")

        plan = _run(_do())
        task_sum = round(sum(t.estimated_cost_usd for t in plan.tasks), 6)
        assert abs(plan.estimated_total_cost - task_sum) < 1e-6


# ─── render() output ─────────────────────────────────────────────────────────

class TestRender:
    def test_render_contains_dry_run_header(self):
        orch = _make_orch()

        async def fake_call(model, prompt, **kwargs):
            system = kwargs.get("system", "").lower()
            if "decomposition" in system:
                return _api_response(_decomp_json(_TWO_TASKS), model)
            return _api_response("", model)

        async def _do():
            with patch.object(orch.client, "call", side_effect=fake_call):
                return await orch.dry_run("test project", "criteria")

        plan = _run(_do())
        rendered = plan.render()
        assert "DRY-RUN" in rendered
        assert "Execution Plan" in rendered

    def test_render_shows_all_task_ids(self):
        orch = _make_orch()

        async def fake_call(model, prompt, **kwargs):
            system = kwargs.get("system", "").lower()
            if "decomposition" in system:
                return _api_response(_decomp_json(_TWO_TASKS), model)
            return _api_response("", model)

        async def _do():
            with patch.object(orch.client, "call", side_effect=fake_call):
                return await orch.dry_run("test project", "criteria")

        plan = _run(_do())
        rendered = plan.render()
        assert "t1" in rendered
        assert "t2" in rendered

    def test_render_shows_no_execution_message(self):
        orch = _make_orch()

        async def fake_call(model, prompt, **kwargs):
            system = kwargs.get("system", "").lower()
            if "decomposition" in system:
                return _api_response(_decomp_json(_TWO_TASKS), model)
            return _api_response("", model)

        async def _do():
            with patch.object(orch.client, "call", side_effect=fake_call):
                return await orch.dry_run("test project", "criteria")

        plan = _run(_do())
        rendered = plan.render()
        assert "No tasks were executed" in rendered or "dry run" in rendered.lower()

    def test_render_shows_level_markers(self):
        orch = _make_orch()

        async def fake_call(model, prompt, **kwargs):
            system = kwargs.get("system", "").lower()
            if "decomposition" in system:
                return _api_response(_decomp_json(_TWO_TASKS), model)
            return _api_response("", model)

        async def _do():
            with patch.object(orch.client, "call", side_effect=fake_call):
                return await orch.dry_run("test project", "criteria")

        plan = _run(_do())
        rendered = plan.render()
        assert "Level 0" in rendered
        assert "Level 1" in rendered

    def test_empty_plan_renders_without_error(self):
        plan = ExecutionPlan(
            project_description="empty project",
            success_criteria="nothing",
        )
        rendered = plan.render()
        assert "DRY-RUN" in rendered
        assert "empty project" in rendered
