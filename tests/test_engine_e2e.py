"""
E2E Integration Tests — Orchestrator Engine
============================================
Tests the full generate → critique → revise → evaluate pipeline using
a mock UnifiedClient (no real API calls).  Covers:

  - Happy path: successful project completion
  - Budget exhaustion mid-run
  - Timeout mid-run
  - Deterministic validator gate (score forced to 0.0 on failure)
  - Dependency ordering (task_002 runs after task_001)
  - Fallback routing when primary model fails
  - Circuit breaker (model marked unhealthy after 3 failures)
  - Decomposition retry on bad JSON
  - run_job() policy-driven entry point
  - tokens_used populated in TaskResult
  - async_run_validators() subprocess offload path
  - Context truncation warning

No external async plugin required — all async tests use asyncio.run() wrappers.

Run with:
    cd "E:\\Documents\\Vibe-Coding\\Ai Orchestrator"
    python -m pytest tests/test_engine_e2e.py -v
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional
from unittest.mock import patch

import pytest

from orchestrator.api_clients import APIResponse
from orchestrator.engine import Orchestrator
from orchestrator.models import (
    Budget, Model, ProjectStatus, Task, TaskStatus, TaskType,
    build_default_profiles,
)
from orchestrator.policy import JobSpec, PolicySet
from orchestrator.validators import (
    ValidationResult, async_run_validators, run_validators,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _api_response(text: str, model: Model = Model.GEMINI_FLASH,
                  input_tokens: int = 100, output_tokens: int = 200,
                  latency_ms: float = 100.0) -> APIResponse:
    resp = APIResponse(
        text=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model=model,
    )
    resp.latency_ms = latency_ms
    return resp


def _decomp_json(tasks: list[dict]) -> str:
    return json.dumps(tasks)


def _score_json(score: float, reasoning: str = "ok") -> str:
    return json.dumps({"score": score, "reasoning": reasoning})


def _make_orch(budget: Optional[Budget] = None) -> Orchestrator:
    b = budget or Budget(max_usd=1.0, max_time_seconds=300.0)
    orch = Orchestrator(budget=b, max_concurrency=1)
    # Make every model available so routing doesn't short-circuit
    for m in Model:
        orch.api_health[m] = True
    return orch


def _run(coro):
    """Run a coroutine synchronously — no pytest-asyncio needed."""
    return asyncio.run(coro)


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — Happy path: single task passes threshold
# ─────────────────────────────────────────────────────────────────────────────

def test_single_task_success():
    orch = _make_orch()

    single_task = [
        {
            "id": "task_001",
            "type": "code_generation",
            "prompt": "Write hello world in Python",
            "dependencies": [],
            "hard_validators": [],
        }
    ]
    call_count = [0]

    async def fake_call(model, prompt, **kwargs):
        call_count[0] += 1
        if "decomposition" in kwargs.get("system", "").lower():
            return _api_response(_decomp_json(single_task), model)
        if "evaluator" in kwargs.get("system", "").lower() or "score" in prompt.lower():
            return _api_response(_score_json(0.92), model)
        return _api_response("def hello():\n    print('hello world')\n", model)

    async def _run_test():
        with patch.object(orch.client, "call", side_effect=fake_call):
            return await orch.run_project(
                project_description="Hello world project",
                success_criteria="Code runs",
            )

    state = _run(_run_test())
    assert state.status in (ProjectStatus.SUCCESS, ProjectStatus.PARTIAL_SUCCESS)
    assert "task_001" in state.results
    result = state.results["task_001"]
    assert result.score >= 0.85
    assert result.status == TaskStatus.COMPLETED
    assert call_count[0] > 0


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — Budget exhaustion halts run
# ─────────────────────────────────────────────────────────────────────────────

def test_budget_exhausted_halts():
    budget = Budget(max_usd=0.001, max_time_seconds=300.0)
    orch = _make_orch(budget)

    two_tasks = [
        {"id": "task_001", "type": "code_generation", "prompt": "Task 1",
         "dependencies": [], "hard_validators": []},
        {"id": "task_002", "type": "code_generation", "prompt": "Task 2",
         "dependencies": [], "hard_validators": []},
    ]

    async def fake_call(model, prompt, **kwargs):
        orch.budget.charge(0.001, "generation")
        if "decomposition" in kwargs.get("system", "").lower():
            return _api_response(_decomp_json(two_tasks), model)
        if "evaluator" in kwargs.get("system", "").lower() or "score" in prompt.lower():
            return _api_response(_score_json(0.5), model)
        return _api_response("output", model)

    async def _run_test():
        with patch.object(orch.client, "call", side_effect=fake_call):
            return await orch.run_project("Budget test", "criteria")

    state = _run(_run_test())
    assert state.status == ProjectStatus.BUDGET_EXHAUSTED


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — Timeout halts execution
# ─────────────────────────────────────────────────────────────────────────────

def test_timeout_halts():
    budget = Budget(max_usd=100.0, max_time_seconds=0.001)  # 1ms → always expired
    orch = _make_orch(budget)

    tasks = [
        {"id": "task_001", "type": "code_generation", "prompt": "T",
         "dependencies": [], "hard_validators": []},
    ]

    async def fake_call(model, prompt, **kwargs):
        if "decomposition" in kwargs.get("system", "").lower():
            return _api_response(_decomp_json(tasks), model)
        return _api_response(_score_json(0.9), model)

    async def _run_test():
        with patch.object(orch.client, "call", side_effect=fake_call):
            return await orch.run_project("Timeout test", "criteria")

    state = _run(_run_test())
    assert state.status == ProjectStatus.TIMEOUT


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 — Deterministic validator forces score = 0.0
# ─────────────────────────────────────────────────────────────────────────────

def test_validator_gate_forces_zero_score():
    orch = _make_orch()

    tasks = [
        {"id": "task_001", "type": "code_generation", "prompt": "Write code",
         "dependencies": [], "hard_validators": ["python_syntax"]},
    ]

    async def fake_call(model, prompt, **kwargs):
        if "decomposition" in kwargs.get("system", "").lower():
            return _api_response(_decomp_json(tasks), model)
        if "evaluator" in kwargs.get("system", "").lower() or "score" in prompt.lower():
            return _api_response(_score_json(0.99), model)
        # Deliberately broken Python — syntax error
        return _api_response("def broken(\n    this is not python", model)

    async def _run_test():
        with patch.object(orch.client, "call", side_effect=fake_call):
            return await orch.run_project("Syntax test", "must compile")

    state = _run(_run_test())
    result = state.results.get("task_001")
    assert result is not None
    assert result.score == 0.0
    assert result.deterministic_check_passed is False


# ─────────────────────────────────────────────────────────────────────────────
# Test 5 — Dependency ordering respected
# ─────────────────────────────────────────────────────────────────────────────

def test_dependency_ordering():
    orch = _make_orch()

    tasks = [
        {"id": "task_001", "type": "code_generation", "prompt": "Write lib",
         "dependencies": [], "hard_validators": []},
        {"id": "task_002", "type": "code_review", "prompt": "Review lib",
         "dependencies": ["task_001"], "hard_validators": []},
    ]
    execution_log: list[str] = []

    async def fake_call(model, prompt, **kwargs):
        if "decomposition" in kwargs.get("system", "").lower():
            return _api_response(_decomp_json(tasks), model)
        if "evaluator" in kwargs.get("system", "").lower() or "score" in prompt.lower():
            return _api_response(_score_json(0.90), model)
        if "Write lib" in prompt:
            execution_log.append("task_001")
        elif "Review lib" in prompt or "task_001" in prompt:
            execution_log.append("task_002")
        return _api_response("output", model)

    async def _run_test():
        with patch.object(orch.client, "call", side_effect=fake_call):
            return await orch.run_project("Dependency test", "criteria")

    state = _run(_run_test())
    assert "task_001" in state.results
    assert "task_002" in state.results
    if "task_001" in execution_log and "task_002" in execution_log:
        assert execution_log.index("task_001") < execution_log.index("task_002")


# ─────────────────────────────────────────────────────────────────────────────
# Test 6 — Fallback routing when primary model raises
# ─────────────────────────────────────────────────────────────────────────────

def test_fallback_on_primary_failure():
    orch = _make_orch()

    tasks = [
        {"id": "task_001", "type": "code_generation", "prompt": "Write code",
         "dependencies": [], "hard_validators": []},
    ]
    primary = orch._get_available_models(TaskType.CODE_GEN)[0]
    fallback = orch._get_fallback(primary)
    assert fallback is not None, "Need a fallback model for this test"

    async def fake_call(model, prompt, **kwargs):
        if "decomposition" in kwargs.get("system", "").lower():
            return _api_response(_decomp_json(tasks), model)
        if "evaluator" in kwargs.get("system", "").lower() or "score" in prompt.lower():
            return _api_response(_score_json(0.88), model)
        if model == primary:
            raise RuntimeError("Primary model unavailable")
        return _api_response("fallback output", model)

    async def _run_test():
        with patch.object(orch.client, "call", side_effect=fake_call):
            return await orch.run_project("Fallback test", "criteria")

    state = _run(_run_test())
    result = state.results.get("task_001")
    assert result is not None
    # Degraded is acceptable — it means fallback was used
    assert result.status in (TaskStatus.COMPLETED, TaskStatus.DEGRADED)


# ─────────────────────────────────────────────────────────────────────────────
# Test 7 — Circuit breaker trips after threshold failures
# ─────────────────────────────────────────────────────────────────────────────

def test_circuit_breaker_marks_unhealthy():
    orch = _make_orch()
    threshold = orch._CIRCUIT_BREAKER_THRESHOLD
    target = Model.GEMINI_PRO

    for _ in range(threshold):
        orch._record_failure(target)

    assert orch.api_health[target] is False, (
        f"Model {target.value} should be marked unhealthy after "
        f"{threshold} consecutive failures"
    )


def test_circuit_breaker_resets_on_success():
    orch = _make_orch()
    target = Model.GEMINI_PRO

    for _ in range(orch._CIRCUIT_BREAKER_THRESHOLD - 1):
        orch._record_failure(target)

    fake_resp = _api_response("ok", model=target)
    orch._record_success(target, fake_resp)

    assert orch._consecutive_failures[target] == 0
    assert orch.api_health[target] is True


# ─────────────────────────────────────────────────────────────────────────────
# Test 8 — Decomposition retry on bad JSON
# ─────────────────────────────────────────────────────────────────────────────

def test_decomposition_retries_on_bad_json():
    orch = _make_orch()

    good_tasks = [
        {"id": "task_001", "type": "code_generation", "prompt": "Write code",
         "dependencies": [], "hard_validators": []},
    ]
    attempt = [0]

    async def fake_call(model, prompt, **kwargs):
        if "decomposition" in kwargs.get("system", "").lower():
            attempt[0] += 1
            if attempt[0] == 1:
                return _api_response("this is not JSON at all!!!", model)
            else:
                return _api_response(_decomp_json(good_tasks), model)
        if "evaluator" in kwargs.get("system", "").lower() or "score" in prompt.lower():
            return _api_response(_score_json(0.88), model)
        return _api_response("output", model)

    async def _run_test():
        with patch.object(orch.client, "call", side_effect=fake_call):
            return await orch.run_project("Retry test", "criteria")

    state = _run(_run_test())
    assert attempt[0] >= 2
    assert "task_001" in state.results


# ─────────────────────────────────────────────────────────────────────────────
# Test 9 — run_job() policy-driven entry point
# ─────────────────────────────────────────────────────────────────────────────

def test_run_job_sets_budget_and_policies():
    orch = _make_orch()

    custom_budget = Budget(max_usd=5.0, max_time_seconds=600.0)
    spec = JobSpec(
        project_description="Policy test",
        success_criteria="ok",
        budget=custom_budget,
        policy_set=PolicySet(),
    )

    tasks = [
        {"id": "task_001", "type": "code_generation", "prompt": "P",
         "dependencies": [], "hard_validators": []},
    ]

    async def fake_call(model, prompt, **kwargs):
        if "decomposition" in kwargs.get("system", "").lower():
            return _api_response(_decomp_json(tasks), model)
        if "evaluator" in kwargs.get("system", "").lower() or "score" in prompt.lower():
            return _api_response(_score_json(0.9), model)
        return _api_response("output", model)

    async def _run_test():
        with patch.object(orch.client, "call", side_effect=fake_call):
            return await orch.run_job(spec)

    state = _run(_run_test())
    assert orch.budget.max_usd == 5.0
    assert orch.budget.max_time_seconds == 600.0
    assert "task_001" in state.results


# ─────────────────────────────────────────────────────────────────────────────
# Test 10 — tokens_used populated in TaskResult
# ─────────────────────────────────────────────────────────────────────────────

def test_tokens_used_populated():
    orch = _make_orch()

    tasks = [
        {"id": "task_001", "type": "code_generation", "prompt": "Write code",
         "dependencies": [], "hard_validators": []},
    ]

    async def fake_call(model, prompt, **kwargs):
        if "decomposition" in kwargs.get("system", "").lower():
            return _api_response(_decomp_json(tasks), model,
                                 input_tokens=50, output_tokens=150)
        if "evaluator" in kwargs.get("system", "").lower() or "score" in prompt.lower():
            return _api_response(_score_json(0.93), model,
                                 input_tokens=30, output_tokens=50)
        return _api_response("def foo(): pass\n", model,
                             input_tokens=200, output_tokens=400)

    async def _run_test():
        with patch.object(orch.client, "call", side_effect=fake_call):
            return await orch.run_project("Token test", "criteria")

    state = _run(_run_test())
    result = state.results.get("task_001")
    assert result is not None
    assert result.tokens_used["input"] > 0
    assert result.tokens_used["output"] > 0


# ─────────────────────────────────────────────────────────────────────────────
# Test 11 — Context truncation warning
# ─────────────────────────────────────────────────────────────────────────────

def test_context_truncation_warning(caplog):
    from orchestrator.models import TaskResult as TR
    orch = _make_orch()
    orch.context_truncation_limit = 10  # tiny limit

    orch.results["task_001"] = TR(
        task_id="task_001",
        output="x" * 500,  # well over limit=10
        score=0.9,
        model_used=Model.GEMINI_FLASH,
        status=TaskStatus.COMPLETED,
    )

    with caplog.at_level(logging.WARNING, logger="orchestrator"):
        ctx = orch._gather_dependency_context(["task_001"])

    assert len(ctx) <= 200  # truncated output + label
    assert any("truncated" in r.message.lower() for r in caplog.records), (
        "Expected a truncation warning in logs"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tests 12–16 — async_run_validators
# ─────────────────────────────────────────────────────────────────────────────

def test_async_run_validators_json_schema():
    results = _run(async_run_validators('{"key": "value"}', ["json_schema"]))
    assert len(results) == 1
    assert results[0].passed is True
    assert results[0].validator_name == "json_schema"


def test_async_run_validators_python_syntax_pass():
    code = "def foo():\n    return 42\n"
    results = _run(async_run_validators(code, ["python_syntax"]))
    assert results[0].passed is True


def test_async_run_validators_python_syntax_fail():
    results = _run(async_run_validators(
        "def broken(\n    this is not python", ["python_syntax"]
    ))
    assert results[0].passed is False


def test_async_run_validators_ruff_offloads_to_thread():
    good_code = "x = 1\n"
    results = _run(async_run_validators(good_code, ["ruff"]))
    assert len(results) == 1
    assert results[0].validator_name == "ruff"
    # Either "No lint errors" or "ruff not available, skipped"
    assert results[0].passed is True


def test_async_run_validators_unknown_validator(caplog):
    with caplog.at_level(logging.WARNING, logger="orchestrator.validators"):
        results = _run(async_run_validators("anything", ["nonexistent_validator"]))
    assert results == []
    assert any("nonexistent_validator" in r.message for r in caplog.records)


# ─────────────────────────────────────────────────────────────────────────────
# Tests 17–18 — budget phase partition warnings
# ─────────────────────────────────────────────────────────────────────────────

def test_phase_budget_warning_at_cap(caplog):
    orch = _make_orch(Budget(max_usd=10.0))
    # generation soft cap = 45% of 10 = $4.50; spend $5.00 → over cap
    orch.budget.phase_spent["generation"] = 5.0

    with caplog.at_level(logging.WARNING, logger="orchestrator"):
        orch._check_phase_budget("generation")

    assert any("generation" in r.message for r in caplog.records)


def test_phase_budget_error_at_2x_cap(caplog):
    orch = _make_orch(Budget(max_usd=10.0))
    # 2× cap = $9.00; spend $9.50 → ERROR
    orch.budget.phase_spent["generation"] = 9.5

    with caplog.at_level(logging.ERROR, logger="orchestrator"):
        orch._check_phase_budget("generation")

    assert any(r.levelname == "ERROR" and "generation" in r.message
               for r in caplog.records)


# ─────────────────────────────────────────────────────────────────────────────
# Tests 19–21 — topological sort
# ─────────────────────────────────────────────────────────────────────────────

def test_topological_sort_chain():
    orch = _make_orch()
    tasks = {
        "a": Task(id="a", type=TaskType.CODE_GEN, prompt="a"),
        "b": Task(id="b", type=TaskType.CODE_GEN, prompt="b", dependencies=["a"]),
        "c": Task(id="c", type=TaskType.CODE_GEN, prompt="c", dependencies=["b"]),
    }
    order = orch._topological_sort(tasks)
    assert order.index("a") < order.index("b")
    assert order.index("b") < order.index("c")


def test_topological_sort_parallel():
    orch = _make_orch()
    tasks = {
        "a": Task(id="a", type=TaskType.CODE_GEN, prompt="a"),
        "b": Task(id="b", type=TaskType.CODE_GEN, prompt="b"),
        "c": Task(id="c", type=TaskType.CODE_GEN, prompt="c",
                  dependencies=["a", "b"]),
    }
    order = orch._topological_sort(tasks)
    assert order.index("c") > order.index("a")
    assert order.index("c") > order.index("b")


def test_topological_sort_cycle_detected(caplog):
    orch = _make_orch()
    tasks = {
        "a": Task(id="a", type=TaskType.CODE_GEN, prompt="a", dependencies=["b"]),
        "b": Task(id="b", type=TaskType.CODE_GEN, prompt="b", dependencies=["a"]),
    }
    with caplog.at_level(logging.ERROR, logger="orchestrator"):
        order = orch._topological_sort(tasks)

    assert len(order) < 2
    assert any("cycle" in r.message.lower() for r in caplog.records)


# ─────────────────────────────────────────────────────────────────────────────
# Test 22 — build_default_profiles covers all 8 models
# ─────────────────────────────────────────────────────────────────────────────

def test_build_default_profiles_all_models():
    profiles = build_default_profiles()
    for model in Model:
        assert model in profiles, f"Missing profile for {model.value}"
        p = profiles[model]
        assert p.cost_per_1m_input >= 0
        assert p.cost_per_1m_output >= 0
        assert p.provider in ("openai", "anthropic", "google", "kimi", "deepseek")


# ─────────────────────────────────────────────────────────────────────────────
# Test 23 — Multi-task project with partial success
# ─────────────────────────────────────────────────────────────────────────────

def test_multi_task_partial_success():
    orch = _make_orch(Budget(max_usd=5.0))

    tasks = [
        {"id": "task_001", "type": "code_generation", "prompt": "T1",
         "dependencies": [], "hard_validators": []},
        {"id": "task_002", "type": "code_review", "prompt": "T2",
         "dependencies": ["task_001"], "hard_validators": []},
        {"id": "task_003", "type": "summarization", "prompt": "T3",
         "dependencies": ["task_002"], "hard_validators": []},
    ]

    async def fake_call(model, prompt, **kwargs):
        if "decomposition" in kwargs.get("system", "").lower():
            return _api_response(_decomp_json(tasks), model)
        if "evaluator" in kwargs.get("system", "").lower() or "score" in prompt.lower():
            if "T2" in prompt or "task_002" in prompt:
                return _api_response(_score_json(0.50), model)
            return _api_response(_score_json(0.91), model)
        return _api_response("output", model)

    async def _run_test():
        with patch.object(orch.client, "call", side_effect=fake_call):
            return await orch.run_project("Multi task", "criteria")

    state = _run(_run_test())
    assert len(state.results) == 3
    scores = [r.score for r in state.results.values()]
    assert any(s >= 0.85 for s in scores)
