"""
Audit Wave A — code-review follow-ups
=====================================
Regression tests for the code review of PR #32 (fix/wave-a-defaults).

R1: a per-call budget passed to run_project/run_job only rebound _run_ctx.budget, while
    every stage kept the container Budget, so a shared orchestrator carried spend from
    one project into the next.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


async def test_r1_per_call_budget_resets_the_shared_budget_in_place() -> None:
    from orchestrator.application.run_context import RunContext
    from orchestrator.budget import Budget

    shared = Budget(max_usd=10.0)
    await shared.charge(9.0, "generation")
    ctx = RunContext(budget=shared)

    ctx.reset(project_id="p2", budget=Budget(max_usd=5.0, max_time_seconds=60.0))

    assert ctx.budget is shared  # stages hold this object; it must not be swapped out
    assert shared.max_usd == 5.0
    assert shared.max_time_seconds == 60.0
    assert shared.spent_usd == 0.0
    assert shared.remaining_usd == 5.0
    assert all(v == 0.0 for v in shared.phase_spent.values())


async def test_r1_run_job_applies_spec_budget_to_the_shared_budget(monkeypatch) -> None:
    from unittest.mock import AsyncMock

    from orchestrator.budget import Budget
    from orchestrator.engine import Orchestrator

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test-key-for-verification-only")
    orch = Orchestrator(budget=Budget(max_usd=8.0))
    shared = orch.budget
    await shared.charge(7.0, "generation")
    orch._project_runner.run_job = AsyncMock(return_value=None)
    spec = type(
        "Spec",
        (),
        {
            "budget": Budget(max_usd=0.5),
            "policy_set": None,
            "quality_mode": "standard",
            "max_parallel_tasks": 0,
        },
    )()
    monkeypatch.setattr("orchestrator.engine.validate_job_spec", lambda s: None)

    await orch.run_job(spec)

    assert orch._run_ctx.budget is shared
    assert shared.max_usd == 0.5 and shared.spent_usd == 0.0
    await orch.close()


async def test_r1_reset_without_budget_keeps_cumulative_spend() -> None:
    from orchestrator.application.run_context import RunContext
    from orchestrator.budget import Budget

    shared = Budget(max_usd=10.0)
    await shared.charge(3.0, "generation")
    ctx = RunContext(budget=shared)

    ctx.reset(project_id="p2")

    assert shared.spent_usd == 3.0
