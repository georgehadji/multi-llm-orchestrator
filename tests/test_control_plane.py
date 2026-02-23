"""
Unit tests for ControlPlane (B3).
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from orchestrator.control_plane import (
    ControlPlane,
    PolicyViolation,
    RoutingPlan,
    SpecValidationError,
)
from orchestrator.models import ProjectState, ProjectStatus
from orchestrator.specs import (
    Constraints,
    InputSpec,
    JobSpecV2,
    PolicySpecV2,
    SLAs,
)


def _minimal_state() -> ProjectState:
    from orchestrator.models import Budget
    return ProjectState(
        project_description="test",
        success_criteria="pass",
        budget=Budget(max_usd=1.0),
        tasks={},
        results={},
        api_health={},
        status=ProjectStatus.SUCCESS,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

def test_validate_empty_goal():
    cp = ControlPlane()
    job = JobSpecV2(goal="")
    errors = cp._validate(job, PolicySpecV2())
    assert any("goal" in e for e in errors)


def test_validate_invalid_cost():
    cp = ControlPlane()
    job = JobSpecV2(goal="test", slas=SLAs(max_cost_usd=-1.0))
    errors = cp._validate(job, PolicySpecV2())
    assert any("max_cost_usd" in e for e in errors)


def test_validate_invalid_quality_tier():
    cp = ControlPlane()
    job = JobSpecV2(goal="test", slas=SLAs(min_quality_tier=1.5))
    errors = cp._validate(job, PolicySpecV2())
    assert any("min_quality_tier" in e for e in errors)


def test_validate_invalid_data_locality():
    cp = ControlPlane()
    job = JobSpecV2(goal="test", inputs=InputSpec(data_locality="mars"))
    errors = cp._validate(job, PolicySpecV2())
    assert any("data_locality" in e for e in errors)


def test_validate_policy_deny_rule_missing_effect():
    cp = ControlPlane()
    job = JobSpecV2(goal="test")
    policy = PolicySpecV2(allow_deny_rules=[{"when": "x == y"}])
    errors = cp._validate(job, policy)
    assert any("effect" in e for e in errors)


def test_validate_valid_job():
    cp = ControlPlane()
    job = JobSpecV2(goal="valid job")
    errors = cp._validate(job, PolicySpecV2())
    assert errors == []


# ─────────────────────────────────────────────────────────────────────────────
# Submit — validation error path
# ─────────────────────────────────────────────────────────────────────────────

def test_submit_raises_on_validation_error():
    cp = ControlPlane()
    job = JobSpecV2(goal="")  # empty goal → validation error
    with pytest.raises(SpecValidationError):
        asyncio.run(cp.submit(job, PolicySpecV2()))


# ─────────────────────────────────────────────────────────────────────────────
# Submit — policy violation path
# ─────────────────────────────────────────────────────────────────────────────

def test_submit_raises_on_policy_violation():
    cp = ControlPlane()
    job = JobSpecV2(
        goal="my job",
        inputs=InputSpec(data_locality="us"),
    )
    policy = PolicySpecV2(allow_deny_rules=[
        {"effect": "deny", "when": "data_locality == us"},
    ])
    with pytest.raises(PolicyViolation):
        asyncio.run(cp.submit(job, policy))


# ─────────────────────────────────────────────────────────────────────────────
# solve_constraints
# ─────────────────────────────────────────────────────────────────────────────

def test_solve_eu_only_constraint():
    cp = ControlPlane()
    job = JobSpecV2(
        goal="test",
        constraints=Constraints(hard=["eu_only"]),
        inputs=InputSpec(data_locality="eu"),
    )
    plan = cp._solve_constraints(job, PolicySpecV2())
    assert any("eu_only" in n or "EU" in n for n in plan.notes)


def test_solve_cost_sla_adjusts_budget():
    cp = ControlPlane()
    job = JobSpecV2(goal="test", slas=SLAs(max_cost_usd=0.5))
    cp._solve_constraints(job, PolicySpecV2())
    assert job.budget.max_usd == 0.5


def test_solve_quality_tier_above_threshold():
    cp = ControlPlane()
    job = JobSpecV2(goal="test", slas=SLAs(min_quality_tier=0.98))
    plan = cp._solve_constraints(job, PolicySpecV2())
    assert any("quality" in n.lower() for n in plan.notes)


# ─────────────────────────────────────────────────────────────────────────────
# Full submit — happy path (mocked orchestrator)
# ─────────────────────────────────────────────────────────────────────────────

def test_submit_happy_path():
    cp = ControlPlane()
    job = JobSpecV2(goal="build something")

    async def _run():
        with patch.object(cp, "_run_workflow", new=AsyncMock(return_value=_minimal_state())):
            return await cp.submit(job, PolicySpecV2())

    state = asyncio.run(_run())
    assert state.status == ProjectStatus.SUCCESS
