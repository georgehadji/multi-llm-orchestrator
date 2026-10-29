"""
Audit Wave B — money path
=========================
Regression tests for docs/plans/2026-10-29-audit-remediation-plan.md, Wave B.

B2: run_job charged the BudgetHierarchy the Budget's total spent_usd, which is never
    reset between runs on a reused Orchestrator, so run 2 was charged run 1's spend too.
B3: a run that raised only released its reservation, so its real spend was never
    charged; a cancelled run (CancelledError is a BaseException) did not even release it.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

pytestmark = pytest.mark.unit


def _runner(budget, hierarchy):
    from orchestrator.application.project_runner import ProjectRunner

    return ProjectRunner(
        callables=MagicMock(warm_start_fn=None, flush_telemetry_fn=None),
        run_state=MagicMock(),
        state_mgr=AsyncMock(),
        budget=budget,
        event_bus=None,
        resumption_svc=AsyncMock(),
        dashboard_bridge=MagicMock(),
        git_bridge=MagicMock(),
        generator=AsyncMock(),
        meta_v2=None,
        cache=AsyncMock(),
        api_health={},
        budget_hierarchy=hierarchy,
    )


def _spec(job_id="job-1"):
    return SimpleNamespace(
        project_description="d",
        success_criteria="c",
        budget=SimpleNamespace(max_usd=5.0),
        job_id=job_id,
        team="team-a",
    )


def _spending_run(budget, amount, exc=None):
    async def run_project(**kwargs):
        await budget.charge(amount, "generation")
        if exc is not None:
            raise exc
        return "state"

    return run_project


async def test_b2_hierarchy_is_charged_each_runs_own_spend() -> None:
    from orchestrator.budget import Budget
    from orchestrator.cost import BudgetHierarchy

    budget = Budget(max_usd=100.0)
    hierarchy = BudgetHierarchy(org_max_usd=1000.0)
    runner = _runner(budget, hierarchy)

    runner.run_project = _spending_run(budget, 1.0)
    await runner.run_job(_spec("job-1"))
    runner.run_project = _spending_run(budget, 1.0)
    await runner.run_job(_spec("job-2"))

    assert hierarchy.to_dict()["org"]["spent"] == pytest.approx(2.0)


@pytest.mark.parametrize(
    "exc", [RuntimeError("provider down"), __import__("asyncio").CancelledError()]
)
async def test_b3_failed_or_cancelled_run_settles_spend_and_reservation(exc) -> None:
    from orchestrator.budget import Budget
    from orchestrator.cost import BudgetHierarchy

    budget = Budget(max_usd=100.0)
    hierarchy = BudgetHierarchy(org_max_usd=1000.0)
    runner = _runner(budget, hierarchy)
    runner.run_project = _spending_run(budget, 4.0, exc=exc)

    with pytest.raises(type(exc)):
        await runner.run_job(_spec())

    assert hierarchy.to_dict()["org"]["spent"] == pytest.approx(4.0)
    assert hierarchy.remaining("org") == pytest.approx(996.0)  # reservation released
