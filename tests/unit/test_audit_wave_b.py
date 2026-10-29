"""
Audit Wave B — money path
=========================
Regression tests for docs/plans/2026-10-29-audit-remediation-plan.md, Wave B.

B2: run_job charged the BudgetHierarchy the Budget's total spent_usd, which is never
    reset between runs on a reused Orchestrator, so run 2 was charged run 1's spend too.
B3: a run that raised only released its reservation, so its real spend was never
    charged; a cancelled run (CancelledError is a BaseException) did not even release it.
B4: the job-level cap ignored in-flight reservations, unlike the org and team caps.
B5: three paid LLM calls never charged the Budget: DesignCritiqueStage, the assumption
    gate, and a failed Instructor decomposition attempt.
B8: cost_optimization kept three private price tables that disagreed with COST_TABLE,
    and TokenBudget divided per-1M prices by 1000 (1000x too high).
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


def test_b4_job_cap_counts_inflight_reservations() -> None:
    from orchestrator.cost import BudgetHierarchy

    hierarchy = BudgetHierarchy(org_max_usd=1000.0, job_budgets={"j": 1.0})

    assert hierarchy.can_afford_job("j", "", 0.8) is True
    assert hierarchy.can_afford_job("j", "", 0.8) is False


def _paid_client(text="{}", cost=0.1):
    client = MagicMock()
    client.call = AsyncMock(return_value=SimpleNamespace(text=text, cost_usd=cost))
    return client


async def test_b5_design_critique_charges_the_budget() -> None:
    from orchestrator.budget import Budget
    from orchestrator.engine_core.pipeline import PipelineContext
    from orchestrator.engine_core.stages.design_critique import DesignCritiqueStage
    from orchestrator.models import Model, Task, TaskType

    budget = Budget(max_usd=10.0)
    stage = DesignCritiqueStage(client=_paid_client("score: 7"), budget=budget)
    task = Task(id="t1", type=TaskType.CODE_GEN, prompt="build a landing page in React")
    ctx = PipelineContext(task=task, output="<div/>", model=Model.GPT_4O_MINI)

    await stage.process(ctx)

    assert budget.spent_usd == pytest.approx(0.1)


async def test_b5_assumption_gate_charges_through_charge_fn() -> None:
    from orchestrator.assumption_gate import surface_assumptions

    charged: list[float] = []

    async def charge(amount):
        charged.append(amount)

    await surface_assumptions(
        "make it better", _paid_client('{"has_ambiguity": false}', 0.05), charge_fn=charge
    )

    assert charged == [pytest.approx(0.05)]


async def test_b5_failed_instructor_decomposition_is_still_charged(monkeypatch) -> None:
    import sys
    import types

    from orchestrator.application.decomposer_service import decompose_project
    from orchestrator.models import Model

    class FailingDecomposer:
        def __init__(self, api_client):
            pass

        async def decompose(self, **kwargs):
            raise RuntimeError("schema validation failed")

        def _calculate_decomposition_tokens(self, project):
            return 1000

    fake = types.ModuleType("orchestrator.structured_outputs")
    fake.TaskDecomposer = FailingDecomposer
    monkeypatch.setitem(sys.modules, "orchestrator.structured_outputs", fake)
    charged: list[float] = []

    async def charge(amount):
        charged.append(amount)

    fallback = MagicMock()
    fallback.decompose = AsyncMock(return_value={})

    await decompose_project(
        project="p",
        criteria="c",
        model=Model.GPT_4O_MINI,
        client=MagicMock(),
        decomposer=fallback,
        api_health={},
        record_failure_fn=MagicMock(),
        charge_fn=charge,
    )

    assert charged and charged[0] > 0


def test_b8_cascader_prices_come_from_cost_table() -> None:
    from orchestrator.cost_optimization.model_cascading import ModelCascader
    from orchestrator.models import Model, estimate_cost

    cost = ModelCascader()._estimate_cost(Model.DEEPSEEK_V4_FLASH.value, "a" * 4000)

    assert cost == pytest.approx(estimate_cost(Model.DEEPSEEK_V4_FLASH, 0, 1000))


def test_b8_token_budget_uses_per_million_prices() -> None:
    from orchestrator.cost_optimization.token_budget import TokenBudget

    # Claude Sonnet output is $15 per 1M tokens, so 1K tokens cost $0.015.
    assert TokenBudget()._calculate_cost("claude-sonnet", 1000, "output") == pytest.approx(0.015)
