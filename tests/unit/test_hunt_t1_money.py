"""
T1 (money tier) proof-of-defect and no-regression tests.

Four VERIFIED DEFECTs from docs/hunts/t1-money/inventory.md:

C1 — orchestrator/costing/core.py was an independent, hand-copied fork of
     orchestrator/cost.py's BudgetHierarchy/CostPredictor/CostForecaster. It
     diverged silently: orchestrator.cost.BudgetHierarchy later gained SQLite
     persistence (db_path) that was never backported, so the two same-named
     classes had different behavior and were not interchangeable.

C2 — BudgetEnforcer.record_cost() mutated `budget.spent_usd` directly,
     bypassing Budget's own asyncio.Lock (the exact TOCTOU protection
     BUG-001/FIX-001a added), and called `budget_hierarchy.record_cost(...)`
     — a method that does not exist on BudgetHierarchy (real method:
     `charge_job(job_id, team, amount)`) — silenced with
     `# type: ignore[attr-defined]`.

C4 — Every registered handler in orchestrator/task_handlers.py hardcoded
     TaskResult(cost_usd=0.0, tokens_used={"input": 0, "output": 0}, ...)
     regardless of the real LLM call made via _BaseHandler._call_llm(),
     which itself discarded response.cost_usd/input_tokens/output_tokens and
     returned only response.text. The `budget` parameter every handler
     accepted was never referenced in any handler body.

C5 — ProjectRunner.run_job()'s cross-run settlement computed
     `actual_spend = self._budget.max_usd - self._budget.remaining_usd`,
     which silently clamps at max_usd whenever a run overspends its own
     per-run cap (Budget.remaining_usd floors at 0.0, and Budget.charge()
     enforces no ceiling) — undercounting what gets charged to
     BudgetHierarchy's cross-run org/team/job totals.

C3 (BudgetEnforcer/TaskExecutor/task_handlers.py never instantiated in the
live pipeline) is intentionally NOT covered here — it is a wiring/
architecture question flagged `[REQUIRES HUMAN REVIEW]`, not a code fix.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]


# --- C1 -----------------------------------------------------------------


@pytest.mark.unit
def test_c1_costing_core_budget_hierarchy_is_the_canonical_class():
    """orchestrator.costing.core.BudgetHierarchy must be the SAME object as
    orchestrator.cost.BudgetHierarchy, not a diverged copy."""
    from orchestrator.cost import BudgetHierarchy as canonical
    from orchestrator.costing.core import BudgetHierarchy as via_costing_core

    assert via_costing_core is canonical


@pytest.mark.unit
def test_c1_costing_package_budget_hierarchy_is_canonical():
    """Same check via the package's own re-export path (orchestrator.costing)."""
    from orchestrator.cost import BudgetHierarchy as canonical
    from orchestrator.costing import BudgetHierarchy as via_package

    assert via_package is canonical


@pytest.mark.unit
def test_c1_costing_core_budget_hierarchy_supports_persistence():
    """No-regression guard: the stale fork had no `db_path` parameter at all.
    Constructing with db_path must not raise TypeError."""
    import tempfile

    from orchestrator.costing.core import BudgetHierarchy

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "budget.db"
        bh = BudgetHierarchy(org_max_usd=10.0, db_path=db_path)
        assert db_path.exists()


# --- C2 -------------------------------------------------------------------


@pytest.mark.unit
def test_c2_budget_hierarchy_has_no_record_cost_method():
    """Pins the fact that motivated the fix: BudgetHierarchy's real method is
    charge_job(job_id, team, amount), not record_cost(task_id, cost_usd)."""
    from orchestrator.cost import BudgetHierarchy

    assert not hasattr(BudgetHierarchy, "record_cost")
    assert hasattr(BudgetHierarchy, "charge_job")


@pytest.mark.asyncio
async def test_c2_record_cost_charges_budget_via_lock_protected_charge():
    """record_cost() must delegate to Budget.charge() (lock-protected), not
    mutate spent_usd directly."""
    from orchestrator.application.budget_enforcer import BudgetEnforcer
    from orchestrator.budget import Budget

    budget = Budget(max_usd=10.0)
    enforcer = BudgetEnforcer(budget=budget)

    await enforcer.record_cost("task-1", 0.5, phase="generation")

    assert budget.spent_usd == pytest.approx(0.5)
    assert budget.phase_spent["generation"] == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_c2_record_cost_does_not_crash_with_a_hierarchy_present():
    """Before the fix, this raised AttributeError the instant budget_hierarchy
    was truthy — record_cost() must not touch the hierarchy at all (that is
    enforce_hierarchy_job's job, which has the job_id/team this method lacks)."""
    from orchestrator.application.budget_enforcer import BudgetEnforcer
    from orchestrator.budget import Budget
    from orchestrator.cost import BudgetHierarchy

    budget = Budget(max_usd=10.0)
    hierarchy = BudgetHierarchy(org_max_usd=100.0)
    enforcer = BudgetEnforcer(budget=budget, budget_hierarchy=hierarchy)

    await enforcer.record_cost("task-1", 0.5, phase="generation")

    # record_cost must not have touched the hierarchy's spend at all.
    assert hierarchy.remaining("org") == pytest.approx(100.0)


@pytest.mark.asyncio
async def test_c2_record_cost_still_updates_enforcers_own_phase_tracking():
    """No-regression: the enforcer's own (separate) phase_spent bookkeeping
    must keep working exactly as before."""
    from orchestrator.application.budget_enforcer import BudgetEnforcer
    from orchestrator.budget import Budget

    budget = Budget(max_usd=10.0)
    enforcer = BudgetEnforcer(budget=budget)

    await enforcer.record_cost("task-1", 0.3, phase="evaluation")

    assert enforcer.phase_spent["evaluation"] == pytest.approx(0.3)


# --- C4 -------------------------------------------------------------------


class _FakeClient:
    """Fake transport at the UnifiedClient seam (DEFECT_HUNT_PLAN.md Step 4:
    no live provider calls; fake the adapter seam, never mock the unit under
    test). Returns a real APIResponse with known, non-zero cost/tokens."""

    def __init__(self, cost_usd: float, input_tokens: int, output_tokens: int):
        from orchestrator.api_clients import APIResponse
        from orchestrator.models import Model

        self.calls = 0
        self._response = APIResponse(
            text="fake output",
            model=Model.GPT_4O_MINI,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        )

    async def call(self, model, prompt, **kwargs):
        self.calls += 1
        return self._response


def _make_task(task_type):
    from orchestrator.models import Task

    return Task(id="t1", type=task_type, prompt="do the thing")


@pytest.mark.asyncio
async def test_c4_call_llm_returns_full_response_not_bare_text():
    """_call_llm must expose cost_usd/input_tokens/output_tokens, not just text."""
    from orchestrator.models import Model
    from orchestrator.task_handlers import _BaseHandler

    fake = _FakeClient(cost_usd=0.0123, input_tokens=111, output_tokens=222)
    handler = _BaseHandler()
    response = await handler._call_llm(fake, "prompt", model=Model.GPT_4O_MINI)

    assert response.text == "fake output"
    assert response.cost_usd == pytest.approx(0.0123)
    assert response.input_tokens == 111
    assert response.output_tokens == 222


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "handler_cls, task_type",
    [
        ("CodeGenerationHandler", "CODE_GEN"),
        ("CodeReviewHandler", "CODE_REVIEW"),
        ("EvaluationHandler", "EVALUATE"),
        ("ReasoningHandler", "REASONING"),
    ],
)
async def test_c4_handler_reports_real_cost_and_tokens_not_hardcoded_zero(handler_cls, task_type):
    """Every registered handler must report the real cost/tokens from the API
    response instead of hardcoding cost_usd=0.0 / tokens_used={0,0}."""
    import orchestrator.task_handlers as th
    from orchestrator.models import TaskType

    fake = _FakeClient(cost_usd=0.042, input_tokens=50, output_tokens=75)
    task = _make_task(getattr(TaskType, task_type))
    handler = getattr(th, handler_cls)()

    result = await handler.execute(task=task, client=fake, budget=None)

    assert fake.calls == 1
    assert result.cost_usd == pytest.approx(0.042)
    assert result.tokens_used == {"input": 50, "output": 75}


@pytest.mark.asyncio
async def test_c4_handler_charges_a_real_budget_when_given_one():
    """When execute() is given a real Budget, the handler must actually
    charge it — previously `budget` was accepted and never referenced."""
    from orchestrator.budget import Budget
    from orchestrator.models import TaskType
    from orchestrator.task_handlers import CodeGenerationHandler

    fake = _FakeClient(cost_usd=0.75, input_tokens=10, output_tokens=20)
    task = _make_task(TaskType.CODE_GEN)
    budget = Budget(max_usd=10.0)

    await CodeGenerationHandler().execute(task=task, client=fake, budget=budget)

    assert budget.spent_usd == pytest.approx(0.75)


@pytest.mark.asyncio
async def test_c4_handler_tolerates_budget_none():
    """C3 means `budget` currently always arrives as None in the live wiring
    (TaskExecutor is never instantiated with a real BudgetEnforcer) — the fix
    must not assume a real Budget is present."""
    from orchestrator.models import TaskType
    from orchestrator.task_handlers import CodeGenerationHandler

    fake = _FakeClient(cost_usd=0.1, input_tokens=1, output_tokens=1)
    task = _make_task(TaskType.CODE_GEN)

    result = await CodeGenerationHandler().execute(task=task, client=fake, budget=None)

    assert result.cost_usd == pytest.approx(0.1)


# --- C5 -------------------------------------------------------------------


@pytest.mark.unit
def test_c5_project_runner_no_longer_derives_spend_from_remaining_usd():
    """No-regression source guard: the clamped-at-max_usd formula must not
    come back at the run_job() settlement call site."""
    src = (REPO_ROOT / "orchestrator" / "application" / "project_runner.py").read_text(
        encoding="utf-8"
    )
    assert (
        re.search(
            r"actual_spend\s*=\s*self\._budget\.max_usd\s*-\s*self\._budget\.remaining_usd", src
        )
        is None
    )
    assert "actual_spend = self._budget.spent_usd" in src


@pytest.mark.asyncio
async def test_c5_run_job_charges_hierarchy_with_true_overspend_not_clamped_value(monkeypatch):
    """Real trigger via the actual ProjectRunner.run_job() call path: when the
    run overspends its own per-run cap, the hierarchy must be charged the
    true spend, not max_usd (what `max_usd - remaining_usd` would clamp to
    once remaining_usd floors at 0.0)."""
    from unittest.mock import AsyncMock, MagicMock

    # Under CI/sandboxed stdin, UnattendedGuard blocks run_project() unless a
    # retry cap / checkpoint is wired — unrelated to C5, same escape hatch
    # tests/integration/conftest.py uses for every run_project() call.
    monkeypatch.setenv("ORCH_UNATTENDED_GUARD", "false")

    from orchestrator.application.project_runner import ProjectRunner
    from orchestrator.application.project_runner_deps import (
        ProjectRunnerCallables,
        ProjectRunState,
    )
    from orchestrator.cost import BudgetHierarchy
    from orchestrator.models import ProjectState, ProjectStatus

    def _make_state(status=ProjectStatus.SUCCESS):
        return ProjectState(
            project_description="build X",
            success_criteria="tests pass",
            budget=None,
            tasks={},
            results={},
            status=status,
            api_health={},
        )

    run_state = ProjectRunState(entered=True)
    callables = ProjectRunnerCallables(
        topological_sort=MagicMock(return_value=["t1"]),
        topological_levels=MagicMock(return_value=[["t1"]]),
        make_state=MagicMock(return_value=_make_state()),
        determine_final_status=MagicMock(return_value=ProjectStatus.SUCCESS),
        log_summary=MagicMock(),
        execute_all=AsyncMock(return_value=_make_state()),
        generate_architecture_rules=AsyncMock(return_value=""),
        analyze_completed_project=AsyncMock(),
        client=MagicMock(),
    )

    # Budget overspent its own $10 cap by $2 — remaining_usd floors at 0.0.
    budget = MagicMock()
    budget.max_usd = 10.0
    budget.max_time_seconds = 3600
    budget.spent_usd = 12.0
    budget.remaining_usd = 0.0
    budget.elapsed_seconds = 10.0
    budget.validate_sufficient_for_tasks = MagicMock(return_value=(True, ""))

    state_mgr = AsyncMock()
    state_mgr.load_project = AsyncMock(return_value=None)
    state_mgr.save_project = AsyncMock()
    state_mgr.close = AsyncMock()

    cache = AsyncMock()
    cache.close = AsyncMock()

    generator = AsyncMock()
    gen_result = MagicMock()
    gen_result.succeeded = True
    gen_result.tasks = {"t1": MagicMock()}
    gen_result.error = ""
    generator.decompose = AsyncMock(return_value=gen_result)

    resumption_svc = AsyncMock()
    resumption_svc.resume = AsyncMock(return_value=_make_state())

    hierarchy = BudgetHierarchy(org_max_usd=1000.0)

    runner = ProjectRunner(
        callables=callables,
        run_state=run_state,
        state_mgr=state_mgr,
        budget=budget,
        event_bus=None,
        resumption_svc=resumption_svc,
        dashboard_bridge=MagicMock(),
        git_bridge=MagicMock(commit_project=MagicMock(return_value="abc123")),
        generator=generator,
        meta_v2=None,
        cache=cache,
        api_health={},
        budget_hierarchy=hierarchy,
    )

    spec = SimpleNamespace(
        project_description="build X",
        success_criteria="tests pass",
        budget=SimpleNamespace(max_usd=10.0),
        job_id="job-1",
        team="team-a",
    )

    await runner.run_job(spec)

    # True spend ($12) must reach the hierarchy, not the clamped $10.
    assert hierarchy.to_dict()["org"]["spent"] == pytest.approx(12.0)


# --- B1-COST-1/2/3 (V3 precision defect audit, docs/audits/v3/T1/batch1) --------
#
# A separate audit campaign (docs/PRECISION_DEFECT_AUDIT_PLAN.md) found two more
# defects in this same tier, independent of C1-C5 above.


@pytest.mark.unit
def test_b1_cost_1_track_usage_known_model_does_not_raise():
    """Fires B1-COST-1 without the fix; passes with it. Violated property:
    track_usage() must price a model already in COST_TABLE, not only run
    the unknown-model fallback branch. CostAnalytics.track_usage() indexed
    the string-keyed CostDict positionally (cost_entry[0]/[1])."""
    from orchestrator.costing.analytics import CostAnalytics
    from orchestrator.models import Model

    analytics = CostAnalytics()
    try:
        cost = analytics.track_usage(Model.GPT_OSS_120B, input_tokens=1000, output_tokens=500)
    except KeyError:
        pytest.fail("defect still present: track_usage() indexed CostDict positionally")
    assert cost > 0


@pytest.mark.unit
def test_b1_cost_2_repeated_preflight_same_job_id_does_not_leak_reservation():
    """Fires B1-COST-2 without the fix; passes with it. Violated property:
    _reserved_usd/_team_reserved must return to 0 once every reservation
    they reflect has been released by a single settling charge_job() call.
    can_afford_job() overwrote (not accumulated) self._reservations[job_id]
    on a second pre-flight check for the same job_id."""
    from orchestrator.cost import BudgetHierarchy

    hier = BudgetHierarchy(org_max_usd=10.0)

    assert hier.can_afford_job("job-1", "eng", 5.0) is True
    assert hier.can_afford_job("job-1", "eng", 3.0) is True  # e.g. an overlapping retry

    hier.charge_job("job-1", "eng", 7.5)  # single settle, as run_job() does

    assert hier._reserved_usd == pytest.approx(0.0)
    assert hier._team_reserved.get("eng", 0.0) == pytest.approx(0.0)


@pytest.mark.unit
def test_b1_cost_3_anon_reservations_release_the_correct_team():
    """Fires B1-COST-3 without the fix; passes with it. Violated property:
    releasing a team-scoped-but-unattributed (job_id="") reservation must
    decrement THAT team's _team_reserved, not whichever team settles next.
    FIFO anon-reservation release used the CALLER's team argument instead of
    the team the reservation was actually made under."""
    from orchestrator.cost import BudgetHierarchy

    hier = BudgetHierarchy(org_max_usd=100.0)

    assert hier.can_afford_job("", "team-a", 5.0) is True
    assert hier.can_afford_job("", "team-b", 7.0) is True

    hier.charge_job("", "team-b", 7.0)  # team-b finishes first — out of FIFO order
    assert hier._team_reserved.get("team-b", 0.0) == pytest.approx(0.0)

    hier.charge_job("", "team-a", 5.0)
    assert hier._team_reserved.get("team-a", 0.0) == pytest.approx(0.0)
    assert hier._reserved_usd == pytest.approx(0.0)
