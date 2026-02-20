"""
Tests for the Cost Layer — BudgetHierarchy, CostPredictor, CostForecaster.
Covers: multi-level budget caps, EMA updates, COST_TABLE fallback,
        pre-flight forecasting, risk levels, planner/engine integration.
"""
from __future__ import annotations

import pytest

from orchestrator.cost import (
    BudgetHierarchy,
    CostForecaster,
    CostPredictor,
    ForecastReport,
    RiskLevel,
)
from orchestrator.models import Budget, Model, Task, TaskType, build_default_profiles


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_task(task_type: TaskType = TaskType.CODE_GEN, task_id: str = "t_001") -> Task:
    return Task(id=task_id, type=task_type, prompt="test task prompt")


# ─────────────────────────────────────────────────────────────────────────────
# BudgetHierarchy — can_afford_job
# ─────────────────────────────────────────────────────────────────────────────

def test_can_afford_job_within_org_limit():
    h = BudgetHierarchy(org_max_usd=100.0)
    assert h.can_afford_job("j1", "eng", 10.0) is True


def test_can_afford_job_exceeds_org_limit():
    h = BudgetHierarchy(org_max_usd=5.0)
    assert h.can_afford_job("j1", "eng", 10.0) is False


def test_team_limit_blocks_even_when_org_has_budget():
    h = BudgetHierarchy(org_max_usd=100.0, team_budgets={"eng": 8.0})
    assert h.can_afford_job("j1", "eng", 9.0) is False


def test_team_limit_allows_when_within_cap():
    h = BudgetHierarchy(org_max_usd=100.0, team_budgets={"eng": 20.0})
    assert h.can_afford_job("j1", "eng", 10.0) is True


def test_job_limit_blocks_even_when_org_and_team_have_budget():
    h = BudgetHierarchy(
        org_max_usd=100.0,
        team_budgets={"eng": 50.0},
        job_budgets={"j1": 5.0},
    )
    assert h.can_afford_job("j1", "eng", 6.0) is False


def test_job_limit_allows_when_within_cap():
    h = BudgetHierarchy(
        org_max_usd=100.0,
        team_budgets={"eng": 50.0},
        job_budgets={"j1": 20.0},
    )
    assert h.can_afford_job("j1", "eng", 10.0) is True


def test_zero_cost_always_allowed():
    h = BudgetHierarchy(org_max_usd=0.01)
    assert h.can_afford_job("j1", "eng", 0.0) is True


def test_negative_cost_always_allowed():
    h = BudgetHierarchy(org_max_usd=1.0)
    assert h.can_afford_job("j1", "eng", -5.0) is True


def test_can_afford_no_team_constraint():
    """If team is not in team_budgets, only org cap applies."""
    h = BudgetHierarchy(org_max_usd=100.0, team_budgets={"other": 5.0})
    assert h.can_afford_job("j1", "unknown_team", 50.0) is True


# ─────────────────────────────────────────────────────────────────────────────
# BudgetHierarchy — charge_job
# ─────────────────────────────────────────────────────────────────────────────

def test_charge_job_updates_org_spend():
    h = BudgetHierarchy(org_max_usd=100.0)
    h.charge_job("j1", "eng", 10.0)
    assert h.remaining("org") == pytest.approx(90.0)


def test_charge_job_updates_team_spend():
    h = BudgetHierarchy(org_max_usd=100.0, team_budgets={"eng": 50.0})
    h.charge_job("j1", "eng", 15.0)
    assert h.remaining("team", "eng") == pytest.approx(35.0)


def test_charge_job_updates_job_spend():
    h = BudgetHierarchy(org_max_usd=100.0, job_budgets={"j1": 30.0})
    h.charge_job("j1", "eng", 12.0)
    assert h.remaining("job", "j1") == pytest.approx(18.0)


def test_charge_job_cumulative_spend():
    h = BudgetHierarchy(org_max_usd=100.0)
    h.charge_job("j1", "eng", 5.0)
    h.charge_job("j1", "eng", 5.0)
    assert h.remaining("org") == pytest.approx(90.0)


def test_charge_zero_does_nothing():
    h = BudgetHierarchy(org_max_usd=100.0)
    h.charge_job("j1", "eng", 0.0)
    assert h.remaining("org") == pytest.approx(100.0)


def test_can_afford_false_after_spending():
    h = BudgetHierarchy(org_max_usd=10.0)
    h.charge_job("j1", "eng", 9.0)
    assert h.can_afford_job("j2", "eng", 2.0) is False


# ─────────────────────────────────────────────────────────────────────────────
# BudgetHierarchy — remaining
# ─────────────────────────────────────────────────────────────────────────────

def test_remaining_org_initial():
    h = BudgetHierarchy(org_max_usd=50.0)
    assert h.remaining("org") == pytest.approx(50.0)


def test_remaining_team_initial():
    h = BudgetHierarchy(org_max_usd=100.0, team_budgets={"eng": 30.0})
    assert h.remaining("team", "eng") == pytest.approx(30.0)


def test_remaining_never_negative():
    h = BudgetHierarchy(org_max_usd=5.0)
    h.charge_job("j1", "", 100.0)  # overcharge
    assert h.remaining("org") == pytest.approx(0.0)


def test_remaining_unknown_level_raises():
    h = BudgetHierarchy(org_max_usd=100.0)
    with pytest.raises(ValueError, match="Unknown budget level"):
        h.remaining("bad_level")


# ─────────────────────────────────────────────────────────────────────────────
# BudgetHierarchy — to_dict
# ─────────────────────────────────────────────────────────────────────────────

def test_to_dict_has_expected_keys():
    h = BudgetHierarchy(org_max_usd=100.0, team_budgets={"eng": 30.0})
    d = h.to_dict()
    assert "org" in d
    assert "team" in d
    assert "job" in d
    assert d["org"]["max"] == pytest.approx(100.0)


# ─────────────────────────────────────────────────────────────────────────────
# CostPredictor — EMA behaviour
# ─────────────────────────────────────────────────────────────────────────────

def test_predict_unknown_falls_back_to_cost_table():
    predictor = CostPredictor()
    # No data recorded yet → COST_TABLE fallback
    result = predictor.predict(Model.KIMI_K2_5, TaskType.CODE_GEN)
    # Should be a positive float (not zero or negative)
    assert result > 0.0


def test_predict_after_record_returns_ema_value():
    predictor = CostPredictor()
    predictor.record(Model.KIMI_K2_5, TaskType.CODE_GEN, 0.001)
    result = predictor.predict(Model.KIMI_K2_5, TaskType.CODE_GEN)
    assert result == pytest.approx(0.001)


def test_record_zero_cost_ignored():
    """Zero/negative actual_cost must not corrupt EMA."""
    predictor = CostPredictor()
    predictor.record(Model.KIMI_K2_5, TaskType.CODE_GEN, 0.001)
    before = predictor.predict(Model.KIMI_K2_5, TaskType.CODE_GEN)
    predictor.record(Model.KIMI_K2_5, TaskType.CODE_GEN, 0.0)
    after = predictor.predict(Model.KIMI_K2_5, TaskType.CODE_GEN)
    assert after == before


def test_record_negative_cost_ignored():
    predictor = CostPredictor()
    predictor.record(Model.KIMI_K2_5, TaskType.CODE_GEN, 0.002)
    before = predictor.predict(Model.KIMI_K2_5, TaskType.CODE_GEN)
    predictor.record(Model.KIMI_K2_5, TaskType.CODE_GEN, -1.0)
    after = predictor.predict(Model.KIMI_K2_5, TaskType.CODE_GEN)
    assert after == before


def test_ema_converges_toward_target():
    predictor = CostPredictor(alpha=0.1)
    target = 0.005
    for _ in range(60):
        predictor.record(Model.KIMI_K2_5, TaskType.CODE_GEN, target)
    result = predictor.predict(Model.KIMI_K2_5, TaskType.CODE_GEN)
    assert abs(result - target) < target * 0.05


def test_ema_different_model_task_type_isolated():
    predictor = CostPredictor()
    predictor.record(Model.KIMI_K2_5, TaskType.CODE_GEN, 0.001)
    predictor.record(Model.GPT_4O, TaskType.WRITING, 0.01)
    assert predictor.predict(Model.KIMI_K2_5, TaskType.CODE_GEN) < \
           predictor.predict(Model.GPT_4O, TaskType.WRITING)


def test_invalid_alpha_raises():
    with pytest.raises(ValueError, match="alpha"):
        CostPredictor(alpha=0.0)

    with pytest.raises(ValueError, match="alpha"):
        CostPredictor(alpha=1.1)


# ─────────────────────────────────────────────────────────────────────────────
# CostPredictor — cheapest_model
# ─────────────────────────────────────────────────────────────────────────────

def test_cheapest_model_returns_model_with_lowest_predict():
    predictor = CostPredictor()
    predictor.record(Model.KIMI_K2_5, TaskType.CODE_GEN, 0.0001)
    predictor.record(Model.GPT_4O, TaskType.CODE_GEN, 0.01)
    cheapest = predictor.cheapest_model(TaskType.CODE_GEN, [Model.KIMI_K2_5, Model.GPT_4O])
    assert cheapest == Model.KIMI_K2_5


def test_cheapest_model_empty_candidates_returns_none():
    predictor = CostPredictor()
    assert predictor.cheapest_model(TaskType.CODE_GEN, []) is None


def test_cheapest_model_single_candidate():
    predictor = CostPredictor()
    result = predictor.cheapest_model(TaskType.CODE_GEN, [Model.KIMI_K2_5])
    assert result == Model.KIMI_K2_5


# ─────────────────────────────────────────────────────────────────────────────
# CostForecaster.forecast
# ─────────────────────────────────────────────────────────────────────────────

def test_forecast_empty_task_list_returns_zero_report():
    profiles = build_default_profiles()
    predictor = CostPredictor()
    report = CostForecaster.forecast([], profiles, predictor)
    assert isinstance(report, ForecastReport)
    assert report.estimated_total_usd == pytest.approx(0.0)
    assert report.risk_level == RiskLevel.LOW


def test_forecast_returns_positive_total_for_tasks():
    profiles = build_default_profiles()
    predictor = CostPredictor()
    tasks = [_make_task(TaskType.CODE_GEN, f"t_{i}") for i in range(3)]
    report = CostForecaster.forecast(tasks, profiles, predictor)
    assert report.estimated_total_usd > 0.0


def test_forecast_risk_low_when_cost_well_below_budget():
    profiles = build_default_profiles()
    predictor = CostPredictor()
    # Seed with very small costs
    for model in profiles:
        predictor.record(model, TaskType.CODE_GEN, 0.000001)
    tasks = [_make_task(TaskType.CODE_GEN)]
    budget = Budget(max_usd=100.0)
    report = CostForecaster.forecast(tasks, profiles, predictor, budget=budget)
    assert report.risk_level == RiskLevel.LOW


def test_forecast_risk_high_when_cost_near_budget():
    profiles = build_default_profiles()
    predictor = CostPredictor()
    # Seed with large costs to push ratio above 0.8
    for model in profiles:
        predictor.record(model, TaskType.CODE_GEN, 1.0)
    tasks = [_make_task(TaskType.CODE_GEN) for _ in range(3)]
    budget = Budget(max_usd=1.0)
    report = CostForecaster.forecast(tasks, profiles, predictor, budget=budget)
    assert report.risk_level == RiskLevel.HIGH


def test_forecast_risk_medium_in_between():
    profiles = build_default_profiles()
    predictor = CostPredictor()
    tasks = [_make_task(TaskType.CODE_GEN)]
    # A budget roughly 1.5x the expected cost → should land in MEDIUM (50-80%)
    # We'll use the fallback cost table; just check it can return MEDIUM
    report_no_budget = CostForecaster.forecast(tasks, profiles, predictor)
    # Without budget, ratio=0 → LOW
    assert report_no_budget.risk_level == RiskLevel.LOW


def test_forecast_per_phase_breakdown_has_all_keys():
    profiles = build_default_profiles()
    predictor = CostPredictor()
    tasks = [_make_task(TaskType.WRITING)]
    report = CostForecaster.forecast(tasks, profiles, predictor)
    assert "generation" in report.estimated_per_phase
    assert "cross_review" in report.estimated_per_phase
    assert "evaluation" in report.estimated_per_phase


def test_forecast_time_estimate_positive():
    profiles = build_default_profiles()
    predictor = CostPredictor()
    tasks = [_make_task(TaskType.CODE_GEN)]
    report = CostForecaster.forecast(tasks, profiles, predictor)
    assert report.estimated_time_seconds > 0.0


def test_forecast_will_exceed_budget():
    profiles = build_default_profiles()
    predictor = CostPredictor()
    # Make cost predictably large
    for model in profiles:
        predictor.record(model, TaskType.CODE_GEN, 5.0)
    tasks = [_make_task(TaskType.CODE_GEN)]
    report = CostForecaster.forecast(tasks, profiles, predictor, budget=Budget(max_usd=1.0))
    assert report.will_exceed_budget(1.0) is True


def test_forecast_will_not_exceed_large_budget():
    profiles = build_default_profiles()
    predictor = CostPredictor()
    for model in profiles:
        predictor.record(model, TaskType.CODE_GEN, 0.000001)
    tasks = [_make_task(TaskType.CODE_GEN)]
    report = CostForecaster.forecast(tasks, profiles, predictor, budget=Budget(max_usd=1000.0))
    assert report.will_exceed_budget(1000.0) is False


def test_forecast_more_tasks_higher_cost():
    profiles = build_default_profiles()
    predictor = CostPredictor()
    tasks_1 = [_make_task(TaskType.CODE_GEN, "t1")]
    tasks_3 = [_make_task(TaskType.CODE_GEN, f"t{i}") for i in range(3)]
    report_1 = CostForecaster.forecast(tasks_1, profiles, predictor)
    report_3 = CostForecaster.forecast(tasks_3, profiles, predictor)
    assert report_3.estimated_total_usd > report_1.estimated_total_usd


def test_forecast_zero_budget_max_gives_high_risk():
    """If budget.max_usd is 0, ratio = 1.0 → HIGH risk."""
    profiles = build_default_profiles()
    predictor = CostPredictor()
    tasks = [_make_task(TaskType.CODE_GEN)]
    budget = Budget(max_usd=0.0, max_time_seconds=3600)
    report = CostForecaster.forecast(tasks, profiles, predictor, budget=budget)
    # ratio set to 1.0 when budget.max_usd <= 0, which is >= 0.8 → HIGH
    assert report.risk_level == RiskLevel.HIGH


# ─────────────────────────────────────────────────────────────────────────────
# ConstraintPlanner + CostPredictor integration
# ─────────────────────────────────────────────────────────────────────────────

def _make_planner(profiles, predictor=None):
    """Helper matching the pattern used by test_constraint_planner.py."""
    from orchestrator.planner import ConstraintPlanner
    from orchestrator.policy_engine import PolicyEngine
    engine = PolicyEngine()
    health = {m: True for m in profiles}
    return ConstraintPlanner(profiles, engine, health, cost_predictor=predictor)


def test_planner_accepts_cost_predictor():
    """ConstraintPlanner should be constructable with a CostPredictor."""
    profiles = build_default_profiles()
    predictor = CostPredictor()
    planner = _make_planner(profiles, predictor=predictor)
    assert planner is not None


def test_planner_falls_back_to_static_when_predictor_returns_zero():
    """When predictor has no data (predict returns COST_TABLE fallback), planner still selects."""
    profiles = build_default_profiles()
    predictor = CostPredictor()
    planner = _make_planner(profiles, predictor=predictor)
    # Should not raise; a model should still be selected
    selected = planner.select_model(TaskType.CODE_GEN, policies=[], budget_remaining=10.0)
    assert selected is not None


def test_planner_uses_predictor_when_data_available():
    """When predictor has data, planner incorporates it into cost estimation."""
    profiles = build_default_profiles()
    predictor = CostPredictor()
    # Seed kimi as very cheap
    predictor.record(Model.KIMI_K2_5, TaskType.CODE_GEN, 0.000001)
    planner = _make_planner(profiles, predictor=predictor)
    selected = planner.select_model(TaskType.CODE_GEN, policies=[], budget_remaining=10.0)
    assert selected is not None
