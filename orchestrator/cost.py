"""
Cost Layer — hierarchical budgets, adaptive cost prediction, project forecasting.
==================================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis
Three independent components:

BudgetHierarchy
    Cross-run organisational budget caps (Org → Team → Job levels).
    Standalone — does NOT replace the per-run Budget dataclass. Both can be
    used together: Budget tracks intra-run spend; BudgetHierarchy tracks
    cross-run caps that persist across multiple Orchestrator.run_job() calls.

CostPredictor
    Per-(model × task_type) EMA cost predictor. Learns from actual charges
    recorded during execution and provides forward-looking estimates.
    Falls back to COST_TABLE static estimates for unknown combinations.

CostForecaster
    Pre-flight cost estimation for a list of tasks before execution.
    Returns a ForecastReport with risk level relative to a Budget.

Usage:
    from orchestrator.cost import BudgetHierarchy, CostPredictor, CostForecaster

    # Cross-run budget
    hier = BudgetHierarchy(org_max_usd=100.0, team_budgets={"eng": 30.0})
    orch = Orchestrator(budget=Budget(max_usd=10.0), budget_hierarchy=hier)

    # Pre-flight forecast
    predictor = CostPredictor()
    report = CostForecaster.forecast(tasks, profiles, predictor, budget=Budget(max_usd=10.0))
    if report.risk_level == RiskLevel.HIGH:
        print("Warning: may exceed budget!")
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .models import COST_TABLE, Budget, Model, Task, TaskType, estimate_cost
from .policy import ModelProfile

logger = logging.getLogger("orchestrator.cost")

# Typical call counts per task: generate(1) + critique(1) + revise(1) + 15% eval overhead
_CALLS_PER_TASK: float = 3.0
_EVAL_OVERHEAD:  float = 0.15   # 15% extra for evaluation passes

# Typical token counts for cost estimation (mirrors planner._TYPICAL_*_TOKENS)
_TYPICAL_INPUT_TOKENS: dict[TaskType, int] = {
    TaskType.CODE_GEN:     800,
    TaskType.CODE_REVIEW:  600,
    TaskType.REASONING:    600,
    TaskType.WRITING:      400,
    TaskType.DATA_EXTRACT: 300,
    TaskType.SUMMARIZE:    600,
    TaskType.EVALUATE:     500,
}
_TYPICAL_OUTPUT_TOKENS: dict[TaskType, int] = {
    TaskType.CODE_GEN:     1000,
    TaskType.CODE_REVIEW:  600,
    TaskType.REASONING:    800,
    TaskType.WRITING:      600,
    TaskType.DATA_EXTRACT: 200,
    TaskType.SUMMARIZE:    300,
    TaskType.EVALUATE:     400,
}

# Average task execution time in seconds (rough estimates for forecasting)
_TYPICAL_TASK_SECONDS: dict[TaskType, float] = {
    TaskType.CODE_GEN:     90.0,
    TaskType.CODE_REVIEW:  60.0,
    TaskType.REASONING:    75.0,
    TaskType.WRITING:      60.0,
    TaskType.DATA_EXTRACT: 30.0,
    TaskType.SUMMARIZE:    30.0,
    TaskType.EVALUATE:     45.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# RiskLevel
# ─────────────────────────────────────────────────────────────────────────────

class RiskLevel(str, Enum):
    """Budget risk level for a forecast."""
    LOW    = "low"      # estimated cost < 50% of budget
    MEDIUM = "medium"   # estimated cost 50–80% of budget
    HIGH   = "high"     # estimated cost ≥ 80% of budget


# ─────────────────────────────────────────────────────────────────────────────
# ForecastReport
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ForecastReport:
    """
    Pre-flight cost and time estimate for a set of tasks.

    Attributes
    ----------
    estimated_total_usd    : total predicted USD spend
    estimated_per_phase    : breakdown by budget phase (generation/cross_review/evaluation)
    estimated_time_seconds : total predicted wall-clock time
    risk_level             : LOW / MEDIUM / HIGH relative to budget (if provided)
    """
    estimated_total_usd:    float
    estimated_per_phase:    dict[str, float]
    estimated_time_seconds: float
    risk_level:             RiskLevel

    def will_exceed_budget(self, budget_max_usd: float) -> bool:
        """Return True if the estimated total exceeds the given budget cap."""
        return self.estimated_total_usd > budget_max_usd


# ─────────────────────────────────────────────────────────────────────────────
# BudgetHierarchy
# ─────────────────────────────────────────────────────────────────────────────

class BudgetHierarchy:
    """
    Tracks cross-run organisational budget caps at three levels:

      Org  — global maximum across all teams and jobs
      Team — per-team cap (team_name → max_usd)
      Job  — per-job cap  (job_id    → max_usd)

    All three are checked independently. The most restrictive constraint wins.

    This class is standalone — it does NOT replace the per-run Budget object.
    Use it alongside Orchestrator.budget for cross-run enforcement:
        orch = Orchestrator(budget=Budget(max_usd=10.0),
                            budget_hierarchy=BudgetHierarchy(org_max_usd=50.0))

    Note: For high-volume long-running deployments, consider periodically
    resetting or persisting `_spent` state to disk.
    """

    def __init__(
        self,
        org_max_usd:   float,
        team_budgets:  Optional[dict[str, float]] = None,
        job_budgets:   Optional[dict[str, float]] = None,
    ) -> None:
        self._org_max   = org_max_usd
        self._team_max  = dict(team_budgets) if team_budgets else {}
        self._job_max   = dict(job_budgets)  if job_budgets  else {}

        # Spent trackers
        self._org_spent:  float = 0.0
        self._team_spent: dict[str, float] = {}
        self._job_spent:  dict[str, float] = {}

    # ── Query ────────────────────────────────────────────────────────────────

    def can_afford_job(
        self,
        job_id:         str,
        team:           str,
        estimated_cost: float,
    ) -> bool:
        """
        Return True if the estimated cost fits within all three budget levels.

        A zero or negative estimated_cost is always allowed.
        """
        if estimated_cost <= 0:
            return True

        # Org level
        if (self._org_spent + estimated_cost) > self._org_max:
            logger.warning(
                "BudgetHierarchy: org cap would be exceeded "
                "(spent=%.4f, est=%.4f, max=%.4f)",
                self._org_spent, estimated_cost, self._org_max,
            )
            return False

        # Team level
        if team and team in self._team_max:
            team_spent = self._team_spent.get(team, 0.0)
            if (team_spent + estimated_cost) > self._team_max[team]:
                logger.warning(
                    "BudgetHierarchy: team '%s' cap would be exceeded "
                    "(spent=%.4f, est=%.4f, max=%.4f)",
                    team, team_spent, estimated_cost, self._team_max[team],
                )
                return False

        # Job level
        if job_id and job_id in self._job_max:
            job_spent = self._job_spent.get(job_id, 0.0)
            if (job_spent + estimated_cost) > self._job_max[job_id]:
                logger.warning(
                    "BudgetHierarchy: job '%s' cap would be exceeded "
                    "(spent=%.4f, est=%.4f, max=%.4f)",
                    job_id, job_spent, estimated_cost, self._job_max[job_id],
                )
                return False

        return True

    def charge_job(self, job_id: str, team: str, amount: float) -> None:
        """Deduct amount from org, team, and job spend trackers."""
        if amount <= 0:
            return
        self._org_spent += amount
        if team:
            self._team_spent[team] = self._team_spent.get(team, 0.0) + amount
        if job_id:
            self._job_spent[job_id] = self._job_spent.get(job_id, 0.0) + amount

    def remaining(self, level: str = "org", key: str = "") -> float:
        """
        Return remaining budget at the specified level.

        Parameters
        ----------
        level : "org" | "team" | "job"
        key   : team name or job id (required for team/job level)
        """
        if level == "org":
            return max(0.0, self._org_max - self._org_spent)
        elif level == "team":
            max_v = self._team_max.get(key, self._org_max)
            spent = self._team_spent.get(key, 0.0)
            return max(0.0, max_v - spent)
        elif level == "job":
            max_v = self._job_max.get(key, self._org_max)
            spent = self._job_spent.get(key, 0.0)
            return max(0.0, max_v - spent)
        else:
            raise ValueError(f"Unknown budget level {level!r}. Use 'org', 'team', or 'job'.")

    def to_dict(self) -> dict:
        """Serialise the hierarchy state to a plain dict."""
        return {
            "org": {"max": self._org_max, "spent": self._org_spent},
            "team": {
                k: {"max": self._team_max.get(k, self._org_max),
                    "spent": self._team_spent.get(k, 0.0)}
                for k in set(list(self._team_max.keys()) + list(self._team_spent.keys()))
            },
            "job": {
                k: {"max": self._job_max.get(k, self._org_max),
                    "spent": self._job_spent.get(k, 0.0)}
                for k in set(list(self._job_max.keys()) + list(self._job_spent.keys()))
            },
        }


# ─────────────────────────────────────────────────────────────────────────────
# CostPredictor
# ─────────────────────────────────────────────────────────────────────────────

class CostPredictor:
    """
    EMA-based adaptive cost predictor per (model, task_type) pair.

    Starts with no data; falls back to COST_TABLE static estimates for unknown
    combinations (guarantees `predict()` always returns a positive value).

    Usage:
        predictor = CostPredictor()
        predictor.record(Model.KIMI_K2_5, TaskType.CODE_GEN, 0.000032)
        est = predictor.predict(Model.KIMI_K2_5, TaskType.CODE_GEN)
        cheapest = predictor.cheapest_model(TaskType.CODE_GEN, candidates)
    """

    def __init__(self, alpha: float = 0.1) -> None:
        """
        Parameters
        ----------
        alpha : EMA decay factor (0 < alpha ≤ 1). Default 0.1 (same as telemetry).
                Higher alpha → faster adaptation to recent costs.
        """
        if not 0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self._alpha = alpha
        # Stores EMA per (model, task_type) tuple
        self._ema: dict[tuple[Model, TaskType], float] = {}

    def record(self, model: Model, task_type: TaskType, actual_cost_usd: float) -> None:
        """Update the EMA for (model, task_type) with the observed cost."""
        if actual_cost_usd <= 0:
            return   # don't corrupt EMA with zero/negative costs (cache hits, free calls)
        key = (model, task_type)
        if key in self._ema:
            self._ema[key] = self._alpha * actual_cost_usd + (1 - self._alpha) * self._ema[key]
        else:
            self._ema[key] = actual_cost_usd   # seed with first observed value

    def predict(self, model: Model, task_type: TaskType) -> float:
        """
        Return the predicted cost for one API call with this model+task_type.

        If no data has been recorded, falls back to COST_TABLE static estimate.
        Returns 0 if neither EMA nor COST_TABLE data is available (callers
        should treat 0 as "no prediction, use static fallback").
        """
        key = (model, task_type)
        if key in self._ema:
            return self._ema[key]
        # Static fallback via COST_TABLE
        return self._static_estimate(model, task_type)

    def _static_estimate(self, model: Model, task_type: TaskType) -> float:
        """Fallback: estimate cost from COST_TABLE + typical token counts."""
        input_t  = _TYPICAL_INPUT_TOKENS.get(task_type, 500)
        output_t = _TYPICAL_OUTPUT_TOKENS.get(task_type, 500)
        try:
            return estimate_cost(model, input_t, output_t)
        except (KeyError, Exception):
            return 0.0

    def cheapest_model(
        self,
        task_type:  TaskType,
        candidates: list[Model],
    ) -> Optional[Model]:
        """Return the candidate model with the lowest predicted cost for task_type."""
        if not candidates:
            return None
        return min(candidates, key=lambda m: self.predict(m, task_type))


# ─────────────────────────────────────────────────────────────────────────────
# CostForecaster
# ─────────────────────────────────────────────────────────────────────────────

class CostForecaster:
    """
    Pre-flight project-level cost and time estimator.

    Uses CostPredictor estimates (or COST_TABLE fallbacks) to project the
    total spend for a list of tasks before execution begins.
    """

    @staticmethod
    def forecast(
        tasks:     list["Task"],
        profiles:  dict[Model, ModelProfile],
        predictor: "CostPredictor",
        budget:    Optional[Budget] = None,
    ) -> ForecastReport:
        """
        Estimate total cost and time for the given task list.

        Parameters
        ----------
        tasks     : list of Task objects (from project decomposition)
        profiles  : current ModelProfile dict (for provider/capability info)
        predictor : CostPredictor to use for per-call estimates
        budget    : if provided, used to compute risk_level

        Returns
        -------
        ForecastReport with estimated_total_usd, per-phase breakdown, time, and risk_level
        """
        if not tasks:
            return ForecastReport(
                estimated_total_usd=0.0,
                estimated_per_phase={"generation": 0.0, "cross_review": 0.0, "evaluation": 0.0},
                estimated_time_seconds=0.0,
                risk_level=RiskLevel.LOW,
            )

        # Pick the cheapest available model per task type for estimation
        all_models = list(profiles.keys())

        total_gen_cost    = 0.0
        total_review_cost = 0.0
        total_eval_cost   = 0.0
        total_time        = 0.0

        for task in tasks:
            task_type = getattr(task, "type", TaskType.CODE_GEN)
            cheapest = predictor.cheapest_model(task_type, all_models) or (all_models[0] if all_models else None)
            if cheapest is None:
                continue

            per_call = predictor.predict(cheapest, task_type)
            max_iter    = getattr(task, "max_iterations", 1) or 1
            gen_cost    = per_call * max_iter              # N generation calls (one per iteration)
            review_cost = per_call * max_iter              # N cross-review calls (one per iteration)
            eval_cost   = per_call * _EVAL_OVERHEAD        # ~15% for evaluation (once per task)

            total_gen_cost    += gen_cost
            total_review_cost += review_cost
            total_eval_cost   += eval_cost
            total_time        += _TYPICAL_TASK_SECONDS.get(task_type, 60.0)

        estimated_total = total_gen_cost + total_review_cost + total_eval_cost

        per_phase = {
            "generation":   total_gen_cost,
            "cross_review": total_review_cost,
            "evaluation":   total_eval_cost,
        }

        # Risk level relative to budget
        if budget is not None and budget.max_usd > 0:
            ratio = estimated_total / budget.max_usd
        elif budget is not None:
            ratio = 1.0
        else:
            ratio = 0.0   # no budget → unknown risk, treat as LOW

        if ratio >= 0.8:
            risk = RiskLevel.HIGH
        elif ratio >= 0.5:
            risk = RiskLevel.MEDIUM
        else:
            risk = RiskLevel.LOW

        return ForecastReport(
            estimated_total_usd=estimated_total,
            estimated_per_phase=per_phase,
            estimated_time_seconds=total_time,
            risk_level=risk,
        )
