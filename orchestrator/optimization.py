"""
OptimizationBackend — pluggable scoring strategies for ConstraintPlanner.
=========================================================================
Three concrete backends ship with the orchestrator:

  GreedyBackend      — current scalar: quality × trust / (cost + ε).
                       Default. Behaviour is identical to the original
                       hard-coded _score() formula in planner.py.

  WeightedSumBackend — scalarizes (cost, latency) with configurable
                       α / β weights before applying quality × trust.
                       Useful when latency is a first-class concern
                       (e.g. interactive / real-time pipelines).

  ParetoBackend      — two-step Pareto selection inspired by Lan et al.
                       (ANIT 2023, NSGA-II for micro-service scheduling):
                       Step 1: filter to the Pareto-optimal front on
                               (estimated_cost, avg_latency_ms) — O(N²).
                       Step 2: among non-dominated models pick the one
                               with the highest quality × trust score.
                       Avoids the greedy trap of picking the cheapest
                       model even when a slightly costlier model is both
                       faster AND higher quality.

All backends share the same interface so ConstraintPlanner can swap them
at construction time or at runtime via set_backend().
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional

from .models import Model, TaskType
from .policy import ModelProfile

# ── Shared constant ────────────────────────────────────────────────────────────
_EPSILON: float = 1e-6   # prevents division by zero when cost rounds to 0


# ── Abstract base ─────────────────────────────────────────────────────────────

class OptimizationBackend(ABC):
    """
    Abstract base class for model-selection scoring strategies.

    Implementors receive the filtered candidate list and must return
    the single best model (or None if candidates is empty).

    Parameters passed to select():
      candidates      — list of Model values that survived all 4 filters
                        (api_health, policy, capability, budget).
      profiles        — full dict[Model → ModelProfile] (live, mutable).
      task_type       — the TaskType being served.
      typical_cost_fn — Callable[[ModelProfile, TaskType], float] that
                        returns the pre-computed typical cost for a model.
                        Already implemented in ConstraintPlanner and passed
                        through so backends do not recompute it.
    """

    @abstractmethod
    def select(
        self,
        candidates: list[Model],
        profiles: dict[Model, ModelProfile],
        task_type: TaskType,
        typical_cost_fn: Callable[[ModelProfile, TaskType], float],
    ) -> Optional[Model]:
        """Return the best model from candidates, or None if empty."""


# ── GreedyBackend ─────────────────────────────────────────────────────────────

class GreedyBackend(OptimizationBackend):
    """
    Current default scoring: quality × trust / (cost + ε).

    Tie-break: subtract priority_rank × 1e-6 so that when two models
    are otherwise equal, the one ranked higher in ROUTING_TABLE wins.
    This preserves the static routing table as a soft prior.
    """

    def select(
        self,
        candidates: list[Model],
        profiles: dict[Model, ModelProfile],
        task_type: TaskType,
        typical_cost_fn: Callable[[ModelProfile, TaskType], float],
    ) -> Optional[Model]:
        if not candidates:
            return None

        def _score(m: Model) -> float:
            p = profiles[m]
            cost = typical_cost_fn(p, task_type)
            base = (p.quality_score * p.trust_factor) / (cost + _EPSILON)
            rank = p.capable_task_types.get(task_type, 99)
            return base - rank * 1e-6

        return max(candidates, key=_score)


# ── WeightedSumBackend ────────────────────────────────────────────────────────

class WeightedSumBackend(OptimizationBackend):
    """
    Scalarize (cost, latency) with configurable α / β weights.

    Score formula:
        combined   = α × estimated_cost + β × (avg_latency_ms / latency_scale_ms)
        base_score = quality × trust / (combined + ε)
        final      = base_score − priority_rank × 1e-6   (tie-break)

    Parameters
    ----------
    alpha : float
        Weight for estimated cost. Default 0.5.
    beta : float
        Weight for normalised latency. Default 0.5.
        α + β should sum to 1.0 for interpretable weights.
    latency_scale_ms : float
        Normalisation denominator for latency (default 5000 ms).
        A latency of `latency_scale_ms` contributes the same to the
        combined score as a cost of $1.0 (with α = β = 0.5 and a
        typical task cost of ~$0.001 the cost term will dominate;
        tune alpha/beta to balance).

    Typical usage
    -------------
    # Favour low-latency models over cheap models:
    backend = WeightedSumBackend(alpha=0.2, beta=0.8)
    orch.set_optimization_backend(backend)
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        latency_scale_ms: float = 5000.0,
    ):
        self.alpha = alpha
        self.beta = beta
        self.latency_scale_ms = latency_scale_ms

    def select(
        self,
        candidates: list[Model],
        profiles: dict[Model, ModelProfile],
        task_type: TaskType,
        typical_cost_fn: Callable[[ModelProfile, TaskType], float],
    ) -> Optional[Model]:
        if not candidates:
            return None

        def _score(m: Model) -> float:
            p = profiles[m]
            cost = typical_cost_fn(p, task_type)
            lat_norm = p.avg_latency_ms / self.latency_scale_ms
            combined = self.alpha * cost + self.beta * lat_norm
            base = (p.quality_score * p.trust_factor) / (combined + _EPSILON)
            rank = p.capable_task_types.get(task_type, 99)
            return base - rank * 1e-6

        return max(candidates, key=_score)


# ── ParetoBackend ─────────────────────────────────────────────────────────────

class ParetoBackend(OptimizationBackend):
    """
    Two-step Pareto-optimal selection (Lan et al., ANIT 2023).

    Background: greedy single-objective scoring may discard solutions
    that lie on the cost–latency Pareto frontier.  For example, a model
    that costs 20 % more but runs 5× faster may still be the best choice
    when latency is the bottleneck — but greedy scoring would reject it.

    Algorithm
    ---------
    Step 1: Pareto dominance filter on (cost, latency) — O(N²).
            Model A *dominates* model B iff:
              cost(A) ≤ cost(B)  AND  latency(A) ≤ latency(B)
              AND at least one inequality is strict.
            Only non-dominated models (the Pareto front) advance.

    Step 2: Among Pareto-optimal candidates, return the model with the
            highest quality × trust score (ignores cost/latency — they
            are already Pareto-optimal, so further discrimination should
            favour quality).

    Safety fallback: if the Pareto front computation produces an empty
    set (degenerate case with all models identical on both objectives),
    all candidates are used for step 2.

    Note: With N ≤ 8 models (current ROUTING_TABLE max), the O(N²)
    dominance check is trivially fast and requires no genetic algorithm.
    """

    def select(
        self,
        candidates: list[Model],
        profiles: dict[Model, ModelProfile],
        task_type: TaskType,
        typical_cost_fn: Callable[[ModelProfile, TaskType], float],
    ) -> Optional[Model]:
        if not candidates:
            return None

        # Step 1: compute objectives for each candidate
        costs = {m: typical_cost_fn(profiles[m], task_type) for m in candidates}
        lats  = {m: profiles[m].avg_latency_ms for m in candidates}

        # Pareto dominance filter
        pareto: list[Model] = [
            m for m in candidates
            if not any(
                costs[o] <= costs[m]
                and lats[o] <= lats[m]
                and (costs[o] < costs[m] or lats[o] < lats[m])
                for o in candidates
                if o != m
            )
        ]

        front = pareto if pareto else candidates  # safety: should always be non-empty

        # Step 2: among Pareto-front, maximise quality × trust
        return max(front, key=lambda m: profiles[m].quality_score * profiles[m].trust_factor)
