"""
ConstraintPlanner — policy-compliant, multi-objective model selection.
======================================================================
Replaces the three static lookups in engine.py:
  _get_available_models()  →  select_model()
  _select_reviewer()       →  select_reviewer()
  _get_fallback()          →  replan()

Selection algorithm (select_model):
  1. Filter: api_health[model] is True
  2. Filter: PolicyEngine.check() passes for the active policies
  3. Filter: task_type in profile.capable_task_types
  4. Filter: estimated_cost(typical_tokens) <= budget_remaining
  5. Score:  quality_score × trust_factor / (estimated_cost + EPSILON)
     Tie-break: subtract priority_rank × 1e-6 to preserve ROUTING_TABLE ordering
                when scores are otherwise equal
  6. Return highest-scored model, or None if nothing survives all filters.

The planner returns None instead of raising PolicyViolationError.
The caller (engine.py) is responsible for deciding how to handle a None return:
either degrade the task, raise PolicyViolationError, or fail fast.

Novelty: this is a constrained multi-objective optimisation at the level of
*semantic tasks* (code_gen, review, evaluate), not just infrastructure routing.
The joint optimisation over cost × latency × compliance × quality × trust
cannot be expressed in a static lookup table.
"""
from __future__ import annotations

import logging
from typing import Optional

from .models import Model, TaskType, get_provider
from .optimization import GreedyBackend, OptimizationBackend
from .policy import ModelProfile, Policy
from .policy_engine import PolicyEngine

logger = logging.getLogger("orchestrator.planner")

# ── Scoring constants ──────────────────────────────────────────────────────────
_EPSILON: float = 1e-6          # prevents division by zero when cost rounds to 0

# ── Typical token volumes per task type ───────────────────────────────────────
# Used to estimate cost *before* the call (for budget filtering and scoring).
# Actual costs are charged via Budget.charge() after the real call completes.
_TYPICAL_INPUT_TOKENS: dict[TaskType, int] = {
    TaskType.CODE_GEN:     800,
    TaskType.CODE_REVIEW:  600,
    TaskType.REASONING:    700,
    TaskType.WRITING:      500,
    TaskType.DATA_EXTRACT: 400,
    TaskType.SUMMARIZE:    300,
    TaskType.EVALUATE:     400,
}
_TYPICAL_OUTPUT_TOKENS: dict[TaskType, int] = {
    TaskType.CODE_GEN:     1200,
    TaskType.CODE_REVIEW:  900,
    TaskType.REASONING:    900,
    TaskType.WRITING:      800,
    TaskType.DATA_EXTRACT: 600,
    TaskType.SUMMARIZE:    300,
    TaskType.EVALUATE:     400,
}


class ConstraintPlanner:
    """
    Selects models subject to policy constraints, budget limits, and
    multi-objective scoring. Stateless between calls; all mutable state
    lives in ModelProfile objects (updated by TelemetryCollector).

    The api_health dict is shared with Orchestrator and reflects live
    provider availability — unhealthy providers are immediately excluded.
    """

    def __init__(
        self,
        profiles: dict[Model, ModelProfile],
        policy_engine: PolicyEngine,
        api_health: dict[Model, bool],
        backend: Optional[OptimizationBackend] = None,  # noqa: F821 — used below
    ):
        self._profiles = profiles
        self._policy_engine = policy_engine
        self._api_health = api_health
        self._backend: OptimizationBackend = backend or GreedyBackend()

    def set_backend(self, backend: OptimizationBackend) -> None:
        """
        Swap the optimization strategy at runtime.

        Example
        -------
            from orchestrator.optimization import ParetoBackend
            planner.set_backend(ParetoBackend())
        """
        self._backend = backend

    # ─────────────────────────────────────────────────────────────────────────
    # Primary selection
    # ─────────────────────────────────────────────────────────────────────────

    def select_model(
        self,
        task_type: TaskType,
        policies: list[Policy],
        budget_remaining: float,
        task_id: str = "",
    ) -> Optional[Model]:
        """
        Multi-objective model selection for a task.

        Returns the best compliant model or None if no model survives filters.
        Does NOT raise PolicyViolationError — that is the caller's responsibility.

        Parameters
        ----------
        task_type : TaskType
            The semantic type of the task (CODE_GEN, REVIEW, etc.)
        policies : list[Policy]
            Merged active policies (global + node-level) from PolicySet.policies_for().
        budget_remaining : float
            Remaining budget in USD. Models whose estimated task cost exceeds
            this value are filtered out.
        task_id : str
            For logging context only.
        """
        candidates = self._apply_filters(task_type, policies, budget_remaining)
        if not candidates:
            logger.warning(
                "select_model: no candidates for task_type=%s task_id=%r",
                task_type.value, task_id,
            )
            return None

        best = self._backend.select(
            candidates, self._profiles, task_type, self._estimate_typical_cost
        )
        if best is not None:
            logger.info(
                "select_model: chose %s for %s (backend=%s, task=%s)",
                best.value, task_type.value,
                type(self._backend).__name__, task_id,
            )
        return best

    def select_reviewer(
        self,
        generator: Model,
        task_type: TaskType,
        policies: list[Policy],
        budget_remaining: float,
    ) -> Optional[Model]:
        """
        Select a cross-provider reviewer for the output of `generator`.

        Invariant preserved from the original engine: reviewer provider ≠ generator
        provider when cross-provider options exist (prevents shared-bias blind spots).
        Falls back to a different model from the same provider if no cross-provider
        options survive all filters.

        Parameters
        ----------
        generator : Model
            The model that produced the output to be reviewed.
        task_type : TaskType
            Used to filter capable reviewer candidates.
        policies : list[Policy]
            Active policies — applied identically for reviewer selection.
        budget_remaining : float
            Budget filter applied to reviewers.
        """
        gen_provider = get_provider(generator)
        candidates = self._apply_filters(task_type, policies, budget_remaining)

        # Prefer cross-provider reviewer
        cross_provider = [
            m for m in candidates
            if get_provider(m) != gen_provider and m != generator
        ]
        if cross_provider:
            best = self._backend.select(
                cross_provider, self._profiles, task_type, self._estimate_typical_cost
            )
            if best is not None:
                logger.info(
                    "select_reviewer: cross-provider %s for generator=%s",
                    best.value, generator.value,
                )
                return best

        # Fallback: any different model (same provider allowed)
        different_model = [m for m in candidates if m != generator]
        if different_model:
            best = self._backend.select(
                different_model, self._profiles, task_type, self._estimate_typical_cost
            )
            if best is not None:
                logger.warning(
                    "select_reviewer: same-provider fallback %s "
                    "(no cross-provider candidates available)", best.value,
                )
                return best

        logger.warning(
            "select_reviewer: no reviewer available for generator=%s",
            generator.value,
        )
        return None

    def replan(
        self,
        failed_model: Model,
        task_type: TaskType,
        policies: list[Policy],
        budget_remaining: float,
    ) -> Optional[Model]:
        """
        Called when a model errors or its output fails deterministic validation.
        Excludes failed_model from the candidate set and re-runs selection.

        This replaces the static FALLBACK_CHAIN lookup. The static chain is
        now only used to initialize default profiles; at runtime, replan()
        makes a fresh multi-objective selection from the surviving candidates,
        so the fallback choice is informed by live telemetry and policy state.

        Returns None if no alternative model survives all filters.
        """
        candidates = self._apply_filters(task_type, policies, budget_remaining)
        remaining = [m for m in candidates if m != failed_model]
        if not remaining:
            logger.warning(
                "replan: no alternative to %s for task_type=%s",
                failed_model.value, task_type.value,
            )
            return None

        best = self._backend.select(
            remaining, self._profiles, task_type, self._estimate_typical_cost
        )
        if best is not None:
            logger.info(
                "replan: selected %s as fallback for failed %s",
                best.value, failed_model.value,
            )
        return best

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_filters(
        self,
        task_type: TaskType,
        policies: list[Policy],
        budget_remaining: float,
    ) -> list[Model]:
        """
        Returns models that survive all four filters:
          1. API health
          2. Policy compliance
          3. Capability for task_type
          4. Budget (estimated cost ≤ remaining)
        """
        result = []
        for model, profile in self._profiles.items():

            # Filter 1: API health (live, shared with Orchestrator)
            if not self._api_health.get(model, False):
                continue

            # Filter 2: Policy compliance
            check = self._policy_engine.check(model, profile, policies)
            if not check.passed:
                logger.debug(
                    "_apply_filters: %s excluded by policy — %s",
                    model.value, check.violations,
                )
                continue

            # Filter 3: Capability — model must appear in ROUTING_TABLE for this task
            if task_type not in profile.capable_task_types:
                continue

            # Filter 4: Budget
            estimated = self._estimate_typical_cost(profile, task_type)
            if estimated > budget_remaining:
                logger.debug(
                    "_apply_filters: %s excluded by budget "
                    "(est $%.4f > remaining $%.4f)",
                    model.value, estimated, budget_remaining,
                )
                continue

            result.append(model)

        return result

    def _estimate_typical_cost(
        self,
        profile: ModelProfile,
        task_type: TaskType,
    ) -> float:
        """
        Estimate the cost of a single call for the given task type using the
        typical token volumes defined at module level.
        """
        input_t = _TYPICAL_INPUT_TOKENS.get(task_type, 500)
        output_t = _TYPICAL_OUTPUT_TOKENS.get(task_type, 600)
        return profile.estimate_cost(input_t, output_t)

    def _score(self, model: Model, task_type: TaskType) -> float:
        """
        Multi-objective score: quality × trust / (estimated_cost + ε)

        Higher is better. The priority_rank from ROUTING_TABLE is used as a
        secondary tie-breaker: score -= priority_rank × 1e-6, so when two
        models are equally good by the primary objective, the one preferred
        by the original routing table wins. This means the static table's
        engineering intuition is preserved as a soft prior.
        """
        profile = self._profiles[model]
        estimated_cost = self._estimate_typical_cost(profile, task_type)
        base_score = (
            profile.quality_score * profile.trust_factor
            / (estimated_cost + _EPSILON)
        )
        priority_rank = profile.capable_task_types.get(task_type, 99)
        return base_score - priority_rank * 1e-6
