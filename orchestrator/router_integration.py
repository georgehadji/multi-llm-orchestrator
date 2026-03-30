"""
Integration: Outcome-Weighted Router with Existing Engine
=========================================================

Connects the new OutcomeWeightedRouter with the existing orchestrator engine
for seamless adoption without breaking changes.

Usage:
    # Option 1: Use new router (recommended)
    from orchestrator.router_integration import get_smart_router
    router = get_smart_router()

    # Option 2: Use legacy routing (backward compatible)
    from orchestrator.models import ROUTING_TABLE

    # Option 3: Hybrid mode
    from orchestrator.router_integration import HybridRouter
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .adaptive_router import AdaptiveRouter, get_adaptive_router
from .log_config import get_logger
from .models import ROUTING_TABLE, Model, Task, TaskType
from .outcome_router import (
    OutcomeWeightedRouter,
    RoutingStrategy,
    create_routing_context,
    get_outcome_router,
)

if TYPE_CHECKING:
    from .feedback_loop import CodebaseFingerprint

logger = get_logger(__name__)


class HybridRouter:
    """
    Hybrid router that blends legacy and outcome-weighted routing.

    Provides smooth migration path:
    - Phase 1: 90% legacy, 10% outcome-weighted
    - Phase 2: 50/50
    - Phase 3: 100% outcome-weighted
    """

    def __init__(
        self,
        legacy_weight: float = 0.3,
        outcome_router: OutcomeWeightedRouter | None = None,
        adaptive_router: AdaptiveRouter | None = None,
    ):
        self.legacy_weight = legacy_weight
        self.outcome = outcome_router or get_outcome_router()
        self.adaptive = adaptive_router or get_adaptive_router()

    async def select_model(
        self,
        task: Task,
        task_type: TaskType,
        budget_remaining: float,
        budget_total: float,
        codebase_fingerprint: CodebaseFingerprint | None = None,
        strategy: RoutingStrategy = RoutingStrategy.BALANCED,
    ) -> Model:
        """
        Select model using hybrid approach.

        If we have production data, weight toward outcome router.
        Otherwise, fall back to legacy routing.
        """
        # Check if we have production data for this task type
        has_production_data = self._has_production_data(task_type)

        if not has_production_data or self.legacy_weight >= 1.0:
            # Pure legacy routing
            return self._legacy_route(task_type)

        if self.legacy_weight <= 0.0:
            # Pure outcome-weighted routing
            model, _ = await self.outcome.select_model(
                create_routing_context(
                    task=task,
                    budget_remaining=budget_remaining,
                    budget_total=budget_total,
                    strategy=strategy,
                    codebase_fingerprint=codebase_fingerprint,
                )
            )
            return model

        # Hybrid: blend both approaches
        legacy_model = self._legacy_route(task_type)
        outcome_model, metadata = await self.outcome.select_model(
            create_routing_context(
                task=task,
                budget_remaining=budget_remaining,
                budget_total=budget_total,
                strategy=strategy,
                codebase_fingerprint=codebase_fingerprint,
            )
        )

        # If they agree, use that model
        if legacy_model == outcome_model:
            return legacy_model

        # Weighted random selection
        import random
        if random.random() < self.legacy_weight:
            logger.debug(f"Hybrid router: selected legacy choice {legacy_model.value}")
            return legacy_model
        else:
            logger.debug(f"Hybrid router: selected outcome-weighted choice {outcome_model.value}")
            return outcome_model

    def _legacy_route(self, task_type: TaskType) -> Model:
        """Legacy routing using ROUTING_TABLE and adaptive router."""
        candidates = ROUTING_TABLE.get(task_type, list(Model))

        # Filter by health
        healthy = [m for m in candidates if self.adaptive.is_available(m)]
        if not healthy:
            healthy = candidates

        # Prefer lowest latency
        return self.adaptive.preferred_model(healthy, task_type) or healthy[0]

    def _has_production_data(self, task_type: TaskType) -> bool:
        """Check if we have production data for a task type."""
        total_samples = sum(
            record.total_deployments
            for (model, tt), record in self.outcome.feedback._performance_records.items()
            if tt == task_type
        )
        return total_samples >= 5  # Minimum threshold

    def set_legacy_weight(self, weight: float) -> None:
        """Adjust legacy routing weight (0.0 - 1.0)."""
        self.legacy_weight = max(0.0, min(1.0, weight))
        logger.info(f"Hybrid router: legacy weight set to {self.legacy_weight}")


class SmartRouter:
    """
    Intelligent router that automatically selects the best strategy.

    - Uses outcome-weighted routing when production data exists
    - Falls back to adaptive routing for new task types
    - Handles circuit breaker integration
    """

    def __init__(self):
        self.outcome = get_outcome_router()
        self.adaptive = get_adaptive_router()
        self._fallback_chain: list[Model] = []

    async def route(
        self,
        task: Task,
        task_type: TaskType,
        budget_remaining: float,
        budget_total: float,
        codebase_fingerprint: CodebaseFingerprint | None = None,
    ) -> tuple[Model, dict[str, Any]]:
        """
        Smart routing with automatic fallback.

        Returns: (selected_model, decision_metadata)
        """
        # Try outcome-weighted routing first
        try:
            model, metadata = await self.outcome.select_model(
                create_routing_context(
                    task=task,
                    budget_remaining=budget_remaining,
                    budget_total=budget_total,
                    strategy=RoutingStrategy.BALANCED,
                    codebase_fingerprint=codebase_fingerprint,
                )
            )

            # Verify model is healthy
            if self.adaptive.is_available(model):
                return model, {**metadata, "router": "outcome-weighted"}

            # Model unhealthy, use fallback
            logger.warning(f"Selected model {model.value} unhealthy, using fallback")

        except Exception as e:
            logger.error(f"Outcome-weighted routing failed: {e}, falling back")

        # Fallback to legacy routing
        candidates = ROUTING_TABLE.get(task_type, list(Model))
        healthy = [m for m in candidates if self.adaptive.is_available(m)]

        if healthy:
            model = self.adaptive.preferred_model(healthy, task_type) or healthy[0]
            return model, {"router": "legacy", "reason": "fallback"}

        # Ultimate fallback: any available model
        for model in Model:
            if self.adaptive.is_available(model):
                return model, {"router": "emergency", "reason": "all_preferred_unavailable"}

        # Nothing available - return first model and hope for the best
        logger.error("No healthy models available!")
        return list(Model)[0], {"router": "emergency", "reason": "no_healthy_models"}

    def get_routing_report(self) -> dict[str, Any]:
        """Get report on routing decisions and health."""
        return {
            "outcome_router": self.outcome.get_nash_stability_report(),
            "adaptive_router": {
                "degraded_models": [
                    m.value for m in Model
                    if self.adaptive.get_state(m).value == "degraded"
                ],
                "disabled_models": [
                    m.value for m in Model
                    if self.adaptive.get_state(m).value == "disabled"
                ],
            },
        }


# Global instances
_hybrid_router: HybridRouter | None = None
_smart_router: SmartRouter | None = None


def get_hybrid_router(legacy_weight: float = 0.3) -> HybridRouter:
    """Get global hybrid router instance."""
    global _hybrid_router
    if _hybrid_router is None:
        _hybrid_router = HybridRouter(legacy_weight=legacy_weight)
    return _hybrid_router


def get_smart_router() -> SmartRouter:
    """Get global smart router instance."""
    global _smart_router
    if _smart_router is None:
        _smart_router = SmartRouter()
    return _smart_router


def reset_routers() -> None:
    """Reset all routers (for testing)."""
    global _hybrid_router, _smart_router
    _hybrid_router = None
    _smart_router = None
