"""
orchestrator/model_selector.py
───────────────────────────────
Health-aware model selection and tiered routing service.

Refactored to use Domain Services (RoutingService, CostService)
instead of hardcoded tables in models.py.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from .models import Model, TaskType, get_provider

if TYPE_CHECKING:
    from .domain.services.config_services import RoutingService, CostService

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants extracted for routing logic
# ---------------------------------------------------------------------------
_MODEL_TIERS: dict[Model, int] = {
    Model.DEEPSEEK_V4_FLASH: 0,
    Model.DEEPSEEK_V4_PRO: 0,
    Model.QWEN_3_7_MAX: 0,
    Model.QWEN_3_6_FLASH: 0,
    Model.XAI_GROK_4_20: 0,
    Model.CLAUDE_SONNET_4_6: 0,
    Model.CLAUDE_OPUS_4_8: 0,
    Model.GPT_4O: 0,
    Model.GPT_5_4: 0,
    Model.GPT_5_4_CODEX: 0,
    Model.GPT_4O_MINI: 0,
    Model.GEMINI_FLASH: 0,
    Model.GEMINI_FLASH_LITE: 0,
    Model.MOONSHOT_KIMI_K2_6: 0,
    Model.XIAOMI_MIMO_V2_FLASH: 0,
    Model.MINIMAX_M2_7: 0,
    Model.ZHIPU_GLM_5_2: 0,
    Model.ZHIPU_GLM_5_TURBO: 0,
    Model.ZHIPU_GLM_5_2: 0,
    Model.STEPFUN_STEP_3_5_FLASH: 0,
    Model.LLAMA_4_MAVERICK: 0,
    Model.PHI_4: 0,
    Model.CLAUDE_HAIKU_4_5: 0,
    # Image generation models
    Model.NANO_BANANA_2: 1,
    Model.RECRAFT_V4_PRO_VECTOR: 1,
    Model.FLUX_2_KLEIN: 1,
}

_COMPLEXITY_KEYWORDS = [
    "microservice",
    "distributed",
    "kubernetes",
    "cluster",
    "scalable",
    "authentication",
    "authorization",
    "OAuth",
    "JWT",
    "RBAC",
    "permissions",
    "database",
    "migration",
    "replication",
    "sharding",
    "caching",
    "redis",
    "real-time",
    "websocket",
    "streaming",
    "queue",
    "kafka",
    "rabbitmq",
    "multi-tenant",
    "SaaS",
    "API gateway",
    "load balancer",
    "CDN",
    "machine learning",
    "ML",
    "AI",
    "neural",
    "embedding",
    "vector",
]

_TECH_STACK_KEYWORDS = [
    "react",
    "next.js",
    "vue",
    "angular",
    "fastapi",
    "django",
    "flask",
    "express",
    "postgresql",
    "mongodb",
    "mysql",
    "docker",
    "terraform",
    "aws",
    "azure",
    "gcp",
]

_RELIABLE_DECOMPOSITION_MODELS: list[Model] = [
    Model.MOONSHOT_KIMI_K2_7_CODE,
    Model.QWEN_3_7_MAX,
    Model.CLAUDE_SONNET_4_6,
    Model.GPT_4O,
    Model.DEEPSEEK_V4_FLASH,
    Model.GEMINI_FLASH,
    Model.GPT_4O_MINI,
]


class ModelSelector:
    """
    Health-aware model selection service.
    """

    def __init__(
        self,
        api_health: dict[Model, bool],
        routing_service: RoutingService,
        cost_service: CostService,
    ) -> None:
        self._health = api_health
        self._routing = routing_service
        self._costs = cost_service

    def available_models(self, task_type: TaskType) -> list[Model]:
        """Get priority-ordered list of healthy models for a task."""
        candidates = self._routing.get_models_for_task(task_type)
        return [m for m in candidates if self._health.get(m, True)]

    def decomposition_model(self, project_description: str) -> Model:
        """Select best model for project decomposition."""
        for m in _RELIABLE_DECOMPOSITION_MODELS:
            if self._health.get(m, True):
                return m

        # Fallback: find any healthy model, preferring cheaper
        healthy = [m for m in Model if self._health.get(m, True)]
        if not healthy:
            return Model.GPT_4O_MINI  # Ultimate fallback
        return min(healthy, key=lambda m: self._costs.get_cost(m)["output"])

    def select(self, task_type: TaskType) -> "Model | None":
        """Select the top-priority healthy model for a task type."""
        models = self.available_models(task_type)
        return models[0] if models else None

    def reviewer(self, generator: Model, task_type: TaskType) -> Model | None:
        """Select a diverse reviewer model."""
        gen_provider = get_provider(generator)
        candidates = self.available_models(task_type)

        # 1. Cross-provider
        for c in candidates:
            if get_provider(c) != gen_provider:
                return c

        # 2. Same provider, different model
        for c in candidates:
            if c != generator:
                return c

        return None

    def fallback(self, failed_model: Model) -> Model | None:
        """Select fallback after failure."""
        fb = self._routing.get_fallback_for_model(failed_model)
        if fb and self._health.get(fb, True):
            return fb

        # Any other healthy model
        for m in Model:
            if m != failed_model and self._health.get(m, True):
                return m

        return None

    def next_tier(self, current_model: Model, task_type: TaskType) -> Model | None:
        """Escalate to higher tier."""
        current_tier = _MODEL_TIERS.get(current_model, 0)
        candidates = [
            m for m in self.available_models(task_type) if _MODEL_TIERS.get(m, 0) > current_tier
        ]

        if not candidates:
            return None

        return min(candidates, key=lambda m: self._costs.get_cost(m)["output"])


class TieredModelRouter:
    """Tier-aware model routing with escalation tracking."""

    def __init__(
        self,
        api_health: dict[Model, bool],
        routing_service: RoutingService,
        cost_service: CostService,
        adaptive_router: Any | None = None,
    ) -> None:
        self._health = api_health
        self._routing = routing_service
        self._costs = cost_service
        self._adaptive = adaptive_router
        self._escalation_count: dict[str, int] = {}

    def available_models(self, task_type: TaskType) -> list[Model]:
        """Get available models, filtered by health and adaptive routing."""
        candidates = self._routing.get_models_for_task(task_type)
        available = [m for m in candidates if self._health.get(m, True)]

        if self._adaptive and hasattr(self._adaptive, "is_available"):
            available = [m for m in available if self._adaptive.is_available(m)]

        return available

    def escalate_tier(self, task_type: TaskType) -> None:
        """Track escalation level for a task type."""
        tier_key = f"{task_type.value}"
        self._escalation_count[tier_key] = self._escalation_count.get(tier_key, 0) + 1
        logger.info(
            f"Tier escalation for {task_type.value}: level {self._escalation_count[tier_key]}"
        )

    def fast_decomposition_model(self) -> Model:
        """Get a reliable model for decomposition."""
        # Reuse ModelSelector logic for consistency
        selector = ModelSelector(self._health, self._routing, self._costs)
        return selector.decomposition_model("")

    def cheapest_available(self) -> Model:
        """Get the cheapest healthy model available."""
        healthy = [m for m in Model if self._health.get(m, True)]
        if not healthy:
            return Model.GPT_4O_MINI
        return min(healthy, key=lambda m: self._costs.get_cost(m)["output"])
