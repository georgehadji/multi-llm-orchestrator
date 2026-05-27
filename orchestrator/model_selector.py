"""
orchestrator/model_selector.py
───────────────────────────────
Health-aware model selection service.

Extracted from engine.py per T1-C of the Architecture Enhancement Plan.
Contains routing business logic that must not live in the Mediator (engine.py).

Rules:
  - No I/O, no asyncio, no engine imports.
  - All selection decisions are deterministic given the same api_health state.
  - `available_models_fn` is injected so this module has no dependency on engine.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from .models import COST_TABLE, FALLBACK_CHAIN, Model, TaskType, get_provider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Single tier — most capable models only (May 2026 optimization)
# All models at same priority; ordering determines selection preference
# ---------------------------------------------------------------------------
# Single tier — most capable models only
# All models at same priority; ordering in _TIER_MODELS determines selection
_MODEL_TIERS: dict[Model, int] = {
    Model.DEEPSEEK_V4_FLASH: 0,
    Model.DEEPSEEK_V4_PRO: 0,
    Model.QWEN_3_7_MAX: 0,
    Model.QWEN_3_6_FLASH: 0,
    Model.XAI_GROK_4_20: 0,
    Model.CLAUDE_SONNET_4_6: 0,
    Model.CLAUDE_OPUS_4_6: 0,
    Model.GPT_4O: 0,
    Model.GPT_5_4: 0,
    Model.GPT_5_4_CODEX: 0,
    Model.GPT_4O_MINI: 0,
    Model.GEMINI_FLASH: 0,
    Model.GEMINI_FLASH_LITE: 0,
    Model.MOONSHOT_KIMI_K2_6: 0,
    Model.XIAOMI_MIMO_V2_FLASH: 0,
    Model.MINIMAX_M2_7: 0,
    Model.ZHIPU_GLM_5_1: 0,
    Model.ZHIPU_GLM_5_TURBO: 0,
    Model.STEPFUN_STEP_3_5_FLASH: 0,
    Model.LLAMA_4_MAVERICK: 0,
    Model.PHI_4: 0,
    Model.CLAUDE_3_HAIKU: 0,
}

# ---------------------------------------------------------------------------
# Complexity keyword lists (used by decomposition_model)
# ---------------------------------------------------------------------------
_COMPLEXITY_KEYWORDS = [
    # Architecture patterns
    "microservice",
    "distributed",
    "kubernetes",
    "cluster",
    "scalable",
    # Security
    "authentication",
    "authorization",
    "OAuth",
    "JWT",
    "RBAC",
    "permissions",
    # Data
    "database",
    "migration",
    "replication",
    "sharding",
    "caching",
    "redis",
    # Real-time
    "real-time",
    "websocket",
    "streaming",
    "queue",
    "kafka",
    "rabbitmq",
    # Advanced features
    "multi-tenant",
    "SaaS",
    "API gateway",
    "load balancer",
    "CDN",
    # ML/AI
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
    "angular",  # Frontend frameworks
    "fastapi",
    "django",
    "flask",
    "express",  # Backend frameworks
    "postgresql",
    "mongodb",
    "mysql",  # Databases
    "docker",
    "terraform",
    "aws",
    "azure",
    "gcp",  # DevOps
]


# ─────────────────────────────────────────────────────────────────────────
# Tier-aware model routing (PHASE 2 of ENGINE_OPTIMIZATION_PLAN)
# ─────────────────────────────────────────────────────────────────────────

# Updated tier definitions (v3.0) — consolidated from engine.py
# Single tier — most capable models, ordered by benchlm.ai coding score
_TIER_MODELS: list[Model] = [
    Model.QWEN_3_7_MAX,  # 92.2 benchlm, #4 globally
    Model.DEEPSEEK_V4_PRO,  # 90.1 benchlm, $1.50/$6.00
    Model.MOONSHOT_KIMI_K2_6,  # 89.2 benchlm, $0.95/$4.00
    Model.GPT_5_4_CODEX,  # 87.8 benchlm, $1.75/$14.00
    Model.GPT_5_4,  # 87.8 benchlm, $2.50/$15.00
    Model.DEEPSEEK_V4_FLASH,  # 83.5 benchlm, $0.27/$1.10
    Model.ZHIPU_GLM_5_1,  # 83.4 benchlm, $0.10/$0.40
    Model.CLAUDE_SONNET_4_6,  # 82.2 benchlm, $3.00/$15.00
    Model.XAI_GROK_4_20,  # lowest hallucination
    Model.QWEN_3_6_FLASH,  # 79.1 benchlm, $0.12/$0.50
    Model.GEMINI_FLASH,  # 78.3 benchlm, $0.15/$0.60
    Model.XIAOMI_MIMO_V2_FLASH,  # $0.09/$0.29 -- ultra-cheap backup
    Model.MINIMAX_M2_7,  # $0.30/$1.20
    Model.CLAUDE_OPUS_4_6,  # 86.0 benchlm, most capable
    Model.GPT_4O,  # $2.50/$10.00 -- reliable
]

# Reliable decomposition models (v3.1)
_RELIABLE_DECOMPOSITION_MODELS: list[Model] = [
    Model.QWEN_3_7_MAX,  # 92.2 benchlm, #4 globally -- BEST
    Model.CLAUDE_SONNET_4_6,  # 82.2 benchlm, excellent structured output
    Model.GPT_4O,  # $2.50/$10.00, reliable JSON, large context
    Model.DEEPSEEK_V4_FLASH,  # 83.5 benchlm, $0.27/$1.10, fast + reliable
    Model.GEMINI_FLASH,  # $0.15/$0.60, cheap reliable backup
    Model.GPT_4O_MINI,  # $0.15/$0.60, cheapest reliable
]


class TieredModelRouter:
    """Tier-aware model routing with escalation tracking.

    Extracted from engine.py per PHASE 2 of the ENGINE_OPTIMIZATION_PLAN.
    Handles three-tier routing (CHEAP -> BALANCED -> PREMIUM), escalation
    tracking, and health-aware candidate filtering.
    """

    def __init__(
        self,
        api_health: dict[Model, bool],
        adaptive_router: object | None = None,
    ) -> None:
        self._health = api_health
        self._adaptive = adaptive_router
        self._escalation_count: dict[str, int] = {}

    def available_models(self, task_type: TaskType) -> list[Model]:
        """Get available models from the single premium tier.

        Uses ROUTING_TABLE as priority order, filtered by health.
        Falls back to the global _TIER_MODELS list if ROUTING_TABLE is empty.
        """
        candidates = ROUTING_TABLE.get(task_type, []) or list(_TIER_MODELS)
        available = [m for m in candidates if self._health.get(m, True)]
        if not available:
            available = [m for m in _TIER_MODELS if self._health.get(m, True)]
        if self._adaptive and hasattr(self._adaptive, "is_available"):
            available = [m for m in available if self._adaptive.is_available(m)]
        return available

    def escalate_tier(self, task_type: TaskType) -> None:
        """Escalate to higher tier after cheap tier failure."""
        tier_key = f"{task_type.value}"
        self._escalation_count[tier_key] = self._escalation_count.get(tier_key, 0) + 1
        logger.info(
            f"Tier escalation for {task_type.value}: level {self._escalation_count[tier_key]}"
        )

    def fast_decomposition_model(self) -> Model:
        """Get a fast, reliable model for task decomposition.

        Prioritizes reliability over cost for decomposition,
        since decomposition is a critical path and happens once per project.
        """
        for m in _RELIABLE_DECOMPOSITION_MODELS:
            if self._health.get(m, True):
                logger.debug(f"Using {m.value} for decomposition")
                return m

        # Fallback to cheapest available
        return self._cheapest_available()

    def cheapest_available(self) -> Model:
        """Return the cheapest healthy model by output cost."""
        healthy = [m for m in Model if self._health.get(m, False)]
        if not healthy:
            raise RuntimeError("No healthy models available")
        return min(healthy, key=lambda m: COST_TABLE[m]["output"])


class ModelSelector:
    """
    Health-aware model selection service.

    Accepts api_health (mutable dict updated by engine) and a callable that
    returns the list of models valid for a given task type. Both are injected
    so this class has zero dependency on the engine or any I/O layer.
    """

    def __init__(
        self,
        api_health: dict[Model, bool],
        available_models_fn: Callable[[TaskType], list[Model]],
    ) -> None:
        # Holds a reference — engine mutates this dict in place, so selector
        # always sees the current health state without needing to be recreated.
        self._health = api_health
        self._available = available_models_fn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decomposition_model(self, project_description: str) -> Model:
        """
        Select the best model for project decomposition based on project
        complexity and model health.

        Decision priority (v3.1 - FIXED for truncation issues):
          1. GPT_4O — reliable JSON, large context
          2. CLAUDE_SONNET_4_6 — excellent structure
          3. GEMINI_FLASH — reliable fallback
          4. QWEN_3_6_FLASH — last resort (truncation issues)
        """
        # Complexity scoring kept for logging/observability
        project_lower = project_description.lower()
        complexity_score = sum(1 for kw in _COMPLEXITY_KEYWORDS if kw in project_lower)
        tech_score = sum(1 for kw in _TECH_STACK_KEYWORDS if kw in project_lower)
        total_complexity = complexity_score + (tech_score // 2)

        # Log complexity for observability
        logger.debug(f"Project complexity score: {total_complexity}")

        # Use reliable models first (avoiding truncation issues)
        if self._health.get(Model.GPT_4O, True):
            logger.debug("P1-2: Using GPT-4o for decomposition (reliable JSON)")
            return Model.GPT_4O

        if self._health.get(Model.CLAUDE_SONNET_4_6, True):
            logger.debug("P1-2: Using Claude Sonnet 4.6 for decomposition")
            return Model.CLAUDE_SONNET_4_6

        if self._health.get(Model.GEMINI_FLASH, True):
            logger.debug("P1-2: Using Gemini Flash for decomposition")
            return Model.GEMINI_FLASH

        # Last resort - any healthy model
        logger.warning("P1-2: Using fallback model for decomposition")
        if self._health.get(Model.DEEPSEEK_V4_FLASH, True):
            return Model.DEEPSEEK_V4_FLASH

        if self._health.get(Model.QWEN_3_6_FLASH, True):
            return Model.QWEN_3_6_FLASH

        logger.error("P1-2: No healthy models available for decomposition")
        return Model.STEPFUN_STEP_3_5_FLASH

    def reviewer(self, generator: Model, task_type: TaskType) -> Model | None:
        """
        Select a reviewer model that is different from the generator.

        Prefers cross-provider diversity for independent review; falls back to
        any healthy model from a different identity if no cross-provider option
        is available.
        """
        gen_provider = get_provider(generator)
        candidates = self._available(task_type)

        # Prefer cross-provider reviewer
        for c in candidates:
            if get_provider(c) != gen_provider and self._health.get(c, False):
                return c

        # Fall back to any healthy model that isn't the generator
        for c in candidates:
            if c != generator and self._health.get(c, False):
                return c

        return None

    def fallback(self, failed_model: Model) -> Model | None:
        """
        Return a fallback model after `failed_model` fails.

        Consults FALLBACK_CHAIN first; scans all healthy models as last resort.
        """
        fb = FALLBACK_CHAIN.get(failed_model)
        if fb and self._health.get(fb, False):
            return fb

        for m in Model:
            if m != failed_model and self._health.get(m, False):
                return m

        return None

    def next_tier(self, current_model: Model, task_type: TaskType) -> Model | None:
        """
        Escalate to a higher-tier model when a quality plateau is detected.

        Tiers: CHEAP (0) → BALANCED (1) → PREMIUM (2).
        Returns the cheapest healthy model one tier above current, or None if
        already at the top tier or no valid candidates exist.
        """
        current_tier = _MODEL_TIERS.get(current_model, 1)

        candidates = [
            m
            for m in Model
            if _MODEL_TIERS.get(m, 1) > current_tier
            and self._health.get(m, False)
            and m in self._available(task_type)
            and m in COST_TABLE  # skip models without cost data
        ]

        if not candidates:
            return None

        return min(candidates, key=lambda m: COST_TABLE[m]["output"])
