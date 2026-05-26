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
# Model tier definitions (used by next_tier)
# Higher index = higher quality / cost
# NOTE: DeepSeek models removed due to timeout issues
#       Replaced with Xiaomi, Zhipu, StepFun (reliable alternatives)
# ---------------------------------------------------------------------------
_MODEL_TIERS: dict[Model, int] = {
    # Cheap tier (0) - Reliable, fast models
    Model.GEMINI_FLASH_LITE: 0,
    Model.GPT_4O_MINI: 0,
    Model.ZHIPU_GLM_5_1: 0,  # z-ai/glm-5.1, canonical GLM model
    Model.PHI_4: 0,  # $0.07/$0.14, very fast
    # Balanced tier (1) - Good quality/price ratio
    Model.GEMINI_FLASH: 1,
    Model.XIAOMI_MIMO_V2_FLASH: 1,  # $0.09/$0.29, #1 SWE-bench, reliable
    Model.STEPFUN_STEP_3_5_FLASH: 1,  # $0.10/$0.30, 196B MoE, reliable
    Model.CLAUDE_3_HAIKU: 1,
    # Premium tier (2) - High quality
    Model.GPT_4O: 2,
    Model.GEMINI_PRO: 2,
    Model.CLAUDE_SONNET_4_6: 2,  # $3/$15, excellent coding
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
_TIER_MODELS_CHEAP: list[Model] = [
    Model.QWEN_3_CODER_NEXT,  # $0.12/$0.75 - Fast coding specialist
    Model.XIAOMI_MIMO_V2_FLASH,  # $0.09/$0.29 - #1 SWE-bench, fast
    Model.ZHIPU_GLM_5_1,
    Model.STEPFUN_STEP_3_5_FLASH,  # $0.10/$0.30 - 196B MoE reasoning
    Model.PHI_4,  # $0.07/$0.14 - Microsoft 14B
    Model.GEMMA_3_27B,  # $0.08/$0.20 - Google open-weights
    Model.LLAMA_3_3_70B,  # $0.12/$0.30 - Meta 70B reliable
    Model.NVIDIA_NEMOTRON_3_SUPER,  # $0.10/$0.50 - 120B MoE efficient
]

_TIER_MODELS_BALANCED: list[Model] = [
    Model.DEEPSEEK_V3_2,  # $0.27/$1.10 - 1.24T tokens
    Model.MOONSHOT_KIMI_K2_5,  # $0.42/$2.20 - Visual coding SOTA
    Model.MINIMAX_M2_7,  # $0.30/$1.20 - 56.2% SWE-Pro
    Model.GEMINI_FLASH,  # $0.15/$0.60 - 1M context, fast
    Model.CLAUDE_3_HAIKU,  # $0.25/$1.25 - Claude budget tier
    Model.DEEPSEEK_CHAT,  # $0.28/$0.42 - Cost effective
    Model.XIAOMI_MIMO_V2_PRO,  # $1.00/$3.00 - 1T+ params (slow)
]

_TIER_MODELS_PREMIUM: list[Model] = [
    Model.XAI_GROK_4_20_BETA,  # $2.00/$6.00 - Lowest hallucination
    Model.CLAUDE_SONNET_4_6,  # $3.00/$15.00 - Best coding
    Model.QWEN_3_5_397B_A17B,  # $0.39/$2.34 - 397B MoE SOTA
    Model.GPT_5_4_CODEX,  # $1.75/$14.00 - SWE-Bench Pro SOTA
    Model.GEMINI_PRO,  # $2.00/$12.00 - Gemini premium
    Model.O4_MINI,  # $1.50/$6.00 - OpenAI reasoning
]

# Reliable decomposition models (v3.1)
_RELIABLE_DECOMPOSITION_MODELS: list[Model] = [
    Model.GPT_4O,  # $2.50/$10.00 - Reliable JSON, large context
    Model.CLAUDE_SONNET_4_6,  # $3.00/$15.00 - Excellent structure
    Model.GEMINI_FLASH,  # $0.15/$0.60 - Reliable JSON output
    Model.GPT_4O_MINI,  # $0.15/$0.60 - Cheap, reliable
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
        """Get available models with tiered selection for cost optimization.

        Uses three-tier routing: CHEAP -> BALANCED -> PREMIUM.
        Starts with cheaper models and escalates if needed based on
        task complexity and previous failures.
        """
        tier_key = f"{task_type.value}"
        escalation = self._escalation_count.get(tier_key, 0)

        if escalation == 0:
            if task_type in (TaskType.DATA_EXTRACT, TaskType.SUMMARIZE):
                candidates = list(_TIER_MODELS_CHEAP + _TIER_MODELS_BALANCED)
            else:
                candidates = list(_TIER_MODELS_BALANCED + _TIER_MODELS_CHEAP)
        elif escalation == 1:
            candidates = list(_TIER_MODELS_BALANCED + _TIER_MODELS_PREMIUM)
        else:
            candidates = list(ROUTING_TABLE.get(task_type, []))

        available = [m for m in candidates if self._health.get(m, True)]
        if not available:
            available = [m for m in Model if self._health.get(m, True)]

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
          4. QWEN_3_CODER_NEXT — last resort (truncation issues)
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

        # Last resort - models with known issues
        logger.warning("P1-2: Using Qwen/Xiaomi as fallback (may have truncation issues)")
        if self._health.get(Model.QWEN_3_CODER_NEXT, True):
            return Model.QWEN_3_CODER_NEXT
        
        if self._health.get(Model.XIAOMI_MIMO_V2_FLASH, True):
            return Model.XIAOMI_MIMO_V2_FLASH
            
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
