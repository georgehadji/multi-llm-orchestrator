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
# ---------------------------------------------------------------------------
_MODEL_TIERS: dict[Model, int] = {
    # Cheap tier (0)
    Model.GEMINI_FLASH_LITE: 0,
    Model.GPT_4O_MINI: 0,
    # Balanced tier (1)
    Model.GEMINI_FLASH: 1,
    Model.DEEPSEEK_CHAT: 1,
    Model.CLAUDE_3_HAIKU: 1,
    # Premium tier (2)
    Model.GPT_4O: 2,
    Model.DEEPSEEK_REASONER: 2,
    Model.GEMINI_PRO: 2,
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

        Decision priority (v3.0):
          1. QWEN_3_CODER_NEXT  — best JSON structure capability
          2. XIAOMI_MIMO_V2_FLASH
          3. GEMINI_FLASH
          4. STEPFUN_STEP_3_5_FLASH  (last resort)
        """
        # Complexity scoring kept for logging/observability (not yet used for
        # routing since v3.0 always prefers Qwen3 Coder Next).
        project_lower = project_description.lower()
        complexity_score = sum(1 for kw in _COMPLEXITY_KEYWORDS if kw in project_lower)
        tech_score = sum(1 for kw in _TECH_STACK_KEYWORDS if kw in project_lower)
        total_complexity = complexity_score + (tech_score // 2)  # noqa: F841

        if self._health.get(Model.QWEN_3_CODER_NEXT, True):
            logger.debug("P1-2: Using Qwen3 Coder Next for decomposition (best JSON structure)")
            return Model.QWEN_3_CODER_NEXT

        if self._health.get(Model.XIAOMI_MIMO_V2_FLASH, True):
            logger.debug("P1-2: Using MiMo-V2-Flash for decomposition")
            return Model.XIAOMI_MIMO_V2_FLASH

        if self._health.get(Model.GEMINI_FLASH, True):
            logger.debug("P1-2: Using Gemini Flash for decomposition")
            return Model.GEMINI_FLASH

        logger.warning("P1-2: Using Step 3.5 Flash as fallback (may have JSON issues)")
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
