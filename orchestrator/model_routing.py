"""
ModelRouting — Tier-based model selection strategy
====================================================
Maps task phases to model quality tiers (PREMIUM/STANDARD/ECONOMY)
and selects the appropriate model for each tier.
Distinct from adaptive_router.py (health state) — this handles tier selection.

Pattern: Strategy + Lookup Tables
Async: No — pure computation
Layer: L1 Infrastructure
"""
from __future__ import annotations

import logging
from enum import Enum

logger = logging.getLogger("orchestrator.model_routing")


class ModelTier(Enum):
    """Quality/cost tiers for model selection."""

    PREMIUM = "premium"   # highest quality, highest cost
    STANDARD = "standard"  # good quality, moderate cost
    ECONOMY = "economy"   # fast and cheap


TIER_ROUTING: dict[ModelTier, list[str]] = {
    # PREMIUM: frontier models — highest quality, higher cost
    ModelTier.PREMIUM: [
        "gpt-4o",
        "claude-3-5-sonnet-20241022",
        "gemini-1.5-pro",
        "meta-llama/llama-3.1-405b-instruct",   # OR: 405B open-source near-frontier
    ],
    # STANDARD: strong models at moderate cost
    ModelTier.STANDARD: [
        "gpt-4o-mini",
        "claude-3-haiku-20240307",
        "deepseek-chat",
        "meta-llama/llama-4-maverick",           # OR: 400B MoE, $0.17 flat
        "meta-llama/llama-3.3-70b-instruct",     # OR: 70B battle-tested
        "nousresearch/hermes-3-llama-3.1-70b",  # OR: tool-use fine-tune
    ],
    # ECONOMY: cheapest capable models — fast and low-cost
    ModelTier.ECONOMY: [
        "deepseek-chat",
        "gemini-1.5-flash",
        "meta-llama/llama-4-scout",              # OR: 109B MoE, $0.11/$0.34
        "meta-llama/llama-3.3-70b-instruct",     # OR: reliable 70B
        "microsoft/phi-4",                       # OR: 14B, excellent $/quality
        "google/gemma-3-27b-it",                 # OR: Google open-weights 27B
        "openrouter/auto",                       # OR: dynamic routing (cheapest fit)
    ],
}

PHASE_TO_TIER: dict[str, ModelTier] = {
    "reasoning": ModelTier.PREMIUM,
    "code_gen": ModelTier.PREMIUM,
    "code_review": ModelTier.STANDARD,
    "writing": ModelTier.STANDARD,
    "summarize": ModelTier.ECONOMY,
    "data_extract": ModelTier.ECONOMY,
    "evaluate": ModelTier.STANDARD,
}


def select_model(tier: ModelTier, preferred: str | None = None) -> str:
    """Return the best model for *tier*, honouring *preferred* if it belongs to that tier.

    Args:
        tier: The quality/cost tier to select from.
        preferred: Optional caller-supplied model name.  Returned as-is when it
                   appears in the tier's model list; ignored otherwise.

    Returns:
        A model name string.  Never raises — always returns a valid entry.
    """
    models = TIER_ROUTING[tier]
    if preferred is not None and preferred in models:
        logger.debug("select_model: preferred=%r accepted for tier=%s", preferred, tier.value)
        return preferred
    selected = models[0]
    if preferred is not None:
        logger.debug(
            "select_model: preferred=%r not in tier=%s, falling back to %r",
            preferred,
            tier.value,
            selected,
        )
    return selected


def get_tier_for_phase(phase: str) -> ModelTier:
    """Return the :class:`ModelTier` mapped to *phase*.

    Falls back to :attr:`ModelTier.STANDARD` for unknown phases (fail-open).

    Args:
        phase: A task-phase string (e.g. ``"reasoning"``, ``"summarize"``).

    Returns:
        The corresponding :class:`ModelTier`.
    """
    tier = PHASE_TO_TIER.get(phase, ModelTier.STANDARD)
    if phase not in PHASE_TO_TIER:
        logger.debug("get_tier_for_phase: unknown phase=%r, defaulting to STANDARD", phase)
    return tier
