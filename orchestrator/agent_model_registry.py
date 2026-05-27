"""
Agent Model Registry — Centralised LLM model assignments per agent role
========================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Single source of truth for which model each agent uses in budget and
premium tiers. All model references use the Model enum from models.py
so they are validated at import time.

Usage:
    from orchestrator.agent_model_registry import AGENT_MODELS, get_model_for

    # Look up models for an agent role
    entry = AGENT_MODELS[AgentRole.DEVELOPER]
    budget_model = entry.budget    # Model enum value
    premium_model = entry.premium  # Model enum value

    # Build model_preferences dict for AgentBase
    prefs = entry.to_model_preferences()

Design principles:
  - Budget tier:  most capable model under $1/1M input, prioritising coding
                  specialists (SWE-bench) and reasoning capability.
  - Premium tier: best available model regardless of cost, favouring
                  low-hallucination, large-context, frontier models.
  - All models MUST exist in the Model enum — verified at import time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from .agents.base import AgentRole
from .models import Model, TaskType


@dataclass(frozen=True)
class AgentModelEntry:
    """Model assignments for one agent role.

    Attributes:
        role:         The AgentRole this entry applies to.
        budget:       Cost-effective model for everyday use.
        premium:      Best available model, cost not a concern.
        task_type:    Which TaskType this agent routes through (for
                      model_preferences dict construction).
        rationale:    Why these models were chosen.
    """

    role: AgentRole
    budget: Model
    premium: Model
    task_type: TaskType
    rationale: str

    def to_model_preferences(self) -> dict[TaskType, Model]:
        """Build the dict consumed by AgentBase.__init__(model_preferences=...).

        Returns a single-entry dict mapping this agent's task_type to its
        budget model. Callers can override the budget tier by passing a
        different dict.
        """
        return {self.task_type: self.budget}

    def premium_preferences(self) -> dict[TaskType, Model]:
        """Same as to_model_preferences() but using the premium tier."""
        return {self.task_type: self.premium}


# ═══════════════════════════════════════════════════════════════════════════════
# Centralised Registry — single place for all agent→model assignments
# ═══════════════════════════════════════════════════════════════════════════════

AGENT_MODELS: dict[AgentRole, AgentModelEntry] = {
    # ═══════════════════════════════════════════════════════════════════════
    # Single tier — most capable model per agent (May 2026)
    # All agents use the same tier; budget == premium (single model)
    # ═══════════════════════════════════════════════════════════════════════
    AgentRole.ARCHITECT: AgentModelEntry(
        role=AgentRole.ARCHITECT,
        budget=Model.QWEN_3_7_MAX,
        premium=Model.QWEN_3_7_MAX,
        task_type=TaskType.REASONING,
        rationale="92.2 benchlm, #4 globally — best reasoning for system design",
    ),
    AgentRole.DEVELOPER: AgentModelEntry(
        role=AgentRole.DEVELOPER,
        budget=Model.DEEPSEEK_V4_FLASH,
        premium=Model.DEEPSEEK_V4_FLASH,
        task_type=TaskType.CODE_GEN,
        rationale="83.5 benchlm coding — beats Claude Sonnet at 10x less cost",
    ),
    AgentRole.TESTER: AgentModelEntry(
        role=AgentRole.TESTER,
        budget=Model.DEEPSEEK_V4_FLASH,
        premium=Model.DEEPSEEK_V4_FLASH,
        task_type=TaskType.CODE_GEN,
        rationale="83.5 benchlm — fast, cheap, reliable test generation",
    ),
    AgentRole.REVIEWER: AgentModelEntry(
        role=AgentRole.REVIEWER,
        budget=Model.DEEPSEEK_V4_PRO,
        premium=Model.DEEPSEEK_V4_PRO,
        task_type=TaskType.CODE_REVIEW,
        rationale="90.1 benchlm — best reasoning chains for bug detection",
    ),
    AgentRole.DEVOPS: AgentModelEntry(
        role=AgentRole.DEVOPS,
        budget=Model.DEEPSEEK_V4_FLASH,
        premium=Model.DEEPSEEK_V4_FLASH,
        task_type=TaskType.CODE_GEN,
        rationale="83.5 benchlm, 1M context — excellent for infra-as-code",
    ),
    AgentRole.RESEARCHER: AgentModelEntry(
        role=AgentRole.RESEARCHER,
        budget=Model.MOONSHOT_KIMI_K2_6,
        premium=Model.MOONSHOT_KIMI_K2_6,
        task_type=TaskType.DATA_EXTRACT,
        rationale="89.2 benchlm, 256K context — best for multi-source research",
    ),
    AgentRole.PRODUCT_MANAGER: AgentModelEntry(
        role=AgentRole.PRODUCT_MANAGER,
        budget=Model.QWEN_3_7_MAX,
        premium=Model.QWEN_3_7_MAX,
        task_type=TaskType.REASONING,
        rationale="92.2 benchlm — #4 globally, best for trade-off analysis",
    ),
    AgentRole.QA: AgentModelEntry(
        role=AgentRole.QA,
        budget=Model.CLAUDE_SONNET_4_6,
        premium=Model.CLAUDE_SONNET_4_6,
        task_type=TaskType.CODE_REVIEW,
        rationale="82.2 benchlm — best quality analysis for critical releases",
    ),
    AgentRole.USER: AgentModelEntry(
        role=AgentRole.USER,
        budget=Model.CLAUDE_SONNET_4_6,
        premium=Model.CLAUDE_SONNET_4_6,
        task_type=TaskType.WRITING,
        rationale="82.2 benchlm — best conversational quality for premium UX",
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════


def get_model_for(role: AgentRole, tier: str = "budget") -> Model:
    """Look up a model for an agent role and tier.

    Args:
        role: The AgentRole to look up.
        tier: "budget" (default) or "premium".

    Returns:
        Model enum value.

    Raises:
        KeyError: If the role is not in AGENT_MODELS.
        ValueError: If tier is not "budget" or "premium".
    """
    entry = AGENT_MODELS[role]
    if tier == "premium":
        return entry.premium
    if tier == "budget":
        return entry.budget
    raise ValueError(f"Unknown tier '{tier}'. Use 'budget' or 'premium'.")


def build_all_model_preferences(tier: str = "budget") -> dict[AgentRole, dict[TaskType, Model]]:
    """Build model_preferences dicts for all agent roles.

    Args:
        tier: "budget" (default) or "premium".

    Returns:
        Dict mapping AgentRole → {TaskType → Model}, suitable for passing
        to AgentBase.__init__(model_preferences=...).
    """
    result: dict[AgentRole, dict[TaskType, Model]] = {}
    for role, entry in AGENT_MODELS.items():
        if tier == "premium":
            result[role] = entry.premium_preferences()
        else:
            result[role] = entry.to_model_preferences()
    return result


def get_default_model_preferences(role: AgentRole, tier: str = "budget") -> dict[TaskType, Model]:
    """Convenience: get the default model_preferences for a single agent.

    Args:
        role: The AgentRole.
        tier: "budget" (default) or "premium".

    Returns:
        Dict {TaskType → Model} for AgentBase.__init__(model_preferences=...).
    """
    return (
        AGENT_MODELS[role].to_model_preferences()
        if tier == "budget"
        else AGENT_MODELS[role].premium_preferences()
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Validation — run at import time to catch stale model references
# ═══════════════════════════════════════════════════════════════════════════════


def _validate_all_models() -> None:
    """Verify every model in AGENT_MODELS is a valid Model enum member."""
    for role, entry in AGENT_MODELS.items():
        assert isinstance(
            entry.budget, Model
        ), f"{role.value}.budget = {entry.budget!r} is not a Model enum value"
        assert isinstance(
            entry.premium, Model
        ), f"{role.value}.premium = {entry.premium!r} is not a Model enum value"
        assert isinstance(
            entry.task_type, TaskType
        ), f"{role.value}.task_type = {entry.task_type!r} is not a TaskType enum value"


_validate_all_models()


__all__ = [
    "AGENT_MODELS",
    "AgentModelEntry",
    "get_model_for",
    "build_all_model_preferences",
    "get_default_model_preferences",
]
