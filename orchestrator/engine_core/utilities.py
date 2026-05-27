"""
Utilities — Shared functions extracted from engine.py to break circular imports
================================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Phase 8 of the Master Architecture Enhancement Plan.
Functions that were previously only accessible by importing from engine.py
(creating circular import chains) live here instead.

Contains:
- _clean_code_output: Remove LLM artifacts from generated code
- _get_available_models: Filter models by health and task type
- _select_reviewer: Select a reviewer model from a different provider
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from ..exceptions import TruncatedResponseError
from ..models import FALLBACK_CHAIN, ROUTING_TABLE, Model, TaskType, get_provider

if TYPE_CHECKING:
    pass

logger = logging.getLogger("orchestrator.engine_core.utilities")


def _clean_code_output(text: str, task_type: TaskType) -> str:
    """Post-process code output to remove common LLM artifacts.

    - Markdown fences (```language...```)
    - Placeholder comments explaining what to add
    - Explanatory text fragments that aren't code

    Args:
        text: Raw LLM output.
        task_type: Type of task that generated this text.

    Returns:
        Cleaned text.
    """
    if task_type != TaskType.CODE_GEN:
        return text

    # Remove markdown code fences
    text = re.sub(r"^```\w*\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n?```\s*$", "", text, flags=re.MULTILINE)

    # Remove trailing whitespace after fences
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    # Remove placeholder markers commonly hallucinated by models
    placeholder_markers = [
        r"#.*?TODO.*?(?=\n|$)",
        r"//.*?TODO.*?(?=\n|$)",
        r"#.*?FIXME.*?(?=\n|$)",
        r"<!--.*?-->",
        r"#.*?Add your.*?here.*?(?=\n|$)",
        r"/\*.*?\*/",
        r"\{\{.*?\}\}",
        r"#.*?Implement.*?here",
    ]
    for pattern in placeholder_markers:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # Remove explanatory fragments at the end
    tail_phrases = [
        r"\nNote:.*$",
        r"\nNote that.*$",
        r"\nThis is a basic.*$",
        r"\nYou can customize.*$",
        r"\nLet me know if.*$",
    ]
    for phrase in tail_phrases:
        text = re.sub(phrase, "", text, flags=re.IGNORECASE | re.DOTALL)

    return text.strip()


def _get_available_models(
    task_type: Any = None, api_health: dict[Model, bool] | None = None
) -> list[Model]:
    """Get available models for a task type, filtered by health.

    Args:
        task_type: Optional task type for routing table lookup.
        api_health: Optional per-model health dict. If None, all models are
                    considered available.

    Returns:
        List of healthy models appropriate for the task type.
    """
    if api_health is None:
        api_health = {}

    routing = ROUTING_TABLE.get(task_type, [])
    return [m for m in routing if api_health.get(m, True)]


def _select_reviewer(
    primary: Model,
    task_type: Any = None,
    api_health: dict[Model, bool] | None = None,
) -> Model | None:
    """Select a reviewer model from a different provider than the primary.

    This provides cross-provider diversity — the reviewer model sees
    the output fresh, without sharing the primary model's biases.

    Args:
        primary: The model that generated the output.
        task_type: Task type for routing table lookup.
        api_health: Optional per-model health dict.

    Returns:
        A reviewer Model from a different provider, or None if unavailable.
    """
    if api_health is None:
        api_health = {}

    primary_provider = get_provider(primary)
    routing = ROUTING_TABLE.get(task_type, [])

    for m in routing:
        if get_provider(m) != primary_provider and api_health.get(m, True):
            return m

    # Fallback: try the fallback chain
    fallback = FALLBACK_CHAIN.get(primary)
    if fallback is not None and api_health.get(fallback, True) is True:
        return fallback

    return None
