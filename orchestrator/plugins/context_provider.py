"""
Context Provider — Pluggable Prompt Enrichment ABC
====================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Domain-specific plugin surface for context enrichment. Implementations
live in ~/.orchestrator/plugins/context/<name>/__init__.py or bundled.

Providers implement enrich() to augment generation prompts with
additional context before the prompt reaches the LLM. This is
ephemeral — the enriched prompt is never persisted to project state.

Integration: Called from engine.py._execute_task() after dependency
context is built but BEFORE the LLM call. The enriched prompt is sent
to the LLM but NOT saved in TaskResult.output or project state.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ContextProviderMetadata:
    """Metadata about a context enrichment provider."""

    name: str
    version: str = "1.0.0"
    description: str = ""
    author: str = ""


@dataclass
class EnrichmentResult:
    """Result of a context enrichment call.

    Attributes:
        enriched_prompt: The prompt with injected enrichment content.
        enrichment_added: Whether any enrichment was actually added.
            False when the provider had nothing to contribute.
    """

    enriched_prompt: str
    enrichment_added: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# ContextProvider ABC
# ─────────────────────────────────────────────────────────────────────────────


class ContextProvider(ABC):
    """Pluggable context enrichment for task prompts.

    Providers implement enrich() to inject additional context into
    the generation prompt. The enriched prompt is ephemeral — it is
    sent to the LLM but NOT stored in TaskResult or project state.

    Lifecycle:
        1. provider.initialize(orchestrator_home)  — on registration
        2. provider.enrich(prompt, task_type, project_context) — before LLM call
        3. provider.shutdown()                     — on orchestrator shutdown
    """

    metadata: ContextProviderMetadata

    @abstractmethod
    async def initialize(self, orchestrator_home: Any) -> None:
        """Initialize provider resources."""
        ...

    @abstractmethod
    async def shutdown(self) -> None:
        """Release provider resources."""
        ...

    @abstractmethod
    async def enrich(
        self,
        prompt: str,
        task_type: str,
        project_context: str = "",
    ) -> EnrichmentResult:
        """Enrich a generation prompt with additional context.

        Args:
            prompt: The current prompt text for this task.
            task_type: Task type string (e.g. "code_generation").
            project_context: Dependency context from prior tasks.

        Returns:
            EnrichmentResult with the enriched prompt. If the provider
            has nothing to contribute, return the original prompt unchanged
            with enrichment_added=False.
        """
        ...
