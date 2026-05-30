"""
ContextService — LLM context construction and management
==========================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Handles system prompt construction and project-wide context gathering.
Extracted from engine.py to dismantle the God Object.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ..prompt_builder import SystemPrompt

if TYPE_CHECKING:
    from ..models import TaskResult

logger = logging.getLogger("orchestrator.engine_core.context_service")


class ContextService:
    """Service for building prompts and gathering project context."""

    def __init__(self, quality_mode: str = "standard"):
        self._quality_mode = quality_mode

    def build_system_prompt(self, task_type: str = "") -> str:
        """Build system prompt based on configured quality mode."""
        return SystemPrompt.build(task_type, self._quality_mode)

    def build_project_context(self, results: dict[str, TaskResult]) -> str:
        """Gather context from successful task results."""
        context_parts = []
        # Limit to first 3 results for context efficiency
        for task_id, result in list(results.items())[:3]:
            if result.status.value in ("completed", "degraded") and result.output:
                snippet = result.output[:2000]
                context_parts.append(f"## {task_id}\n```python\n{snippet}\n```")

        if context_parts:
            return "## Existing Code Context\n\n" + "\n\n".join(context_parts)
        return ""
