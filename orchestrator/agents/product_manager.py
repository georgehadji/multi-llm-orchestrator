"""
ProductManagerAgent — User stories, requirements, prioritization
==================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Converts high-level goals into prioritized user stories with acceptance criteria.
"""

from __future__ import annotations

import logging
from typing import Any

from .base import AgentBase, AgentRole, AgentTask, AgentTaskResult
from ..models import TaskType

logger = logging.getLogger("orchestrator.agents.product_manager")


class ProductManagerAgent(AgentBase):
    """Generates user stories, maps to modules, prioritizes.

    Models (from orchestrator/agent_model_registry.py):
      Budget:  QWEN_3_7_MAX       ($0.78/M in, $3.90/M out) — flagship reasoning + coding
      Premium: CLAUDE_SONNET_4_6  ($3.00/M in, $15.00/M out) — best requirement analysis
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(role=AgentRole.PRODUCT_MANAGER, **kwargs)

    @property
    def system_prompt(self) -> str:
        return (
            "You are a product manager. Convert high-level goals into "
            "user stories using the format: 'As a <role>, I want <feature> "
            "so that <value>'. Assign priorities (P0-P3) and acceptance criteria. "
            "Be specific and actionable."
        )

    async def handle_task(self, task: AgentTask) -> AgentTaskResult:
        logger.info("ProductManager: processing %s", task.id)
        if self.client is None:
            return AgentTaskResult(task_id=task.id, success=False, output="No LLM")

        try:
            response, _ = await self.client.call(
                model=self.model_preferences.get(TaskType.REASONING),
                system_prompt=self.system_prompt,
                user_prompt=f"Generate user stories and requirements for:\n\n{task.goal}\n\n{task.context}",
                max_tokens=2048,
                temperature=0.3,
            )
            return AgentTaskResult(task_id=task.id, success=True, output=response.text)
        except Exception as exc:
            return AgentTaskResult(task_id=task.id, success=False, output=str(exc))
