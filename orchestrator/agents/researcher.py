"""
ResearcherAgent — Web search and research agent
==================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Searches the web for documentation, best practices, libraries,
and solutions. Uses Perplexity Sonar for search-enhanced generation.
"""

from __future__ import annotations

import logging
from typing import Any

from .base import AgentBase, AgentRole, AgentTask, AgentTaskResult
from ..models import Model, TaskType

logger = logging.getLogger("orchestrator.agents.researcher")


class ResearcherAgent(AgentBase):
    """Searches the web and synthesizes research findings.

    Budget:  SONAR_PRO ($3.00/M) — web-search enhanced
    Premium: SONAR_DEEP_RESEARCH ($2.00/M) — multi-source deep research
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(role=AgentRole.RESEARCHER, **kwargs)

    @property
    def system_prompt(self) -> str:
        return (
            "You are a research assistant. Search the web for the most "
            "up-to-date information, documentation, and best practices. "
            "Cite your sources and summarize findings clearly."
        )

    async def handle_task(self, task: AgentTask) -> AgentTaskResult:
        logger.info("ResearcherAgent: researching %s", task.id)
        if self.client is None:
            return AgentTaskResult(task_id=task.id, success=False, output="No LLM client")
        try:
            response, _ = await self.client.call(
                model=self.model_preferences.get(TaskType.DATA_EXTRACT),
                system_prompt=self.system_prompt,
                user_prompt=task.goal + "\n\n" + task.context,
                max_tokens=2048,
                temperature=0.3,
            )
            return AgentTaskResult(task_id=task.id, success=True, output=response.text)
        except Exception as exc:
            return AgentTaskResult(task_id=task.id, success=False, output=str(exc))
