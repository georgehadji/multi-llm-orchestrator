"""
ReviewerAgent — Code review and security audit agent
======================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Reviews generated code for bugs, security issues, and quality.
Uses cross-model critique: budget models for standard tasks,
premium models for critical security review.
"""

from __future__ import annotations

import logging
from typing import Any

from .base import AgentBase, AgentRole, AgentTask, AgentTaskResult
from ..models import Model, TaskType

logger = logging.getLogger("orchestrator.agents.reviewer")


class ReviewerAgent(AgentBase):
    """Reviews code for bugs, security issues, and quality.

    Budget:  QWEN_3_6_PLUS ($0.325/M) — thorough review, large context
    Premium: XAI_GROK_4_3 ($1.25/M) — strong security analysis
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(role=AgentRole.REVIEWER, **kwargs)

    @property
    def system_prompt(self) -> str:
        return (
            "You are a senior code reviewer. Analyze the code for: "
            "bugs, security vulnerabilities, performance issues, "
            "and style violations. Be specific and actionable."
        )

    async def handle_task(self, task: AgentTask) -> AgentTaskResult:
        logger.info("ReviewerAgent: reviewing %s", task.id)
        if self.client is None:
            return AgentTaskResult(task_id=task.id, success=False, output="No LLM client")
        try:
            response, _ = await self.client.call(
                model=self.model_preferences.get(TaskType.CODE_REVIEW),
                system_prompt=self.system_prompt,
                user_prompt=task.goal + "\n\n" + task.context,
                max_tokens=2048,
                temperature=0.2,
            )
            return AgentTaskResult(task_id=task.id, success=True, output=response.text)
        except Exception as exc:
            return AgentTaskResult(task_id=task.id, success=False, output=str(exc))
