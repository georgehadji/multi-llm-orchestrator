"""
DevOpsAgent — Infrastructure and deployment agent
====================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Handles Docker, CI/CD, configuration files, and deployment scripts.
"""

from __future__ import annotations

import logging
from typing import Any

from .base import AgentBase, AgentRole, AgentTask, AgentTaskResult
from ..models import Model, TaskType

logger = logging.getLogger("orchestrator.agents.devops")


class DevOpsAgent(AgentBase):
    """Handles infrastructure, Docker, CI/CD, and deployment.

    Models (from orchestrator/agent_model_registry.py):
      Budget:  DEEPSEEK_V4_FLASH   ($0.27/M in, $1.10/M out) — large context for configs
      Premium: GPT_5_4_CODEX   ($1.75/M in, $14.00/M out) — best infra-as-code
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(role=AgentRole.DEVOPS, **kwargs)

    @property
    def system_prompt(self) -> str:
        return (
            "You are a DevOps engineer. Write Dockerfiles, CI/CD pipelines, "
            "and infrastructure config. Follow security best practices: "
            "don't run as root, use multi-stage builds, pin versions."
        )

    async def handle_task(self, task: AgentTask) -> AgentTaskResult:
        logger.info("DevOpsAgent: handling %s", task.id)
        if self.client is None:
            return AgentTaskResult(task_id=task.id, success=False, output="No LLM client")
        try:
            response, _ = await self.client.call(
                model=self.model_preferences.get(TaskType.CODE_GEN),
                system_prompt=self.system_prompt,
                user_prompt=task.goal + "\n\n" + task.context,
                max_tokens=2048,
                temperature=0.3,
            )
            return AgentTaskResult(task_id=task.id, success=True, output=response.text)
        except Exception as exc:
            return AgentTaskResult(task_id=task.id, success=False, output=str(exc))
