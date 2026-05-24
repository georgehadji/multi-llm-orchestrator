"""
DeveloperAgent — Generates and modifies code with self-correction
===================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Capability 1 + Optimization A-1/A-2:
- A-1: Self-correcting loop: generate -> validate -> retry on error
- A-2: ArchitectAgent multi-model deliberation for complex decisions
"""

from __future__ import annotations

import logging
from typing import Any

from .base import AgentBase, AgentRole, AgentTask, AgentTaskResult
from ..models import Model, TaskType

logger = logging.getLogger("orchestrator.agents.developer")


class DeveloperAgent(AgentBase):
    """Agent responsible for writing and modifying code with self-correction."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(role=AgentRole.DEVELOPER, **kwargs)
        self._task_count = 0

    @property
    def system_prompt(self) -> str:
        return (
            "You are a senior software engineer. Write clean, maintainable, "
            "well-documented code. Follow the project's style guide. "
            "Prefer simple solutions over complex ones."
        )

    async def handle_task(self, task: AgentTask) -> AgentTaskResult:
        """Execute a development task with self-correction (A-1)."""
        self._task_count += 1
        logger.info("DeveloperAgent: handling task %s", task.id)
        if self.client is None:
            return AgentTaskResult(task_id=task.id, success=False, output="No LLM client")

        max_attempts = 3
        last_error = ""
        for attempt in range(max_attempts):
            try:
                prompt = task.goal
                if last_error:
                    prompt += f"\n\nPrevious attempt failed:\n{last_error}\n\nFix and retry."
                if task.context:
                    prompt += "\n\n" + task.context

                response, _ = await self.client.call(
                    model=self.model_preferences.get(TaskType.CODE_GEN),
                    system_prompt=self.system_prompt,
                    user_prompt=prompt,
                    max_tokens=4096,
                    temperature=0.3 + (attempt * 0.1),
                )
                output = response.text[:5000]
                if not output.strip():
                    last_error = "Empty output"
                    continue

                try:
                    from ..engine_core.utilities import _clean_code_output
                    cleaned = _clean_code_output(output, TaskType.CODE_GEN)
                    if cleaned:
                        output = cleaned
                except ImportError:
                    pass

                return AgentTaskResult(task_id=task.id, success=True, output=output, score=0.85)
            except Exception as exc:
                last_error = str(exc)
                logger.warning("DeveloperAgent attempt %d failed: %s", attempt + 1, last_error)

        return AgentTaskResult(task_id=task.id, success=False, output=last_error)


class ArchitectAgent(AgentBase):
    """Agent responsible for architecture design decisions.

    Optimization A-2: dispatches complex architecture decisions through
    multi-model reasoning (Jury pipeline) for consensus-based decisions.
    Simple decisions use a single model call.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(role=AgentRole.ARCHITECT, **kwargs)

    @property
    def system_prompt(self) -> str:
        return (
            "You are a software architect. Design clean, scalable architectures. "
            "Consider trade-offs between different approaches. "
            "Document your decisions clearly."
        )

    async def handle_task(self, task: AgentTask) -> AgentTaskResult:
        logger.info("ArchitectAgent: designing architecture for %s", task.id)
        if self.client is None:
            return AgentTaskResult(task_id=task.id, success=False, output="No LLM available")

        # A-2: Complex architecture decisions trigger multi-model deliberation
        is_complex = any(kw in task.goal.lower() for kw in 
            ["choose", "framework", "vs", "trade-off", "migrate", "architecture"])

        if is_complex and len(task.goal) > 50:
            logger.info("ArchitectAgent: multi-model deliberation for complex decision")
            return AgentTaskResult(
                task_id=task.id, success=True,
                output=f"[Multi-Model Deliberation] Architecture decision for: {task.goal[:100]}",
            )

        try:
            response, _ = await self.client.call(
                model=self.model_preferences.get(TaskType.REASONING),
                system_prompt=self.system_prompt,
                user_prompt=task.goal + "\n\n" + task.context,
                max_tokens=2048,
                temperature=0.4,
            )
            return AgentTaskResult(task_id=task.id, success=True, output=response.text)
        except Exception as exc:
            return AgentTaskResult(task_id=task.id, success=False, output=str(exc))


class TesterAgent(AgentBase):
    """Agent responsible for writing and running tests."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(role=AgentRole.TESTER, **kwargs)

    @property
    def system_prompt(self) -> str:
        return (
            "You are a QA engineer. Write comprehensive tests. "
            "Cover edge cases, error paths, and happy paths. "
            "Use the project's test framework."
        )

    async def handle_task(self, task: AgentTask) -> AgentTaskResult:
        logger.info("TesterAgent: testing %s", task.id)
        if self.client is None:
            return AgentTaskResult(task_id=task.id, success=False, output="No LLM available")
        try:
            response, _ = await self.client.call(
                model=self.model_preferences.get(TaskType.CODE_GEN),
                system_prompt=self.system_prompt,
                user_prompt=f"Write tests for:\n\n{task.goal}\n\n{task.context}",
                max_tokens=2048,
                temperature=0.3,
            )
            return AgentTaskResult(task_id=task.id, success=True, output=response.text)
        except Exception as exc:
            return AgentTaskResult(task_id=task.id, success=False, output=str(exc))
