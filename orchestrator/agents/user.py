"""
UserAgent — Talks to the user in natural language
====================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Translates between the AgentOrchestrator and the human user.
Explains plans, answers questions, presents decisions for input,
and summarizes completion. Uses `print()` and `input()` for
interactive chat via the CLI.
"""

from __future__ import annotations

import logging
from typing import Any

from .base import AgentBase, AgentRole, AgentTask, AgentTaskResult
from ..models import TaskType

logger = logging.getLogger("orchestrator.agents.user")


class UserAgent(AgentBase):
    """Talks to the user. Explains plans, asks questions, reports progress.

    Budget/Free:  OWL_ALPHA ($0.00/M) — free, never costs anything
    Premium:      CLAUDE_SONNET_4_6 ($3.00/M) — best conversational quality
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(role=AgentRole.USER, **kwargs)

    @property
    def system_prompt(self) -> str:
        return (
            "You are the user-facing voice of an AI development team. "
            "Your job is to explain plans clearly, answer user questions "
            "about what the system is doing and why, and present decisions "
            "for human approval. Be concise and honest."
        )

    async def handle_task(self, task: AgentTask) -> AgentTaskResult:
        """Handle a user-facing task — present info or ask a question."""
        if task.goal.startswith("ask:"):
            return await self._ask_user(task)
        return await self._inform_user(task)

    async def _ask_user(self, task: AgentTask) -> AgentTaskResult:
        """Present a decision to the user and collect their input."""
        question = task.goal.replace("ask:", "").strip()
        print(f"\n[Agent] {question}")
        response = input("You: ")
        return AgentTaskResult(
            task_id=task.id,
            success=True,
            output=response,
            messages=[f"Asked: {question}", f"User said: {response}"],
        )

    async def _inform_user(self, task: AgentTask) -> AgentTaskResult:
        """Explain something to the user — no input needed."""
        print(f"\n[Agent] {task.goal}")
        if task.context:
            print(f"  {task.context}")
        return AgentTaskResult(
            task_id=task.id,
            success=True,
            output=task.goal,
        )

    async def present_plan(self, plan_summary: str) -> None:
        """Present a plan to the user before execution starts."""
        print(f"\n{'='*60}")
        print("  PLAN")
        print(f"{'='*60}")
        print(plan_summary)
        print(f"{'='*60}")

    async def confirm(self, question: str) -> bool:
        """Ask the user yes/no and return their answer."""
        print(f"\n[Agent] {question}")
        response = input("Yes/no: ").strip().lower()
        return response in ("yes", "y", "yeah", "sure", "ok", "proceed")

    async def report_progress(self, task_id: str, status: str, detail: str = "") -> None:
        """Report task progress to the user."""
        icon = {"completed": "DONE", "failed": "FAIL", "running": "..."}.get(status, "?")
        print(f"  [{icon}] {task_id}: {detail or status}")
