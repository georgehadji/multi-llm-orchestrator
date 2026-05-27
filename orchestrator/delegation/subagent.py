"""
SubAgent — Lightweight Isolated Agent for Parallel Task Execution
====================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

A SubAgent wraps a single task with an isolated context, budget slice,
and role gating. Leaf agents cannot delegate further; orchestrator
agents can (up to max_spawn_depth).

Integration: Created and managed by BatchRunner. Each SubAgent gets:
- A proportional budget slice from the parent
- A unique task_id for isolation
- Role gating (leaf vs orchestrator)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..models import Task, TaskResult
    from ..budget import Budget

logger = logging.getLogger("orchestrator.delegation.subagent")


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SubAgentConfig:
    """Configuration for a single SubAgent.

    Attributes:
        max_iterations: Maximum LLM iterations for the execution loop.
        role: "leaf" (default) — cannot delegate further.
              "orchestrator" — can spawn its own subagents.
        inherit_memory: Whether to pass memory context to the subagent.
        enabled_toolsets: Subset of tools to expose. None = all.
    """

    max_iterations: int = 90
    role: str = "leaf"
    inherit_memory: bool = True
    enabled_toolsets: list[str] | None = None


# ─────────────────────────────────────────────────────────────────────────────
# SubAgent
# ─────────────────────────────────────────────────────────────────────────────


class SubAgent:
    """Lightweight isolated agent for parallel task execution.

    Each SubAgent receives a budget slice and a single task. It calls
    the provided execute_fn (which is the Orchestrator's _execute_task
    method) but wrapped with role enforcement.

    Usage:
        config = SubAgentConfig(role="leaf")
        agent = SubAgent(config, budget_slice, execute_fn)
        result = await agent.execute(task, dependency_context)
    """

    def __init__(
        self,
        config: SubAgentConfig,
        budget_slice: "Budget",
        execute_fn: Any = None,
    ) -> None:
        """Initialize subagent.

        Args:
            config: SubAgentConfig with role, iterations, toolsets.
            budget_slice: A Budget instance with a fraction of the parent budget.
            execute_fn: Async callable ``(Task) -> TaskResult``. Typically
                ``Orchestrator._execute_task``.
        """
        self.config = config
        self.budget = budget_slice
        self._execute_fn = execute_fn

    async def execute(
        self,
        task: "Task",
        dependency_context: str = "",
    ) -> "TaskResult":
        """Execute a single task in an isolated context.

        Args:
            task: The Task to execute.
            dependency_context: Context string from dependency outputs.

        Returns:
            The completed TaskResult.
        """
        # Update task context with dependency info
        if dependency_context:
            task.context = dependency_context

        if self._execute_fn is not None:
            result = await self._execute_fn(task)
        else:
            # Standalone mode: raise if no execute_fn provided
            raise RuntimeError(
                f"SubAgent '{task.id}' has no execute_fn — " f"cannot execute task in isolation"
            )

        return result

    @property
    def is_leaf(self) -> bool:
        """True if this agent cannot delegate further."""
        return self.config.role == "leaf"
