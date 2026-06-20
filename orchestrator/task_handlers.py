"""
Task Handler Protocol — Typed Dispatch for Task Execution
=========================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Replaces string-keyed ROUTING_TABLE dispatch with a typed TaskHandler
protocol that the AST can trace. Every call path becomes EXTRACTED
with confidence 1.0 instead of INFERRED at 0.54.

Design:
  - TaskHandler is a Protocol — structural typing, no base class coupling
  - Registry populated at import time via @register decorator
  - Handlers are stateless — all dependencies injected at construction
  - Fallback model selection lives in the handler, not in engine.py

Usage:
    from .task_handlers import get_handler, register

    @register(TaskType.CODE_GEN)
    class CodeGenerationHandler:
        task_type = TaskType.CODE_GEN
        async def execute(self, task, client, budget, **kwargs) -> TaskResult:
            ...
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

from .models import Model, Task, TaskResult, TaskType, ROUTING_TABLE

logger = logging.getLogger("orchestrator.task_handlers")

# ═══════════════════════════════════════════════════════════════════════════════
# Protocol
# ═══════════════════════════════════════════════════════════════════════════════


@runtime_checkable
class TaskHandler(Protocol):
    """
    Protocol for task-type-specific execution handlers.

    Each handler implements ``execute()`` which receives the task, client,
    and budget, and returns a TaskResult. Handlers are registered for a
    TaskType via the ``@register`` decorator.

    Example::

        @register(TaskType.CODE_GEN)
        class CodeGenHandler:
            task_type = TaskType.CODE_GEN

            async def execute(self, task, client, budget, **kwargs) -> TaskResult:
                ...
    """

    task_type: TaskType

    async def execute(
        self,
        task: Task,
        client: Any,
        budget: Any,
        **kwargs: Any,
    ) -> TaskResult: ...


# ═══════════════════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════════════

_HANDLER_REGISTRY: dict[TaskType, type[TaskHandler]] = {}


def register(task_type: TaskType):
    """
    Decorator to register a handler class for a TaskType.

    Example::

        @register(TaskType.CODE_GEN)
        class CodeGenerationHandler:
            ...
    """

    def decorator(cls: type[TaskHandler]) -> type[TaskHandler]:
        cls.task_type = task_type
        _HANDLER_REGISTRY[task_type] = cls
        logger.debug("Registered handler %s for %s", cls.__name__, task_type.value)
        return cls

    return decorator


def get_handler(task_type: TaskType) -> type[TaskHandler]:
    """
    Get the registered handler class for a TaskType.

    Raises ``KeyError`` if no handler is registered for the given task type.
    """
    if task_type not in _HANDLER_REGISTRY:
        raise KeyError(
            f"No handler registered for {task_type.value}. "
            f"Registered: {list(_HANDLER_REGISTRY.keys())}"
        )
    return _HANDLER_REGISTRY[task_type]


def get_handler_or_none(task_type: TaskType) -> type[TaskHandler] | None:
    """Like ``get_handler()`` but returns None instead of raising."""
    return _HANDLER_REGISTRY.get(task_type)


def registered_types() -> list[TaskType]:
    """Return all TaskTypes that have registered handlers."""
    return list(_HANDLER_REGISTRY.keys())


# ═══════════════════════════════════════════════════════════════════════════════
# Built-in Handlers
# ═══════════════════════════════════════════════════════════════════════════════

from .api_clients import UnifiedClient
from .budget import Budget
from .prompt_builder import SystemPrompt


class _BaseHandler:
    """Common logic for all handlers — not registered directly."""

    task_type: TaskType

    def _get_model(self, task: Task) -> Model:
        """Select model for this task type from ROUTING_TABLE."""
        return ROUTING_TABLE.get(self.task_type, Model.GPT_4O_MINI)

    async def _call_llm(
        self,
        client: UnifiedClient,
        prompt: str,
        system: str = "",
        model: Model | None = None,
        max_tokens: int = 1500,
        temperature: float = 0.3,
        timeout: int = 120,
        **kwargs,
    ) -> str:
        """Single LLM call with standard error handling."""
        response = await client.call(
            model or self._get_model(None),  # type: ignore
            prompt,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            **kwargs,
        )
        return response.text


@register(TaskType.CODE_GEN)
class CodeGenerationHandler(_BaseHandler):
    """Handler for code generation tasks — produces runnable source files."""

    task_type = TaskType.CODE_GEN

    async def execute(
        self,
        task: Task,
        client: UnifiedClient,
        budget: Budget,
        **kwargs,
    ) -> TaskResult:
        model = self._get_model(task)
        system = SystemPrompt.build("code_gen", kwargs.get("mode", "standard"))
        prompt = task.prompt

        # Insert dependency context if provided
        dep_context = kwargs.get("dependency_context", "")
        if dep_context:
            prompt = f"{prompt}\n\n--- CONTEXT FROM PRIOR TASKS ---\n{dep_context}"

        text = await self._call_llm(
            client,
            prompt,
            system=system,
            model=model,
            max_tokens=task.max_output_tokens,
            temperature=0.0,
            timeout=120,
        )

        # Clean code output
        from .engine_core.utilities import _clean_code_output

        output = _clean_code_output(text, task.type)

        return TaskResult(
            task_id=task.id,
            output=output,
            score=0.0,  # To be set by evaluator
            model_used=model,
            reviewer_model=None,
            tokens_used={"input": 0, "output": 0},
            iterations=1,
            cost_usd=0.0,
            status=(
                TaskResult.TaskStatus.PENDING.value
                if hasattr(TaskResult, "TaskStatus")
                else "completed"
            ),
            critique="",
            deterministic_check_passed=False,
            degraded_fallback_count=0,
            attempt_history=[],
            task_type=task.type.value,
        )


@register(TaskType.CODE_REVIEW)
class CodeReviewHandler(_BaseHandler):
    """Handler for code review tasks — reviews generated source files."""

    task_type = TaskType.CODE_REVIEW

    async def execute(
        self,
        task: Task,
        client: UnifiedClient,
        budget: Budget,
        **kwargs,
    ) -> TaskResult:
        model = self._get_model(task)
        system = "You are a critical code reviewer. Find flaws, be specific."
        prompt = task.prompt

        dep_context = kwargs.get("dependency_context", "")
        if dep_context:
            prompt = (
                f"{prompt}\n\n--- SOURCE CODE TO REVIEW ---\n"
                f"The following is the actual generated source code you must "
                f"review. Do NOT claim the code was not provided.\n\n"
                f"{dep_context}"
            )

        text = await self._call_llm(
            client,
            prompt,
            system=system,
            model=model,
            max_tokens=2000,
            temperature=0.2,
            timeout=180,
        )

        return TaskResult(
            task_id=task.id,
            output=text,
            score=0.0,
            model_used=model,
            reviewer_model=None,
            tokens_used={"input": 0, "output": 0},
            iterations=1,
            cost_usd=0.0,
            status=(
                TaskResult.TaskStatus.PENDING.value
                if hasattr(TaskResult, "TaskStatus")
                else "completed"
            ),
            critique="",
            deterministic_check_passed=False,
            degraded_fallback_count=0,
            attempt_history=[],
            task_type=task.type.value,
        )


@register(TaskType.EVALUATE)
class EvaluationHandler(_BaseHandler):
    """Handler for evaluation tasks — scores and critiques output."""

    task_type = TaskType.EVALUATE

    async def execute(
        self,
        task: Task,
        client: UnifiedClient,
        budget: Budget,
        **kwargs,
    ) -> TaskResult:
        model = self._get_model(task)
        prompt = task.prompt

        text = await self._call_llm(
            client,
            prompt,
            system="You are a precise evaluator. Score exactly, return only JSON.",
            model=model,
            max_tokens=500,
            temperature=0.1,
            timeout=60,
        )

        return TaskResult(
            task_id=task.id,
            output=text,
            score=0.0,
            model_used=model,
            reviewer_model=None,
            tokens_used={"input": 0, "output": 0},
            iterations=1,
            cost_usd=0.0,
            status=(
                TaskResult.TaskStatus.PENDING.value
                if hasattr(TaskResult, "TaskStatus")
                else "completed"
            ),
            critique="",
            deterministic_check_passed=False,
            degraded_fallback_count=0,
            attempt_history=[],
            task_type=task.type.value,
        )


@register(TaskType.REASONING)
class ReasoningHandler(_BaseHandler):
    """Handler for reasoning tasks — planning, analysis, architecture."""

    task_type = TaskType.REASONING

    async def execute(
        self,
        task: Task,
        client: UnifiedClient,
        budget: Budget,
        **kwargs,
    ) -> TaskResult:
        model = self._get_model(task)
        prompt = task.prompt

        dep_context = kwargs.get("dependency_context", "")
        if dep_context:
            prompt = f"{prompt}\n\n--- CONTEXT ---\n{dep_context}"

        text = await self._call_llm(
            client,
            prompt,
            system="You are a reasoning expert. Think step by step.",
            model=model,
            max_tokens=task.max_output_tokens,
            temperature=0.3,
            timeout=240,
        )

        return TaskResult(
            task_id=task.id,
            output=text,
            score=0.0,
            model_used=model,
            reviewer_model=None,
            tokens_used={"input": 0, "output": 0},
            iterations=1,
            cost_usd=0.0,
            status=(
                TaskResult.TaskStatus.PENDING.value
                if hasattr(TaskResult, "TaskStatus")
                else "completed"
            ),
            critique="",
            deterministic_check_passed=False,
            degraded_fallback_count=0,
            attempt_history=[],
            task_type=task.type.value,
        )
