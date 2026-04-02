"""
orchestrator/task_factory.py
────────────────────────────
Factory for creating Task instances with task-type-appropriate defaults.

Extracted from Task.__post_init__ per T1-D of the Architecture Enhancement Plan.
Keeps models.py as pure data (no behavioral methods).

Rules:
  - No I/O, no asyncio, no engine imports.
  - All constants (DEFAULT_THRESHOLDS, MAX_OUTPUT_TOKENS) live here, not in models.py.
  - Call sites use TaskFactory.create() instead of Task() directly.
"""

from __future__ import annotations

from .models import Task, TaskType

# ---------------------------------------------------------------------------
# Type-specific defaults (previously living in Task.__post_init__)
# ---------------------------------------------------------------------------
_DEFAULT_THRESHOLDS: dict[TaskType, float] = {
    TaskType.DATA_EXTRACT: 0.90,
    TaskType.SUMMARIZE: 0.80,
    TaskType.CODE_GEN: 0.85,
    TaskType.CODE_REVIEW: 0.75,
    TaskType.REASONING: 0.90,
    TaskType.WRITING: 0.80,
    TaskType.EVALUATE: 0.80,
}

_MAX_OUTPUT_TOKENS: dict[TaskType, int] = {
    TaskType.CODE_GEN: 8192,
    TaskType.CODE_REVIEW: 4096,
    TaskType.REASONING: 4096,
    TaskType.WRITING: 4096,
    TaskType.DATA_EXTRACT: 2048,
    TaskType.SUMMARIZE: 1024,
    TaskType.EVALUATE: 2048,
}

_DEFAULT_ACCEPTANCE_THRESHOLD = 0.85
_DEFAULT_MAX_OUTPUT_TOKENS = 1500


def _max_iterations(task_type: TaskType) -> int:
    if task_type == TaskType.CODE_GEN:
        return 3
    if task_type == TaskType.CODE_REVIEW:
        return 4
    if task_type == TaskType.REASONING:
        return 3
    return 2


class TaskFactory:
    """
    Factory that creates Task instances with task-type-appropriate defaults.

    Use instead of instantiating Task() directly so that acceptance thresholds,
    iteration counts, and token limits are set correctly for each task type.
    """

    @staticmethod
    def create(
        id: str,  # noqa: A002
        task_type: TaskType,
        prompt: str,
        *,
        context: str = "",
        dependencies: list[str] | None = None,
        hard_validators: list[str] | None = None,
        target_path: str = "",
        module_name: str = "",
        tech_context: str = "",
    ) -> Task:
        """
        Create a Task with defaults computed from task_type.

        All keyword-only arguments mirror Task dataclass fields.
        Mutable defaults (dependencies, hard_validators) are safely handled.
        """
        return Task(
            id=id,
            type=task_type,
            prompt=prompt,
            context=context,
            dependencies=dependencies if dependencies is not None else [],
            acceptance_threshold=_DEFAULT_THRESHOLDS.get(task_type, _DEFAULT_ACCEPTANCE_THRESHOLD),
            max_iterations=_max_iterations(task_type),
            max_output_tokens=_MAX_OUTPUT_TOKENS.get(task_type, _DEFAULT_MAX_OUTPUT_TOKENS),
            hard_validators=hard_validators if hard_validators is not None else [],
            target_path=target_path,
            module_name=module_name,
            tech_context=tech_context,
        )
