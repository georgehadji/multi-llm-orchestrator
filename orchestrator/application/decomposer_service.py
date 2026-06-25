"""
DecomposerService — Project decomposition with Instructor fast path + fallback.
================================================================================
Part of Application Layer. Extracted from engine.py._decompose.

Instructor (structured LLM output) is tried first; falls back to the DAG-based
Decomposer from engine_core if Instructor is unavailable or the project is too
long for its context window.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from ..models import Task, Model

logger = logging.getLogger(__name__)

_INSTRUCTOR_MAX_CHARS = 8_000


async def decompose_project(
    project: str,
    criteria: str,
    model: Model,
    client: Any,
    decomposer: Any,
    api_health: dict[Any, bool],
    record_failure_fn: Callable[..., Any],
    charge_fn: Callable[..., Any],
    app_profile: Any = None,
    policy: Any = None,
) -> dict[str, Task]:
    """Decompose a project description into atomic tasks.

    Tries Instructor (structured LLM output) first. Falls back to the DAG-based
    Decomposer if Instructor is unavailable, the project exceeds its context
    window, or the Instructor attempt fails.

    Returns:
        Dict mapping ``task.id → Task``.
    """
    # ── Fast path: Instructor structured decomposition ──────────────────
    try:
        from ..structured_outputs import TaskDecomposer

        if len(project) <= _INSTRUCTOR_MAX_CHARS:
            decomposer_inst = TaskDecomposer(api_client=client)
            decomp_model = (
                "deepseek/deepseek-v4-flash" if "free" in model.value.lower() else model.value
            )
            logger.info("Using Instructor for structured decomposition with %s", decomp_model)
            result = await decomposer_inst.decompose(
                project_description=project,
                success_criteria=criteria,
                model=decomp_model,
                max_retries=1,
            )
            tasks = {task.id: task for task in result.to_tasks()}
            logger.info("Instructor decomposition succeeded: %d tasks", len(tasks))
            return tasks
    except ImportError:
        logger.warning("Instructor not available, using Decomposer")
    except Exception as e:
        logger.warning(
            "Instructor decomposition failed (%s), using Decomposer",
            type(e).__name__,
        )

    # ── Fallback: DAG-based Decomposer from engine_core ────────────────
    return await decomposer.decompose(
        project=project,
        criteria=criteria,
        app_profile=app_profile,
        policy=policy,
        api_health=api_health,
        record_failure_fn=record_failure_fn,
        charge_fn=charge_fn,
    )
