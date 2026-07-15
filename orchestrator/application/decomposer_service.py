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
    project_context: Any = None,
) -> dict[str, Task]:
    """Decompose a project description into atomic tasks.

    Tries Instructor (structured LLM output) first. Falls back to the DAG-based
    Decomposer if Instructor is unavailable, the project exceeds its context
    window, or the Instructor attempt fails.

    If ``project_context`` is provided and contains a primary language, it is
    injected into the project description so the decomposition LLM knows what
    language to target.

    Returns:
        Dict mapping ``task.id → Task``.
    """
    # ── Inject language hint from project_context into description ──────
    enhanced_project = project
    if project_context is not None:
        try:
            tech_stack = getattr(project_context, "tech_stack", None) or []
            if tech_stack:
                primary_lang = tech_stack[0] if isinstance(tech_stack, list) else str(tech_stack)
                if primary_lang and primary_lang not in ("python",):
                    enhanced_project = (
                        f"{project}\n\n"
                        f"IMPORTANT: This is a {primary_lang} project. "
                        f"All code_generation tasks MUST produce {primary_lang} code output. "
                        f"Do NOT generate Python code for this project."
                    )
                    logger.info("Injected target language into decomposition: %s", primary_lang)
        except Exception:
            pass

    # ── Fast path: Instructor structured decomposition ──────────────────
    try:
        from ..structured_outputs import TaskDecomposer

        if len(enhanced_project) <= _INSTRUCTOR_MAX_CHARS:
            decomposer_inst = TaskDecomposer(api_client=client)  # type: ignore[no-untyped-call]
            decomp_model = model.value
            logger.info("Using Instructor for structured decomposition with %s", decomp_model)
            result = await decomposer_inst.decompose(
                project_description=enhanced_project,
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
    fallback_tasks: dict[str, Task] = await decomposer.decompose(
        project=enhanced_project,
        criteria=criteria,
        app_profile=app_profile,
        policy=policy,
        api_health=api_health,
        record_failure_fn=record_failure_fn,
        charge_fn=charge_fn,
        project_context=project_context,
    )
    return fallback_tasks
