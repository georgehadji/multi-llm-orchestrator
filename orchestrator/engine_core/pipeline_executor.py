"""
PipelineExecutor — Task execution loop extracted from engine.py
=================================================================
Extracted via Strangler Fig pattern from Orchestrator._execute_task
and its helper methods (_build_skill_prefix, _enrich_with_visual_context,
_check_anti_slop, _record_trajectory, _get_enricher).

Owns the core generate → critique → revise → evaluate loop with
self-consistency retry and ARA algorithm integration.

Usage:
    executor = PipelineExecutor(
        pipeline=task_pipeline,
        selector=model_selector,
        skill_manager=skill_manager,
        taste_skill_service=taste_svc,
        client=unified_client,
        background_tasks=bg_tasks_set,
    )
    result = await executor.execute(task)
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .pipeline import TaskPipeline
    from ..model_selector import ModelSelector
    from ..models import Task, TaskResult, ResiliencePolicy

logger = logging.getLogger("orchestrator.engine_core.pipeline_executor")


class PipelineExecutor:
    """
    Executes a single task through the TaskPipeline stages.

    Responsibilities:
    1. Model selection (via injected ModelSelector)
    2. Skill prefix building + visual context enrichment (via TaskContextEnricher)
    3. Pipeline loop with self-consistency / ARA retries
    4. Anti-slop check + trajectory recording (via TaskContextEnricher)
    """

    def __init__(
        self,
        pipeline: TaskPipeline,
        selector: ModelSelector,
        skill_manager: Any = None,
        taste_skill_service: Any = None,
        client: Any = None,
        background_tasks: set[asyncio.Task] | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._selector = selector
        self._skill_manager = skill_manager
        self._taste_skill_service = taste_skill_service
        self._client = client
        self._background_tasks: set[asyncio.Task] = (
            background_tasks if background_tasks is not None else set()
        )
        self._ctx_enricher: Any = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def execute(
        self,
        task: Task,
        policy: ResiliencePolicy | None = None,
    ) -> TaskResult:
        """
        Execute a single task via the TaskPipeline.

        Stages run: Generate → Critique → Evaluate → Validate →
        PersuasionDefense → Preflight → SelfConsistency (with retries).

        Args:
            task: The task to execute.
            policy: Optional resilience policy (unused — preserved for API compat).

        Returns:
            TaskResult with final status and output.
        """
        from .pipeline import PipelineContext
        from ..models import TaskStatus

        # Select initial model
        model = task.preferred_model
        if not model and self._selector is not None:
            model = self._selector.select(task.type)

        # Build skill prefix (SkillOpt + taste-skill)
        skill_prefix = await self._build_skill_prefix(task)

        # Optional image-reference visual-context enrichment
        task = await self._enrich_with_visual_context(task)

        ctx = PipelineContext(
            task=task,
            model=model,
            tokens_used={"input": 0, "output": 0},
            skill_prefix=skill_prefix,
        )

        # Loop for self-consistency / ARA retries
        while True:
            ctx = await self._pipeline.run(ctx)
            if ctx.abort_reason not in ("retry_for_quality", "ara_retry"):
                break
            # Reset for next attempt
            ctx.reset_for_retry()

        # Determine final status
        status = TaskStatus.COMPLETED
        if ctx.score < task.acceptance_threshold:
            status = TaskStatus.DEGRADED
        if ctx.abort_reason and ctx.abort_reason.startswith("stage_error"):
            status = TaskStatus.FAILED

        result = ctx.to_task_result(status=status)

        # taste-skill: soft anti-slop check (WARN only, never blocks)
        self._check_anti_slop(task, ctx)

        # SkillOpt: record trajectory for optimizer (fire-and-forget)
        await self._record_trajectory(task, ctx)

        return result

    # ------------------------------------------------------------------
    # Helper methods (moved verbatim from engine.py _execute_task cluster)
    # ------------------------------------------------------------------

    async def _build_skill_prefix(self, task: Task) -> str:
        """Build combined skill prefix — delegates to TaskContextEnricher."""
        enricher = self._get_enricher()
        return await enricher.build_prefix(task)

    async def _enrich_with_visual_context(self, task: Task) -> Task:
        """Enrich task with visual context — delegates to TaskContextEnricher."""
        enricher = self._get_enricher()
        return await enricher.enrich_with_visual_context(task)

    def _check_anti_slop(self, task: Task, ctx: Any) -> None:
        """Soft anti-slop WARN check — delegates to TaskContextEnricher."""
        enricher = self._get_enricher()
        enricher.check_anti_slop(task, ctx)

    async def _record_trajectory(self, task: Task, ctx: Any) -> None:
        """Record SkillOpt trajectory — delegates to TaskContextEnricher."""
        enricher = self._get_enricher()
        await enricher.record_trajectory(task, ctx, self._background_tasks)

    def _get_enricher(self) -> Any:
        """Lazy-init TaskContextEnricher for backward compatibility."""
        if self._ctx_enricher is None:
            from .stages.context_enricher import TaskContextEnricher

            self._ctx_enricher = TaskContextEnricher(
                skill_manager=self._skill_manager,
                taste_skill_service=self._taste_skill_service,
                client=self._client,
            )
        return self._ctx_enricher
