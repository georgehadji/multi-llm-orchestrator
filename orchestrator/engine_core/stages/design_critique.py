"""
DesignCritiqueStage — LLM-based design quality review.
======================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Runs a design-specific critique pass on frontend-generated output,
scoring typography, colour, layout, interactivity, and content realism.

Inserted between CritiqueStage and EvaluateStage in the TaskPipeline.
"""

from __future__ import annotations

import logging

from ..pipeline import PipelineContext
from ...design.design_rubric import DefaultDesignRubric
from ...design.frontend_detect import is_web_frontend_task
from ...domain.ports import LLMClient
from ...models import Model

logger = logging.getLogger("orchestrator.engine_core.stages.design_critique")


class DesignCritiqueStage:
    """Run a design-focused critique on frontend output."""

    # Pipeline stage ordering — lower values run first
    priority: int = 250

    def __init__(
        self,
        client: LLMClient,
        rubric: DefaultDesignRubric | None = None,
        get_reviewer_fn: object = None,
    ) -> None:
        self._client = client
        self._rubric = rubric or DefaultDesignRubric()
        self._get_reviewer_fn = get_reviewer_fn

    async def process(self, ctx: PipelineContext) -> PipelineContext:
        """Run design critique if task is a frontend design task."""
        if not ctx.output:
            return ctx

        if not self._is_design_task(ctx.task):
            return ctx

        reviewer = self._resolve_reviewer(ctx)
        if reviewer is None:
            return ctx

        prompt = self._rubric.build(ctx.task.prompt, ctx.output)

        try:
            response = await self._client.call(
                model=reviewer,
                prompt=prompt,
                system=DefaultDesignRubric.SYSTEM_PROMPT,
                max_tokens=1500,
                temperature=0.2,
                timeout=60,
                retries=1,
            )
            ctx.design_critique = response.text[:2000]
            ctx.design_score = self._rubric.parse_score(response.text)

            # Merge into main critique for downstream EvaluateStage
            if ctx.critique:
                ctx.critique = f"{ctx.critique}\n\n[DESIGN CRITIQUE]\n{ctx.design_critique}"
            else:
                ctx.critique = f"[DESIGN CRITIQUE]\n{ctx.design_critique}"

            logger.info(
                "Design critique for %s: score=%.2f",
                ctx.task.id,
                ctx.design_score,
            )
        except Exception as exc:
            logger.warning("Design critique failed for task %s: %s", ctx.task.id, exc)

        return ctx

    def _is_design_task(self, task: object) -> bool:
        """Return True when task produces frontend output."""
        return is_web_frontend_task(task.prompt, getattr(task, "target_path", ""))

    def _resolve_reviewer(self, ctx: PipelineContext) -> Model | None:
        """Resolve the reviewer model for design critique."""
        if self._get_reviewer_fn is not None:
            reviewer = self._get_reviewer_fn(ctx.model, ctx.task.type)
            if reviewer is not None and reviewer != ctx.model:
                return reviewer

        # Default: use the same reviewer model as the main critique stage
        if ctx.reviewer_model is not None:
            return ctx.reviewer_model

        # Fallback: use the same model as generation
        return ctx.model
