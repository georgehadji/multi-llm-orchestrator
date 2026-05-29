"""
CritiqueStage — Cross-model critique/review
==============================================
"""

from __future__ import annotations

import logging
from ..pipeline import PipelineContext
from ...domain.ports import LLMClient
from ...models import Model, TaskType

logger = logging.getLogger("orchestrator.engine_core.stages.critique")


class CritiqueStage:
    """Run cross-model critique on generated output.

    Uses a different (often higher-quality) model to review the
    generated output and produce structured feedback.
    """

    def __init__(
        self,
        client: LLMClient,
        get_reviewer_fn: object = None,
    ) -> None:
        self._client = client
        self._get_reviewer_fn = get_reviewer_fn

    async def process(self, ctx: PipelineContext) -> PipelineContext:
        """Run critique if a reviewer model is available."""
        if ctx.model is None:
            return ctx

        reviewer = self._resolve_reviewer(ctx)
        if reviewer is None or reviewer == ctx.model:
            return ctx

        ctx.reviewer_model = reviewer

        critique_prompt = (
            "Review the following generated code for correctness, "
            "performance, security, and style:\n\n"
            f"{ctx.output[:4000]}"
        )

        try:
            response = await self._client.call(
                model=reviewer,
                prompt=critique_prompt,
                system=(
                    "You are a code reviewer. Provide constructive, "
                    "specific feedback. Be concise."
                ),
                max_tokens=2048,
                temperature=0.3,
                timeout=60,
                retries=1,
            )
            ctx.critique = response.text[:2000]
        except Exception as e:
            logger.warning("Critique failed for task %s: %s", ctx.task.id, e)

        return ctx

    def _resolve_reviewer(self, ctx: PipelineContext):
        """Resolve the reviewer model."""
        if self._get_reviewer_fn is not None:
            return self._get_reviewer_fn(ctx.model, ctx.task.type)
        return None
