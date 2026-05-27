"""
EvaluateStage — Score/Analyze task output quality
====================================================
"""

from __future__ import annotations

import logging
from ..pipeline import PipelineContext

logger = logging.getLogger("orchestrator.engine_core.stages.evaluate")


class EvaluateStage:
    """Evaluate the quality of generated output.

    Uses the configured EvaluatorService to produce a CritiqueReport
    with score (0.0-1.0) and structured critique items.
    """

    def __init__(self, evaluator: object) -> None:
        self._evaluator = evaluator

    async def process(self, ctx: PipelineContext) -> PipelineContext:
        """Run evaluation if output exists and score not already set."""
        if not ctx.output:
            return ctx

        try:
            critique_report = await self._evaluator.evaluate(
                task=ctx.task,
                output=ctx.output,
            )
            ctx.score = critique_report.score / 10.0  # normalize to 0.0-1.0
            ctx.critique = critique_report.to_prompt_context(max_items=10)
        except Exception as exc:
            logger.warning("Evaluation failed for task %s: %s", ctx.task.id, exc)

        return ctx
