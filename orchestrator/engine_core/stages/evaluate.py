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

    # Pipeline stage ordering — lower values run first
    priority: int = 300

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
            # CritiqueReport.score is already in [0.0, 1.0] (see EvaluatorService);
            # the previous `/ 10.0` double-normalized it, collapsing real scores
            # (0.85 -> 0.085, the 0.5 default -> 0.05) and degrading every task.
            ctx.score = critique_report.score
            ctx.critique = critique_report.to_prompt_context(max_items=10)
        except Exception as exc:
            logger.error(
                "Evaluation failed for task %s: %s",
                ctx.task.id,
                exc,
                exc_info=True,
            )
            ctx.evaluation_failed = True
            ctx.evaluation_error = str(exc)
            ctx.score = 0.0

        return ctx
