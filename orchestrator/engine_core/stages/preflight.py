"""
PreflightStage — Final quality gate before returning result
=============================================================
"""

from __future__ import annotations

import logging
from ..pipeline import PipelineContext
from ...engine_core.validator import TaskValidator

logger = logging.getLogger("orchestrator.engine_core.stages.preflight")


class PreflightStage:
    """Run preflight quality check on the final output.

    Delegates to TaskValidator.run_preflight_check() for the
    PASS/WARN/ENRICH/BLOCK gate logic.
    """

    # Pipeline stage ordering — lower values run first
    priority: int = 600

    def __init__(self, validator: TaskValidator) -> None:
        self._validator = validator

    async def process(self, ctx: PipelineContext) -> PipelineContext:
        """Run preflight check on the best output."""
        if ctx.model is None or not ctx.output:
            return ctx

        result, new_score, pf_result = await self._validator.run_preflight_check(
            task=ctx.task,
            output=ctx.output,
            score=ctx.score,
            primary=ctx.model,
        )
        ctx.output = result
        ctx.score = new_score
        ctx.preflight_result = pf_result

        return ctx
