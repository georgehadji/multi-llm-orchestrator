"""
ValidateStage — Deterministic validation of task output
=========================================================
"""

from __future__ import annotations

import logging
from ..pipeline import PipelineContext
from ...validators import all_validators_pass

logger = logging.getLogger("orchestrator.engine_core.stages.validate")


class ValidateStage:
    """Run deterministic validators on generated output.

    Checks syntax, tests, and other hard validators configured
    on the task.
    """

    def __init__(self) -> None:
        pass

    async def process(self, ctx: PipelineContext) -> PipelineContext:
        """Run deterministic validators."""
        if not ctx.output:
            return ctx

        try:
            passed, failures = all_validators_pass(ctx.task, ctx.output.strip())
            if not passed:
                logger.warning(
                    "Deterministic validation failed for task %s: %s",
                    ctx.task.id, failures,
                )
        except Exception as exc:
            logger.warning(
                "Validation error for task %s: %s", ctx.task.id, exc
            )

        return ctx
