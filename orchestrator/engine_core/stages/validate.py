"""
ValidateStage — Deterministic validation of task output
=========================================================
"""

from __future__ import annotations

import logging
from ..pipeline import PipelineContext
from ...validators import all_validators_pass, run_validators

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
            hard_validators = getattr(ctx.task, "hard_validators", []) or []
            if hard_validators:
                results = run_validators(ctx.output.strip(), hard_validators)
                passed = all_validators_pass(results)
                if not passed:
                    failures = [r.details for r in results if not r.passed]
                    logger.warning(
                        "Deterministic validation failed for task %s: %s",
                        ctx.task.id,
                        failures,
                    )
        except Exception as exc:
            logger.warning("Validation error for task %s: %s", ctx.task.id, exc)

        return ctx
