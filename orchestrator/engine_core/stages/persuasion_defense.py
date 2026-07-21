"""
PersuasionDefenseStage — Post-generation hallucination verification
=====================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

ARA Phase 3: Runs PersuasionDefense on CODE_GEN outputs to catch
unsupported claims before delivery. Wraps the PersuasionDefensePipeline
from ara_pipelines.py into a pipeline stage.

Extract claims -> NLI verify -> conflict surface -> decide pass/fail.
"""

from __future__ import annotations

import logging

from ...ara_pipelines import ReasoningMethod

logger = logging.getLogger("orchestrator.engine_core.stages.persuasion_defense")

from ..pipeline import PipelineContext


class PersuasionDefenseStage:
    """Post-generation hallucination verification for code tasks.

    Runs PersuasionDefense on CODE_GEN task outputs to verify
    factual claims against supported context. Flags outputs with
    unverified claims for revision.
    """

    # Pipeline stage ordering — lower values run first
    priority: int = 500

    def __init__(self, ara_integration: object = None) -> None:
        self._ara = ara_integration

    async def process(self, ctx: PipelineContext) -> PipelineContext:
        """Run PersuasionDefense verification on task output."""
        from ...models import TaskType

        if ctx.task.type != TaskType.CODE_GEN:
            return ctx
        if not ctx.output:
            return ctx
        if self._ara is None:
            return ctx

        try:
            # Create a verification sub-task
            from ...models import Task

            verify_task = Task(
                id=f"{ctx.task.id}_verify",
                type=TaskType.EVALUATE,
                prompt=ctx.output[:4000],
                max_output_tokens=1000,
            )

            result = await self._ara.execute_task_with_pipeline(
                task=verify_task,
                method=ReasoningMethod.PERSUASION_DEFENSE,
            )

            # Store verification metadata
            ctx_meta = getattr(ctx, "metadata", {}) or {}
            ctx_meta["persuasion_defense"] = {
                "claims": result.metadata.get("claims", 0),
                "verified": result.metadata.get("verified", 0),
                "conflicts": result.metadata.get("conflicts", 0),
            }
            ctx.metadata = ctx_meta

            # Block delivery if verification score is critically low
            if result.score < 0.5:
                logger.warning(
                    "PersuasionDefense BLOCKED task %s: score=%.3f, claims=%d, verified=%d, conflicts=%d",
                    ctx.task.id,
                    result.score,
                    result.metadata.get("claims", 0),
                    result.metadata.get("verified", 0),
                    result.metadata.get("conflicts", 0),
                )
                ctx.should_abort = True
                ctx.abort_reason = "verification_failed"

        except Exception as exc:
            logger.warning("PersuasionDefense failed for task %s: %s", ctx.task.id, exc)
            # Fail-open: don't block delivery on verification error

        return ctx
