"""
EnhancedSelfConsistencyStage — ARA-aware retry when quality is below threshold
=================================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Extends SelfConsistencyStage with ARA-powered improvement. Instead of just
retrying with a fallback model, eligible tasks get routed through ARA
reasoning pipelines (CoVE, Debate, Jury) for structured improvement.
"""

from __future__ import annotations

import logging
from typing import Any

from ..pipeline import PipelineContext
from ...models import FALLBACK_CHAIN, TaskType

logger = logging.getLogger("orchestrator.engine_core.stages.self_consistency")


class EnhancedSelfConsistencyStage:
    """Check if output quality meets threshold; signal retry if not.

    When score is below the configured quality threshold and we
    haven't exhausted max_attempts:
    1. If ARA strategy is available and configured for this task type,
       sets abort_reason to "ara_retry" for ARA pipeline routing.
    2. Otherwise falls back to standard model-diversity retry.

    ARA-powered retry methods per task type:
    - CODE_REVIEW -> Debate (multi-agent debate for deadlocked review)
    - CODE_GEN    -> CoVE (Chain-of-Verification for factual accuracy)
    - REASONING   -> CoVE (fact-check complex reasoning claims)
    """

    def __init__(
        self,
        max_attempts: int = 2,
        quality_threshold: float = 0.7,
        ara_strategy: Any = None,
    ) -> None:
        self._max_attempts = max_attempts
        self._quality_threshold = quality_threshold
        self._ara_strategy = ara_strategy

    async def process(self, ctx: PipelineContext) -> PipelineContext:
        """Check quality and signal retry if needed."""
        if ctx.score >= self._quality_threshold:
            return ctx

        if ctx.attempt >= self._max_attempts:
            logger.info(
                "Max attempts (%d) reached for task %s, best score=%.3f",
                self._max_attempts, ctx.task.id, ctx.score,
            )
            return ctx

        # Record this attempt for diagnostics
        ctx.attempt_history.append({
            "attempt": ctx.attempt,
            "score": ctx.score,
            "model": ctx.model.value if ctx.model else "none",
            "output_snippet": ctx.output[:200],
        })

        ctx.attempt += 1

        # Try ARA retry if strategy is available
        if self._ara_strategy is not None:
            method = self._ara_strategy.get_retry_method(ctx.task)
            if method is not None:
                logger.info(
                    "Attempt %d score=%.3f below threshold=%.3f, "
                    "retrying with ARA method %s",
                    ctx.attempt, ctx.score, self._quality_threshold,
                    method.value,
                )
                ctx.task.revision_context = ctx.critique
                ctx.reset_for_retry()
                ctx.should_abort = True
                ctx.abort_reason = "ara_retry"
                return ctx

        # Standard fallback: switch to fallback model for diversity
        if ctx.model is not None:
            fallback = FALLBACK_CHAIN.get(ctx.model, ctx.model)
            ctx.task.preferred_model = fallback
            logger.info(
                "Attempt %d score=%.3f below threshold=%.3f, "
                "retrying with fallback model %s",
                ctx.attempt, ctx.score, self._quality_threshold,
                fallback.value,
            )

        ctx.task.revision_context = ctx.critique
        ctx.reset_for_retry()
        ctx.should_abort = True
        ctx.abort_reason = "retry_for_quality"

        return ctx
