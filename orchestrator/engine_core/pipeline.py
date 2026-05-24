"""
TaskPipeline — Composable task execution pipeline
===================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Core abstractions for the Strangler Fig extraction of engine.py _execute_task.

Orchestrator._execute_task() delegates to TaskPipeline which runs
pluggable PipelineStage instances in order.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..models import Model, Task, TaskResult, TaskStatus

logger = logging.getLogger("orchestrator.engine_core.pipeline")


@dataclass
class PipelineContext:
    """Mutable state carried through pipeline stages.

    Each stage reads from and writes to this context. The pipeline
    orchestrator loops until ctx.should_abort is set or all stages
    have been visited.

    Attributes:
        task: The task being executed.
        attempt: Current self-consistency attempt (0-based).
        output: Latest generated output text.
        revised_output: Output from the revision stage.
        score: Latest evaluation score (0.0-1.0).
        critique: Latest critique text.
        model: The model used for generation.
        reviewer_model: The model used for critique/review.
        tokens_used: Cumulative token counts per attempt.
        cost_usd: Cumulative cost.
        should_abort: If True, stop pipeline after this stage.
        abort_reason: Reason for abort (e.g. "retry_for_quality").
        attempt_history: Records of each attempt for diagnostics.
        preflight_result: Result from preflight check stage.
    """

    task: Task
    attempt: int = 0
    output: str = ""
    revised_output: str = ""
    score: float = 0.0
    critique: str = ""
    model: Model | None = None
    reviewer_model: Model | None = None
    tokens_used: dict[str, int] = field(default_factory=lambda: {"input": 0, "output": 0})
    cost_usd: float = 0.0
    should_abort: bool = False
    abort_reason: str = ""
    attempt_history: Any = field(default_factory=list)
    preflight_result: Any = None

    def reset_for_retry(self) -> None:
        """Reset mutable state for a retry attempt while preserving task context."""
        self.should_abort = False
        self.abort_reason = ""

    def to_task_result(self, status: TaskStatus = TaskStatus.COMPLETED) -> TaskResult:
        """Convert accumulated pipeline state into a TaskResult.

        Args:
            status: Final task status (COMPLETED, FAILED, DEGRADED).

        Returns:
            TaskResult populated from pipeline context.
        """
        return TaskResult(
            task_id=self.task.id,
            output=self.output,
            score=self.score,
            model_used=self.model or Model.GPT_4O_MINI,
            reviewer_model=self.reviewer_model,
            tokens_used=dict(self.tokens_used),
            iterations=self.attempt + 1,
            cost_usd=self.cost_usd,
            status=status,
            critique=self.critique,
            attempt_history=[
                {"attempt": a.get("attempt", 0), "score": a.get("score", 0.0)}
                for a in self.attempt_history
            ],
            preflight_result=self.preflight_result,
            preflight_passed=(self.preflight_result is None
                              or getattr(self.preflight_result, "passed", True)),
            task_type=self.task.type.value,
        )


class PipelineStage(Protocol):
    """A single stage in the task execution pipeline.

    Implementations receive a PipelineContext and return it (possibly
    with modifications). Set ctx.should_abort to True to stop the
    pipeline after this stage completes.
    """

    async def process(self, ctx: PipelineContext) -> PipelineContext:
        """Transform the PipelineContext.

        Args:
            ctx: Current pipeline state.

        Returns:
            Updated pipeline context.
        """
        ...


class TaskPipeline:
    """Composable task execution pipeline.

    Runs a sequence of PipelineStage instances in order. After each
    stage, checks ctx.should_abort. If set, stops early.

    Args:
        stages: Ordered list of pipeline stages.
    """

    def __init__(self, stages: list[PipelineStage]) -> None:
        self._stages = list(stages)

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        """Execute all stages in order.

        Args:
            ctx: Initial pipeline context.

        Returns:
            PipelineContext after all (non-aborted) stages have run.
        """
        for stage in self._stages:
            if ctx.should_abort:
                break
            name = type(stage).__name__
            logger.debug("pipeline stage: %s (task=%s)", name, ctx.task.id)
            try:
                ctx = await stage.process(ctx)
            except Exception as exc:
                logger.error(
                    "pipeline stage %s failed for task %s: %s",
                    name, ctx.task.id, exc,
                )
                ctx.should_abort = True
                ctx.abort_reason = f"stage_error:{name}:{exc}"
                break
        return ctx
