"""Profiling wrappers for TaskPipeline stages."""
from __future__ import annotations
from typing import TYPE_CHECKING, Any

from .profiler import NexusScopeProfiler

if TYPE_CHECKING:
    from ...engine_core.pipeline import PipelineContext, PipelineStage


class ProfilingStageWrapper:
    """Wraps a single PipelineStage with profiling."""
    def __init__(self, stage: "PipelineStage", profiler: NexusScopeProfiler):
        self._stage = stage
        self._profiler = profiler

    async def process(self, ctx: "PipelineContext") -> "PipelineContext":
        async with self._profiler.async_session(
            f"pipeline.stage.{type(self._stage).__name__}"
        ):
            return await self._stage.process(ctx)


class ProfilingTaskPipeline:
    """Drop-in replacement for TaskPipeline with profiling wrappers."""
    def __init__(self, stages: list["PipelineStage"], profiler: NexusScopeProfiler | None = None):
        self._profiler = profiler or NexusScopeProfiler()
        self._stages = [ProfilingStageWrapper(s, self._profiler) for s in stages]

    async def run(self, ctx: "PipelineContext") -> "PipelineContext":
        for stage in self._stages:
            ctx = await stage.process(ctx)
            if ctx.should_abort:
                break
        return ctx
