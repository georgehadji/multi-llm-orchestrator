"""
Tests for the Pipeline and Stage modules (extracted from engine.py Phase 5).
==============================================================================
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.engine_core.pipeline import PipelineContext, TaskPipeline
from orchestrator.engine_core.stages import EvaluateStage, GenerateStage, ValidateStage
from orchestrator.models import Model, Task, TaskType, TaskResult, TaskStatus


@pytest.fixture
def mock_task():
    return Task(
        id="test_task",
        type=TaskType.CODE_GEN,
        prompt="Write a hello world program",
        max_output_tokens=500,
    )


@pytest.fixture
def ctx(mock_task):
    return PipelineContext(task=mock_task)


class TestPipelineContext:
    """Test PipelineContext data class."""

    def test_initial_state(self, ctx):
        assert ctx.attempt == 0
        assert ctx.output == ""
        assert ctx.score == 0.0
        assert ctx.cost_usd == 0.0
        assert ctx.should_abort is False

    def test_reset_for_retry(self, ctx):
        ctx.should_abort = True
        ctx.abort_reason = "test"
        ctx.reset_for_retry()
        assert ctx.should_abort is False
        assert ctx.abort_reason == ""

    def test_to_task_result_default(self, ctx):
        result = ctx.to_task_result()
        assert isinstance(result, TaskResult)
        assert result.task_id == "test_task"
        assert hasattr(result, "preflight_result")

    def test_tokens_used_default(self, ctx):
        assert ctx.tokens_used == {"input": 0, "output": 0}

    def test_to_task_result_with_state(self, ctx):
        ctx.output = "def hello(): pass"
        ctx.score = 0.85
        ctx.model = Model.GPT_4O_MINI
        ctx.cost_usd = 0.5
        ctx.tokens_used = {"input": 100, "output": 50}
        result = ctx.to_task_result(status=TaskStatus.COMPLETED)
        assert result.output == "def hello(): pass"
        assert result.score == 0.85
        assert result.model_used == Model.GPT_4O_MINI
        assert result.cost_usd == 0.5


class TestTaskPipeline:
    """Test TaskPipeline execution."""

    @pytest.mark.asyncio
    async def test_sequential_stages(self, ctx):
        """Verify stages execute in order."""
        calls = []

        class Stage1:
            async def process(self, ctx):
                calls.append("stage1")
                return ctx

        class Stage2:
            async def process(self, ctx):
                calls.append("stage2")
                return ctx

        pipeline = TaskPipeline([Stage1(), Stage2()])
        result = await pipeline.run(ctx)
        assert calls == ["stage1", "stage2"]

    @pytest.mark.asyncio
    async def test_abort_stops_pipeline(self, ctx):
        """Verify a stage that sets should_abort stops the pipeline."""

        class Stage1:
            async def process(self, ctx):
                ctx.should_abort = True
                ctx.abort_reason = "early_exit"
                return ctx

        class Stage2:
            async def process(self, ctx):
                raise AssertionError("Should not be reached")

        pipeline = TaskPipeline([Stage1(), Stage2()])
        result = await pipeline.run(ctx)
        assert result.should_abort is True
        assert result.abort_reason == "early_exit"

    @pytest.mark.asyncio
    async def test_stage_error_caught(self, ctx):
        """Verify a stage that raises is caught by pipeline."""

        class FailingStage:
            async def process(self, ctx):
                raise RuntimeError("Something broke")

        pipeline = TaskPipeline([FailingStage()])
        result = await pipeline.run(ctx)
        assert result.should_abort is True
        assert "stage_error" in result.abort_reason

    @pytest.mark.asyncio
    async def test_generation_stage(self):
        """GenerateStage produces output."""
        mock_client = MagicMock()
        mock_client.call = AsyncMock(
            return_value=MagicMock(
                text="Generated output",
                cost_usd=0.01,
            )
        )
        mock_budget = MagicMock()
        mock_selector = MagicMock()
        mock_selector.select = MagicMock(return_value=Model.GPT_4O_MINI)

        stage = GenerateStage(client=mock_client, budget=mock_budget, selector=mock_selector)
        task = Task(id="g1", type=TaskType.CODE_GEN, prompt="Write code")
        result = await stage.process(PipelineContext(task=task))
        assert result.output == "Generated output"
        assert result.model == Model.GPT_4O_MINI

    @pytest.mark.asyncio
    async def test_validate_stage(self, ctx):
        """ValidateStage doesn't crash."""
        stage = ValidateStage()
        ctx.output = "valid code"
        result = await stage.process(ctx)
        assert result == ctx
