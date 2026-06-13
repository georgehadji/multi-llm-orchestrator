"""Unit tests for DesignCritiqueStage."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.engine_core.pipeline import PipelineContext
from orchestrator.engine_core.stages.design_critique import DesignCritiqueStage
from orchestrator.models import Model, Task, TaskType


@pytest.mark.unit
class TestDesignCritiqueStage:
    def _make_ctx(self, prompt="build a React UI", target_path="App.tsx", output="<div>hello</div>"):
        task = Task(
            id="t1",
            type=TaskType.CODE_GEN,
            prompt=prompt,
            target_path=target_path,
        )
        ctx = PipelineContext(task=task, output=output)
        ctx.model = Model.GPT_4O_MINI
        return ctx

    @pytest.mark.asyncio
    async def test_skips_non_frontend_task(self):
        stage = DesignCritiqueStage(client=MagicMock())
        ctx = self._make_ctx(prompt="build a FastAPI endpoint", target_path="api.py", output="def hello(): pass")
        result = await stage.process(ctx)
        assert result.design_score == 0.0
        assert result.design_critique == ""

    @pytest.mark.asyncio
    async def test_skips_empty_output(self):
        stage = DesignCritiqueStage(client=MagicMock())
        ctx = self._make_ctx(output="")
        result = await stage.process(ctx)
        assert result.design_score == 0.0

    @pytest.mark.asyncio
    async def test_runs_critique_for_frontend_task(self):
        client = MagicMock()
        client.call = AsyncMock(return_value=MagicMock(text='{"score": 0.82}'))
        stage = DesignCritiqueStage(client=client)
        ctx = self._make_ctx()
        result = await stage.process(ctx)
        assert result.design_score == 0.82
        assert result.design_critique == '{"score": 0.82}'
        assert "[DESIGN CRITIQUE]" in result.critique
        client.call.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_uses_reviewer_model_when_set(self):
        client = MagicMock()
        client.call = AsyncMock(return_value=MagicMock(text='{"score": 0.9}'))
        stage = DesignCritiqueStage(client=client)
        ctx = self._make_ctx()
        ctx.reviewer_model = Model.CLAUDE_SONNET
        await stage.process(ctx)
        call_kwargs = client.call.await_args.kwargs
        assert call_kwargs["model"] == Model.CLAUDE_SONNET

    @pytest.mark.asyncio
    async def test_merges_with_existing_critique(self):
        client = MagicMock()
        client.call = AsyncMock(return_value=MagicMock(text='{"score": 0.7}'))
        stage = DesignCritiqueStage(client=client)
        ctx = self._make_ctx()
        ctx.critique = "Existing code critique."
        result = await stage.process(ctx)
        assert "Existing code critique." in result.critique
        assert "[DESIGN CRITIQUE]" in result.critique

    @pytest.mark.asyncio
    async def test_handles_client_error_gracefully(self):
        client = MagicMock()
        client.call = AsyncMock(side_effect=Exception("API down"))
        stage = DesignCritiqueStage(client=client)
        ctx = self._make_ctx()
        result = await stage.process(ctx)
        assert result.design_score == 0.0
        assert result.design_critique == ""
