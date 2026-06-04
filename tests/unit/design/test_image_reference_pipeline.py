"""Unit tests for ImageReferencePipeline."""

from unittest.mock import AsyncMock, MagicMock

import pytest


def _make_task(prompt, target_path=""):
    from orchestrator.models import Task, TaskType

    return Task(id="t1", type=TaskType.CODE_GEN, prompt=prompt, target_path=target_path)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_build_visual_context_empty_when_flag_off(tmp_path):
    from orchestrator.design.image_reference_pipeline import ImageReferencePipeline
    from orchestrator.design.taste_skill_loader import TasteSkillLoader

    flags = MagicMock()
    flags.image_reference_pipeline = False
    client = AsyncMock()

    pipeline = ImageReferencePipeline(
        loader=TasteSkillLoader(skill_dir=tmp_path),
        client=client,
        flags=flags,
    )
    task = _make_task("build a landing page in HTML")
    result = await pipeline.build_visual_context(task)
    assert result == ""
    client.call.assert_not_called()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_build_visual_context_empty_when_skill_missing(tmp_path):
    from orchestrator.design.image_reference_pipeline import ImageReferencePipeline
    from orchestrator.design.taste_skill_loader import TasteSkillLoader

    flags = MagicMock()
    flags.image_reference_pipeline = True
    client = AsyncMock()

    pipeline = ImageReferencePipeline(
        loader=TasteSkillLoader(skill_dir=tmp_path),  # empty dir
        client=client,
        flags=flags,
    )
    task = _make_task("build a landing page in HTML")
    result = await pipeline.build_visual_context(task)
    assert result == ""


@pytest.mark.unit
@pytest.mark.asyncio
async def test_build_visual_context_calls_llm_when_flag_on(tmp_path):
    from orchestrator.design.image_reference_pipeline import ImageReferencePipeline
    from orchestrator.design.taste_skill_loader import TasteSkillLoader

    (tmp_path / "imagegen_web.SKILL.md").write_text("# Imagegen skill", encoding="utf-8")
    flags = MagicMock()
    flags.image_reference_pipeline = True

    mock_response = MagicMock()
    mock_response.text = "Bold typography, dark palette, editorial layout"
    client = MagicMock()
    client.call = AsyncMock(return_value=mock_response)

    pipeline = ImageReferencePipeline(
        loader=TasteSkillLoader(skill_dir=tmp_path),
        client=client,
        flags=flags,
    )
    task = _make_task("build a luxury portfolio site in HTML/CSS")
    result = await pipeline.build_visual_context(task)
    assert "Bold typography" in result
    assert "Visual direction" in result or "visual" in result.lower()
    client.call.assert_called_once()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_build_visual_context_handles_llm_failure_gracefully(tmp_path):
    from orchestrator.design.image_reference_pipeline import ImageReferencePipeline
    from orchestrator.design.taste_skill_loader import TasteSkillLoader

    (tmp_path / "imagegen_web.SKILL.md").write_text("# Skill", encoding="utf-8")
    flags = MagicMock()
    flags.image_reference_pipeline = True

    client = MagicMock()
    client.call = AsyncMock(side_effect=RuntimeError("LLM timeout"))

    pipeline = ImageReferencePipeline(
        loader=TasteSkillLoader(skill_dir=tmp_path),
        client=client,
        flags=flags,
    )
    task = _make_task("build a React landing page")
    result = await pipeline.build_visual_context(task)
    assert result == ""
