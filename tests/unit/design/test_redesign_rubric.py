"""Unit tests for RedesignRubric."""

import json

import pytest


@pytest.mark.unit
def test_build_score_returns_string():
    from orchestrator.design.redesign_rubric import RedesignRubric

    rubric = RedesignRubric()
    result = rubric.build_score("redesign the homepage", "<div>...</div>", "code_generation")
    assert isinstance(result, str)
    assert len(result) > 50


@pytest.mark.unit
def test_build_score_contains_audit_sections():
    from orchestrator.design.redesign_rubric import RedesignRubric

    rubric = RedesignRubric()
    result = rubric.build_score("redesign", "<html>", "code_generation")
    # Must cover major audit domains
    lower = result.lower()
    assert "typography" in lower or "font" in lower
    assert "color" in lower or "colour" in lower or "palette" in lower
    assert "layout" in lower


@pytest.mark.unit
def test_build_score_preserves_json_score_contract():
    """The prompt must instruct the LLM to emit {"score": ..., "reasoning": ...}."""
    from orchestrator.design.redesign_rubric import RedesignRubric

    rubric = RedesignRubric()
    prompt = rubric.build_score("redesign", "<html>", "code_generation")
    # The prompt must mention "score" so the LLM knows the contract
    assert '"score"' in prompt or "'score'" in prompt or "score" in prompt.lower()
    assert "reasoning" in prompt.lower()


@pytest.mark.unit
def test_build_score_includes_original_prompt():
    from orchestrator.design.redesign_rubric import RedesignRubric

    rubric = RedesignRubric()
    original = "redesign the pricing section to feel more premium"
    result = rubric.build_score(original, "<div>pricing</div>", "code_generation")
    assert original in result


@pytest.mark.unit
def test_build_score_uses_fallback_when_loader_returns_empty(tmp_path):
    from orchestrator.design.redesign_rubric import RedesignRubric
    from orchestrator.design.taste_skill_loader import TasteSkillLoader

    loader = TasteSkillLoader(skill_dir=tmp_path)  # empty dir
    rubric = RedesignRubric(loader=loader)
    result = rubric.build_score("improve ui", "<html>", "code_generation")
    assert isinstance(result, str)
    assert "typography" in result.lower() or "font" in result.lower()


@pytest.mark.unit
def test_build_score_with_real_bundled_skill():
    """When the redesign skill is bundled, the rubric should be richer."""
    from orchestrator.design.redesign_rubric import RedesignRubric
    from orchestrator.design.taste_skill_loader import TasteSkillLoader

    loader = TasteSkillLoader()  # uses real bundled skills
    rubric = RedesignRubric(loader=loader)
    result = rubric.build_score("redesign the hero section", "<section>...</section>", "code_generation")
    assert len(result) > 200  # real skill content is substantial
