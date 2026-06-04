"""Integration tests — taste-skill prefix injection through TasteSkillService."""

from unittest.mock import MagicMock

import pytest


def _make_task(prompt, target_path="", design_variant=None):
    from orchestrator.models import Task, TaskType

    return Task(
        id="t-integration",
        type=TaskType.CODE_GEN,
        prompt=prompt,
        target_path=target_path,
        design_variant=design_variant,
    )


@pytest.mark.integration
def test_frontend_task_gets_taste_prefix():
    """A frontend task with taste_skill_enabled=True should get a non-empty prefix."""
    from orchestrator.design.taste_skill_service import TasteSkillService

    flags = MagicMock()
    flags.taste_skill_enabled = True
    svc = TasteSkillService(flags=flags)

    task = _make_task("build a landing page in HTML, CSS, and JavaScript", "index.html")
    prefix = svc.build_prefix(task)
    assert prefix, "Frontend task should produce a non-empty skill prefix"
    assert "DESIGN_VARIANCE" in prefix


@pytest.mark.integration
def test_python_backend_task_gets_no_prefix():
    """A Python backend task must not receive any taste-skill prefix."""
    from orchestrator.design.taste_skill_service import TasteSkillService

    flags = MagicMock()
    flags.taste_skill_enabled = True
    svc = TasteSkillService(flags=flags)

    task = _make_task("build a FastAPI authentication endpoint", "app/auth.py")
    prefix = svc.build_prefix(task)
    assert prefix == "", f"Backend task should get empty prefix, got: {prefix[:100]}"


@pytest.mark.integration
def test_flag_disabled_produces_no_prefix():
    """When taste_skill_enabled=False, no prefix for any task."""
    from orchestrator.design.taste_skill_service import TasteSkillService

    flags = MagicMock()
    flags.taste_skill_enabled = False
    svc = TasteSkillService(flags=flags)

    task = _make_task("build a landing page in HTML", "index.html")
    assert svc.build_prefix(task) == ""


@pytest.mark.integration
def test_redesign_task_uses_redesign_variant():
    """A task with REDESIGN variant should produce a prefix containing redesign rubric content."""
    from orchestrator.design.taste_skill_service import TasteSkillService
    from orchestrator.models import DesignVariant

    flags = MagicMock()
    flags.taste_skill_enabled = True
    svc = TasteSkillService(flags=flags)

    task = _make_task(
        "redesign the pricing page to feel more premium",
        "pricing.html",
        design_variant=DesignVariant.REDESIGN,
    )
    prefix = svc.build_prefix(task)
    assert prefix, "Redesign task should get a prefix"
    # variant block should be included
    assert "redesign" in prefix.lower() or "Style variant" in prefix


@pytest.mark.integration
def test_prompt_cue_infers_redesign():
    """A task without explicit variant but with 'redesign' in prompt should auto-resolve to REDESIGN."""
    from orchestrator.design.taste_skill_service import TasteSkillService
    from orchestrator.models import DesignVariant

    flags = MagicMock()
    flags.taste_skill_enabled = True
    svc = TasteSkillService(flags=flags)

    task = _make_task("redesign the homepage hero section in HTML", "index.html")
    variant = svc.resolve_variant(task)
    assert variant == DesignVariant.REDESIGN


@pytest.mark.integration
def test_dials_from_settings_appear_in_prefix():
    """Custom dial values from settings must appear in the injected prefix."""
    from orchestrator.design.taste_skill_service import TasteSkillService

    flags = MagicMock()
    flags.taste_skill_enabled = True
    settings = MagicMock()
    settings.design_variance = 9
    settings.motion_intensity = 1
    settings.visual_density = 7
    svc = TasteSkillService(flags=flags, settings=settings)

    task = _make_task("build a React SaaS landing page", "src/App.tsx")
    prefix = svc.build_prefix(task)
    assert "DESIGN_VARIANCE=9" in prefix
    assert "MOTION_INTENSITY=1" in prefix
    assert "VISUAL_DENSITY=7" in prefix
