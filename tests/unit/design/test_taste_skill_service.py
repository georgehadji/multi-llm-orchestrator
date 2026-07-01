"""Unit tests for TasteSkillService."""

from unittest.mock import MagicMock

import pytest


def _make_task(prompt, target_path="", design_variant=None):
    from orchestrator.models import Task, TaskType

    return Task(
        id="t1",
        type=TaskType.CODE_GEN,
        prompt=prompt,
        target_path=target_path,
        design_variant=design_variant,
    )


@pytest.mark.unit
def test_resolve_variant_uses_task_field():
    from orchestrator.design.taste_skill_service import TasteSkillService
    from orchestrator.models import DesignVariant

    svc = TasteSkillService()
    task = _make_task("build a UI", design_variant=DesignVariant.BRUTALIST)
    assert svc.resolve_variant(task) == DesignVariant.BRUTALIST


@pytest.mark.unit
def test_resolve_variant_infers_redesign_from_prompt():
    from orchestrator.design.taste_skill_service import TasteSkillService
    from orchestrator.models import DesignVariant

    svc = TasteSkillService()
    task = _make_task("please redesign the homepage to look more modern")
    assert svc.resolve_variant(task) == DesignVariant.REDESIGN


@pytest.mark.unit
def test_resolve_variant_infers_redesign_improve_ui():
    from orchestrator.design.taste_skill_service import TasteSkillService
    from orchestrator.models import DesignVariant

    svc = TasteSkillService()
    task = _make_task("improve ui for the settings screen")
    assert svc.resolve_variant(task) == DesignVariant.REDESIGN


@pytest.mark.unit
def test_resolve_variant_defaults_to_default():
    from orchestrator.design.taste_skill_service import TasteSkillService
    from orchestrator.models import DesignVariant

    svc = TasteSkillService()
    task = _make_task("build a landing page with hero section")
    assert svc.resolve_variant(task) == DesignVariant.DEFAULT


@pytest.mark.unit
def test_build_prefix_empty_when_flag_disabled(tmp_path):
    from orchestrator.design.taste_skill_service import TasteSkillService

    flags = MagicMock()
    flags.taste_skill_enabled = False
    svc = TasteSkillService(flags=flags)
    task = _make_task("build a landing page in HTML", "index.html")
    assert svc.build_prefix(task) == ""


@pytest.mark.unit
def test_build_prefix_empty_for_non_frontend_task():
    from orchestrator.design.taste_skill_service import TasteSkillService

    flags = MagicMock()
    flags.taste_skill_enabled = True
    svc = TasteSkillService(flags=flags)
    task = _make_task("write a Python FastAPI endpoint", "app/api.py")
    assert svc.build_prefix(task) == ""


@pytest.mark.unit
def test_build_prefix_returns_content_for_frontend_task(tmp_path):
    from orchestrator.design.taste_skill_injector import TasteSkillInjector
    from orchestrator.design.taste_skill_loader import TasteSkillLoader
    from orchestrator.design.taste_skill_service import TasteSkillService

    (tmp_path / "default.SKILL.md").write_text("# Anti-slop", encoding="utf-8")
    loader = TasteSkillLoader(skill_dir=tmp_path)
    injector = TasteSkillInjector(loader)

    flags = MagicMock()
    flags.taste_skill_enabled = True
    svc = TasteSkillService(injector=injector, flags=flags)
    task = _make_task("build a landing page in HTML and CSS", "index.html")
    prefix = svc.build_prefix(task)
    assert "Anti-slop" in prefix


@pytest.mark.unit
def test_build_prefix_dials_come_from_settings(tmp_path):
    from orchestrator.design.taste_skill_injector import TasteSkillInjector
    from orchestrator.design.taste_skill_loader import TasteSkillLoader
    from orchestrator.design.taste_skill_service import TasteSkillService

    (tmp_path / "default.SKILL.md").write_text("# Skill", encoding="utf-8")
    loader = TasteSkillLoader(skill_dir=tmp_path)
    injector = TasteSkillInjector(loader)

    flags = MagicMock()
    flags.taste_skill_enabled = True
    settings = MagicMock()
    settings.design_variance = 9
    settings.motion_intensity = 2
    settings.visual_density = 6

    svc = TasteSkillService(injector=injector, flags=flags, settings=settings)
    task = _make_task("build a React UI", "App.tsx")
    prefix = svc.build_prefix(task)
    assert "DESIGN_VARIANCE=9" in prefix
    assert "MOTION_INTENSITY=2" in prefix
