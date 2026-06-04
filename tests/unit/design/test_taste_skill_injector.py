"""Unit tests for TasteSkillInjector and DesignDials."""

import pytest


@pytest.mark.unit
def test_dials_default_values():
    from orchestrator.design.taste_skill_injector import DesignDials

    d = DesignDials()
    assert d.design_variance == 5
    assert d.motion_intensity == 5
    assert d.visual_density == 5


@pytest.mark.unit
def test_dials_clamped_below_min():
    from orchestrator.design.taste_skill_injector import DesignDials

    d = DesignDials(design_variance=0, motion_intensity=-5, visual_density=0)
    assert d.design_variance == 1
    assert d.motion_intensity == 1
    assert d.visual_density == 1


@pytest.mark.unit
def test_dials_clamped_above_max():
    from orchestrator.design.taste_skill_injector import DesignDials

    d = DesignDials(design_variance=99, motion_intensity=11, visual_density=100)
    assert d.design_variance == 10
    assert d.motion_intensity == 10
    assert d.visual_density == 10


@pytest.mark.unit
def test_dials_frozen():
    from orchestrator.design.taste_skill_injector import DesignDials

    d = DesignDials()
    with pytest.raises((AttributeError, TypeError)):
        d.design_variance = 9  # type: ignore[misc]


@pytest.mark.unit
def test_get_prefix_default_variant_includes_ban_list(tmp_path):
    from orchestrator.design.taste_skill_injector import DesignDials, TasteSkillInjector
    from orchestrator.design.taste_skill_loader import TasteSkillLoader
    from orchestrator.models import DesignVariant

    skill_file = tmp_path / "default.SKILL.md"
    skill_file.write_text("# Anti-slop rules\nBanned: Inter", encoding="utf-8")
    loader = TasteSkillLoader(skill_dir=tmp_path)
    injector = TasteSkillInjector(loader)

    prefix = injector.get_prefix(DesignVariant.DEFAULT, DesignDials())
    assert "Anti-slop" in prefix or "Banned" in prefix


@pytest.mark.unit
def test_get_prefix_includes_dials_block(tmp_path):
    from orchestrator.design.taste_skill_injector import DesignDials, TasteSkillInjector
    from orchestrator.design.taste_skill_loader import TasteSkillLoader
    from orchestrator.models import DesignVariant

    skill_file = tmp_path / "default.SKILL.md"
    skill_file.write_text("# Skill", encoding="utf-8")
    loader = TasteSkillLoader(skill_dir=tmp_path)
    injector = TasteSkillInjector(loader)

    dials = DesignDials(design_variance=8, motion_intensity=3, visual_density=7)
    prefix = injector.get_prefix(DesignVariant.DEFAULT, dials)
    assert "DESIGN_VARIANCE=8" in prefix
    assert "MOTION_INTENSITY=3" in prefix
    assert "VISUAL_DENSITY=7" in prefix


@pytest.mark.unit
def test_get_prefix_brutalist_pairs_default_plus_variant(tmp_path):
    from orchestrator.design.taste_skill_injector import DesignDials, TasteSkillInjector
    from orchestrator.design.taste_skill_loader import TasteSkillLoader
    from orchestrator.models import DesignVariant

    (tmp_path / "default.SKILL.md").write_text("# Default", encoding="utf-8")
    (tmp_path / "brutalist.SKILL.md").write_text("# Brutalist rules", encoding="utf-8")
    loader = TasteSkillLoader(skill_dir=tmp_path)
    injector = TasteSkillInjector(loader)

    prefix = injector.get_prefix(DesignVariant.BRUTALIST, DesignDials())
    assert "# Default" in prefix
    assert "# Brutalist rules" in prefix


@pytest.mark.unit
def test_get_prefix_returns_empty_when_default_not_bundled(tmp_path):
    from orchestrator.design.taste_skill_injector import DesignDials, TasteSkillInjector
    from orchestrator.design.taste_skill_loader import TasteSkillLoader
    from orchestrator.models import DesignVariant

    loader = TasteSkillLoader(skill_dir=tmp_path)  # empty dir
    injector = TasteSkillInjector(loader)

    assert injector.get_prefix(DesignVariant.DEFAULT, DesignDials()) == ""


@pytest.mark.unit
def test_get_prefix_variant_missing_uses_default_only(tmp_path):
    from orchestrator.design.taste_skill_injector import DesignDials, TasteSkillInjector
    from orchestrator.design.taste_skill_loader import TasteSkillLoader
    from orchestrator.models import DesignVariant

    (tmp_path / "default.SKILL.md").write_text("# Default only", encoding="utf-8")
    # No soft.SKILL.md
    loader = TasteSkillLoader(skill_dir=tmp_path)
    injector = TasteSkillInjector(loader)

    prefix = injector.get_prefix(DesignVariant.SOFT, DesignDials())
    assert "# Default only" in prefix
    assert "soft" not in prefix.lower() or "Style variant" not in prefix
