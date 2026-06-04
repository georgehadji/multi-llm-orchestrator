"""Unit tests for TasteSkillLoader."""

from pathlib import Path

import pytest


@pytest.mark.unit
def test_load_returns_default_skill_content():
    from orchestrator.design.taste_skill_loader import TasteSkillLoader

    loader = TasteSkillLoader()
    content = loader.load("default")
    assert content, "default skill should be bundled"
    assert "Anti-Slop" in content or "anti" in content.lower() or "taste" in content.lower()


@pytest.mark.unit
def test_load_missing_variant_returns_empty_string(tmp_path):
    from orchestrator.design.taste_skill_loader import TasteSkillLoader

    loader = TasteSkillLoader(skill_dir=tmp_path)
    assert loader.load("nonexistent") == ""


@pytest.mark.unit
def test_load_unknown_key_returns_empty_string():
    from orchestrator.design.taste_skill_loader import TasteSkillLoader

    loader = TasteSkillLoader()
    result = loader.load("totally_unknown_key_xyz")
    assert result == ""


@pytest.mark.unit
def test_load_is_cached_no_double_read(tmp_path):
    """Second load() call should return cached value without re-reading file."""
    from orchestrator.design.taste_skill_loader import TasteSkillLoader

    skill_file = tmp_path / "default.SKILL.md"
    skill_file.write_text("# Test skill content", encoding="utf-8")
    loader = TasteSkillLoader(skill_dir=tmp_path)

    first = loader.load("default")
    # Overwrite the file — cached loader should not see the new content
    skill_file.write_text("# MODIFIED", encoding="utf-8")
    second = loader.load("default")
    assert first == second == "# Test skill content"


@pytest.mark.unit
def test_available_lists_only_bundled_files():
    from orchestrator.design.taste_skill_loader import TasteSkillLoader

    loader = TasteSkillLoader()
    available = loader.available()
    assert isinstance(available, list)
    assert "default" in available  # default must always be bundled


@pytest.mark.unit
def test_available_empty_for_empty_dir(tmp_path):
    from orchestrator.design.taste_skill_loader import TasteSkillLoader

    loader = TasteSkillLoader(skill_dir=tmp_path)
    assert loader.available() == []


@pytest.mark.unit
def test_is_bundled_true_for_default():
    from orchestrator.design.taste_skill_loader import TasteSkillLoader

    loader = TasteSkillLoader()
    assert loader.is_bundled("default") is True


@pytest.mark.unit
def test_is_bundled_false_for_missing():
    from orchestrator.design.taste_skill_loader import TasteSkillLoader

    loader = TasteSkillLoader()
    assert loader.is_bundled("does_not_exist") is False


@pytest.mark.unit
def test_content_trimmed_to_max_chars(tmp_path):
    from orchestrator.design.taste_skill_loader import TasteSkillLoader

    skill_file = tmp_path / "default.SKILL.md"
    skill_file.write_text("x" * 10_000, encoding="utf-8")
    loader = TasteSkillLoader(skill_dir=tmp_path, max_chars=100)
    content = loader.load("default")
    assert len(content) <= 200  # truncation note adds some chars
    assert "[...truncated" in content
