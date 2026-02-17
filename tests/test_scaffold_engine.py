"""
Tests for ScaffoldEngine and scaffold templates (Task 3).
No filesystem side effects — uses tmp_path fixture.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator.app_detector import AppProfile
from orchestrator.scaffold import ScaffoldEngine


# ─────────────────────────────────────────────────────────────────────────────
# ScaffoldEngine — basic interface
# ─────────────────────────────────────────────────────────────────────────────

def test_scaffold_returns_dict(tmp_path):
    """scaffold() must return a dict mapping relative paths to file content strings."""
    engine = ScaffoldEngine()
    profile = AppProfile(app_type="fastapi")
    result = engine.scaffold(profile, tmp_path)
    assert isinstance(result, dict)
    assert len(result) > 0


def test_scaffold_writes_files_to_disk(tmp_path):
    """scaffold() must create the files in the output_dir."""
    engine = ScaffoldEngine()
    profile = AppProfile(app_type="fastapi")
    result = engine.scaffold(profile, tmp_path)
    for rel_path in result:
        assert (tmp_path / rel_path).exists(), f"Expected {rel_path} to exist"


def test_scaffold_fastapi_has_main_py(tmp_path):
    """FastAPI template must include src/main.py."""
    engine = ScaffoldEngine()
    profile = AppProfile(app_type="fastapi")
    result = engine.scaffold(profile, tmp_path)
    assert "src/main.py" in result


def test_scaffold_fastapi_has_init_py(tmp_path):
    """FastAPI template must include src/__init__.py."""
    engine = ScaffoldEngine()
    profile = AppProfile(app_type="fastapi")
    result = engine.scaffold(profile, tmp_path)
    assert "src/__init__.py" in result


def test_scaffold_cli_has_cli_py(tmp_path):
    """CLI template must include cli.py as the entry point."""
    engine = ScaffoldEngine()
    profile = AppProfile(app_type="cli")
    result = engine.scaffold(profile, tmp_path)
    assert "cli.py" in result


def test_scaffold_library_has_src_init(tmp_path):
    """Library template must include src/__init__.py."""
    engine = ScaffoldEngine()
    profile = AppProfile(app_type="library")
    result = engine.scaffold(profile, tmp_path)
    assert "src/__init__.py" in result


def test_scaffold_generic_fallback(tmp_path):
    """Unknown app_type must fall back to generic template without raising."""
    engine = ScaffoldEngine()
    profile = AppProfile(app_type="unknown_type_xyz")
    result = engine.scaffold(profile, tmp_path)
    assert isinstance(result, dict)
    assert len(result) > 0


def test_scaffold_all_types_have_gitignore(tmp_path):
    """Every template must include a .gitignore file."""
    engine = ScaffoldEngine()
    for app_type in ["fastapi", "cli", "library", "generic"]:
        out = tmp_path / app_type
        out.mkdir()
        profile = AppProfile(app_type=app_type)
        result = engine.scaffold(profile, out)
        assert ".gitignore" in result, f"{app_type} template missing .gitignore"


def test_scaffold_all_types_have_readme(tmp_path):
    """Every template must include a README.md file."""
    engine = ScaffoldEngine()
    for app_type in ["fastapi", "cli", "library", "generic"]:
        out = tmp_path / app_type
        out.mkdir()
        profile = AppProfile(app_type=app_type)
        result = engine.scaffold(profile, out)
        assert "README.md" in result, f"{app_type} template missing README.md"


def test_scaffold_does_not_overwrite_existing_file(tmp_path):
    """scaffold() must NOT overwrite an existing file that already has content."""
    engine = ScaffoldEngine()
    profile = AppProfile(app_type="fastapi")
    # Pre-create src/main.py with custom content
    (tmp_path / "src").mkdir(parents=True, exist_ok=True)
    (tmp_path / "src" / "main.py").write_text("# custom content\n", encoding="utf-8")

    result = engine.scaffold(profile, tmp_path)

    # The returned dict still has src/main.py in it (template content)
    # But the file on disk must still have the custom content
    assert (tmp_path / "src" / "main.py").read_text(encoding="utf-8") == "# custom content\n"
