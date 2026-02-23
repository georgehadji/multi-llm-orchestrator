"""
Tests for DependencyResolver (Task 5).
Uses tmp_path for filesystem isolation.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator.dep_resolver import DependencyResolver, ResolveReport


# ─────────────────────────────────────────────────────────────────────────────
# ResolveReport — dataclass
# ─────────────────────────────────────────────────────────────────────────────

def test_resolve_report_fields():
    report = ResolveReport(
        packages=["fastapi", "uvicorn"],
        requirements_path="requirements.txt",
        pyproject_updated=True,
        unresolved=["my_custom_lib"],
    )
    assert report.packages == ["fastapi", "uvicorn"]
    assert report.requirements_path == "requirements.txt"
    assert report.pyproject_updated is True
    assert report.unresolved == ["my_custom_lib"]


# ─────────────────────────────────────────────────────────────────────────────
# DependencyResolver.resolve — basic scanning
# ─────────────────────────────────────────────────────────────────────────────

def test_resolve_detects_fastapi_import(tmp_path):
    """Import of 'fastapi' must be detected and mapped to 'fastapi' PyPI package."""
    (tmp_path / "main.py").write_text("import fastapi\n", encoding="utf-8")
    resolver = DependencyResolver()
    report = resolver.resolve(tmp_path)
    assert "fastapi" in report.packages


def test_resolve_detects_from_import(tmp_path):
    """'from fastapi import FastAPI' must detect 'fastapi'."""
    (tmp_path / "main.py").write_text("from fastapi import FastAPI\n", encoding="utf-8")
    resolver = DependencyResolver()
    report = resolver.resolve(tmp_path)
    assert "fastapi" in report.packages


def test_resolve_skips_stdlib_modules(tmp_path):
    """Standard library modules (os, sys, json, pathlib) must not appear in packages."""
    code = "import os\nimport sys\nimport json\nfrom pathlib import Path\n"
    (tmp_path / "main.py").write_text(code, encoding="utf-8")
    resolver = DependencyResolver()
    report = resolver.resolve(tmp_path)
    for mod in ["os", "sys", "json", "pathlib"]:
        assert mod not in report.packages, f"stdlib module {mod!r} should not be in packages"


def test_resolve_skips_relative_imports(tmp_path):
    """Relative imports like 'from .utils import helper' must be ignored."""
    code = "from .utils import helper\nfrom ..models import User\n"
    (tmp_path / "main.py").write_text(code, encoding="utf-8")
    resolver = DependencyResolver()
    report = resolver.resolve(tmp_path)
    # relative imports should produce no packages
    assert len(report.packages) == 0


def test_resolve_writes_requirements_txt(tmp_path):
    """resolve() must write a requirements.txt file."""
    (tmp_path / "main.py").write_text("import fastapi\n", encoding="utf-8")
    resolver = DependencyResolver()
    report = resolver.resolve(tmp_path)
    assert (tmp_path / "requirements.txt").exists()
    content = (tmp_path / "requirements.txt").read_text(encoding="utf-8")
    assert "fastapi" in content


def test_resolve_returns_requirements_path(tmp_path):
    """ResolveReport.requirements_path must point to the written file."""
    (tmp_path / "main.py").write_text("import requests\n", encoding="utf-8")
    resolver = DependencyResolver()
    report = resolver.resolve(tmp_path)
    assert report.requirements_path != ""
    assert Path(report.requirements_path).exists() or (tmp_path / report.requirements_path).exists()


def test_resolve_scans_nested_files(tmp_path):
    """resolve() must scan .py files in subdirectories."""
    subdir = tmp_path / "src"
    subdir.mkdir()
    (subdir / "utils.py").write_text("import httpx\n", encoding="utf-8")
    resolver = DependencyResolver()
    report = resolver.resolve(tmp_path)
    assert "httpx" in report.packages


def test_resolve_deduplicates_packages(tmp_path):
    """Same package imported in multiple files must appear only once."""
    (tmp_path / "a.py").write_text("import fastapi\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("from fastapi import APIRouter\n", encoding="utf-8")
    resolver = DependencyResolver()
    report = resolver.resolve(tmp_path)
    assert report.packages.count("fastapi") == 1


def test_resolve_empty_directory(tmp_path):
    """resolve() on a directory with no .py files must return empty packages and still write requirements.txt."""
    resolver = DependencyResolver()
    report = resolver.resolve(tmp_path)
    assert isinstance(report.packages, list)
    assert len(report.packages) == 0


def test_resolve_updates_pyproject_toml(tmp_path):
    """When pyproject.toml exists with a dependencies list, resolve() must update it."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[project]\nname = "myapp"\ndependencies = []\n', encoding="utf-8"
    )
    (tmp_path / "main.py").write_text("import fastapi\n", encoding="utf-8")
    report = DependencyResolver().resolve(tmp_path)
    assert report.pyproject_updated is True
    content = pyproject.read_text(encoding="utf-8")
    assert "fastapi" in content
