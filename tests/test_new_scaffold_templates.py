"""
Unit tests for new scaffold templates (A1):
  nextjs.py, react_vite.py, html.py

Each template must have the required files and valid structure.
"""
from __future__ import annotations

import json

import pytest

from orchestrator.app_detector import AppProfile
from orchestrator.scaffold import ScaffoldEngine
from orchestrator.scaffold.templates import html, nextjs, react_vite


# ─────────────────────────────────────────────────────────────────────────────
# File-structure assertions
# ─────────────────────────────────────────────────────────────────────────────

class TestNextjsTemplate:
    def test_has_package_json(self):
        assert "package.json" in nextjs.FILES

    def test_has_tsconfig(self):
        assert "tsconfig.json" in nextjs.FILES

    def test_has_next_config(self):
        assert "next.config.js" in nextjs.FILES

    def test_has_tailwind_config(self):
        assert "tailwind.config.ts" in nextjs.FILES

    def test_has_app_layout(self):
        assert "app/layout.tsx" in nextjs.FILES

    def test_has_app_page(self):
        assert "app/page.tsx" in nextjs.FILES

    def test_has_globals_css(self):
        assert "app/globals.css" in nextjs.FILES

    def test_has_gitignore(self):
        assert ".gitignore" in nextjs.FILES

    def test_has_readme(self):
        assert "README.md" in nextjs.FILES

    def test_package_json_valid(self):
        data = json.loads(nextjs.FILES["package.json"])
        assert "next" in data["dependencies"]
        assert "react" in data["dependencies"]
        assert "framer-motion" in data["dependencies"]

    def test_package_json_has_build_script(self):
        data = json.loads(nextjs.FILES["package.json"])
        assert "build" in data["scripts"]

    def test_app_page_uses_framer(self):
        assert "framer-motion" in nextjs.FILES["app/page.tsx"]

    def test_all_files_are_strings(self):
        for path, content in nextjs.FILES.items():
            assert isinstance(content, str), f"{path} content is not a string"
            assert len(content) > 0, f"{path} content is empty"


class TestReactViteTemplate:
    def test_has_package_json(self):
        assert "package.json" in react_vite.FILES

    def test_has_tsconfig(self):
        assert "tsconfig.json" in react_vite.FILES

    def test_has_vite_config(self):
        assert "vite.config.ts" in react_vite.FILES

    def test_has_tailwind_config(self):
        assert "tailwind.config.ts" in react_vite.FILES

    def test_has_index_html(self):
        assert "index.html" in react_vite.FILES

    def test_has_src_main(self):
        assert "src/main.tsx" in react_vite.FILES

    def test_has_src_app(self):
        assert "src/App.tsx" in react_vite.FILES

    def test_has_index_css(self):
        assert "src/index.css" in react_vite.FILES

    def test_has_gitignore(self):
        assert ".gitignore" in react_vite.FILES

    def test_has_readme(self):
        assert "README.md" in react_vite.FILES

    def test_package_json_valid(self):
        data = json.loads(react_vite.FILES["package.json"])
        assert "react" in data["dependencies"]
        assert "vite" in data["devDependencies"]
        assert "build" in data["scripts"]

    def test_all_files_are_strings(self):
        for path, content in react_vite.FILES.items():
            assert isinstance(content, str)
            assert len(content) > 0


class TestHtmlTemplate:
    def test_has_index_html(self):
        assert "index.html" in html.FILES

    def test_has_main_css(self):
        assert "styles/main.css" in html.FILES

    def test_has_main_js(self):
        assert "scripts/main.js" in html.FILES

    def test_has_gitignore(self):
        assert ".gitignore" in html.FILES

    def test_has_readme(self):
        assert "README.md" in html.FILES

    def test_index_html_links_css(self):
        assert "styles/main.css" in html.FILES["index.html"]

    def test_index_html_links_js(self):
        assert "scripts/main.js" in html.FILES["index.html"]

    def test_all_files_are_strings(self):
        for path, content in html.FILES.items():
            assert isinstance(content, str)
            assert len(content) > 0


# ─────────────────────────────────────────────────────────────────────────────
# ScaffoldEngine integration
# ─────────────────────────────────────────────────────────────────────────────

def test_scaffold_nextjs_writes_files(tmp_path):
    engine = ScaffoldEngine()
    profile = AppProfile(app_type="nextjs", tech_stack=["typescript", "nextjs"])
    result = engine.scaffold(profile, tmp_path)
    assert "app/layout.tsx" in result
    assert "app/page.tsx" in result
    assert (tmp_path / "app" / "layout.tsx").exists()


def test_scaffold_react_vite_writes_files(tmp_path):
    engine = ScaffoldEngine()
    profile = AppProfile(app_type="react-fastapi", tech_stack=["react", "vite"])
    result = engine.scaffold(profile, tmp_path)
    assert "src/App.tsx" in result
    assert (tmp_path / "src" / "App.tsx").exists()


def test_scaffold_html_writes_files(tmp_path):
    engine = ScaffoldEngine()
    profile = AppProfile(app_type="html")
    result = engine.scaffold(profile, tmp_path)
    assert "index.html" in result
    assert (tmp_path / "index.html").exists()


def test_scaffold_nextjs_has_gitignore(tmp_path):
    engine = ScaffoldEngine()
    profile = AppProfile(app_type="nextjs")
    result = engine.scaffold(profile, tmp_path)
    assert ".gitignore" in result
