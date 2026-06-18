"""Integration test: _assemble_html_page produces ONE valid HTML document.

Regression guard for the bug where each section component (a full standalone
HTML doc) was injected verbatim into the body, yielding nested documents and
multiple importmaps.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from orchestrator.generators.website_generator import WebsiteGenerator


def _component(section: str, three_version: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{section}</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter&display=swap" rel="stylesheet">
  <style>.{section} {{ padding: 1rem; }}</style>
</head>
<body>
  <section id="{section}" class="{section}"><h2>{section} heading</h2></section>
  <script type="importmap">
  {{ "imports": {{ "three": "https://unpkg.com/three@{three_version}/build/three.module.js" }} }}
  </script>
  <script type="module">import * as THREE from 'three'; console.log('{section}');</script>
</body>
</html>
"""


@pytest.fixture
def assembled(tmp_path: Path) -> str:
    components = tmp_path / "components"
    components.mkdir()
    # Write out-of-order on disk to prove ordering follows the sections list.
    (components / "about.html").write_text(_component("about", "0.160.0"), encoding="utf-8")
    (components / "hero.html").write_text(_component("hero", "0.160.0"), encoding="utf-8")

    design_system = SimpleNamespace(
        colors=SimpleNamespace(
            primary="#818cf8",
            accent="#6366f1",
            surface="#0b0d1a",
            surface_alt="#111827",
            text_primary="#f1f5f9",
            text_secondary="#94a3b8",
        ),
        font_heading="Inter",
        font_body="Inter",
        spacing=SimpleNamespace(unit="8px"),
        shadow=SimpleNamespace(sm="0 1px 2px", md="0 4px 8px", lg="0 8px 24px"),
    )
    config = SimpleNamespace(
        framework="html",
        brand_name="Test Clinic",
        client_name="Test Clinic",
        page_type="landing",
        description="A clinic.",
        sections=["hero", "about"],
    )

    gen = WebsiteGenerator.__new__(WebsiteGenerator)
    gen._assemble_html_page(tmp_path, ["hero", "about"], design_system, config)
    return (tmp_path / "index.html").read_text(encoding="utf-8")


@pytest.mark.unit
def test_single_document(assembled: str):
    assert assembled.lower().count("<!doctype html>") == 1
    assert assembled.lower().count("<html") == 1
    assert assembled.lower().count("<head>") == 1
    assert assembled.lower().count("<body>") == 1


@pytest.mark.unit
def test_single_importmap(assembled: str):
    assert assembled.count('type="importmap"') == 1
    assert "three" in assembled


@pytest.mark.unit
def test_sections_present_in_configured_order(assembled: str):
    assert "hero heading" in assembled
    assert "about heading" in assembled
    # hero must precede about (configured order), not alphabetical.
    assert assembled.index("hero heading") < assembled.index("about heading")


@pytest.mark.unit
def test_module_scripts_after_importmap(assembled: str):
    assert assembled.index('type="importmap"') < assembled.index("import * as THREE")
