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


def _build(tmp_path: Path, *, with_images: bool) -> str:
    components = tmp_path / "components"
    components.mkdir()
    # Write out-of-order on disk to prove ordering follows the sections list.
    (components / "about.html").write_text(_component("about", "0.160.0"), encoding="utf-8")
    (components / "hero.html").write_text(_component("hero", "0.160.0"), encoding="utf-8")

    if with_images:
        img_dir = tmp_path / "public" / "images"
        img_dir.mkdir(parents=True)
        (img_dir / "favicon.svg").write_text("<svg/>", encoding="utf-8")
        (img_dir / "og-image.webp").write_bytes(b"RIFF....WEBP")
        (img_dir / "apple-touch-icon.webp").write_bytes(b"RIFF....WEBP")

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


@pytest.fixture
def assembled(tmp_path: Path) -> str:
    return _build(tmp_path, with_images=False)


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


@pytest.mark.unit
def test_icon_links_reference_existing_files(tmp_path: Path):
    html = _build(tmp_path, with_images=True)
    # Real generated files are referenced; the old hardcoded names are gone.
    assert 'href="public/images/favicon.svg"' in html
    assert 'href="public/images/apple-touch-icon.webp"' in html
    assert "/images/favicon.png" not in html
    assert "/favicon.svg" not in html.replace("public/images/favicon.svg", "")


@pytest.mark.unit
def test_og_image_points_to_generated_file(tmp_path: Path):
    html = _build(tmp_path, with_images=True)
    assert 'content="public/images/og-image.webp"' in html
    assert "/og-image.png" not in html


@pytest.mark.unit
def test_no_icon_links_when_no_images(assembled: str):
    # When no image files exist, no broken icon links are emitted.
    assert 'rel="icon"' not in assembled
    assert 'rel="apple-touch-icon"' not in assembled
