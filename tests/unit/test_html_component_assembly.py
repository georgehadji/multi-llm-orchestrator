"""Unit tests for HTML component extraction during page assembly.

The website generator emits each section as a *standalone* HTML document.
``_extract_html_component_parts`` must split those into reusable pieces so the
final ``index.html`` is a single valid document — not nested documents.
"""

import pytest

pytestmark = pytest.mark.unit

from orchestrator.generators.website_generator import WebsiteGenerator

_COMPONENT = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Hero</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter&display=swap" rel="stylesheet">
  <style>
    .hero { color: red; }
  </style>
</head>
<body>
  <section class="hero" id="hero">
    <canvas id="hero-canvas"></canvas>
    <h1>Clear Vision</h1>
  </section>
  <script type="importmap">
  { "imports": { "three": "https://unpkg.com/three@0.160.0/build/three.module.js" } }
  </script>
  <script type="module">
    import * as THREE from 'three';
    console.log('hero');
  </script>
</body>
</html>
"""


@pytest.fixture
def parts():
    return WebsiteGenerator._extract_html_component_parts(_COMPONENT)


@pytest.mark.unit
def test_body_inner_has_section_but_no_document_scaffolding(parts):
    body = parts["body_inner"]
    assert '<section class="hero"' in body
    assert "<h1>Clear Vision</h1>" in body
    # No nested-document tags should survive.
    assert "<!doctype" not in body.lower()
    assert "<html" not in body.lower()
    assert "<head" not in body.lower()
    assert "<body" not in body.lower()


@pytest.mark.unit
def test_styles_extracted(parts):
    assert any(".hero" in s for s in parts["styles"])
    # Styles must not remain inline in the body.
    assert "<style" not in parts["body_inner"].lower()


@pytest.mark.unit
def test_scripts_extracted_and_removed_from_body(parts):
    joined = "\n".join(parts["scripts"])
    assert "import * as THREE" in joined
    assert "<script" not in parts["body_inner"].lower()


@pytest.mark.unit
def test_importmap_captured_separately(parts):
    assert any("three" in im for im in parts["importmaps"])
    # The importmap must not be double-counted as a regular script.
    assert all("importmap" not in s for s in parts["scripts"])


@pytest.mark.unit
def test_font_links_captured(parts):
    joined = " ".join(parts["font_links"])
    assert "fonts.googleapis.com" in joined
