"""
Unit tests for WebsiteGenerator._namespace_colliding_ids().

When multiple section components are merged into one page, ids that were unique
*within* each standalone component collide *across* components (e.g. every
section using id="three-canvas"). Browsers' getElementById returns only the
first match, so all-but-one section's 3D/canvas init silently targets the wrong
element. This pass prefixes only the *colliding* ids per-section and rewrites
every reference form within that same component so intra-component logic stays
correct.
"""

import pytest


@pytest.fixture(scope="module")
def namespace():
    from orchestrator.generators.website_generator import WebsiteGenerator

    return WebsiteGenerator._namespace_colliding_ids


@pytest.mark.unit
class TestIdNamespacing:
    def test_colliding_id_is_prefixed_in_both_components(self, namespace):
        comps = [
            ("hero", '<div id="three-canvas"></div><script>document.getElementById("three-canvas")</script>'),
            ("tech", '<div id="three-canvas"></div><script>document.getElementById("three-canvas")</script>'),
        ]
        out = dict(namespace(comps))
        assert 'id="hero-three-canvas"' in out["hero"]
        assert 'getElementById("hero-three-canvas")' in out["hero"]
        assert 'id="tech-three-canvas"' in out["tech"]
        assert 'getElementById("tech-three-canvas")' in out["tech"]
        # No bare colliding id left.
        assert 'id="three-canvas"' not in out["hero"] and 'id="three-canvas"' not in out["tech"]

    def test_non_colliding_ids_untouched(self, namespace):
        comps = [
            ("hero", '<div id="hero-only"></div>'),
            ("tech", '<div id="tech-only"></div>'),
        ]
        out = dict(namespace(comps))
        assert out["hero"] == '<div id="hero-only"></div>'
        assert out["tech"] == '<div id="tech-only"></div>'

    def test_css_and_anchor_references_rewritten(self, namespace):
        comps = [
            ("a", '<style>#box{color:red}</style><a href="#box">x</a><div id="box"></div>'),
            ("b", '<div id="box"></div>'),
        ]
        out = dict(namespace(comps))
        assert "#a-box{color:red}" in out["a"]
        assert 'href="#a-box"' in out["a"]
        assert 'id="a-box"' in out["a"]

    def test_prefix_collision_not_corrupted(self, namespace):
        # 'three-canvas' must not corrupt 'three-canvas-bg' (shared prefix).
        comps = [
            ("hero", '<div id="three-canvas"></div><div id="three-canvas-bg"></div>'
                     '<style>#three-canvas{}#three-canvas-bg{}</style>'),
            ("tech", '<div id="three-canvas"></div><div id="three-canvas-bg"></div>'),
        ]
        out = dict(namespace(comps))
        assert 'id="hero-three-canvas"' in out["hero"]
        assert 'id="hero-three-canvas-bg"' in out["hero"]
        assert "#hero-three-canvas{}" in out["hero"]
        assert "#hero-three-canvas-bg{}" in out["hero"]
        # The shorter id's rewrite must not have eaten the longer one.
        assert "hero-three-canvas-bg-bg" not in out["hero"]
        assert "hero-three-hero-three" not in out["hero"]

    def test_no_collision_returns_input_unchanged(self, namespace):
        comps = [("hero", "<div id='x'></div>"), ("tech", "<div id='y'></div>")]
        assert namespace(comps) == comps
