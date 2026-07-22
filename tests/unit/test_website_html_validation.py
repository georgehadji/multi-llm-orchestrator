"""
Unit tests for WebsiteGenerator._validate_html_structure().

These cover the structural bug classes found shipping in generated output:
1. Duplicate id attributes (invalid HTML; breaks getElementById + anchors).
2. Internal anchor links (href="#x") with no matching element id (dead link).
3. Mis-nested / stray closing tags that prematurely close a structural element
   (the real FAQ bug: a stray </div> appeared before </summary>).
A clean, well-formed document must produce zero issues (no false positives).
"""

import pytest

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def validate():
    from orchestrator.generators.website_generator import WebsiteGenerator

    return WebsiteGenerator._validate_html_structure


@pytest.mark.unit
class TestDuplicateIds:
    def test_flags_duplicate_id(self, validate):
        html = (
            "<!DOCTYPE html><html><body>"
            '<div id="three-canvas"></div>'
            '<section><div id="three-canvas"></div></section>'
            "</body></html>"
        )
        issues = validate(html)
        assert any("three-canvas" in i and "duplicate" in i.lower() for i in issues)

    def test_unique_ids_ok(self, validate):
        html = (
            "<!DOCTYPE html><html><body>"
            '<div id="hero"></div><div id="footer"></div>'
            "</body></html>"
        )
        assert not any("duplicate" in i.lower() for i in validate(html))


@pytest.mark.unit
class TestInternalAnchors:
    def test_flags_dead_anchor(self, validate):
        html = (
            "<!DOCTYPE html><html><body>"
            '<a href="#contact">Contact</a>'
            "<section><h2>Hero</h2></section>"
            "</body></html>"
        )
        issues = validate(html)
        assert any("#contact" in i for i in issues)

    def test_live_anchor_ok(self, validate):
        html = (
            "<!DOCTYPE html><html><body>"
            '<a href="#contact">Contact</a>'
            '<section id="contact"><h2>Contact</h2></section>'
            "</body></html>"
        )
        assert not any("#contact" in i for i in validate(html))

    def test_ignores_external_and_root_links(self, validate):
        html = (
            "<!DOCTYPE html><html><body>"
            '<a href="https://x.io">x</a><a href="/contact">c</a><a href="#">top</a>'
            "</body></html>"
        )
        # None of these are internal id references -> no anchor issues.
        assert not any("dead anchor" in i.lower() for i in validate(html))


@pytest.mark.unit
class TestMisNesting:
    def test_flags_stray_div_before_summary_close(self, validate):
        # Mirrors the shipped FAQ bug: </div> closes the open list div while the
        # <summary>/<details> are still open.
        html = (
            "<!DOCTYPE html><html><body>"
            '<div class="faq-list">'
            "<details><summary><span>Q?</span>"
            "</div>"  # <-- stray: prematurely closes faq-list under open summary
            "</summary><div>A.</div></details>"
            "</div></body></html>"
        )
        issues = validate(html)
        assert any("nest" in i.lower() or "stray" in i.lower() for i in issues)

    def test_well_formed_details_ok(self, validate):
        html = (
            "<!DOCTYPE html><html><body>"
            '<div class="faq-list">'
            "<details><summary><span>Q?</span></summary><div>A.</div></details>"
            "</div></body></html>"
        )
        assert validate(html) == []
