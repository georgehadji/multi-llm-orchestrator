"""
Unit tests for WebsiteGenerator._inject_section_id().

Generated section components carry their name as a class (``services-section``)
but often no ``id``, so cross-section CTAs like ``href="#services"`` resolve to
nothing. This pass injects ``id="<section>"`` onto the section's wrapper element
when it lacks one, making intra-page navigation work.
"""

import pytest

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def inject():
    from orchestrator.generators.website_generator import WebsiteGenerator

    return WebsiteGenerator._inject_section_id


@pytest.mark.unit
class TestInjectSectionId:
    def test_injects_id_into_section_without_id(self, inject):
        out = inject('<section class="services-section"><h2>x</h2></section>', "services")
        assert 'id="services"' in out
        assert '<section class="services-section" id="services">' in out

    def test_preserves_existing_id(self, inject):
        body = '<section class="about-section" id="about"><h2>x</h2></section>'
        assert inject(body, "about") == body

    def test_skips_leading_skip_link_targets_section(self, inject):
        # testimonials-style: a skip-link <a> precedes the real <section>.
        body = '<a href="#c" class="skip-link">Skip</a><section class="t"><h2>x</h2></section>'
        out = inject(body, "testimonials")
        assert '<section class="t" id="testimonials">' in out
        # the skip-link anchor must be untouched
        assert '<a href="#c" class="skip-link">Skip</a>' in out

    def test_falls_back_to_first_block_when_no_section(self, inject):
        out = inject('<div class="cta-wrap"><a>x</a></div>', "cta")
        assert '<div class="cta-wrap" id="cta">' in out

    def test_no_block_element_returns_unchanged(self, inject):
        body = "just text, no tags"
        assert inject(body, "footer") == body
