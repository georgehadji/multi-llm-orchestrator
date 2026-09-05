"""Check implementations, exercised against markup that is right and markup
that is wrong.

The regression class at the end pins four false positives found by running the
auditor against a real generated site. Each one had the same shape: the check
was measuring something adjacent to what it claimed to measure.
"""

from __future__ import annotations

import pytest

from orchestrator.generators.wf100.auditor import audit
from orchestrator.generators.wf100.checks import implementation_for, implemented_ids
from orchestrator.generators.wf100.evidence import BusinessRecord, Page, SiteEvidence
from orchestrator.generators.wf100.report import Status

pytestmark = pytest.mark.unit


def _ev(markup: str = "", *, styles: str = "", scripts: str = "", pages=None, **kwargs):
    if pages is None:
        pages = [Page(path="index.html", markup=markup)]
    return SiteEvidence(pages=tuple(pages), styles=styles, scripts=scripts, **kwargs)


def _run(check_id: str, evidence: SiteEvidence):
    impl = implementation_for(check_id)
    assert impl is not None, f"{check_id} has no implementation"
    return impl(evidence)


class TestArchitecture:
    def test_a1_wants_a_main_landmark(self):
        assert _run("A1", _ev("<body><div>hi</div></body>")).status is Status.FAIL

    def test_a1_accepts_a_full_landmark_set(self):
        markup = "<body><header>h</header><nav>n</nav><main>m</main><footer>f</footer></body>"
        assert _run("A1", _ev(markup)).status is Status.PASS

    def test_a2_flags_two_h1_elements(self):
        assert _run("A2", _ev("<h1>a</h1><h1>b</h1>")).status is Status.FAIL

    def test_a2_accepts_exactly_one(self):
        assert _run("A2", _ev("<h1>a</h1><h2>b</h2>")).status is Status.PASS

    def test_a3_flags_a_skipped_heading_level(self):
        assert _run("A3", _ev("<h1>a</h1><h4>b</h4>")).status is Status.FAIL

    def test_a3_allows_descending_back_up(self):
        assert _run("A3", _ev("<h1>a</h1><h2>b</h2><h3>c</h3><h2>d</h2>")).status is Status.PASS

    def test_a10_finds_console_logging(self):
        assert _run("A10", _ev("<p>x</p>", scripts="console.log('x')")).status is Status.FAIL

    def test_a10_passes_a_clean_build(self):
        assert _run("A10", _ev("<p>x</p>", scripts="const a=1")).status is Status.PASS


class TestPerformance:
    def test_b4_flags_an_oversized_image(self):
        ev = _ev("<img src='a.jpg'>", assets={"a.jpg": 900 * 1024})
        assert _run("B4", ev).status is Status.FAIL

    def test_b4_is_not_applicable_without_images(self):
        assert _run("B4", _ev("<p>x</p>", assets={"a.css": 10})).status is Status.NOT_APPLICABLE

    def test_b7_wants_later_images_lazy(self):
        markup = "<img src='a.webp'><img src='b.webp'>"
        assert _run("B7", _ev(markup)).status is Status.FAIL

    def test_b7_keeps_the_first_image_eager(self):
        markup = "<img src='a.webp'><img src='b.webp' loading='lazy'>"
        assert _run("B7", _ev(markup)).status is Status.PASS

    def test_b9_passes_a_system_font_stack(self):
        assert (
            _run("B9", _ev("<p>x</p>", styles="body{font-family:system-ui}")).status is Status.PASS
        )

    def test_b15_flags_a_blocking_head_script(self):
        markup = "<head><script src='a.js'></script></head>"
        assert _run("B15", _ev(markup)).status is Status.FAIL

    def test_b15_accepts_a_deferred_script(self):
        markup = "<head><script src='a.js' defer></script></head>"
        assert _run("B15", _ev(markup)).status is Status.PASS


class TestAccessibility:
    def test_c3_flags_an_outline_removed_without_replacement(self):
        assert _run("C3", _ev(styles="a:focus{outline:none}")).status is Status.FAIL

    def test_c3_accepts_a_replacement_indicator(self):
        css = "a:focus-visible{outline:none;box-shadow:0 0 0 3px #000}"
        assert _run("C3", _ev(styles=css)).status is Status.PASS

    def test_c4_flags_positive_tabindex(self):
        assert _run("C4", _ev("<a href='/' tabindex='3'>x</a>")).status is Status.FAIL

    def test_c6_flags_an_unlabelled_input(self):
        assert _run("C6", _ev("<form><input id='a' name='a'></form>")).status is Status.FAIL

    def test_c6_accepts_a_bound_label(self):
        markup = "<form><label for='a'>Name</label><input id='a'></form>"
        assert _run("C6", _ev(markup)).status is Status.PASS

    def test_c8_flags_an_image_with_no_alt_attribute(self):
        assert _run("C8", _ev("<img src='a.webp'>")).status is Status.FAIL

    def test_c8_accepts_an_empty_alt_as_decorative(self):
        assert _run("C8", _ev("<img src='a.webp' alt=''>")).status is Status.PASS

    def test_c9_measures_real_contrast(self):
        assert _run("C9", _ev(styles="p{color:#777;background:#fff}")).status is Status.FAIL
        assert _run("C9", _ev(styles="p{color:#111;background:#fff}")).status is Status.PASS

    def test_c12_flags_click_here(self):
        assert _run("C12", _ev("<a href='/a'>click here</a>")).status is Status.FAIL

    def test_c14_wants_reduced_motion_when_the_sheet_animates(self):
        assert _run("C14", _ev(styles="a{transition:all .2s}")).status is Status.FAIL

    def test_c14_is_not_applicable_to_a_static_sheet(self):
        assert _run("C14", _ev(styles="a{color:red}")).status is Status.NOT_APPLICABLE

    def test_c15_wants_a_declared_language(self):
        assert _run("C15", _ev("<html><body>x</body></html>")).status is Status.FAIL
        assert _run("C15", _ev("<html lang='el'><body>x</body></html>")).status is Status.PASS


class TestSeo:
    def test_d9_flags_a_noindex_directive(self):
        markup = "<meta name='robots' content='noindex, nofollow'>"
        assert _run("D9", _ev(markup)).status is Status.FAIL

    def test_d13_rejects_json_ld_that_does_not_parse(self):
        markup = '<script type="application/ld+json">{ oops }</script>'
        assert _run("D13", _ev(markup)).status is Status.FAIL

    def test_d13_accepts_valid_json_ld(self):
        markup = (
            '<script type="application/ld+json">'
            '{"@context":"https://schema.org","@type":"Dentist"}</script>'
        )
        assert _run("D13", _ev(markup)).status is Status.PASS


class TestSecurity:
    def test_g6_wants_integrity_on_external_scripts(self):
        markup = "<script src='https://cdn.example.com/a.js'></script>"
        assert _run("G6", _ev(markup)).status is Status.FAIL

    def test_g6_passes_a_self_hosted_build(self):
        assert _run("G6", _ev("<script src='/app.js'></script>")).status is Status.PASS

    def test_g7_is_not_applicable_without_forms(self):
        assert _run("G7", _ev("<p>x</p>")).status is Status.NOT_APPLICABLE

    def test_g7_wants_protection_on_a_real_form(self):
        assert _run("G7", _ev("<form><input name='a'></form>")).status is Status.FAIL

    def test_g9_is_not_applicable_without_trackers(self):
        assert _run("G9", _ev("<p>x</p>")).status is Status.NOT_APPLICABLE

    def test_g9_flags_analytics_without_consent(self):
        markup = "<script src='https://www.googletagmanager.com/gtag/js'></script>"
        assert _run("G9", _ev(markup)).status is Status.FAIL


class TestContent:
    def test_h1_finds_lorem_ipsum(self):
        assert _run("H1", _ev("<p>Lorem ipsum dolor sit amet</p>")).status is Status.FAIL

    def test_h1_finds_unresolved_template_variables(self):
        finding = _run("H1", _ev("<p>Call us on $phone today</p>"))
        assert finding.status is Status.FAIL
        assert "$phone" in finding.detail

    def test_h1_passes_real_copy(self):
        assert _run("H1", _ev("<p>We fit implants on Tsimiski street.</p>")).status is Status.PASS

    def test_h5_needs_a_record_to_verify_against(self):
        assert _run("H5", _ev("<a href='tel:+302310000000'>call</a>")).status is Status.OUTSTANDING

    def test_h5_compares_published_contact_with_the_record(self):
        ev = _ev(
            "<a href='tel:+302310111222'>call</a>", record=BusinessRecord(phone="+30 2310 999888")
        )
        assert _run("H5", ev).status is Status.FAIL


class TestFalsePositiveRegressions:
    """Each of these was a real false positive from auditing a generated site."""

    def test_a8_treats_a_root_link_as_the_home_page(self):
        # `href="/"` is the site root, which is index.html — not a broken link.
        ev = _ev("<a href='/'>Home</a>", assets={"index.html": 100})
        assert _run("A8", ev).status is Status.PASS

    def test_b8_does_not_count_canonical_as_a_critical_path_origin(self):
        # rel=canonical is metadata. The browser never fetches it, so demanding
        # a preconnect for its origin is measuring the wrong thing.
        markup = "<link rel='canonical' href='https://clinic.example.gr/'>" "<img src='/hero.webp'>"
        assert _run("B8", _ev(markup)).status is Status.PASS

    def test_b8_still_wants_a_preconnect_for_a_fetched_origin(self):
        markup = "<link rel='stylesheet' href='https://cdn.example.com/a.css'><img src='/h.webp'>"
        assert _run("B8", _ev(markup)).status is Status.FAIL

    def test_e8_does_not_count_the_business_name_as_stuffing(self):
        # A practice called "Thessaloniki Dental" repeating its own name in the
        # header and footer is branding, not keyword stuffing.
        body = "Thessaloniki Dental " * 6 + "we fit implants carefully " * 20
        ev = _ev(
            f"<body>{body}</body>",
            record=BusinessRecord(name="Thessaloniki Dental", city="Thessaloniki"),
        )
        assert _run("E8", ev).status is Status.PASS

    def test_e8_still_flags_the_bare_locality_repeated(self):
        body = "Thessaloniki " * 12 + "we fit implants carefully " * 20
        ev = _ev(
            f"<body>{body}</body>", record=BusinessRecord(name="Bright Smiles", city="Thessaloniki")
        )
        assert _run("E8", ev).status is Status.FAIL

    def test_f10_reads_testimonial_markup_not_only_the_word_testimonial(self):
        markup = (
            "<h2>What patients say</h2>"
            "<blockquote>Painless and quick.<cite>Dimitris P.</cite></blockquote>"
            "<p>Registration no. 12345</p>"
        )
        assert _run("F10", _ev(markup)).status is Status.PASS


class TestAuditorHonesty:
    def test_a_check_whose_evidence_is_absent_is_outstanding(self):
        report = audit(_ev("<html lang='el'><main><h1>x</h1></main></html>"))
        for cid in ("B1", "B2", "B3", "G1", "G3", "H3", "H4"):
            assert report.finding(cid).status is Status.OUTSTANDING, cid

    def test_the_outstanding_detail_says_how_to_supply_the_evidence(self):
        report = audit(_ev("<main><h1>x</h1></main>"))
        assert "CrUX" in report.finding("B1").detail
        assert "deployed URL" in report.finding("G1").detail
        assert "person" in report.finding("H3").detail

    def test_a_crashing_check_is_outstanding_never_passed(self, monkeypatch):
        import orchestrator.generators.wf100.auditor as auditor_module

        def boom(_ev):
            raise RuntimeError("parser exploded")

        monkeypatch.setattr(
            auditor_module, "implementation_for", lambda cid: boom if cid == "A1" else None
        )
        report = audit(_ev("<main><h1>x</h1></main>"))
        assert report.finding("A1").status is Status.OUTSTANDING
        assert "parser exploded" in report.finding("A1").detail

    def test_directory_mode_never_claims_https_is_configured(self, tmp_path):
        (tmp_path / "index.html").write_text("<html lang='el'><main><h1>x</h1></main></html>")
        (tmp_path / "_headers").write_text("/*\n  Strict-Transport-Security: max-age=63072000\n")
        from orchestrator.generators.wf100.auditor import audit_directory

        report = audit_directory(tmp_path)
        # The _headers file states an intention. It is not a served response.
        assert report.finding("G1").status is Status.OUTSTANDING
        assert report.finding("G3").status is Status.OUTSTANDING


class TestNoCheckCrashes:
    """No check may raise, on any input.

    A crash is reported OUTSTANDING rather than FAIL — which is right, since a
    stack trace is not evidence about the site — but it also means a broken
    check looks like an unverified one and every test above it still passes.
    F11 crashed on every page without a qualification in the copy, and the
    end-to-end suite went green through it for exactly that reason. This test
    is the one that would have said so.
    """

    @pytest.fixture(
        params=["empty", "bare", "nothing-matches", "everything-matches"], ids=lambda p: p
    )
    def hostile(self, request):
        if request.param == "empty":
            return SiteEvidence()
        if request.param == "bare":
            return _ev("<html><body><p>hi</p></body></html>")
        if request.param == "nothing-matches":
            return _ev(
                "<html lang='el'><body><main><h1>x</h1>"
                "<form><input name='a'></form></main></body></html>",
                styles="a{transition:all .2s}b{color:#111;background:#fff}",
                scripts="const a=1",
                assets={"index.html": 10, "a.css": 10, "hero.jpg": 999999},
                files={"robots.txt": "User-agent: *"},
                record=BusinessRecord(name="X", city="Y", phone="123"),
            )
        return _ev(
            "<html lang='el'><head><title>A dental practice on Tsimiski street</title>"
            "<meta name='description' content='x'></head><body><header><nav>n</nav></header>"
            "<main><h1>Implants</h1><blockquote>Great<cite>A</cite></blockquote>"
            "<img src='hero.webp' alt='The clinic' width='8' height='6'>"
            "<form><label for='a'>Name</label><input id='a' required></form>"
            "<a href='tel:+30231000'>Call</a></main><footer>f</footer></body></html>",
            styles=":root{--ink:#111;--paper:#fff}a:focus{outline:2px solid}"
            "@media (prefers-reduced-motion: no-preference){a{transition:all .2s}}",
            scripts="const a=1",
            assets={"index.html": 10, "hero.webp": 4096},
            files={"robots.txt": "User-agent: *\nSitemap: https://x/s.xml"},
            record=BusinessRecord(name="X", city="Tsimiski", phone="+30231000"),
        )

    @pytest.mark.parametrize("check_id", sorted(implemented_ids()))
    def test_check_returns_a_finding_rather_than_raising(self, check_id, hostile):
        finding = implementation_for(check_id)(hostile)
        assert finding.check.id == check_id
        assert finding.detail.strip()
