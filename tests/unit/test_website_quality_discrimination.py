"""
The website quality gate must DISCRIMINATE, not just report honestly.

Measured baseline before this suite (13-site labelled corpus: 3 hand-crafted
sites vs 10 generator outputs titled "Untitled Project"):

    GOOD  n=3  mean 0.872
    BAD   n=10 mean 0.895
    mean separation: -0.023   <- the gate scored BAD sites HIGHER than GOOD

5 of 8 checks produced exactly ONE distinct value across all 13 sites and so
carried zero information: Accessibility (always 1.00), Performance (always
0.96), Responsive Design (always 0.50), Email Verification (always 1.00),
Secret Exposure (always 1.00). Because the aggregate is a mean over 8 checks,
those constants pinned ~0.56 of every score before anything was measured.

Root causes, one per check:
  * _check_accessibility globbed only **/*.tsx and **/*.jsx, so on a static
    HTML site it scanned zero files and returned a perfect score — the same
    "passes because it measured nothing" bug content quality had.
  * _check_performance only ever recorded an issue when total JS exceeded
    500KB; everything else was a non-scoring "recommendation", and images (the
    dominant real weight) were never measured at all.
  * _check_responsive scored `breakpoints/4 if has_responsive_classes else 0.5`,
    and has_responsive_classes was set only by Tailwind prefixes in .tsx files.
    A static site with eight media queries scored the same 0.50 as one with none.
  * Email Verification and Secret Exposure correctly report "not applicable" on
    a brochure site, but contributed 1.00 to the mean, inflating every score.

These tests pin the fixed behaviour: applicability must not inflate, each check
must grade rather than saturate, and a good site must clearly outscore a bad one.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


# ── Fixtures: two sites that any reasonable reviewer would rank apart ─────────

_GOOD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Gadini Barberia — Italian Barbering in Thessaloniki</title>
  <meta name="description" content="Traditional Italian barbering in central Thessaloniki: bespoke cuts, hot-towel shaves and beard grooming, by appointment six days a week.">
  <meta property="og:title" content="Gadini Barberia">
  <meta property="og:image" content="/public/images/og-image.webp">
  <link rel="canonical" href="https://example-barber.gr/">
  <link rel="stylesheet" href="styles.css">
</head>
<body>
  <header><nav aria-label="Main"><a href="#services">Services</a> <a href="#contact">Contact</a></nav></header>
  <main>
    <h1>Italian barbering, done properly</h1>
    <section id="services">
      <h2>Services</h2>
      <img src="/public/images/cut.webp" alt="A barber finishing a scissor cut" loading="lazy" width="800" height="600">
      <p>Bespoke cuts shaped to how your hair actually grows, finished with a straight razor at the neckline. Hot-towel shaves use a badger brush and a single-blade razor.</p>
    </section>
    <section id="contact">
      <h2>Book a chair</h2>
      <form method="post" action="/api/contact">
        <label for="name">Your name</label><input id="name" name="name">
        <label for="email">Email</label><input id="email" name="email" type="email">
        <button type="submit">Request an appointment</button>
      </form>
    </section>
  </main>
  <footer><p>Tsimiski 42, Thessaloniki — +30 2310 000000</p></footer>
</body>
</html>
"""

_GOOD_CSS = """:root { --ink: #14110f; }
body { color: var(--ink); margin: 0; }
a:focus-visible, button:focus-visible { outline: 2px solid #b47c3c; outline-offset: 2px; }
@media (min-width: 480px) { .grid { grid-template-columns: 1fr 1fr; } }
@media (min-width: 768px) { .grid { grid-template-columns: repeat(3, 1fr); } }
@media (min-width: 1024px) { .grid { gap: 2rem; } }
@media (min-width: 1280px) { .wrap { max-width: 1200px; } }
"""

_BAD_HTML = """<!DOCTYPE html>
<html>
<head>
  <title>Untitled Project</title>
  <script src="https://cdn.example.com/a.js"></script>
  <script src="https://cdn.example.com/b.js"></script>
  <script src="https://cdn.example.com/c.js"></script>
  <link rel="stylesheet" href="styles.css">
</head>
<body>
  <div class="container">
    <div class="hero"><span>Welcome</span></div>
    <img src="hero.png">
    <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit.</p>
    <form method="post" action="/api/contact"><input name="email"><button></button></form>
  </div>
</body>
</html>
"""

_BAD_CSS = """.container { width: 1200px; }
.hero { color: #999; background: #aaa; }
"""


def _write(tmp_path, html: str, css: str, *, image_bytes: int = 40_000):
    (tmp_path / "index.html").write_text(html, encoding="utf-8")
    (tmp_path / "styles.css").write_text(css, encoding="utf-8")
    images = tmp_path / "public" / "images"
    images.mkdir(parents=True, exist_ok=True)
    (images / "hero.webp").write_bytes(b"\0" * image_bytes)
    return tmp_path


@pytest.fixture
def validator():
    from orchestrator.generators.website_validator import WebsiteQualityValidator

    return WebsiteQualityValidator()


@pytest.fixture
def good_dir(tmp_path):
    d = tmp_path / "good"
    d.mkdir(parents=True, exist_ok=True)
    return _write(d, _GOOD_HTML, _GOOD_CSS)


@pytest.fixture
def bad_dir(tmp_path):
    d = tmp_path / "bad"
    d.mkdir(parents=True, exist_ok=True)
    # 6 MB of images: a real page-weight problem, not a rounding difference.
    return _write(d, _BAD_HTML, _BAD_CSS, image_bytes=6_000_000)


# ── The headline property ────────────────────────────────────────────────────


@pytest.mark.unit
class TestDiscrimination:
    async def test_good_site_scores_well(self, validator, good_dir):
        report = await validator.validate(good_dir)
        assert report.score >= 0.80, _explain("good site scored low", report)

    async def test_bad_site_scores_poorly(self, validator, bad_dir):
        report = await validator.validate(bad_dir)
        assert report.score <= 0.60, _explain("bad site scored high", report)

    async def test_separation_is_wide(self, validator, good_dir, bad_dir):
        good = (await validator.validate(good_dir)).score
        bad = (await validator.validate(bad_dir)).score
        assert good - bad >= 0.30, (
            f"gate does not discriminate: good={good:.3f} bad={bad:.3f} "
            f"separation={good - bad:+.3f} (need >= 0.30)"
        )

    async def test_most_checks_carry_information(self, validator, good_dir, bad_dir):
        """A check scoring identically on both sites tells you nothing."""
        good = {c.name: c.score for c in (await validator.validate(good_dir)).checks}
        bad = {c.name: c.score for c in (await validator.validate(bad_dir)).checks}
        shared = set(good) & set(bad)
        informative = {n for n in shared if abs(good[n] - bad[n]) > 0.01}
        assert len(informative) >= 4, (
            f"only {len(informative)} of {len(shared)} checks distinguish these two "
            f"very different sites: informative={sorted(informative)}, "
            f"flat={sorted(shared - informative)}"
        )


def _explain(msg: str, report) -> str:
    rows = ", ".join(f"{c.name}={c.score:.2f}" for c in report.checks)
    return f"{msg}: aggregate={report.score:.3f} [{rows}]"


# ── Aggregation: 'not applicable' must not inflate ────────────────────────────


@pytest.mark.unit
class TestApplicabilityDoesNotInflate:
    def test_inapplicable_check_is_excluded_from_the_aggregate(self):
        from orchestrator.design_system import QualityCheck, QualityReport

        real = QualityCheck(name="real", passed=False, score=0.40, details="")
        na = QualityCheck(name="n/a", passed=True, score=1.0, details="", applicable=False)
        assert QualityReport(checks=[real, na]).score == pytest.approx(
            0.40
        ), "a check that does not apply must not pull the aggregate up"

    def test_applicable_defaults_to_true(self):
        from orchestrator.design_system import QualityCheck

        assert QualityCheck(name="x", passed=True, score=1.0, details="").applicable is True

    def test_inapplicable_check_does_not_fail_the_report(self):
        from orchestrator.design_system import QualityCheck, QualityReport

        ok = QualityCheck(name="ok", passed=True, score=1.0, details="")
        na = QualityCheck(name="n/a", passed=False, score=0.0, details="", applicable=False)
        assert QualityReport(checks=[ok, na]).passed is True

    def test_all_inapplicable_is_not_a_pass(self):
        from orchestrator.design_system import QualityCheck, QualityReport

        na = QualityCheck(name="n/a", passed=True, score=1.0, details="", applicable=False)
        report = QualityReport(checks=[na])
        assert report.score == 0.0 and report.passed is False

    async def test_static_site_marks_backend_checks_inapplicable(self, validator, good_dir):
        report = await validator.validate(good_dir)
        by_name = {c.name: c for c in report.checks}
        # Email Verification has no auth pages here; it must say so rather than
        # silently contributing a free 1.00.
        assert by_name["Email Verification"].applicable is False


# ── Per-check grading ────────────────────────────────────────────────────────


@pytest.mark.unit
class TestAccessibilityReadsHtml:
    async def test_penalises_missing_lang_alt_and_labels(self, validator, bad_dir):
        check = await validator._check_accessibility(bad_dir)
        assert check.score <= 0.6, f"expected a low a11y score, got {check.score}: {check.details}"

    async def test_rewards_a_well_formed_page(self, validator, good_dir):
        check = await validator._check_accessibility(good_dir)
        assert (
            check.score >= 0.85
        ), f"expected a high a11y score, got {check.score}: {check.details}"

    async def test_is_graded_not_binary(self, validator, good_dir, bad_dir):
        hi = (await validator._check_accessibility(good_dir)).score
        lo = (await validator._check_accessibility(bad_dir)).score
        assert hi - lo >= 0.3


@pytest.mark.unit
class TestPerformanceMeasuresRealWeight:
    async def test_penalises_heavy_images_and_blocking_scripts(self, validator, bad_dir):
        check = await validator._check_performance(bad_dir)
        assert check.score <= 0.6, f"expected a low perf score, got {check.score}: {check.details}"

    async def test_rewards_a_lean_page(self, validator, good_dir):
        check = await validator._check_performance(good_dir)
        assert (
            check.score >= 0.85
        ), f"expected a high perf score, got {check.score}: {check.details}"


@pytest.mark.unit
class TestResponsiveCountsRealBreakpoints:
    async def test_css_media_queries_count_without_tailwind(self, validator, good_dir):
        check = await validator._check_responsive(good_dir)
        assert check.score >= 0.85, (
            f"four CSS breakpoints must score well on a static site, got "
            f"{check.score}: {check.details}"
        )

    async def test_no_breakpoints_scores_low(self, validator, bad_dir):
        check = await validator._check_responsive(bad_dir)
        assert check.score <= 0.4, f"expected a low responsive score, got {check.score}"

    async def test_is_graded_across_breakpoint_counts(self, validator, tmp_path):
        scores = []
        for n in (0, 1, 2, 4):
            d = tmp_path / f"bp{n}"
            d.mkdir(parents=True, exist_ok=True)
            css = "body{color:#111}\n" + "".join(
                f"@media (min-width: {480 + i * 256}px) {{ .g {{ gap: 1rem; }} }}\n"
                for i in range(n)
            )
            _write(d, _GOOD_HTML, css)
            scores.append((await validator._check_responsive(d)).score)
        assert scores == sorted(scores), f"score must be monotonic in breakpoints: {scores}"
        assert scores[0] < scores[-1], f"no grading across 0..4 breakpoints: {scores}"


# ── Defect injection: causal sensitivity, independent of subjective labels ────
#
# Ranking real sites needs GOOD/BAD labels, and those labels are a judgement.
# Measuring this repo's own sites showed why that is dangerous: the three
# hand-crafted "premium" sites score WORSE than the generator outputs on
# performance and responsiveness — and they should. gadini-barberia ships 6.1MB
# of images; thessaloniki-dental declares one breakpoint and loads three
# render-blocking scripts; the generator outputs ship no images at all. Visual
# polish is not engineering quality, and the gate was right where the label was
# wrong.
#
# Defect injection avoids the problem entirely: take one page, change exactly
# one thing for the worse, and require the score to fall. No labels needed.


def _inject(tmp_path, name, *, html=None, css=None, image_bytes=40_000):
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    return _write(d, html or _GOOD_HTML, css or _GOOD_CSS, image_bytes=image_bytes)


@pytest.mark.unit
class TestDefectInjectionLowersTheScore:
    """Each defect must measurably lower the aggregate against an identical baseline."""

    async def _baseline(self, validator, tmp_path):
        return (await validator.validate(_inject(tmp_path, "baseline"))).score

    @pytest.mark.parametrize(
        "defect,mutate",
        [
            ("missing lang", lambda h: h.replace('<html lang="en">', "<html>")),
            (
                "image without alt",
                lambda h: h.replace('alt="A barber finishing a scissor cut" ', ""),
            ),
            (
                "placeholder title",
                lambda h: h.replace(
                    "Gadini Barberia — Italian Barbering in Thessaloniki", "Untitled Project"
                ),
            ),
            (
                "lorem ipsum body",
                lambda h: h.replace(
                    "Bespoke cuts shaped to how your hair actually grows",
                    "Lorem ipsum dolor sit amet",
                ),
            ),
            (
                "no landmarks",
                lambda h: h.replace("<main>", "<div>")
                .replace("</main>", "</div>")
                .replace("<header>", "<div>")
                .replace("</header>", "</div>")
                .replace("<footer>", "<div>")
                .replace("</footer>", "</div>")
                .replace("<nav ", "<div ")
                .replace("</nav>", "</div>"),
            ),
            (
                "unlabelled inputs",
                lambda h: h.replace('<label for="name">Your name</label>', "").replace(
                    '<label for="email">Email</label>', ""
                ),
            ),
            (
                "render-blocking scripts",
                lambda h: h.replace(
                    '<link rel="stylesheet" href="styles.css">',
                    '<script src="https://cdn.example.com/a.js"></script>'
                    '<script src="https://cdn.example.com/b.js"></script>'
                    '<link rel="stylesheet" href="styles.css">',
                ),
            ),
        ],
    )
    async def test_html_defect_lowers_score(self, validator, tmp_path, defect, mutate):
        base = await self._baseline(validator, tmp_path)
        site = _inject(tmp_path, defect.replace(" ", "_"), html=mutate(_GOOD_HTML))
        after = (await validator.validate(site)).score
        assert after < base, (
            f"injecting '{defect}' did not lower the score: "
            f"baseline={base:.3f} after={after:.3f}"
        )

    async def test_dropping_breakpoints_lowers_score(self, validator, tmp_path):
        base = await self._baseline(validator, tmp_path)
        one_bp = "body{color:#111}\n@media (min-width: 768px) { .g { gap: 1rem; } }\n"
        after = (await validator.validate(_inject(tmp_path, "one_bp", css=one_bp))).score
        assert after < base, f"dropping to one breakpoint did not lower: {base:.3f} -> {after:.3f}"

    async def test_page_weight_lowers_score(self, validator, tmp_path):
        base = await self._baseline(validator, tmp_path)
        heavy = _inject(tmp_path, "heavy", image_bytes=6_000_000)
        after = (await validator.validate(heavy)).score
        assert after < base, f"6MB of images did not lower the score: {base:.3f} -> {after:.3f}"

    async def test_every_defect_is_attributed_to_a_named_check(self, validator, tmp_path):
        """A lower score must come with a failing check that says why."""
        site = _inject(
            tmp_path, "attributed", html=_GOOD_HTML.replace('<html lang="en">', "<html>")
        )
        report = await validator.validate(site)
        failing = [c.name for c in report.failed_checks()]
        assert (
            "Accessibility (WCAG 2.1 AA)" in failing
        ), f"missing lang must be attributed to the accessibility check; failing={failing}"


# ── Conservative aggregation ─────────────────────────────────────────────────
#
# A plain mean cannot express "this site is unshippable for one reason". With
# six applicable checks, a single check collapsing from 1.00 to 0.00 moves the
# mean by 0.17 — and measured on real defects it was far less than that: a
# placeholder <title> cost 0.021, six megabytes of images cost 0.029. The gate
# noticed every defect and cared about none of them.
#
# The aggregate is therefore half average, half worst: `0.5*mean + 0.5*min`.
# A site is only as shippable as its weakest dimension, which is the same
# conservative-aggregation doctrine the evaluator uses for self-consistency.


@pytest.mark.unit
class TestConservativeAggregation:
    @staticmethod
    def _report(*scores):
        from orchestrator.design_system import QualityCheck, QualityReport

        return QualityReport(
            checks=[
                QualityCheck(name=f"c{i}", passed=s >= 0.7, score=s, details="")
                for i, s in enumerate(scores)
            ]
        )

    def test_all_high_scores_high(self):
        assert self._report(1.0, 1.0, 1.0, 1.0).score == pytest.approx(1.0)

    def test_one_fatal_check_sinks_the_aggregate(self):
        """Five perfect checks must not hide one that scored zero."""
        report = self._report(1.0, 1.0, 1.0, 1.0, 1.0, 0.0)
        # mean = 0.833, min = 0.0 -> 0.417. A plain mean would have said 0.833.
        assert report.score == pytest.approx(0.42, abs=0.01)
        assert report.score < 0.5

    def test_worst_check_matters_as_much_as_the_average(self):
        one_bad = self._report(1.0, 1.0, 1.0, 0.5)
        all_mid = self._report(0.85, 0.85, 0.85, 0.85)
        assert one_bad.score < all_mid.score, (
            "a site with one weak dimension must rank below one that is "
            "uniformly acceptable at the same mean"
        )

    def test_still_bounded_zero_to_one(self):
        assert self._report(0.0, 0.0).score == 0.0
        assert self._report(1.0).score == pytest.approx(1.0)

    def test_single_check_is_its_own_aggregate(self):
        assert self._report(0.6).score == pytest.approx(0.6)

    async def test_defects_now_cost_real_score(self, validator, tmp_path):
        """Sensitivity: the worst single defect must move the score materially."""
        base = (await validator.validate(_inject(tmp_path, "cbase"))).score
        one_bp = _inject(
            tmp_path, "cbp", css="body{color:#111}\n@media (min-width:768px){.g{gap:1rem}}\n"
        )
        after = (await validator.validate(one_bp)).score
        assert base - after >= 0.15, (
            f"dropping from four breakpoints to one should cost real score: "
            f"{base:.3f} -> {after:.3f} (delta {after - base:+.3f})"
        )


@pytest.mark.unit
class TestSecretExposureIsATrueNegative:
    """Secret Exposure scores 1.00 on every site in the corpus.

    That is a check finding nothing, not a check measuring nothing — the
    distinction the rest of this suite exists to enforce. This test proves it
    still fires, so it can be trusted rather than assumed dead.
    """

    async def test_detects_a_leaked_key(self, validator, tmp_path):
        d = tmp_path / "leaky"
        d.mkdir(parents=True, exist_ok=True)
        _write(d, _GOOD_HTML, _GOOD_CSS)
        (d / "script.js").write_text(
            'const OPENAI_API_KEY = "sk-proj-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";\n',
            encoding="utf-8",
        )
        check = await validator._check_secret_exposure(d)
        assert check.passed is False, f"leaked key not detected: {check.details}"
        assert check.score < 1.0

    async def test_clean_site_passes(self, validator, good_dir):
        check = await validator._check_secret_exposure(good_dir)
        assert check.passed is True


# ── Mobile-first ─────────────────────────────────────────────────────────────
#
# "Responsive" and "mobile-first" are not the same claim. A desktop-first sheet
# is responsive — it has breakpoints — but its BASE styles target a wide screen
# and every `max-width` query walks them back down. On a phone the browser
# parses the desktop layout first and then overrides it.
#
# The generator emitted zero `min-width` queries before this change, so every
# site it produced was desktop-first by construction.


def _css_with(queries: str) -> str:
    return "body{margin:0}\na:focus-visible{outline:2px solid}\n" + queries


_MOBILE_FIRST_CSS = "".join(
    f"@media (min-width: {w}) {{ .g {{ gap: 1rem; }} }}\n" for w in ("30em", "48em", "64em", "80em")
)
_DESKTOP_FIRST_CSS = "".join(
    f"@media (max-width: {w}) {{ .g {{ display: block; }} }}\n"
    for w in ("80em", "64em", "48em", "30em")
)


@pytest.mark.unit
class TestMobileFirstIsEnforced:
    async def test_mobile_first_sheet_scores_higher_than_desktop_first(self, validator, tmp_path):
        mf = _inject(tmp_path, "mf", css=_css_with(_MOBILE_FIRST_CSS))
        df = _inject(tmp_path, "df", css=_css_with(_DESKTOP_FIRST_CSS))
        mf_score = (await validator._check_responsive(mf)).score
        df_score = (await validator._check_responsive(df)).score
        assert mf_score > df_score, (
            f"a desktop-first sheet must not score the same as a mobile-first one: "
            f"mobile-first={mf_score} desktop-first={df_score}"
        )

    async def test_desktop_first_is_named_in_the_details(self, validator, tmp_path):
        df = _inject(tmp_path, "df2", css=_css_with(_DESKTOP_FIRST_CSS))
        check = await validator._check_responsive(df)
        assert (
            "mobile" in check.details.lower()
        ), f"the failure must say what is wrong, got: {check.details}"

    async def test_mobile_first_sheet_is_not_penalised(self, validator, tmp_path):
        mf = _inject(tmp_path, "mf2", css=_css_with(_MOBILE_FIRST_CSS))
        check = await validator._check_responsive(mf)
        assert check.score >= 0.85 and check.passed, f"{check.score}: {check.details}"

    async def test_a_few_max_width_queries_are_tolerated(self, validator, tmp_path):
        """Mobile-first sheets legitimately use the occasional max-width."""
        mixed = _css_with(_MOBILE_FIRST_CSS + "@media (max-width: 30em){.h{display:none}}\n")
        site = _inject(tmp_path, "mixed", css=mixed)
        check = await validator._check_responsive(site)
        assert check.passed, f"a mostly-min-width sheet must pass: {check.details}"
