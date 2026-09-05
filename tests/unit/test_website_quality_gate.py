"""
Unit tests for the website quality gate.

Background — the defect these tests lock down:
`WebsiteQualityValidator.validate()` ran every check, produced real per-check
scores, then built `QualityReport(checks=..., lighthouse_score=..., ...)`
WITHOUT passing `score=` or `passed=`. Both fell back to their `__init__`
defaults (0.0 / False), so the aggregate verdict was a constant regardless of
output quality. Nothing downstream could gate on it, and the generator's
"quality score: %.2f" log line printed 0.00 for every run ever made.

These tests assert that:
1. QualityReport derives its aggregate score from its checks.
2. QualityReport derives `passed` from its checks (conservative: all must pass).
3. Explicitly-supplied score/passed still win (back-compat for callers).
4. Both shipped validators (root + generators) return a real aggregate.
5. WebsiteBuildResult exposes a quality-gate verdict the CLI can exit on.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


@pytest.fixture
def QC():
    from orchestrator.design_system import QualityCheck

    return QualityCheck


@pytest.fixture
def QR():
    from orchestrator.design_system import QualityReport

    return QualityReport


@pytest.mark.unit
class TestQualityReportAggregation:
    def test_score_is_conservative_not_a_plain_mean(self, QC, QR):
        """Aggregate is 0.5*mean + 0.5*min — see QualityReport._derive_score.

        This asserted the plain mean (0.75) until 2026-09-05. A mean let one
        fatal dimension hide behind good siblings: measured against a clean
        fixture, a placeholder <title> moved the aggregate by 0.021 and six
        megabytes of images by 0.029. The gate noticed every defect and cared
        about none of them.
        """
        checks = [
            QC(name="a", passed=True, score=1.0, details=""),
            QC(name="b", passed=True, score=0.5, details=""),
        ]
        report = QR(checks=checks)
        # mean 0.75, min 0.50 -> 0.625
        assert report.score == pytest.approx(0.625)

    def test_score_not_constant_zero_for_good_output(self, QC, QR):
        checks = [QC(name=n, passed=True, score=1.0, details="") for n in "abc"]
        report = QR(checks=checks)
        assert report.score > 0.0, "aggregate score must reflect the checks, not default to 0.0"

    def test_passed_is_false_when_any_check_fails(self, QC, QR):
        checks = [
            QC(name="a", passed=True, score=1.0, details=""),
            QC(name="responsive", passed=False, score=0.5, details=""),
        ]
        report = QR(checks=checks)
        assert report.passed is False

    def test_passed_is_true_when_all_checks_pass(self, QC, QR):
        checks = [QC(name=n, passed=True, score=0.9, details="") for n in "ab"]
        report = QR(checks=checks)
        assert report.passed is True

    def test_explicit_score_overrides_derivation(self, QC, QR):
        checks = [QC(name="a", passed=True, score=1.0, details="")]
        report = QR(checks=checks, score=0.25, passed=False)
        assert report.score == pytest.approx(0.25)
        assert report.passed is False

    def test_no_checks_is_zero_and_not_passed(self, QR):
        report = QR(checks=[])
        assert report.score == 0.0
        assert report.passed is False

    def test_failed_check_names_are_reportable(self, QC, QR):
        checks = [
            QC(name="SEO Basics", passed=True, score=1.0, details=""),
            QC(name="Responsive Design", passed=False, score=0.5, details="only 1 breakpoint"),
        ]
        report = QR(checks=checks)
        assert [c.name for c in report.checks if not c.passed] == ["Responsive Design"]


@pytest.mark.unit
class TestValidatorsReturnRealAggregate:
    """Both shipped validator copies must produce a non-constant aggregate."""

    @staticmethod
    def _write_site(tmp_path):
        (tmp_path / "index.html").write_text(
            "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width, initial-scale=1'>"
            "<title>Real Title</title><meta name='description' content='A real description "
            "of the page contents for search engines.'></head><body><h1>Heading</h1>"
            "<img src='a.png' alt='descriptive alt'><p>Body copy that is long enough.</p>"
            "</body></html>",
            encoding="utf-8",
        )
        (tmp_path / "styles.css").write_text(
            "@media (min-width:640px){body{color:#111}}"
            "@media (min-width:768px){body{color:#222}}"
            "@media (min-width:1024px){body{color:#333}}"
            "@media (min-width:1280px){body{color:#444}}",
            encoding="utf-8",
        )

    @pytest.mark.asyncio
    async def test_root_validator_returns_nonzero_score(self, tmp_path):
        from orchestrator.website_validator import WebsiteQualityValidator

        self._write_site(tmp_path)
        report = await WebsiteQualityValidator().validate(tmp_path)
        assert report.score > 0.0

    @pytest.mark.asyncio
    async def test_generators_validator_returns_nonzero_score(self, tmp_path):
        from orchestrator.generators.website_validator import WebsiteQualityValidator

        self._write_site(tmp_path)
        report = await WebsiteQualityValidator().validate(tmp_path)
        assert report.score > 0.0


@pytest.mark.unit
class TestBuildResultGateVerdict:
    """The factory needs a machine-readable verdict, separate from 'the pipeline ran'."""

    def test_result_exposes_gate_fields(self):
        from orchestrator.generators.website_generator import WebsiteBuildResult

        result = WebsiteBuildResult(output_dir="/tmp/x")
        assert hasattr(result, "quality_gate_passed")
        assert hasattr(result, "gate_failures")

    def test_gate_verdict_defaults_to_unknown_not_pass(self):
        """An un-validated build must never report a passing gate."""
        from orchestrator.generators.website_generator import WebsiteBuildResult

        result = WebsiteBuildResult(output_dir="/tmp/x")
        assert result.quality_gate_passed is not True


@pytest.mark.unit
class TestContentQualityScansHtml:
    """`_check_content_quality` globbed only **/*.tsx and **/*.jsx.

    On a static HTML site it therefore scanned ZERO files, found zero issues and
    returned a perfect 1.00 — which is how a page titled "Untitled Project"
    passed content quality. A check that passes because it examined nothing is
    the same defect class as the discarded aggregate.
    """

    @staticmethod
    def _site(tmp_path, title, body="<h1>Hello</h1><p>Some real copy here.</p>"):
        (tmp_path / "index.html").write_text(
            f"<!DOCTYPE html><html lang='en'><head><title>{title}</title></head>"
            f"<body>{body}</body></html>",
            encoding="utf-8",
        )
        return tmp_path

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "validator_path",
        [
            "orchestrator.website_validator",
            "orchestrator.generators.website_validator",
        ],
    )
    async def test_placeholder_title_fails_content_quality(self, tmp_path, validator_path):
        import importlib

        mod = importlib.import_module(validator_path)
        site = self._site(tmp_path, "Untitled Project")
        check = await mod.WebsiteQualityValidator()._check_content_quality(site)
        assert check.passed is False, "'Untitled Project' must be caught as placeholder content"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "validator_path",
        [
            "orchestrator.website_validator",
            "orchestrator.generators.website_validator",
        ],
    )
    async def test_lorem_ipsum_in_html_body_fails(self, tmp_path, validator_path):
        import importlib

        mod = importlib.import_module(validator_path)
        site = self._site(tmp_path, "Real Brand", body="<p>Lorem ipsum dolor sit amet.</p>")
        check = await mod.WebsiteQualityValidator()._check_content_quality(site)
        assert check.passed is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "validator_path",
        [
            "orchestrator.website_validator",
            "orchestrator.generators.website_validator",
        ],
    )
    async def test_real_content_still_passes(self, tmp_path, validator_path):
        import importlib

        mod = importlib.import_module(validator_path)
        site = self._site(tmp_path, "Gadini Barberia — Italian Grooming")
        check = await mod.WebsiteQualityValidator()._check_content_quality(site)
        assert check.passed is True, "must not fire on genuine copy (no false positives)"


@pytest.mark.unit
class TestRateLimitApplicability:
    """A static brochure site has no endpoints to rate-limit.

    Without an applicability guard this check fails EVERY static site forever,
    which makes the aggregate score useless as a discriminator. `_check_auth_flow`
    already models the right behaviour ("no auth pages found — not applicable").
    """

    @pytest.mark.asyncio
    async def test_static_site_is_not_applicable(self, tmp_path):
        from orchestrator.generators.website_validator import WebsiteQualityValidator

        (tmp_path / "index.html").write_text(
            "<!DOCTYPE html><html><body><h1>Barber shop</h1>"
            "<a href='tel:+301234567'>Call us</a></body></html>",
            encoding="utf-8",
        )
        (tmp_path / "styles.css").write_text("body{color:#111}", encoding="utf-8")
        check = await WebsiteQualityValidator()._check_rate_limiting(tmp_path)
        assert check.passed is True
        assert "not applicable" in check.details.lower()

    @pytest.mark.asyncio
    async def test_site_with_form_post_still_requires_rate_limiting(self, tmp_path):
        from orchestrator.generators.website_validator import WebsiteQualityValidator

        (tmp_path / "index.html").write_text(
            "<!DOCTYPE html><html><body>"
            "<form method='post' action='/api/contact'><input name='email'></form>"
            "</body></html>",
            encoding="utf-8",
        )
        (tmp_path / "server.js").write_text(
            "app.post('/api/contact', (req,res)=>{ res.send('ok') })", encoding="utf-8"
        )
        check = await WebsiteQualityValidator()._check_rate_limiting(tmp_path)
        assert check.passed is False, "a real POST endpoint must still demand rate limiting"


@pytest.mark.unit
class TestApplyQualityGate:
    """The gate translates a report into a ship / don't-ship verdict."""

    @staticmethod
    def _fixture(scores_passed):
        from orchestrator.design_system import QualityCheck, QualityReport
        from orchestrator.generators.website_generator import WebsiteBuildResult, WebsiteConfig

        checks = [
            QualityCheck(name=f"check{i}", passed=p, score=s, details="d")
            for i, (s, p) in enumerate(scores_passed)
        ]
        return WebsiteBuildResult(output_dir="/tmp/x"), QualityReport(checks=checks), WebsiteConfig

    def test_gate_disabled_yields_no_verdict(self):
        from orchestrator.generators.website_generator import WebsiteGenerator

        result, report, Config = self._fixture([(1.0, True)])
        WebsiteGenerator._apply_quality_gate(result, report, Config(min_quality=0.0))
        assert result.quality_gate_passed is None, "disabled gate must withhold a verdict, not pass"

    def test_score_above_threshold_passes(self):
        from orchestrator.generators.website_generator import WebsiteGenerator

        result, report, Config = self._fixture([(1.0, True), (0.9, True)])
        WebsiteGenerator._apply_quality_gate(result, report, Config(min_quality=0.8))
        assert result.quality_gate_passed is True

    def test_score_below_threshold_fails(self):
        from orchestrator.generators.website_generator import WebsiteGenerator

        result, report, Config = self._fixture([(0.5, False), (0.6, True)])
        WebsiteGenerator._apply_quality_gate(result, report, Config(min_quality=0.8))
        assert result.quality_gate_passed is False
        assert result.gate_failures, "a failing gate must say which checks failed"

    def test_require_all_checks_overrides_a_clearing_score(self):
        """A failing check must block even when the aggregate clears min_quality.

        The premise moved on 2026-09-05: the aggregate is now 0.5*mean + 0.5*min,
        so a 0.5 among three 1.0s no longer clears 0.8 on its own (it scores
        0.6875). The behaviour under test is unchanged — require_all_checks
        still rejects a build with any failing check — so the fixture uses a
        weak-but-failing 0.90 to keep the score above the bar.
        """
        from orchestrator.generators.website_generator import WebsiteGenerator

        # mean 0.975, min 0.90 -> conservative aggregate 0.9375, above the bar.
        result, report, Config = self._fixture(
            [(1.0, True), (1.0, True), (1.0, True), (0.9, False)]
        )
        cfg = Config(min_quality=0.8, require_all_checks=True)
        assert report.score > 0.8, f"premise broken: score={report.score}"
        WebsiteGenerator._apply_quality_gate(result, report, cfg)
        assert result.quality_gate_passed is False

    def test_gate_failures_name_the_check(self):
        from orchestrator.generators.website_generator import WebsiteGenerator

        result, report, Config = self._fixture([(0.5, False)])
        WebsiteGenerator._apply_quality_gate(result, report, Config(min_quality=0.9))
        assert "check0" in result.gate_failures[0]


@pytest.mark.unit
class TestWebsiteCliGateFlags:
    def test_cli_exposes_gate_flags(self):
        import argparse

        from orchestrator.commands import website as website_cmd

        parser = argparse.ArgumentParser()
        website_cmd.register(parser.add_subparsers())
        args = parser.parse_args(["website", "-d", "x", "--min-quality", "0.85"])
        assert args.min_quality == pytest.approx(0.85)
        assert hasattr(args, "require_all_checks")

    def test_min_quality_defaults_to_disabled(self):
        import argparse

        from orchestrator.commands import website as website_cmd

        parser = argparse.ArgumentParser()
        website_cmd.register(parser.add_subparsers())
        args = parser.parse_args(["website", "-d", "x"])
        assert args.min_quality == 0.0, "gate is opt-in; existing invocations must not change"
