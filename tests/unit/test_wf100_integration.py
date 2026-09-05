"""End to end: render the shipped template, then audit what came out.

This is the loop the factory actually runs — build a site from a template and a
client file, then hold it to WF-100 before anyone deploys it. It is also the
test that would catch the generator and the auditor drifting apart, which no
amount of testing either one alone would find.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from orchestrator.generators.website_template import apply_template, load_template
from orchestrator.generators.wf100.auditor import audit_directory
from orchestrator.generators.wf100.render import exit_code_for, render_markdown, render_text
from orchestrator.generators.wf100.report import Status, Verdict

pytestmark = [pytest.mark.unit, pytest.mark.integration]

TEMPLATE = Path(__file__).resolve().parents[2] / "templates" / "websites" / "dentist"


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    template = load_template(TEMPLATE)
    client = yaml.safe_load((TEMPLATE / "client.example.yaml").read_text(encoding="utf-8"))
    out = tmp_path_factory.mktemp("audited-site")
    apply_template(template, client, out)
    return out


@pytest.fixture(scope="module")
def report(rendered):
    return audit_directory(rendered)


class TestTheGeneratorMeetsItsOwnStandard:
    def test_the_generated_site_has_no_critical_failures(self, report):
        assert report.blockers == ()
        assert report.critical_failures() == ()

    def test_the_generated_site_passes_every_automated_check_it_can(self, report):
        # G8 is the one expected failure and it is expected on purpose: a privacy
        # policy is a legal document about this practice, and the generator will
        # not write a plausible fake to make a check go green.
        assert [f.check.id for f in report.failures()] == ["G8"]

    def test_the_privacy_gap_is_reported_rather_than_papered_over(self, report):
        assert "privacy policy" in report.finding("G8").detail

    def test_it_scores_well_above_half_on_verified_points_alone(self, report):
        assert report.score > 60

    def test_resolving_the_outstanding_work_would_reach_the_threshold(self, report):
        assert report.ceiling >= 90


class TestItStillRefusesToSignOff:
    def test_the_verdict_is_pending_never_launch(self, report):
        # Seven critical checks need a person or a live deployment. No static
        # audit may clear a site while those are unanswered.
        assert report.verdict is Verdict.PENDING

    def test_the_unverified_criticals_are_named(self, report):
        unverified = {f.check.id for f in report.critical_unverified()}
        assert {"H3", "H4", "H7", "F7", "G1"} <= unverified

    def test_fabrication_checks_are_never_passed_by_the_tool(self, report):
        for cid in ("H2", "H3", "H4", "H6", "H7"):
            assert report.finding(cid).status is Status.OUTSTANDING, cid

    def test_core_web_vitals_are_never_passed_by_the_tool(self, report):
        for cid in ("B1", "B2", "B3"):
            assert report.finding(cid).status is Status.OUTSTANDING, cid

    def test_pending_exits_two_so_a_pipeline_can_tell_it_from_broken(self, report):
        assert exit_code_for(report) == 2


class TestGeneratedArtifacts:
    def test_the_generator_ships_a_404_page(self, rendered):
        assert (rendered / "404.html").is_file()

    def test_the_404_is_noindex_and_keeps_the_navigation(self, rendered):
        markup = (rendered / "404.html").read_text(encoding="utf-8")
        assert "noindex" in markup
        assert "<nav" in markup

    def test_the_404_does_not_count_as_an_accidental_noindex(self, report):
        assert report.finding("D9").status is Status.PASS
        assert [b.code for b in report.blockers] == []

    def test_content_sits_inside_a_main_landmark(self, rendered):
        markup = (rendered / "index.html").read_text(encoding="utf-8")
        assert '<main id="main">' in markup
        assert markup.index("<header") < markup.index('<main id="main"')
        assert markup.index("</main>") < markup.index("<footer")

    def test_the_booking_form_carries_a_honeypot(self, rendered):
        assert 'class="hp"' in (rendered / "index.html").read_text(encoding="utf-8")

    def test_the_booking_form_names_its_own_error_messages(self, report):
        assert report.finding("F8").status is Status.PASS

    def test_directions_are_derivable_from_the_address_alone(self, rendered):
        markup = (rendered / "index.html").read_text(encoding="utf-8")
        assert "google.com/maps" in markup

    def test_an_faq_the_client_wrote_becomes_faqpage_schema(self, rendered):
        assert "FAQPage" in (rendered / "index.html").read_text(encoding="utf-8")


class TestReportsAreProducible:
    def test_the_terminal_report_renders(self, report):
        text = render_text(report)
        assert "WF-100" in text
        assert "OUTSTANDING" in text

    def test_the_client_report_renders(self, report):
        markdown = render_markdown(report)
        assert markdown.startswith("# Website Quality Report")
        assert "Outstanding" in markdown
