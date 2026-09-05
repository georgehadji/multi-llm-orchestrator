"""Rendering: the report must not read as better than it is."""

from __future__ import annotations

import pytest

from orchestrator.generators.wf100.render import (
    exit_code_for,
    render_json,
    render_markdown,
    render_text,
)
from orchestrator.generators.wf100.report import AuditReport, Blocker, Finding, Status, Verdict
from orchestrator.generators.wf100.standard import STANDARD, get

pytestmark = pytest.mark.unit


def _report(**by_id):
    findings = [
        Finding(check=c, status=by_id.get(c.id, Status.PASS), detail="fixture") for c in STANDARD
    ]
    return AuditReport(findings=findings, site="example", mode="directory")


class TestTextReport:
    def test_leads_with_the_verdict(self):
        assert "LAUNCH" in render_text(_report()).splitlines()[1]

    def test_states_outstanding_points_next_to_the_score(self):
        text = render_text(_report(B1=Status.OUTSTANDING, H3=Status.OUTSTANDING))
        assert "outstanding 2" in text
        assert "ceiling" in text

    def test_says_outstanding_checks_are_not_passes(self):
        text = render_text(_report(H3=Status.OUTSTANDING))
        assert "not passes" in text

    def test_groups_outstanding_work_by_verification_level(self):
        text = render_text(_report(H3=Status.OUTSTANDING, B1=Status.OUTSTANDING))
        assert "Level 3 — human review" in text
        assert "Level 2 — machine-assisted" in text

    def test_marks_unverified_critical_checks(self):
        text = render_text(_report(H4=Status.OUTSTANDING))
        assert "H4 [CRITICAL]" in text

    def test_prints_a_remedy_for_every_failure(self):
        text = render_text(_report(B4=Status.FAIL))
        assert get("B4").remedy in text

    def test_explains_outstanding_checks_when_verbose(self):
        report = _report(H3=Status.OUTSTANDING)
        assert "fixture" not in render_text(report)
        assert "fixture" in render_text(report, verbose=True)


class TestMarkdownReport:
    def test_breaks_the_hundred_points_down_by_status(self):
        markdown = render_markdown(_report(B1=Status.OUTSTANDING, B4=Status.FAIL))
        assert "| **Verified and earned** | **98** |" in markdown
        assert "| Outstanding — not yet verified | 1 |" in markdown
        assert "| Failing | 1 |" in markdown

    def test_states_plainly_that_outstanding_is_not_a_pass(self):
        markdown = render_markdown(_report(H3=Status.OUTSTANDING))
        assert "not** been assessed" in markdown
        assert "not passes" in markdown

    def test_records_why_a_check_was_ruled_not_applicable(self):
        report = AuditReport(
            findings=[
                Finding(check=get("G9"), status=Status.NOT_APPLICABLE, detail="no cookies are set")
            ]
        )
        markdown = render_markdown(report)
        assert "no cookies are set" in markdown

    def test_shows_critical_failures_before_the_scores(self):
        markdown = render_markdown(
            AuditReport(
                findings=[Finding(check=c, status=Status.PASS, detail="x") for c in STANDARD],
                blockers=[
                    Blocker("MIXED_CONTENT", "Mixed content", "2 http refs", remedy="use https")
                ],
            )
        )
        assert markdown.index("Critical failures") < markdown.index("Category scores")

    def test_lists_the_limits_of_the_audit(self):
        report = AuditReport(
            findings=[Finding(check=get("A1"), status=Status.PASS, detail="ok")],
            tool_notes=("Core Web Vitals are field metrics.",),
        )
        assert "Limits of this audit" in render_markdown(report)


class TestJsonReport:
    def test_round_trips(self):
        import json

        data = json.loads(render_json(_report()))
        assert data["verdict"] == "launch"
        assert len(data["findings"]) == 100


class TestExitCode:
    def test_launch_is_zero(self):
        assert exit_code_for(_report()) == 0

    def test_broken_is_one_and_unfinished_is_two(self):
        broken = _report(**{c.id: Status.FAIL for c in STANDARD[:15]})
        assert broken.verdict is Verdict.NO_LAUNCH
        assert exit_code_for(broken) == 1
        assert exit_code_for(_report(H3=Status.OUTSTANDING)) == 2
