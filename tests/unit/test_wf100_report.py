"""WF-100 scoring: what an unmeasured check is worth.

The rule the whole tool exists to enforce: a check the auditor could not decide
earns nothing and is reported as outstanding. It is not a pass, and it is not
quietly dropped from the denominator either — it is a debt against the score,
visible until someone clears it.
"""

from __future__ import annotations

import pytest

from orchestrator.generators.wf100.report import (
    AuditReport,
    Finding,
    Status,
    Verdict,
)
from orchestrator.generators.wf100.standard import STANDARD, Category, get

pytestmark = pytest.mark.unit


def _all(status: Status, detail: str = "fixture") -> list[Finding]:
    return [Finding(check=c, status=status, detail=detail) for c in STANDARD]


def _mixed(**by_id: Status) -> list[Finding]:
    """Every check passes except the ids named, which take the status given."""
    return [
        Finding(check=c, status=by_id.get(c.id, Status.PASS), detail="fixture") for c in STANDARD
    ]


class TestPerfectAndEmpty:
    def test_all_passing_scores_one_hundred(self):
        report = AuditReport(findings=_all(Status.PASS))
        assert report.score == 100.0

    def test_all_passing_launches(self):
        report = AuditReport(findings=_all(Status.PASS))
        assert report.verdict is Verdict.LAUNCH

    def test_all_failing_scores_zero(self):
        report = AuditReport(findings=_all(Status.FAIL))
        assert report.score == 0.0

    def test_an_empty_report_never_launches(self):
        report = AuditReport(findings=[])
        assert report.verdict is Verdict.NO_LAUNCH
        assert report.score == 0.0


class TestOutstandingIsNotAPass:
    def test_outstanding_earns_no_points(self):
        report = AuditReport(findings=_mixed(B1=Status.OUTSTANDING))
        assert report.score == 99.0

    def test_outstanding_stays_in_the_denominator(self):
        # Contrast with NOT_APPLICABLE below: an unverified check is a debt, so
        # it must not be excused by shrinking what the score is measured against.
        report = AuditReport(findings=_mixed(B1=Status.OUTSTANDING))
        assert report.assessable == 100

    def test_outstanding_is_reported_separately_from_failure(self):
        report = AuditReport(findings=_mixed(B1=Status.OUTSTANDING, B4=Status.FAIL))
        assert report.outstanding_points == 1
        assert report.failed_points == 1
        assert [f.check.id for f in report.outstanding()] == ["B1"]
        assert [f.check.id for f in report.failures()] == ["B4"]

    def test_ceiling_shows_what_resolving_the_outstanding_would_buy(self):
        report = AuditReport(findings=_mixed(B1=Status.OUTSTANDING, B2=Status.OUTSTANDING))
        assert report.score == 98.0
        assert report.ceiling == 100.0

    def test_a_site_that_cannot_reach_the_threshold_is_refused_outright(self):
        # 15 hard failures: even if every outstanding check passed, 85 < 90.
        failing = {c.id: Status.FAIL for c in STANDARD[:15]}
        report = AuditReport(findings=_mixed(**failing))
        assert report.ceiling == 85.0
        assert report.verdict is Verdict.NO_LAUNCH

    def test_a_site_that_could_still_reach_the_threshold_is_pending(self):
        outstanding = {c.id: Status.OUTSTANDING for c in STANDARD[:15]}
        report = AuditReport(findings=_mixed(**outstanding))
        assert report.score == 85.0
        assert report.ceiling == 100.0
        assert report.verdict is Verdict.PENDING


class TestNotApplicable:
    def test_not_applicable_leaves_the_denominator(self):
        report = AuditReport(findings=_mixed(G9=Status.NOT_APPLICABLE))
        assert report.assessable == 99
        assert report.not_applicable_points == 1

    def test_not_applicable_does_not_inflate_the_score(self):
        # 98 passes, one N/A, one failure: 98 of 99 assessable points.
        report = AuditReport(findings=_mixed(G9=Status.NOT_APPLICABLE, B4=Status.FAIL))
        assert report.score == pytest.approx(98 / 99 * 100, abs=0.01)

    def test_not_applicable_must_be_justified(self):
        with pytest.raises(ValueError, match="justif"):
            Finding(check=get("G9"), status=Status.NOT_APPLICABLE, detail="")

    def test_a_report_of_only_not_applicable_checks_never_launches(self):
        report = AuditReport(findings=_all(Status.NOT_APPLICABLE, detail="nothing to judge"))
        assert report.assessable == 0
        assert report.score == 0.0
        assert report.verdict is Verdict.NO_LAUNCH


class TestCriticalFailures:
    def test_a_critical_failure_blocks_launch_at_any_score(self):
        report = AuditReport(findings=_mixed(H1=Status.FAIL))
        assert report.score == 99.0
        assert report.verdict is Verdict.NO_LAUNCH
        assert [f.check.id for f in report.critical_failures()] == ["H1"]

    def test_a_non_critical_failure_at_ninety_still_launches(self):
        # The standard's own rule: score >= 90 and zero critical failures.
        non_critical = [c for c in STANDARD if not c.critical][:10]
        report = AuditReport(findings=_mixed(**{c.id: Status.FAIL for c in non_critical}))
        assert report.score == 90.0
        assert report.critical_failures() == ()
        assert report.verdict is Verdict.LAUNCH

    def test_an_unverified_critical_check_blocks_launch(self):
        # An unread smoke alarm is not an absence of fire. H3 (fabricated
        # testimonials) outstanding means nobody has checked, so nobody may sign off.
        report = AuditReport(findings=_mixed(H3=Status.OUTSTANDING))
        assert report.score == 99.0
        assert report.critical_failures() == ()
        assert [f.check.id for f in report.critical_unverified()] == ["H3"]
        assert report.verdict is Verdict.PENDING

    def test_verdict_reason_names_the_blocker(self):
        report = AuditReport(findings=_mixed(H4=Status.FAIL))
        assert "H4" in report.verdict_reason


class TestCategoryBreakdown:
    def test_every_category_is_reported(self):
        report = AuditReport(findings=_all(Status.PASS))
        assert set(report.by_category()) == set(Category)

    def test_a_category_reports_earned_against_its_weight(self):
        report = AuditReport(findings=_mixed(B1=Status.OUTSTANDING, B2=Status.FAIL))
        perf = report.by_category()[Category.PERFORMANCE]
        assert perf.weight == 15
        assert perf.earned == 13
        assert perf.outstanding == 1
        assert perf.failed == 1

    def test_category_scores_are_independent(self):
        report = AuditReport(findings=_mixed(B1=Status.FAIL))
        assert report.by_category()[Category.ACCESSIBILITY].earned == 15


class TestDuplicateAndUnknownFindings:
    def test_a_duplicate_finding_is_rejected(self):
        findings = _all(Status.PASS)
        findings.append(Finding(check=get("A1"), status=Status.FAIL, detail="again"))
        with pytest.raises(ValueError, match="duplicate"):
            AuditReport(findings=findings)

    def test_missing_checks_are_recorded_as_outstanding(self):
        # A partial run must not look like a short standard. Anything the
        # auditor never got to is outstanding, not absent.
        report = AuditReport(findings=[Finding(check=get("A1"), status=Status.PASS, detail="ok")])
        assert len(report.findings) == 100
        assert report.outstanding_points == 99
        assert report.score == 1.0


class TestSerialisation:
    def test_report_round_trips_to_a_plain_dict(self):
        report = AuditReport(findings=_mixed(B1=Status.OUTSTANDING, H1=Status.FAIL), site="x")
        data = report.to_dict()
        assert data["score"] == 98.0
        assert data["verdict"] == "no_launch"
        assert data["site"] == "x"
        assert len(data["findings"]) == 100
        assert {"id", "status", "detail", "category", "level", "critical"} <= set(
            data["findings"][0]
        )

    def test_dict_is_json_serialisable(self):
        import json

        report = AuditReport(findings=_all(Status.PASS))
        assert json.loads(json.dumps(report.to_dict()))["score"] == 100.0


class TestAnAuditThatDidNotRun:
    """Zero decided checks is not a soft result — it is the absence of a result."""

    def test_nothing_decided_is_refused_not_pending(self):
        report = AuditReport(findings=[])
        assert report.coverage == 0.0
        assert report.ceiling == 100.0  # everything *could* pass; nothing was looked at
        assert report.verdict is Verdict.NO_LAUNCH

    def test_the_reason_says_the_audit_did_not_run(self):
        report = AuditReport(findings=[])
        assert "not an audit result" in report.verdict_reason

    def test_one_decided_check_is_enough_to_read_the_numbers(self):
        report = AuditReport(findings=[Finding(check=get("A1"), status=Status.PASS, detail="ok")])
        assert report.verdict is Verdict.PENDING


class TestBlockers:
    """Critical failures the standard lists apart from the hundred points."""

    def test_a_blocker_refuses_launch_at_a_perfect_score(self):
        from orchestrator.generators.wf100.report import Blocker

        report = AuditReport(
            findings=_all(Status.PASS),
            blockers=[
                Blocker(code="MIXED_CONTENT", title="Mixed content", detail="2 http:// refs")
            ],
        )
        assert report.score == 100.0
        assert report.verdict is Verdict.NO_LAUNCH
        assert "MIXED_CONTENT" in report.verdict_reason

    def test_blockers_reach_the_serialised_report(self):
        from orchestrator.generators.wf100.report import Blocker

        report = AuditReport(
            findings=_all(Status.PASS), blockers=[Blocker("X", "t", "d", remedy="fix it")]
        )
        assert report.to_dict()["blockers"] == [
            {"code": "X", "title": "t", "detail": "d", "evidence": [], "remedy": "fix it"}
        ]

    def test_no_blockers_leaves_the_verdict_alone(self):
        assert AuditReport(findings=_all(Status.PASS), blockers=[]).verdict is Verdict.LAUNCH
