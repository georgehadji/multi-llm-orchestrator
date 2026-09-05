"""WF-100 scoring and the launch decision.

Four statuses, and the difference between the last two is the whole point:

``PASS`` / ``FAIL``
    The auditor decided. The points are earned or they are not.

``OUTSTANDING``
    The auditor could *not* decide — the evidence it needed was not available.
    The points are not earned, and the check stays in the denominator. An
    outstanding check is a debt against the score, visible until someone clears
    it. It is never silently upgraded to a pass because the tool ran cleanly.

``NOT_APPLICABLE``
    The check has nothing to judge — cookie consent on a site that sets no
    cookies. Its points leave the denominator, and the reason is recorded and
    printed. This is the one status that can raise the score, so it is the one
    that must be justified in writing.

The verdict follows the standard: ``score >= 90 and critical failures == 0``,
with one addition the standard implies but does not spell out — an *unverified*
critical check also blocks launch. Nobody has looked at whether the
testimonials are real, so nobody can sign off that they are.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .standard import LAUNCH_THRESHOLD, STANDARD, Category, Check, Level, category_order


class Status(Enum):
    PASS = "pass"
    FAIL = "fail"
    OUTSTANDING = "outstanding"
    NOT_APPLICABLE = "not_applicable"


class Verdict(Enum):
    """The launch decision. Only ``LAUNCH`` clears a site to go live."""

    LAUNCH = "launch"
    PENDING = "pending"
    NO_LAUNCH = "no_launch"

    @property
    def headline(self) -> str:
        return {
            Verdict.LAUNCH: "LAUNCH — the site meets WF-100",
            Verdict.PENDING: "PENDING — verification outstanding before launch",
            Verdict.NO_LAUNCH: "NO LAUNCH — blocking problems must be fixed",
        }[self]


@dataclass(frozen=True)
class Finding:
    """One check, decided or explicitly not.

    ``detail`` is what was observed, in the auditor's own words, and it is
    mandatory: a status without an observation behind it is an assertion, and
    this tool does not make assertions it cannot show.
    """

    check: Check
    status: Status
    detail: str
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.detail.strip():
            if self.status is Status.NOT_APPLICABLE:
                raise ValueError(
                    f"{self.check.id}: NOT_APPLICABLE needs a written justification — "
                    "excusing a check from the score without saying why is how a "
                    "gate stops measuring anything"
                )
            raise ValueError(
                f"{self.check.id}: every finding needs a detail describing what was seen"
            )

    @property
    def points(self) -> int:
        return self.check.points if self.status is Status.PASS else 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.check.id,
            "title": self.check.title,
            "category": self.check.category.label,
            "level": self.check.level.value,
            "critical": self.check.critical,
            "status": self.status.value,
            "detail": self.detail,
            "evidence": list(self.evidence),
            "remedy": self.check.remedy if self.status is Status.FAIL else "",
            "notes": self.check.notes,
        }


@dataclass(frozen=True)
class Blocker:
    """A critical failure: no launch, at any score.

    The standard lists these separately from the hundred points, and so does
    this implementation. Some map onto a check (an accidental noindex is D9);
    others do not (mixed content, exposed personal data, a production error)
    and would have no home in a per-check score. Modelling them apart also
    stops a blocker from being outvoted by ninety-nine passing points, which is
    the entire reason the standard names them.
    """

    code: str
    title: str
    detail: str
    evidence: tuple[str, ...] = ()
    remedy: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "title": self.title,
            "detail": self.detail,
            "evidence": list(self.evidence),
            "remedy": self.remedy,
        }


@dataclass(frozen=True)
class CategoryScore:
    """One category's share of the hundred, broken out by status."""

    category: Category
    weight: int
    earned: int
    failed: int
    outstanding: int
    not_applicable: int

    @property
    def assessable(self) -> int:
        return self.weight - self.not_applicable

    @property
    def percent(self) -> float:
        if self.assessable <= 0:
            return 0.0
        return round(100.0 * self.earned / self.assessable, 1)


@dataclass
class AuditReport:
    """The result of auditing one site against WF-100.

    Any check the auditor never produced a finding for is filled in as
    OUTSTANDING. A partial run must read as a partial run — never as a short
    standard that happened to pass everything it looked at.
    """

    findings: tuple[Finding, ...] = ()
    site: str = ""
    mode: str = ""
    generated_at: str = ""
    tool_notes: tuple[str, ...] = ()
    blockers: tuple[Blocker, ...] = ()
    _index: dict[str, Finding] = field(default_factory=dict, repr=False)

    def __init__(
        self,
        findings: list[Finding] | tuple[Finding, ...] = (),
        site: str = "",
        mode: str = "",
        generated_at: str = "",
        tool_notes: tuple[str, ...] = (),
        blockers: list[Blocker] | tuple[Blocker, ...] = (),
    ) -> None:
        index: dict[str, Finding] = {}
        for finding in findings:
            if finding.check.id in index:
                raise ValueError(
                    f"duplicate finding for {finding.check.id}: a check is decided once, "
                    "or the score depends on which copy was appended last"
                )
            index[finding.check.id] = finding

        for check in STANDARD:
            if check.id not in index:
                index[check.id] = Finding(
                    check=check,
                    status=Status.OUTSTANDING,
                    detail="not assessed in this run",
                )

        self.findings = tuple(index[c.id] for c in STANDARD)
        self._index = index
        self.site = site
        self.mode = mode
        self.generated_at = generated_at
        self.tool_notes = tuple(tool_notes)
        self.blockers = tuple(blockers)

    # ── points ───────────────────────────────────────────────────────────────

    def _points(self, status: Status) -> int:
        return sum(f.check.points for f in self.findings if f.status is status)

    @property
    def earned(self) -> int:
        return self._points(Status.PASS)

    @property
    def failed_points(self) -> int:
        return self._points(Status.FAIL)

    @property
    def outstanding_points(self) -> int:
        return self._points(Status.OUTSTANDING)

    @property
    def not_applicable_points(self) -> int:
        return self._points(Status.NOT_APPLICABLE)

    @property
    def assessable(self) -> int:
        """Points in play. Only NOT_APPLICABLE leaves; OUTSTANDING stays."""
        return sum(f.check.points for f in self.findings) - self.not_applicable_points

    @property
    def score(self) -> float:
        """Points earned, on the standard's 100-point scale."""
        if self.assessable <= 0:
            return 0.0
        return round(100.0 * self.earned / self.assessable, 2)

    @property
    def ceiling(self) -> float:
        """The best this site could score if every outstanding check passed.

        The number that separates "not verified yet" from "cannot get there".
        A ceiling below the threshold is a definitive refusal; no amount of
        further verification rescues it.
        """
        if self.assessable <= 0:
            return 0.0
        return round(100.0 * (self.earned + self.outstanding_points) / self.assessable, 2)

    @property
    def coverage(self) -> float:
        """Share of assessable points the auditor was actually able to decide."""
        if self.assessable <= 0:
            return 0.0
        decided = self.earned + self.failed_points
        return round(100.0 * decided / self.assessable, 1)

    # ── selections ───────────────────────────────────────────────────────────

    def _select(self, status: Status) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.status is status)

    def failures(self) -> tuple[Finding, ...]:
        return self._select(Status.FAIL)

    def outstanding(self) -> tuple[Finding, ...]:
        return self._select(Status.OUTSTANDING)

    def passes(self) -> tuple[Finding, ...]:
        return self._select(Status.PASS)

    def not_applicable(self) -> tuple[Finding, ...]:
        return self._select(Status.NOT_APPLICABLE)

    def critical_failures(self) -> tuple[Finding, ...]:
        """Failures that block launch at any score."""
        return tuple(f for f in self.failures() if f.check.critical)

    def critical_unverified(self) -> tuple[Finding, ...]:
        """Critical checks nobody has decided. These block launch too."""
        return tuple(f for f in self.outstanding() if f.check.critical)

    def outstanding_by_level(self, level: Level) -> tuple[Finding, ...]:
        return tuple(f for f in self.outstanding() if f.check.level is level)

    def finding(self, check_id: str) -> Finding:
        return self._index[check_id]

    def by_category(self) -> dict[Category, CategoryScore]:
        scores: dict[Category, CategoryScore] = {}
        for category in category_order():
            in_cat = [f for f in self.findings if f.check.category is category]
            scores[category] = CategoryScore(
                category=category,
                weight=sum(f.check.points for f in in_cat),
                earned=sum(f.check.points for f in in_cat if f.status is Status.PASS),
                failed=sum(f.check.points for f in in_cat if f.status is Status.FAIL),
                outstanding=sum(f.check.points for f in in_cat if f.status is Status.OUTSTANDING),
                not_applicable=sum(
                    f.check.points for f in in_cat if f.status is Status.NOT_APPLICABLE
                ),
            )
        return scores

    # ── the decision ─────────────────────────────────────────────────────────

    @property
    def verdict(self) -> Verdict:
        if self.assessable <= 0:
            return Verdict.NO_LAUNCH
        if self.earned + self.failed_points == 0:
            # Nothing was decided, so the audit did not run. PENDING would claim
            # "no blockers found", and an auditor that assessed nothing has no
            # basis for that claim — it found nothing at all, blockers included.
            return Verdict.NO_LAUNCH
        if self.blockers:
            return Verdict.NO_LAUNCH
        if self.critical_failures():
            return Verdict.NO_LAUNCH
        if math.floor(self.ceiling * 100) < LAUNCH_THRESHOLD * 100:
            # Even a clean sweep of everything outstanding lands short.
            return Verdict.NO_LAUNCH
        if self.critical_unverified():
            return Verdict.PENDING
        if self.score < LAUNCH_THRESHOLD:
            return Verdict.PENDING
        return Verdict.LAUNCH

    @property
    def verdict_reason(self) -> str:
        """One line naming what decided the verdict, with the check ids."""
        if self.assessable <= 0:
            return "Nothing was assessed: no check in the standard had anything to judge."

        if self.earned + self.failed_points == 0:
            return (
                "No check was decided, so this is not an audit result. Point the auditor "
                "at a build directory or a live URL before reading the score."
            )

        if self.blockers:
            codes = ", ".join(b.code for b in self.blockers)
            return (
                f"{len(self.blockers)} critical failure(s) block launch regardless of the "
                f"{self.score:g}/100 score: {codes}."
            )

        criticals = self.critical_failures()
        if criticals:
            ids = ", ".join(f.check.id for f in criticals)
            return (
                f"Critical failure blocks launch regardless of score "
                f"({self.score:g}/100): {ids}."
            )

        if math.floor(self.ceiling * 100) < LAUNCH_THRESHOLD * 100:
            return (
                f"Score {self.score:g}/100 with a ceiling of {self.ceiling:g}: even if every "
                f"outstanding check passed, the site would still fall short of "
                f"{LAUNCH_THRESHOLD:g}. {self.failed_points} points are failing."
            )

        unverified = self.critical_unverified()
        if unverified:
            ids = ", ".join(f.check.id for f in unverified)
            return (
                f"Score {self.score:g}/100, no failures — but {len(unverified)} critical "
                f"check(s) are unverified and must be signed off before launch: {ids}."
            )

        if self.score < LAUNCH_THRESHOLD:
            return (
                f"Score {self.score:g}/100 is below {LAUNCH_THRESHOLD:g}. "
                f"{self.outstanding_points} points are outstanding and {self.failed_points} "
                f"are failing; resolving the outstanding checks could reach "
                f"{self.ceiling:g}."
            )

        return (
            f"Score {self.score:g}/100 with no critical failures and no unverified "
            f"critical checks: clear to launch."
        )

    # ── output ───────────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        cats = self.by_category()
        return {
            "standard": "WF-100 v1.0",
            "site": self.site,
            "mode": self.mode,
            "generated_at": self.generated_at,
            "score": self.score,
            "ceiling": self.ceiling,
            "coverage": self.coverage,
            "threshold": LAUNCH_THRESHOLD,
            "verdict": self.verdict.value,
            "verdict_reason": self.verdict_reason,
            "points": {
                "earned": self.earned,
                "failed": self.failed_points,
                "outstanding": self.outstanding_points,
                "not_applicable": self.not_applicable_points,
                "assessable": self.assessable,
            },
            "blockers": [b.to_dict() for b in self.blockers],
            "critical_failures": [f.check.id for f in self.critical_failures()],
            "critical_unverified": [f.check.id for f in self.critical_unverified()],
            "categories": {
                c.label: {
                    "weight": s.weight,
                    "earned": s.earned,
                    "failed": s.failed,
                    "outstanding": s.outstanding,
                    "not_applicable": s.not_applicable,
                    "percent": s.percent,
                }
                for c, s in cats.items()
            },
            "tool_notes": list(self.tool_notes),
            "findings": [f.to_dict() for f in self.findings],
        }


__all__ = [
    "AuditReport",
    "Blocker",
    "CategoryScore",
    "Finding",
    "Status",
    "Verdict",
]
