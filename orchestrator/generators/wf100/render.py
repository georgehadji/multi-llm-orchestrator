"""Rendering a WF-100 audit for the two people who read it.

The terminal view is for whoever is fixing the site: verdict first, then the
blocking problems, then the failures with their remedies.

The Markdown view is for the client. It leads with the same three numbers the
terminal does — earned, outstanding, ceiling — because a client who is shown a
single score will read it as complete, and it is not. The outstanding section
is not an appendix; it is the list of work still owed before anyone can say the
site is ready.
"""

from __future__ import annotations

from .report import AuditReport, Finding, Status, Verdict
from .standard import (
    CATEGORY_WEIGHTS,
    LAUNCH_THRESHOLD,
    STANDARD,
    Evidence,
    Level,
    category_order,
    checks_for,
)

_MARK = {
    Status.PASS: "PASS",
    Status.FAIL: "FAIL",
    Status.OUTSTANDING: "OPEN",
    Status.NOT_APPLICABLE: "n/a ",
}

_LEVEL_LABEL = {
    Level.AUTOMATED: "Level 1 — automated",
    Level.ASSISTED: "Level 2 — machine-assisted, needs confirmation",
    Level.HUMAN: "Level 3 — human review",
}


def _bar(percent: float, width: int = 20) -> str:
    filled = int(round(width * max(0.0, min(100.0, percent)) / 100.0))
    return "#" * filled + "." * (width - filled)


def render_text(report: AuditReport, verbose: bool = False) -> str:
    lines: list[str] = []
    add = lines.append

    add("=" * 72)
    add(f"WF-100 Website Quality Standard v1.0 — {report.verdict.headline}")
    add("=" * 72)
    add(f"Site      {report.site or '(unnamed)'}   [{report.mode} audit]")
    add(
        f"Score     {report.score:g}/100    "
        f"outstanding {report.outstanding_points}    "
        f"ceiling {report.ceiling:g}    "
        f"threshold {LAUNCH_THRESHOLD:g}"
    )
    add(f"Coverage  {report.coverage:g}% of assessable points were actually decided")
    add("")
    add(report.verdict_reason)
    add("")

    if report.blockers:
        add("-" * 72)
        add(f"CRITICAL FAILURES — {len(report.blockers)} (no launch at any score)")
        add("-" * 72)
        for blocker in report.blockers:
            add(f"  [{blocker.code}] {blocker.title}")
            add(f"      {blocker.detail}")
            for item in blocker.evidence:
                add(f"        - {item}")
            if blocker.remedy:
                add(f"      Fix: {blocker.remedy}")
        add("")

    add("-" * 72)
    add("CATEGORY BREAKDOWN")
    add("-" * 72)
    for category, score in report.by_category().items():
        na = f"  ({score.not_applicable} n/a)" if score.not_applicable else ""
        add(
            f"  {category.prefix}  {category.label:<32} "
            f"{score.earned:>2}/{score.assessable:<2} {_bar(score.percent)}"
            f"  open {score.outstanding}{na}"
        )
    add("")

    failures = report.failures()
    if failures:
        add("-" * 72)
        add(f"FAILING — {len(failures)} point(s) lost")
        add("-" * 72)
        for finding in failures:
            flag = " [CRITICAL]" if finding.check.critical else ""
            add(f"  {finding.check.id}{flag} {finding.check.title}")
            add(f"      {finding.detail}")
            for item in finding.evidence[:3]:
                add(f"        - {item}")
            add(f"      Fix: {finding.check.remedy}")
        add("")

    outstanding = report.outstanding()
    if outstanding:
        add("-" * 72)
        add(
            f"OUTSTANDING — {len(outstanding)} check(s) not verified, {report.outstanding_points} point(s)"
        )
        add("  These are not passes. Nothing has been established about them.")
        add("-" * 72)
        for level in Level:
            group = report.outstanding_by_level(level)
            if not group:
                continue
            add(f"  {_LEVEL_LABEL[level]} ({len(group)})")
            for finding in group:
                flag = " [CRITICAL]" if finding.check.critical else ""
                add(f"    {finding.check.id}{flag} {finding.check.title}")
                if verbose:
                    add(f"        {finding.detail}")
        add("")

    na = report.not_applicable()
    if na:
        add("-" * 72)
        add(f"NOT APPLICABLE — {len(na)} check(s) removed from the denominator")
        add("-" * 72)
        for finding in na:
            add(f"  {finding.check.id} {finding.check.title}")
            add(f"      {finding.detail}")
        add("")

    if report.tool_notes:
        add("-" * 72)
        add("WHAT THIS AUDIT COULD NOT SEE")
        add("-" * 72)
        for note in report.tool_notes:
            add(f"  - {note}")
        add("")

    return "\n".join(lines)


def _table_row(finding: Finding) -> str:
    remedy = finding.check.remedy if finding.status is Status.FAIL else ""
    detail = finding.detail.replace("|", "\\|")
    return (
        f"| {finding.check.id} | {finding.check.title} | {_MARK[finding.status].strip()} "
        f"| {detail} | {remedy} |"
    )


def render_markdown(report: AuditReport) -> str:
    lines: list[str] = []
    add = lines.append

    add("# Website Quality Report")
    add("")
    add(
        f"**Standard:** WF-100 v1.0 · **Site:** {report.site or '(unnamed)'} · "
        f"**Audited:** {report.generated_at or 'now'} ({report.mode})"
    )
    add("")
    add(f"## {report.verdict.headline}")
    add("")
    add(report.verdict_reason)
    add("")
    add("| | Points |")
    add("|---|---|")
    add(f"| **Verified and earned** | **{report.earned}** |")
    add(f"| Failing | {report.failed_points} |")
    add(f"| Outstanding — not yet verified | {report.outstanding_points} |")
    add(f"| Not applicable to this site | {report.not_applicable_points} |")
    add(f"| **Assessable total** | **{report.assessable}** |")
    add("")
    add(
        f"**Score {report.score:g}/100** against a launch threshold of {LAUNCH_THRESHOLD:g}. "
        f"If every outstanding check were verified and passed, the site would reach "
        f"**{report.ceiling:g}**."
    )
    add("")

    if report.blockers:
        add("## Critical failures — no launch")
        add("")
        add("These block launch regardless of the score.")
        add("")
        for blocker in report.blockers:
            add(f"### {blocker.title}")
            add("")
            add(blocker.detail)
            if blocker.evidence:
                add("")
                for item in blocker.evidence:
                    add(f"- `{item}`")
            if blocker.remedy:
                add("")
                add(f"**Fix:** {blocker.remedy}")
            add("")

    add("## Category scores")
    add("")
    add("| | Category | Earned | Weight | Outstanding |")
    add("|---|---|---|---|---|")
    for category in category_order():
        score = report.by_category()[category]
        add(
            f"| {category.prefix} | {category.label} | {score.earned} | "
            f"{score.assessable}{f' (of {score.weight})' if score.not_applicable else ''} | "
            f"{score.outstanding} |"
        )
    add("")

    failures = report.failures()
    if failures:
        add(f"## Failing checks ({len(failures)})")
        add("")
        add("| ID | Check | Status | What was found | How to fix it |")
        add("|---|---|---|---|---|")
        for finding in failures:
            add(_table_row(finding))
        add("")

    outstanding = report.outstanding()
    if outstanding:
        add(f"## Outstanding — {len(outstanding)} checks still to verify")
        add("")
        add(
            "These have **not** been assessed. They are not passes and they are not "
            "failures: nothing has been established about them, and each is still owed "
            "before this site can be signed off."
        )
        add("")
        for level in Level:
            group = report.outstanding_by_level(level)
            if not group:
                continue
            add(f"### {_LEVEL_LABEL[level]} — {len(group)} check(s)")
            add("")
            add("| ID | Check | What is needed |")
            add("|---|---|---|")
            for finding in group:
                critical = " **(critical)**" if finding.check.critical else ""
                add(
                    f"| {finding.check.id} | {finding.check.title}{critical} | "
                    f"{finding.detail.replace('|', chr(92) + '|')} |"
                )
            add("")

    na = report.not_applicable()
    if na:
        add(f"## Not applicable ({len(na)})")
        add("")
        add("Removed from the denominator, with the reason recorded.")
        add("")
        add("| ID | Check | Why it does not apply |")
        add("|---|---|---|")
        for finding in na:
            add(f"| {finding.check.id} | {finding.check.title} | {finding.detail} |")
        add("")

    passes = report.passes()
    if passes:
        add(f"<details><summary>Passing checks ({len(passes)})</summary>")
        add("")
        add("| ID | Check | What was measured |")
        add("|---|---|---|")
        for finding in passes:
            add(f"| {finding.check.id} | {finding.check.title} | {finding.detail} |")
        add("")
        add("</details>")
        add("")

    if report.tool_notes:
        add("## Limits of this audit")
        add("")
        for note in report.tool_notes:
            add(f"- {note}")
        add("")

    return "\n".join(lines)


_LEVEL_DECIDER = {
    Level.AUTOMATED: "the auditor, from the build or the response",
    Level.ASSISTED: "the auditor observes; a person or a live deployment confirms",
    Level.HUMAN: "a person, reading the site and knowing the business",
}

_EVIDENCE_LABEL = {
    Evidence.MARKUP: "HTML",
    Evidence.STYLES: "CSS",
    Evidence.SCRIPTS: "JS",
    Evidence.ASSETS: "files",
    Evidence.HTTP: "live response",
    Evidence.BROWSER: "browser",
    Evidence.RECORD: "business record",
    Evidence.FIELD_DATA: "field data",
    Evidence.HUMAN: "a person",
}


def catalogue_markdown() -> str:
    """The standard itself, rendered from the catalogue.

    Generated rather than written, so the published document cannot drift away
    from the checks that actually run. `tests/unit/test_wf100_docs.py` fails if
    the file on disk stops matching this output.
    """
    from .checks import implemented_ids

    implemented = implemented_ids()
    lines: list[str] = []
    add = lines.append

    add("# WF-100 — Website Factory Quality Standard v1.0")
    add("")
    add("<!-- Generated from orchestrator/generators/wf100/standard.py. Do not edit by")
    add("     hand: regenerate with `python -m orchestrator website-audit --catalogue`. -->")
    add("")
    add("A hundred checks, one point each, applied before a site launches.")
    add("")
    add(f"> **Launch rule:** score >= {LAUNCH_THRESHOLD:g} **and** zero critical failures.")
    add("> An *unverified* critical check blocks launch too — an unread smoke alarm is not")
    add("> an absence of fire.")
    add("")
    add("## Scoring")
    add("")
    add("| Status | Points | In the denominator? |")
    add("|---|---|---|")
    add("| Pass | earned | yes |")
    add("| Fail | none | yes |")
    add("| **Outstanding** — could not be decided | **none** | **yes** |")
    add("| Not applicable — nothing to judge | none | no, with the reason recorded |")
    add("")
    add("Outstanding is the status that matters. A check the auditor could not decide")
    add("earns nothing and stays in the denominator: it is a debt against the score,")
    add("visible until someone clears it. It is never upgraded to a pass because the")
    add("tool ran cleanly.")
    add("")
    add("## Verification levels")
    add("")
    add("| Level | Who decides | Checks |")
    add("|---|---|---|")
    for level in Level:
        count = sum(1 for c in STANDARD if c.level is level)
        add(f"| {_LEVEL_LABEL[level]} | {_LEVEL_DECIDER[level]} | {count} |")
    add("")
    add("## Coverage by this auditor")
    add("")
    add(
        f"{len(implemented)} of {len(STANDARD)} checks have an implementation. "
        f"The remaining {len(STANDARD) - len(implemented)} need a browser, field data from "
        "real users, or a person — and are reported outstanding rather than assumed."
    )
    add("")
    add("## The checks")
    add("")
    for category in category_order():
        add(f"### {category.prefix}. {category.label} — {CATEGORY_WEIGHTS[category]} points")
        add("")
        add("| ID | Check | Level | Needs | Automated | Critical |")
        add("|---|---|---|---|---|---|")
        for check in checks_for(category):
            needs = ", ".join(sorted(_EVIDENCE_LABEL[e] for e in check.requires))
            add(
                f"| {check.id} | {check.title} | {check.level.value} | {needs} "
                f"| {'yes' if check.id in implemented else 'no'} "
                f"| {'**yes**' if check.critical else ''} |"
            )
        add("")

    add("## Critical failures")
    add("")
    add("These block launch regardless of the score. Some map onto a check; the rest are")
    add("detected independently, because a blocker outvoted by ninety-nine passing points")
    add("would defeat the purpose of naming it.")
    add("")
    add("| Code | What it means |")
    add("|---|---|")
    for code, meaning in (
        ("MIXED_CONTENT", "Insecure subresources on a secure page"),
        ("EXPOSED_SECRET", "Credentials readable in the published build"),
        ("EXPOSED_PERSONAL_DATA", "Database or contact exports reachable from the site"),
        ("NO_CONTACT_ROUTE", "No phone link, email link or form anywhere"),
        ("ACCIDENTAL_NOINDEX", "Content pages tell search engines not to index them"),
        ("NO_VIEWPORT", "No viewport meta tag, so phones render the desktop layout"),
        ("BROKEN_HTTPS", "The site does not answer over HTTPS"),
    ):
        add(f"| `{code}` | {meaning} |")
    add("")
    add("Fabricated testimonials (H3), fabricated credentials (H4) and unreviewed medical")
    add("claims (H7) are critical too, and are the checks no tool will ever close: they")
    add("ask whether something is *true*.")
    add("")
    return "\n".join(lines)


def render_json(report: AuditReport, indent: int = 2) -> str:
    import json

    return json.dumps(report.to_dict(), indent=indent, ensure_ascii=False)


def exit_code_for(report: AuditReport) -> int:
    """0 launch, 1 refused, 2 pending verification.

    A pipeline gate should treat 1 and 2 differently: 1 is broken, 2 is unfinished.
    """
    return {Verdict.LAUNCH: 0, Verdict.NO_LAUNCH: 1, Verdict.PENDING: 2}[report.verdict]


__all__ = [
    "catalogue_markdown",
    "exit_code_for",
    "render_json",
    "render_markdown",
    "render_text",
]
