"""The auditor: evidence in, report out.

The loop is deliberately dull, because the interesting decision is made before
any check runs:

    if check.requires - evidence.available:   ->  OUTSTANDING

That line is the standard's Level 1/2/3 split enforced mechanically. An author
cannot forget to be careful about a check they implemented against evidence the
run does not hold, because the check is never called. And a check with no
implementation is outstanding too — an unimplemented check is an unanswered
question, not a satisfied one.

A check that raises is reported OUTSTANDING with the error, never PASS and
never FAIL. A crashing check has produced no evidence about the site, and
inferring either verdict from a stack trace would be inventing a result.
"""

from __future__ import annotations

import datetime as _dt

from .checks import implementation_for
from .detectors import run_detectors
from .evidence import BusinessRecord, SiteEvidence
from .report import AuditReport, Finding, Status
from .standard import STANDARD, Evidence, Level

# What a caller has to go and get for each kind of missing evidence. These
# strings end up in the client report as the outstanding-work list, so they are
# written as instructions to a person, not as internal diagnostics.
_HOW_TO_SUPPLY = {
    Evidence.HTTP: "audit the deployed URL instead of the build directory",
    Evidence.BROWSER: "run the page in a browser and watch the console and network tab",
    Evidence.FIELD_DATA: "export field Core Web Vitals from CrUX or your RUM tool",
    Evidence.RECORD: "supply the client's business record (--record)",
    Evidence.HUMAN: "have a person review this and record the outcome",
    Evidence.MARKUP: "point the auditor at pages that contain HTML",
    Evidence.STYLES: "make the stylesheet reachable from the audited source",
    Evidence.SCRIPTS: "make the site's JavaScript reachable from the audited source",
    Evidence.ASSETS: "audit a complete build, including its static files",
}


def audit(evidence: SiteEvidence) -> AuditReport:
    """Run WF-100 against collected evidence."""
    available = evidence.available
    findings: list[Finding] = []

    for check in STANDARD:
        missing = check.requires - available
        if missing:
            findings.append(
                Finding(
                    check=check,
                    status=Status.OUTSTANDING,
                    detail=_missing_evidence_detail(check.level, missing),
                )
            )
            continue

        implementation = implementation_for(check.id)
        if implementation is None:
            findings.append(
                Finding(
                    check=check,
                    status=Status.OUTSTANDING,
                    detail=(
                        "this auditor has no implementation for this check — "
                        "it must be verified by hand"
                    ),
                )
            )
            continue

        try:
            findings.append(implementation(evidence))
        except Exception as exc:  # noqa: BLE001 - a crash is not a verdict
            findings.append(
                Finding(
                    check=check,
                    status=Status.OUTSTANDING,
                    detail=(
                        f"the check raised {type(exc).__name__}: {exc}. It produced no "
                        "evidence about the site, so it is unresolved, not failed."
                    ),
                )
            )

    return AuditReport(
        findings=findings,
        site=evidence.source,
        mode=evidence.mode,
        generated_at=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        tool_notes=_tool_notes(evidence),
        blockers=run_detectors(evidence),
    )


def _missing_evidence_detail(level: Level, missing: frozenset[Evidence]) -> str:
    wanted = sorted(missing, key=lambda e: e.value)
    steps = "; ".join(_HOW_TO_SUPPLY.get(kind, kind.value) for kind in wanted)
    if level is Level.HUMAN:
        return f"human verification required — {steps}"
    return "not decidable from the evidence collected: " + steps


def _tool_notes(evidence: SiteEvidence) -> tuple[str, ...]:
    """What this run could not see, stated once at the top of the report."""
    notes = [
        "Core Web Vitals (B1-B3) are field metrics. A lab score is a different "
        "measurement and is never substituted for them here.",
        "Dependency CVE scanning is outside this tool's scope; G6 checks pinning and "
        "subresource integrity only.",
    ]
    if evidence.mode == "directory":
        notes.append(
            "Audited a build directory. Anything about the served response — TLS, "
            "security headers, compression, caching, redirects — is outstanding by "
            "construction: a _headers file states an intention, not a served header."
        )
    if evidence.record is None:
        notes.append(
            "No business record was supplied, so the auditor could not check the "
            "published name, address and phone against what the business actually is."
        )
    return tuple(notes)


def audit_directory(path: str, record: BusinessRecord | None = None) -> AuditReport:
    return audit(SiteEvidence.from_directory(path, record=record))


def audit_url(url: str, record: BusinessRecord | None = None, **kwargs: object) -> AuditReport:
    return audit(SiteEvidence.from_url(url, record=record, **kwargs))  # type: ignore[arg-type]


__all__ = ["audit", "audit_directory", "audit_url"]
