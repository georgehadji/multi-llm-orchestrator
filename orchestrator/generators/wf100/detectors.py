"""Critical-failure detectors: no launch, at any score.

The standard lists critical failures apart from the hundred points, and so do
these. A detector answers one question — is this specific, disqualifying thing
present? — and reports nothing when the answer is no. Silence here is not a
pass; it means this particular detector found nothing, which is all it is
entitled to say.

Only failures a static read or a live response can *prove* are detected. The
rest of the standard's critical list — fabricated credentials, a GDPR review,
whether a production error occurs at runtime — belongs to the human and
browser checks in the catalogue, and the report carries them there as
unverified rather than pretending a detector settled them.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from .evidence import SiteEvidence
from .report import Blocker

Detector = Callable[[SiteEvidence], Blocker | None]

_DETECTORS: list[Detector] = []


def detector(fn: Detector) -> Detector:
    _DETECTORS.append(fn)
    return fn


def run_detectors(ev: SiteEvidence) -> tuple[Blocker, ...]:
    return tuple(b for b in (d(ev) for d in _DETECTORS) if b is not None)


_INSECURE_SUBRESOURCE = re.compile(
    r"""<(?:script|img|iframe|source|video|audio|embed)\b[^>]*\b(?:src|data)\s*=\s*["'](http://[^"']+)"""
    r"""|<link\b[^>]*\bhref\s*=\s*["'](http://[^"']+)""",
    re.I,
)


@detector
def _mixed_content(ev: SiteEvidence) -> Blocker | None:
    urls = [a or b for a, b in _INSECURE_SUBRESOURCE.findall(ev.markup)]
    # localhost over http is a development reference, not shipped mixed content,
    # but it has no business in a production build either — report it as such.
    if not urls:
        return None
    return Blocker(
        code="MIXED_CONTENT",
        title="Mixed content: insecure resources on a secure page",
        detail=(
            f"{len(urls)} subresource(s) load over plain http://. Browsers block or "
            "downgrade these, and the padlock is lost."
        ),
        evidence=tuple(sorted(set(urls))[:5]),
        remedy="Serve every subresource over https:// or self-host it.",
    )


_SECRET_SHAPES = (
    (r"\bsk-[A-Za-z0-9]{20,}", "OpenAI-style secret key"),
    (r"\bAKIA[0-9A-Z]{16}\b", "AWS access key id"),
    (r"\bAIza[0-9A-Za-z_\-]{35}\b", "Google API key"),
    (r"\bgh[pousr]_[A-Za-z0-9]{30,}", "GitHub token"),
    (r"\bxox[baprs]-[A-Za-z0-9-]{10,}", "Slack token"),
    (r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----", "private key"),
    (r"""(?:password|passwd|secret)\s*[:=]\s*["'][^"'\s]{8,}["']""", "hard-coded password"),
)


@detector
def _exposed_secrets(ev: SiteEvidence) -> Blocker | None:
    haystack = "\n".join([ev.markup, ev.scripts, *ev.files.values()])
    found = []
    for pattern, label in _SECRET_SHAPES:
        for match in re.findall(pattern, haystack, re.I):
            text = match if isinstance(match, str) else match[0]
            found.append(f"{label}: {text[:12]}...")
    if not found:
        return None
    return Blocker(
        code="EXPOSED_SECRET",
        title="Credentials shipped in the published build",
        detail=(
            f"{len(found)} credential-shaped value(s) are readable by anyone who views "
            "source. Treat every one as compromised."
        ),
        evidence=tuple(sorted(set(found))[:5]),
        remedy=(
            "Rotate the credentials first, then move them server-side. "
            "Never store a password or key in code."
        ),
    )


_DUMP_SUFFIXES = (".sql", ".bak", ".dump", ".sqlite", ".db", ".csv", ".xlsx", ".pst")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]{2,}")


@detector
def _exposed_personal_data(ev: SiteEvidence) -> Blocker | None:
    dumps = [p for p in ev.assets if p.lower().endswith(_DUMP_SUFFIXES)]
    bulk_emails = []
    for path, content in ev.files.items():
        addresses = set(_EMAIL.findall(content))
        if len(addresses) > 10:
            bulk_emails.append(f"{path} ({len(addresses)} addresses)")
    if not dumps and not bulk_emails:
        return None
    return Blocker(
        code="EXPOSED_PERSONAL_DATA",
        title="Personal data reachable from the published site",
        detail=(
            f"{len(dumps)} database or spreadsheet export(s) and {len(bulk_emails)} file(s) "
            "holding bulk contact details are part of the build."
        ),
        evidence=tuple((dumps + bulk_emails)[:5]),
        remedy="Remove them from the build and treat the exposure as a reportable incident.",
    )


@detector
def _no_contact_route(ev: SiteEvidence) -> Blocker | None:
    if not ev.pages:
        return None
    markup = ev.markup.lower()
    if "tel:" in markup or "mailto:" in markup or re.search(r"<form\b", markup):
        return None
    return Blocker(
        code="NO_CONTACT_ROUTE",
        title="No way to contact the business",
        detail="The site publishes no phone link, no email link and no form.",
        remedy="Add a tel: link and a contact form before launch.",
    )


@detector
def _accidental_noindex(ev: SiteEvidence) -> Blocker | None:
    blocked = [
        p.path
        for p in ev.pages
        # A 404 SHOULD be noindex. Flagging correct practice as an accidental
        # noindex made a well-built site unlaunchable.
        if not p.is_error_page
        and re.search(
            r"""<meta\b[^>]*\bname\s*=\s*["']?(?:robots|googlebot)["']?[^>]*\bcontent\s*="""
            r"""\s*["'][^"']*noindex""",
            p.markup,
            re.I,
        )
    ]
    if ev.http is not None:
        blocked += [
            url
            for url, r in ev.http.responses.items()
            if "noindex" in r.header("x-robots-tag").lower()
        ]
    if not blocked:
        return None
    return Blocker(
        code="ACCIDENTAL_NOINDEX",
        title="Pages are set to noindex",
        detail=(
            f"{len(blocked)} page(s) instruct search engines not to index them. A site "
            "nobody can find is a site that was not launched."
        ),
        evidence=tuple(blocked[:5]),
        remedy="Remove the staging noindex from the production build.",
    )


@detector
def _no_viewport(ev: SiteEvidence) -> Blocker | None:
    missing = [
        p.path
        for p in ev.pages
        if not re.search(r"""<meta\b[^>]*\bname\s*=\s*["']?viewport""", p.markup, re.I)
    ]
    if not missing:
        return None
    return Blocker(
        code="NO_VIEWPORT",
        title="Mobile layout is broken: no viewport meta tag",
        detail=(
            f"{len(missing)} page(s) have no viewport meta tag, so phones render them at "
            "desktop width and zoom out. Every tap target becomes unusable."
        ),
        evidence=tuple(missing[:5]),
        remedy='Add <meta name="viewport" content="width=device-width, initial-scale=1">.',
    )


@detector
def _broken_https(ev: SiteEvidence) -> Blocker | None:
    if ev.http is None:
        return None
    entry = ev.http.entry()
    if entry is None:
        return None
    url = entry.final_url or entry.url
    if url.lower().startswith("https://") and entry.status < 400:
        return None
    if entry.status == 0:
        return Blocker(
            code="BROKEN_HTTPS",
            title="The site did not answer over HTTPS",
            detail=f"The entry point {entry.url} could not be reached: {entry.text[:120]}",
            remedy="Fix the certificate or the origin before anything else is measured.",
        )
    if not url.lower().startswith("https://"):
        return Blocker(
            code="BROKEN_HTTPS",
            title="The site is served over plain HTTP",
            detail=f"{entry.url} resolved to {url}, which is not encrypted.",
            remedy="Install a certificate and redirect all HTTP traffic to HTTPS.",
        )
    return None


__all__ = ["Detector", "detector", "run_detectors"]
