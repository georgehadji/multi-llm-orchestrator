"""
Hunt T22 — a WF-100 check asserting a verdict it had not established.

Every check declares the evidence it needs (`Check.requires`), and
`auditor.py` marks a check OUTSTANDING rather than running it when that
evidence is absent:

    if check.requires - evidence.available:  ->  OUTSTANDING

G7 (form abuse protection) declared only MARKUP but read `ev.scripts` as
well, unguarded:

    combined = ev.markup + ev.scripts

Captcha, honeypot and rate limiting are normally wired up in JavaScript. With
the scripts uncollected, `ev.scripts` is `""`, the gate does not fire because
SCRIPTS was never declared, and the check reports a definite

    "public forms with no captcha, honeypot or rate limiting —
     they will be found by bots"

about a site whose protection it simply could not see. Same site, same markup,
verdict flipped by whether the auditor happened to collect the JS.

Declaring SCRIPTS in `requires` would have been the wrong fix: a site with no
JavaScript at all has no SCRIPTS evidence, and G7 can still decide those from
markup. The fix narrows to the ambiguous case only — markup that loads scripts
the auditor does not have.
"""

from __future__ import annotations

import pytest

from orchestrator.generators.wf100.checks import implementation_for
from orchestrator.generators.wf100.evidence import Page, SiteEvidence
from orchestrator.generators.wf100.standard import get

pytestmark = pytest.mark.unit

_FORM_WITH_JS = (
    "<html><body>"
    '<form action="/contact"><input name="email"><button>Send</button></form>'
    '<script src="/assets/app.js"></script>'
    "</body></html>"
)


def _g7(ev: SiteEvidence):
    return implementation_for("G7")(ev)


def test_uncollected_scripts_do_not_become_a_security_finding():
    """The defect: a FAIL asserting absence the check could not establish."""
    finding = _g7(SiteEvidence(pages=(Page("index.html", _FORM_WITH_JS),)))

    assert finding.status.name == "OUTSTANDING", (
        "markup that loads uncollected scripts cannot decide form protection; "
        f"got {finding.status.name}: {finding.detail}"
    )
    assert "did not collect" in finding.detail


def test_protection_in_the_scripts_is_found_when_the_scripts_are_there():
    finding = _g7(
        SiteEvidence(pages=(Page("index.html", _FORM_WITH_JS),), scripts="grecaptcha.render()")
    )
    assert finding.status.name == "PASS"
    assert "captcha" in finding.detail


def test_a_real_hole_still_fails_when_the_scripts_were_collected():
    """The fix must not turn genuine findings into OUTSTANDING."""
    finding = _g7(
        SiteEvidence(pages=(Page("index.html", _FORM_WITH_JS),), scripts="console.log('hi')")
    )
    assert finding.status.name == "FAIL"


def test_a_site_with_no_javascript_is_still_decided_from_markup():
    """Why `requires = {MARKUP, SCRIPTS}` would have been the wrong fix.

    A static site has no SCRIPTS evidence, so declaring SCRIPTS would make the
    auditor skip G7 entirely — losing a verdict it can make perfectly well.
    """
    unprotected = _g7(SiteEvidence(pages=(Page("i.html", "<form><input name=e></form>"),)))
    protected = _g7(SiteEvidence(pages=(Page("i.html", '<form><input name="_honey"></form>'),)))

    assert unprotected.status.name == "FAIL"
    assert protected.status.name == "PASS"
    assert "SCRIPTS" not in {e.name for e in get("G7").requires}


def test_no_form_is_still_not_applicable():
    finding = _g7(SiteEvidence(pages=(Page("index.html", "<html><body><p>hi</p></body></html>"),)))
    assert finding.status.name == "NOT_APPLICABLE"
