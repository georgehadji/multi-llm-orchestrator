"""The published standard must match the checks that actually run.

A document describing a hundred checks is worth exactly as much as its
agreement with the catalogue. This test is what makes `docs/WF100_STANDARD.md`
a description rather than a claim.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator.generators.wf100.render import catalogue_markdown

pytestmark = pytest.mark.unit

DOC = Path(__file__).resolve().parents[2] / "docs" / "WF100_STANDARD.md"


def test_the_published_standard_is_current():
    assert DOC.is_file(), f"{DOC} is missing"
    assert DOC.read_text(encoding="utf-8") == catalogue_markdown(), (
        "docs/WF100_STANDARD.md has drifted from the catalogue. Regenerate it:\n"
        "  python -m orchestrator website-audit --catalogue -o docs/WF100_STANDARD.md"
    )


def test_the_catalogue_lists_every_check():
    from orchestrator.generators.wf100.standard import STANDARD

    text = catalogue_markdown()
    for check in STANDARD:
        assert f"| {check.id} |" in text, check.id


def test_the_catalogue_states_the_launch_rule():
    assert "score >= 90" in catalogue_markdown()


def test_the_catalogue_is_honest_about_its_own_coverage():
    text = catalogue_markdown()
    assert "reported outstanding rather than assumed" in text
