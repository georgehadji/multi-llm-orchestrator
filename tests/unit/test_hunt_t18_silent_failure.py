"""
T18 (silent-failure sweep at scale) proof-of-defect and no-regression tests.

The wave AST-classified all 916 broad exception handlers in `orchestrator/`.
627 log, 67 re-raise; 222 are silent. Ranked by money / validation-gate /
persistence / config adjacency, the tier-1 and tier-2 sets were audited.

The headline result is a **negative** one: zero handlers anywhere in the
codebase are *fail-open* — none returns `True` or an affirmative result object
(`passed=True`, `healthy=True`, …) from a handler that neither logs nor
re-raises. That is the high-severity variant of this pattern, and the tiers
that came before (T6, T8, T9, T13, T16) appear to have eliminated it.
`scripts/check_silent_failure.py` keeps it at zero.

C1/C2 — `website_validator.py`'s `_check_rate_limiting` and `_check_auth_flow`
     each scan a bounded list of files and `continue` past any file they cannot
     read, with no log and no record. Unlike T8's C5 in the same file (the
     secret scanner, which reported *clean*), these fail **closed**: an
     unreadable file leaves `found=False` so the check reports failure. The
     defect is therefore a misreport rather than a security hole — the result
     says "no rate limiting found" / "no email verification detected" when the
     truth is "some files could not be read", sending a developer to add
     protection that may already exist. Fixed to log the read failure and say
     the scan was incomplete, matching T8 C5's established pattern in this file.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
GATE = REPO_ROOT / "scripts" / "check_silent_failure.py"


def _validator():
    from orchestrator.generators.website_validator import WebsiteQualityValidator

    return WebsiteQualityValidator()


def _unreadable_tree(tmp_path: Path, filename: str, body: str) -> Path:
    """Build an output dir whose one interesting file raises on read."""
    out = tmp_path / "site"
    out.mkdir()
    (out / filename).write_text(body, encoding="utf-8")
    # A server endpoint so _check_rate_limiting's applicability guard passes.
    api = out / "api"
    api.mkdir()
    (api / "route.js").write_text("export async function POST(req) {}\n", encoding="utf-8")
    return out


def _break_reads(monkeypatch, *, only_suffixes: tuple[str, ...]) -> None:
    """Make Path.read_text raise for the scanned source files."""
    real_read_text = Path.read_text

    def fake(self, *args, **kwargs):
        if self.suffix in only_suffixes:
            raise OSError("simulated unreadable file")
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fake)


# --- C1 -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_c1_rate_limit_scan_reports_unreadable_files(tmp_path, monkeypatch, caplog):
    out = _unreadable_tree(tmp_path, "contact.js", "export function contact() {}\n")
    _break_reads(monkeypatch, only_suffixes=(".js",))

    with caplog.at_level("WARNING"):
        result = await _validator()._check_rate_limiting(out)

    assert not result.passed
    assert "could not be read" in result.details, (
        "an unreadable file must not be reported the same as a file that was "
        f"read and found to lack rate limiting; got: {result.details!r}"
    )
    assert any("could not read" in r.message.lower() for r in caplog.records)


# --- C2 -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_c2_auth_flow_scan_reports_unreadable_files(tmp_path, monkeypatch, caplog):
    out = _unreadable_tree(tmp_path, "login.tsx", "export default function Login() {}\n")
    _break_reads(monkeypatch, only_suffixes=(".tsx",))

    with caplog.at_level("WARNING"):
        result = await _validator()._check_auth_flow(out)

    if not result.applicable:
        pytest.skip("no auth pages discovered in this fixture")

    assert not result.passed
    assert (
        "could not be read" in result.details
    ), f"unreadable auth file reported as 'no verification flow'; got: {result.details!r}"


# --- Gate ---------------------------------------------------------------


def _run_gate(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(GATE), *args],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )


def test_gate_reports_zero_fail_open_handlers():
    result = _run_gate()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "none report success on failure" in result.stdout


def test_gate_catches_both_fail_open_shapes_but_not_a_logged_one():
    probe = REPO_ROOT / "orchestrator" / "__t18_gate_probe.py"
    probe.write_text(
        "import logging\n\n\n"
        "class R:\n"
        "    def __init__(self, passed=False):\n"
        "        self.passed = passed\n\n\n"
        "def bare():\n"
        "    try:\n"
        "        return R()\n"
        "    except Exception:\n"
        "        return True\n\n\n"
        "def obj():\n"
        "    try:\n"
        "        return R()\n"
        "    except Exception:\n"
        "        return R(passed=True)\n\n\n"
        "def logged():\n"
        "    try:\n"
        "        return R()\n"
        "    except Exception:\n"
        "        logging.warning('failed')\n"
        "        return True\n",
        encoding="utf-8",
    )
    try:
        result = _run_gate()
        assert result.returncode == 1
        assert "return True" in result.stderr
        assert "R(passed=True)" in result.stderr
        # The handler that logs surfaces its failure and must not be flagged.
        assert result.stderr.count("__t18_gate_probe.py") == 2
    finally:
        probe.unlink(missing_ok=True)
