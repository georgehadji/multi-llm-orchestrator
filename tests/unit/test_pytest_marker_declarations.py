"""
Guard: every pytest marker used in tests/ must be declared in pyproject.toml.

Why this exists. `--strict-markers` is enabled, so a marker that is used but
never declared is not a warning — it is a COLLECTION ERROR that aborts the
entire pytest run before a single test body executes. On 2026-07-22 commit
9250edb added tests/regression/test_wbs1_verification_regression.py with
`pytestmark = [pytest.mark.regression, pytest.mark.wbs1, ...]` without
declaring either marker. The CI Test job went red on that collection error and
stayed red; those 44 tests never ran once, and neither did the rest of the
suite's reporting. It was found ~6 weeks later.

This test turns that class of outage into a single obvious local failure.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

# Markers pytest itself provides; they never need declaring.
_BUILTIN_MARKERS = {
    "parametrize",
    "skip",
    "skipif",
    "xfail",
    "usefixtures",
    "filterwarnings",
    "tryfirst",
    "trylast",
    "timeout",
    "asyncio",
}

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _declared_markers() -> set[str]:
    if sys.version_info >= (3, 11):
        import tomllib
    else:  # pragma: no cover - CI and dev both run >= 3.11
        pytest.skip("tomllib requires Python 3.11+")

    with (_REPO_ROOT / "pyproject.toml").open("rb") as handle:
        config = tomllib.load(handle)
    markers = config["tool"]["pytest"]["ini_options"]["markers"]
    return {entry.split(":", 1)[0].strip() for entry in markers}


def _used_markers() -> set[str]:
    """Every `pytest.mark.<name>` written anywhere under tests/.

    Uses a text scan rather than pytest introspection on purpose: an undeclared
    marker breaks collection, so anything that needs collection to succeed
    cannot diagnose this failure.
    """
    result = subprocess.run(
        ["grep", "-rhoE", r"pytest\.mark\.[a-zA-Z_0-9]+", str(_REPO_ROOT / "tests")],
        capture_output=True,
        text=True,
        check=False,
    )
    return {line.rsplit(".", 1)[-1] for line in result.stdout.splitlines() if line.strip()}


@pytest.mark.unit
def test_every_used_marker_is_declared():
    undeclared = sorted(_used_markers() - _declared_markers() - _BUILTIN_MARKERS)
    assert not undeclared, (
        f"Undeclared pytest marker(s): {undeclared}. With --strict-markers this "
        f"aborts the whole test run at collection. Declare them under "
        f"[tool.pytest.ini_options] markers in pyproject.toml."
    )


@pytest.mark.unit
def test_marker_scan_finds_known_markers():
    """Sanity-check the scanner itself, so a broken regex can't make the guard vacuous."""
    used = _used_markers()
    assert (
        "unit" in used and "integration" in used
    ), f"marker scan looks broken: {sorted(used)[:10]}"


@pytest.mark.unit
def test_declared_markers_are_wellformed():
    for marker in _declared_markers():
        assert re.fullmatch(r"[a-z_][a-z_0-9]*", marker), f"malformed marker name: {marker!r}"
