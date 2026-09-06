"""
Hunt T13 — generators/, appbuilder/, design/, scaffold/, output/, quality/
============================================================================
Regression tests for the defects fixed in wave T13 of the backend-remainder
defect hunt (docs/hunts/BACKEND_REMAINDER_WAVES_PLAN.md).

C1: quality_control.py::TestRunner._run_security_checks silently skipped
    unreadable files with a bare `except Exception: pass`, reporting a
    false-clean "No security issues found" even when every file failed to
    read — the same false-clean-scan pattern already fixed in
    generators/website_validator.py (T8) and safety/generated_output_scanner.py
    (T9), never applied here. Confirmed dead today (TestLevel.SECURITY is
    never requested by either real caller), fixed anyway per this hunt's
    established "cheap, self-contained fix even in dead code" precedent.
C2: orchestrator/quality/quality_control.py was an unshimmed, byte-for-byte
    duplicate of the canonical orchestrator/quality_control.py (module-depth
    import comments aside) carrying the same C1 bug. Converted to a shim.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from orchestrator.quality_control import TestRunner

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_c1_security_scan_counts_and_reports_unreadable_files(tmp_path: Path) -> None:
    (tmp_path / "good.py").write_text("x = 1\n")
    (tmp_path / "bad.py").mkdir()  # a directory named *.py -> IsADirectoryError on read_text()

    runner = TestRunner()
    results = await runner._run_security_checks(tmp_path)

    assert results[0].passed is False
    assert "could not be read" in results[0].message


@pytest.mark.asyncio
async def test_c1_no_regression_clean_project_still_passes(tmp_path: Path) -> None:
    (tmp_path / "clean.py").write_text("x = 1\ny = 2\n")

    runner = TestRunner()
    results = await runner._run_security_checks(tmp_path)

    assert results[0].passed is True
    assert results[0].message == "No security issues found"


@pytest.mark.asyncio
async def test_c1_still_detects_a_real_secret(tmp_path: Path) -> None:
    (tmp_path / "leaky.py").write_text('api_key = "sk-super-secret-value"\n')

    runner = TestRunner()
    results = await runner._run_security_checks(tmp_path)

    assert results[0].passed is False
    assert "Hardcoded API key" in results[0].message


def test_c2_quality_quality_control_shim_matches_canonical() -> None:
    from orchestrator.quality.quality_control import QualityController as ViaShim
    from orchestrator.quality_control import QualityController as Canonical

    assert ViaShim is Canonical


def test_c2_no_circular_import() -> None:
    import orchestrator.quality.quality_control  # noqa: F401
