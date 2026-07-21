"""
Tests for orchestrator/output/formatter.py — generated output must pass
black + ruff after the formatting pass.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

from orchestrator.output.formatter import FormatReport, format_output_dir

# Deliberately messy but syntactically valid Python: bad spacing, bad quotes,
# unsorted/unused imports, missing final newline — all auto-fixable.
MESSY_PY = (
    "import os\n"
    "import sys\n"
    "x={'a':1,'b':2}\n"
    "def  foo( a,b ):\n"
    "    return a+b\n"
    "print(foo(1,2))"
)


def _black_clean(path: Path) -> bool:
    proc = subprocess.run(
        [sys.executable, "-m", "black", "--check", "-q", str(path)],
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0


def _ruff_clean(path: Path) -> bool:
    proc = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--isolated", "-q", str(path)],
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0


@pytest.mark.unit
class TestFormatOutputDir:
    def test_returns_report(self, tmp_path: Path):
        report = format_output_dir(tmp_path)
        assert isinstance(report, FormatReport)

    def test_missing_dir_is_error_not_crash(self, tmp_path: Path):
        report = format_output_dir(tmp_path / "nope")
        assert not report.is_clean
        assert report.errors

    def test_messy_python_becomes_black_clean(self, tmp_path: Path):
        src = tmp_path / "app.py"
        src.write_text(MESSY_PY, encoding="utf-8")
        assert not _black_clean(src)  # precondition: starts messy

        report = format_output_dir(tmp_path, format_web=False)

        assert report.python_files == 1
        assert "black" in report.tools_used
        assert _black_clean(src), "file should be black-clean after formatting"

    def test_messy_python_becomes_ruff_clean(self, tmp_path: Path):
        # Unsorted + unused imports are ruff-fixable (I + F401 via --fix).
        src = tmp_path / "mod.py"
        src.write_text(MESSY_PY, encoding="utf-8")

        report = format_output_dir(tmp_path, format_web=False)

        assert "ruff" in report.tools_used
        assert _ruff_clean(src), f"ruff issues remain: {report.remaining_issues}"
        assert report.is_clean

    def test_nested_dirs_and_exclusions(self, tmp_path: Path):
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "a.py").write_text(MESSY_PY, encoding="utf-8")
        # Excluded dir must be ignored.
        (tmp_path / "node_modules").mkdir()
        (tmp_path / "node_modules" / "lib.py").write_text(MESSY_PY, encoding="utf-8")

        report = format_output_dir(tmp_path, format_web=False)

        assert report.python_files == 1
        assert _black_clean(tmp_path / "src" / "a.py")
        # The excluded file must be left untouched (still messy).
        assert not _black_clean(tmp_path / "node_modules" / "lib.py")

    def test_unfixable_lint_is_reported(self, tmp_path: Path):
        # F821 undefined name is NOT auto-fixable → must surface as an issue.
        src = tmp_path / "bad.py"
        src.write_text("y = undefined_name + 1\n", encoding="utf-8")

        report = format_output_dir(tmp_path, format_web=False)

        assert not report.is_clean
        assert any("ruff" in issue for issue in report.remaining_issues)


@pytest.mark.integration
class TestOrganizerFormattingIntegration:
    """OutputOrganizer must format generated code as part of its pipeline."""

    @pytest.mark.asyncio
    async def test_organizer_formats_source(self, tmp_path: Path):
        from orchestrator.output_organizer import OutputOrganizer

        src_dir = tmp_path / "src"
        src_dir.mkdir()
        messy = src_dir / "app.py"
        messy.write_text(MESSY_PY, encoding="utf-8")
        assert not _black_clean(messy)

        organizer = OutputOrganizer(
            output_dir=tmp_path,
            auto_generate_tests=False,
            run_tests=False,
            fix_tests=False,
            format_code=True,
        )
        report = await organizer.organize_project()

        assert report.format_report is not None
        assert _black_clean(messy), "organizer should leave source black-clean"
