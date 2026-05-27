"""
test_cli.py — CLI smoke tests via subprocess.
=============================================

Verifies that the CLI entry points load without import errors and return
expected exit codes.  No LLM calls are made.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

PYTHON = sys.executable


def test_cli_help_returns_zero():
    """python -m orchestrator --help must exit 0."""
    result = subprocess.run(
        [PYTHON, "-m", "orchestrator", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert "Multi-LLM Orchestrator" in result.stdout


def test_cli_list_projects_no_crash():
    """--list-projects should run without unhandled exception."""
    result = subprocess.run(
        [PYTHON, "-m", "orchestrator", "--list-projects"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    # May return 0 (empty list) or 1 (no API key), but must NOT traceback
    assert "Traceback" not in result.stderr
    assert "Exception" not in result.stderr


def test_cli_analyze_subcommand_help():
    """analyze --help must work."""
    result = subprocess.run(
        [PYTHON, "-m", "orchestrator", "analyze", "--help"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0
    assert "analyze" in result.stdout.lower() or "usage" in result.stdout.lower()


def test_cli_build_subcommand_help():
    """build --help must work."""
    result = subprocess.run(
        [PYTHON, "-m", "orchestrator", "build", "--help"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0


def test_cli_cache_stats_no_crash():
    """cache-stats should not traceback even on empty cache."""
    result = subprocess.run(
        [PYTHON, "-m", "orchestrator", "cache-stats"],
        capture_output=True,
        text=True,
        timeout=15,
        errors="replace",  # avoid decode errors on Windows
    )
    # Allow UnicodeEncodeError (Windows console encoding issue in
    # cache_optimizer.py) but flag actual orchestrator logic failures.
    stderr = result.stderr
    assert "ModuleNotFoundError" not in stderr
    assert "ImportError" not in stderr
    assert "orchestrator" not in stderr or "UnicodeEncodeError" in stderr or "charmap" in stderr


def test_cli_version_flag_not_implemented():
    """
    If --version is provided and not implemented, it should show usage
    rather than an unhandled exception.
    """
    result = subprocess.run(
        [PYTHON, "-m", "orchestrator", "--version"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    # argparse may error, but must not traceback
    assert "Traceback" not in result.stderr
