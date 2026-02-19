"""
Tests for AppVerifier — local and Docker verification (Task 6).
ALL subprocess and Docker calls are mocked — no real processes spawned.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from orchestrator.app_detector import AppProfile
from orchestrator.app_verifier import AppVerifier, VerifyReport


# ─────────────────────────────────────────────────────────────────────────────
# VerifyReport — dataclass
# ─────────────────────────────────────────────────────────────────────────────

def test_verify_report_fields():
    report = VerifyReport(
        local_install_ok=True,
        tests_passed=True,
        startup_ok=True,
        docker_build_ok=False,
        docker_run_ok=False,
        errors=["docker not available"],
    )
    assert report.local_install_ok is True
    assert report.tests_passed is True
    assert report.startup_ok is True
    assert report.docker_build_ok is False
    assert report.docker_run_ok is False
    assert report.errors == ["docker not available"]


def test_verify_report_success_property_true():
    """success property must be True when all local checks pass."""
    report = VerifyReport(
        local_install_ok=True,
        tests_passed=True,
        startup_ok=True,
    )
    assert report.success is True


def test_verify_report_success_property_false_if_tests_fail():
    report = VerifyReport(local_install_ok=True, tests_passed=False, startup_ok=True)
    assert report.success is False


# ─────────────────────────────────────────────────────────────────────────────
# AppVerifier.verify_local — mocked subprocess
# ─────────────────────────────────────────────────────────────────────────────

def test_verify_local_all_pass(tmp_path):
    """verify_local() must return VerifyReport with all local flags True when subprocess succeeds."""
    profile = AppProfile(app_type="fastapi", test_command="pytest", run_command="uvicorn src.main:app")
    (tmp_path / "requirements.txt").write_text("fastapi\n", encoding="utf-8")

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = ""
    mock_result.stderr = ""

    mock_proc = MagicMock()
    mock_proc.poll.return_value = None  # still running = startup ok

    verifier = AppVerifier()
    with patch("subprocess.run", return_value=mock_result) as mock_run, \
         patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
        report = verifier.verify_local(tmp_path, profile)

    assert report.local_install_ok is True
    assert report.tests_passed is True
    assert report.startup_ok is True
    assert report.errors == []


def test_verify_local_install_fails(tmp_path):
    """If pip install fails (returncode != 0), local_install_ok must be False."""
    profile = AppProfile(app_type="fastapi", test_command="pytest", run_command="")
    (tmp_path / "requirements.txt").write_text("fastapi\n", encoding="utf-8")

    mock_fail = MagicMock()
    mock_fail.returncode = 1
    mock_fail.stdout = ""
    mock_fail.stderr = "ERROR: could not install"

    verifier = AppVerifier()
    with patch("subprocess.run", return_value=mock_fail):
        report = verifier.verify_local(tmp_path, profile)

    assert report.local_install_ok is False
    assert len(report.errors) >= 1


def test_verify_local_tests_fail(tmp_path):
    """If pytest exits non-zero, tests_passed must be False."""
    profile = AppProfile(app_type="fastapi", test_command="pytest", run_command="")

    def side_effect(cmd, **kwargs):
        r = MagicMock()
        # pip install succeeds, pytest fails
        if "pip" in cmd[0] or (len(cmd) > 1 and "pip" in cmd[1]):
            r.returncode = 0
        else:
            r.returncode = 1
        r.stdout = ""
        r.stderr = "FAILED"
        return r

    verifier = AppVerifier()
    with patch("subprocess.run", side_effect=side_effect):
        report = verifier.verify_local(tmp_path, profile)

    assert report.tests_passed is False


def test_verify_local_no_run_command_skips_startup(tmp_path):
    """If profile.run_command is empty, startup check must be skipped (startup_ok=True by default)."""
    profile = AppProfile(app_type="library", test_command="pytest", run_command="")

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = ""
    mock_result.stderr = ""

    verifier = AppVerifier()
    with patch("subprocess.run", return_value=mock_result), \
         patch("subprocess.Popen") as mock_popen:
        report = verifier.verify_local(tmp_path, profile)

    mock_popen.assert_not_called()
    assert report.startup_ok is True


def test_verify_local_startup_fails_when_process_exits_immediately(tmp_path):
    """If the app process exits immediately (poll() returns non-None), startup_ok must be False."""
    profile = AppProfile(app_type="fastapi", test_command="pytest", run_command="uvicorn src.main:app")

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = ""
    mock_result.stderr = ""

    # Process exits immediately (returncode = 1)
    mock_proc = MagicMock()
    mock_proc.poll.return_value = 1  # non-None = exited
    mock_proc.returncode = 1

    verifier = AppVerifier()
    with patch("subprocess.run", return_value=mock_result), \
         patch("subprocess.Popen", return_value=mock_proc), \
         patch("time.sleep"):  # skip the 0.5s sleep
        report = verifier.verify_local(tmp_path, profile)

    assert report.startup_ok is False
    assert len(report.errors) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# AppVerifier.verify_docker — mocked Docker
# ─────────────────────────────────────────────────────────────────────────────

def test_verify_docker_skipped_when_docker_unavailable(tmp_path):
    """If 'docker info' fails, docker checks must be skipped gracefully."""
    profile = AppProfile(app_type="fastapi", requires_docker=False)

    mock_fail = MagicMock()
    mock_fail.returncode = 1
    mock_fail.stderr = "Cannot connect to the Docker daemon"

    verifier = AppVerifier()
    with patch("subprocess.run", return_value=mock_fail):
        report = verifier.verify_docker(tmp_path, profile)

    assert report.docker_build_ok is False
    assert report.docker_run_ok is False
    assert any("docker" in e.lower() for e in report.errors)


def test_verify_docker_build_and_run_success(tmp_path):
    """If docker is available and build+run succeed, both docker flags must be True."""
    profile = AppProfile(app_type="fastapi", requires_docker=True, run_command="uvicorn src.main:app")

    mock_ok = MagicMock()
    mock_ok.returncode = 0
    mock_ok.stdout = ""
    mock_ok.stderr = ""

    verifier = AppVerifier()
    with patch("subprocess.run", return_value=mock_ok):
        report = verifier.verify_docker(tmp_path, profile)

    assert report.docker_build_ok is True
    assert report.docker_run_ok is True


def test_verify_docker_build_fails(tmp_path):
    """If docker build fails, docker_build_ok must be False."""
    profile = AppProfile(app_type="fastapi", requires_docker=True)

    call_count = [0]
    def side_effect(cmd, **kwargs):
        r = MagicMock()
        call_count[0] += 1
        # First call = docker info (succeeds), second = docker build (fails)
        r.returncode = 0 if call_count[0] == 1 else 1
        r.stdout = ""
        r.stderr = "build failed" if call_count[0] > 1 else ""
        return r

    verifier = AppVerifier()
    with patch("subprocess.run", side_effect=side_effect):
        report = verifier.verify_docker(tmp_path, profile)

    assert report.docker_build_ok is False


def test_verify_returns_verify_report_instance(tmp_path):
    """verify_local() and verify_docker() must return VerifyReport instances."""
    profile = AppProfile(app_type="generic")

    mock_ok = MagicMock()
    mock_ok.returncode = 0
    mock_ok.stdout = ""
    mock_ok.stderr = ""

    verifier = AppVerifier()
    with patch("subprocess.run", return_value=mock_ok), \
         patch("subprocess.Popen", return_value=MagicMock(poll=MagicMock(return_value=None))):
        local_report = verifier.verify_local(tmp_path, profile)
        docker_report = verifier.verify_docker(tmp_path, profile)

    assert isinstance(local_report, VerifyReport)
    assert isinstance(docker_report, VerifyReport)
