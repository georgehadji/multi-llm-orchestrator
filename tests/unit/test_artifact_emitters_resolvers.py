"""Tests for the lock-file and digest resolvers used by DockerEmitter (Phase 7, P-4)."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from orchestrator.infrastructure.artifact_emitters._digest import resolve_digest
from orchestrator.infrastructure.artifact_emitters._lockfile import resolve_lockfile

_MODULE = "orchestrator.infrastructure.artifact_emitters._lockfile"


@pytest.mark.unit
class TestResolveLockfile:
    def test_no_pyproject_returns_not_ok(self, tmp_path):
        result = resolve_lockfile(tmp_path)
        assert result.ok is False
        assert result.tool == "none"

    def test_uv_lock_success_reads_components(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\n')

        def _fake_run(cmd, **kwargs):
            (tmp_path / "uv.lock").write_text(
                '[[package]]\nname = "requests"\nversion = "2.32.0"\n'
            )
            return MagicMock(returncode=0)

        with (
            patch(
                f"{_MODULE}.shutil.which",
                side_effect=lambda tool: "/usr/bin/uv" if tool == "uv" else None,
            ),
            patch(f"{_MODULE}.subprocess.run", side_effect=_fake_run),
        ):
            result = resolve_lockfile(tmp_path)

        assert result.ok is True
        assert result.filename == "uv.lock"
        assert result.components == (("requests", "2.32.0"),)

    def test_uv_absent_falls_back_to_pip_compile(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\n')

        def _fake_run(cmd, **kwargs):
            (tmp_path / "requirements.txt").write_text(
                "requests==2.32.0 \\\n    --hash=sha256:deadbeef\n"
            )
            return MagicMock(returncode=0, stderr="")

        with (
            patch(
                f"{_MODULE}.shutil.which",
                side_effect=lambda tool: "/usr/bin/pip-compile" if tool == "pip-compile" else None,
            ),
            patch(f"{_MODULE}.subprocess.run", side_effect=_fake_run),
        ):
            result = resolve_lockfile(tmp_path)

        assert result.ok is True
        assert result.filename == "requirements.txt"
        assert result.components == (("requests", "2.32.0"),)

    def test_neither_tool_available_degrades_without_inventing_a_lockfile(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\n')

        with patch(f"{_MODULE}.shutil.which", return_value=None):
            result = resolve_lockfile(tmp_path)

        assert result.ok is False
        assert result.filename is None
        assert "PATH" in result.log

    def test_subprocess_timeout_degrades_gracefully(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\n')

        with (
            patch(
                f"{_MODULE}.shutil.which",
                side_effect=lambda tool: "/usr/bin/uv" if tool == "uv" else None,
            ),
            patch(
                f"{_MODULE}.subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd="uv lock", timeout=60.0),
            ),
        ):
            result = resolve_lockfile(tmp_path)

        assert result.ok is False


_DIGEST_MODULE = "orchestrator.infrastructure.artifact_emitters._digest"


@pytest.mark.unit
class TestResolveDigest:
    def test_no_docker_cli_degrades_without_a_fake_digest(self):
        with patch(f"{_DIGEST_MODULE}.shutil.which", return_value=None):
            result = resolve_digest("python:3.12-slim")

        assert result.ok is False
        assert result.digest is None
        assert "docker CLI" in result.log

    def test_offline_manifest_inspect_failure_degrades_without_a_fake_digest(self):
        with (
            patch(f"{_DIGEST_MODULE}.shutil.which", return_value="/usr/bin/docker"),
            patch(
                f"{_DIGEST_MODULE}.subprocess.run",
                return_value=MagicMock(returncode=1, stderr="no such host"),
            ),
        ):
            result = resolve_digest("python:3.12-slim")

        assert result.ok is False
        assert result.digest is None

    def test_successful_resolution_extracts_digest(self):
        payload = '{"Descriptor": {"digest": "sha256:abc123"}}'
        with (
            patch(f"{_DIGEST_MODULE}.shutil.which", return_value="/usr/bin/docker"),
            patch(
                f"{_DIGEST_MODULE}.subprocess.run",
                return_value=MagicMock(returncode=0, stdout=payload),
            ),
        ):
            result = resolve_digest("python:3.12-slim")

        assert result.ok is True
        assert result.digest == "sha256:abc123"

    def test_unparseable_output_degrades_without_a_fake_digest(self):
        with (
            patch(f"{_DIGEST_MODULE}.shutil.which", return_value="/usr/bin/docker"),
            patch(
                f"{_DIGEST_MODULE}.subprocess.run",
                return_value=MagicMock(returncode=0, stdout="not json"),
            ),
        ):
            result = resolve_digest("python:3.12-slim")

        assert result.ok is False
        assert result.digest is None
