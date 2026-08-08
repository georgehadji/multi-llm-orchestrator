"""Tests for the GitHub Actions SHA-pin resolver used by CicdEmitter (Phase 7, P-5)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from orchestrator.infrastructure.artifact_emitters._action_pins import resolve_action_sha

_MODULE = "orchestrator.infrastructure.artifact_emitters._action_pins"


@pytest.mark.unit
class TestResolveActionSha:
    def test_no_git_degrades_without_a_fake_sha(self):
        with patch(f"{_MODULE}.shutil.which", return_value=None):
            result = resolve_action_sha("actions/checkout", "v4")

        assert result.ok is False
        assert result.sha is None
        assert "git" in result.log

    def test_network_failure_degrades_without_a_fake_sha(self):
        with (
            patch(f"{_MODULE}.shutil.which", return_value="/usr/bin/git"),
            patch(
                f"{_MODULE}.subprocess.run",
                return_value=MagicMock(returncode=128, stdout="", stderr="unable to access"),
            ),
        ):
            result = resolve_action_sha("actions/checkout", "v4")

        assert result.ok is False
        assert result.sha is None

    def test_ref_not_found_degrades_without_a_fake_sha(self):
        with (
            patch(f"{_MODULE}.shutil.which", return_value="/usr/bin/git"),
            patch(
                f"{_MODULE}.subprocess.run",
                return_value=MagicMock(returncode=0, stdout="", stderr=""),
            ),
        ):
            result = resolve_action_sha("actions/checkout", "v4")

        assert result.ok is False
        assert result.sha is None

    def test_lightweight_tag_resolves_to_its_commit_sha(self):
        sha = "a" * 40
        stdout = f"{sha}\trefs/tags/v4\n"
        with (
            patch(f"{_MODULE}.shutil.which", return_value="/usr/bin/git"),
            patch(
                f"{_MODULE}.subprocess.run",
                return_value=MagicMock(returncode=0, stdout=stdout, stderr=""),
            ),
        ):
            result = resolve_action_sha("actions/checkout", "v4")

        assert result.ok is True
        assert result.sha == sha

    def test_annotated_tag_prefers_the_peeled_commit_sha(self):
        tag_object_sha = "b" * 40
        commit_sha = "c" * 40
        stdout = f"{tag_object_sha}\trefs/tags/v4\n{commit_sha}\trefs/tags/v4^{{}}\n"
        with (
            patch(f"{_MODULE}.shutil.which", return_value="/usr/bin/git"),
            patch(
                f"{_MODULE}.subprocess.run",
                return_value=MagicMock(returncode=0, stdout=stdout, stderr=""),
            ),
        ):
            result = resolve_action_sha("actions/checkout", "v4")

        assert result.ok is True
        assert result.sha == commit_sha

    def test_timeout_degrades_gracefully(self):
        import subprocess

        with (
            patch(f"{_MODULE}.shutil.which", return_value="/usr/bin/git"),
            patch(
                f"{_MODULE}.subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd="git ls-remote", timeout=15.0),
            ),
        ):
            result = resolve_action_sha("actions/checkout", "v4")

        assert result.ok is False
        assert result.sha is None
