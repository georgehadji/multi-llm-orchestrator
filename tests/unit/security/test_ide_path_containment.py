"""SEC-005 — user-controlled paths must stay inside the session output root.

Two layers are covered:

* T0 — `orchestrator.safety.secure_execution` shipped without any test file at
  all. The fixes below depend on it, so its containment behaviour is pinned
  here first.
* T2 — the legacy IDE server built ``Path.cwd() / "ide_outputs" / session_id /
  file_path`` at three sites with no containment check. Both ``session_id`` and
  ``file_path`` arrive over the WebSocket, so both are untrusted.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator.safety.secure_execution import (
    PathTraversalError,
    SecurePath,
    sanitize_path,
)

pytestmark = pytest.mark.unit


ESCAPES = [
    "../secrets.txt",
    "../../etc/passwd",
    "..\\..\\windows\\win.ini",
    "a/../../../outside.txt",
    "/etc/passwd",
    "C:\\Windows\\win.ini",
    "with\x00null.txt",
]


class TestSecurePathContainment:
    """T0 — pin the guard the SEC-005 fix relies on."""

    @pytest.mark.parametrize("candidate", ESCAPES)
    def test_escapes_are_blocked(self, tmp_path: Path, candidate: str) -> None:
        with pytest.raises(PathTraversalError):
            SecurePath(tmp_path, candidate)

    @pytest.mark.parametrize("candidate", ["notes.txt", "src/main.py", "deep/nested/file.json", ""])
    def test_legitimate_paths_resolve_inside_root(self, tmp_path: Path, candidate: str) -> None:
        resolved = SecurePath(tmp_path, candidate).resolved
        assert resolved.is_relative_to(tmp_path.resolve())

    def test_sanitize_path_is_the_same_check(self, tmp_path: Path) -> None:
        assert sanitize_path(tmp_path, "ok.txt").is_relative_to(tmp_path.resolve())
        with pytest.raises(PathTraversalError):
            sanitize_path(tmp_path, "../ok.txt")

    def test_symlink_escape_is_blocked(self, tmp_path: Path) -> None:
        root = tmp_path / "root"
        root.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.txt").write_text("secret", encoding="utf-8")
        try:
            (root / "link").symlink_to(outside, target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("symlink creation not permitted on this platform/account")

        # resolve() follows the link, so containment must reject it.
        with pytest.raises(PathTraversalError):
            SecurePath(root, "link/secret.txt")


class TestSessionOutputDir:
    """T2 — `session_id` arrives over the socket and is not trusted."""

    @pytest.mark.parametrize("bad_session", ["../escape", "..\\escape", "/abs", "a/../.."])
    def test_traversing_session_id_is_blocked(
        self, tmp_path: Path, monkeypatch, bad_session: str
    ) -> None:
        from orchestrator.ide_backend.ide_orchestrator_server import session_output_dir

        monkeypatch.chdir(tmp_path)
        with pytest.raises(PathTraversalError):
            session_output_dir(bad_session)

    def test_normal_session_id_resolves_under_ide_outputs(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        from orchestrator.ide_backend.ide_orchestrator_server import session_output_dir

        monkeypatch.chdir(tmp_path)
        resolved = session_output_dir("session-abc123")
        assert resolved.is_relative_to((tmp_path / "ide_outputs").resolve())
        assert resolved.name == "session-abc123"


class TestSessionFilePath:
    """T2 — this is the exact SEC-005 read primitive."""

    @pytest.mark.parametrize("candidate", ESCAPES)
    def test_traversing_file_path_is_blocked(
        self, tmp_path: Path, monkeypatch, candidate: str
    ) -> None:
        from orchestrator.ide_backend.ide_orchestrator_server import session_file_path

        monkeypatch.chdir(tmp_path)
        with pytest.raises(PathTraversalError):
            session_file_path("session-abc123", candidate)

    def test_legitimate_file_resolves_under_the_session(self, tmp_path: Path, monkeypatch) -> None:
        from orchestrator.ide_backend.ide_orchestrator_server import session_file_path

        monkeypatch.chdir(tmp_path)
        resolved = session_file_path("session-abc123", "src/app.py")
        session_root = (tmp_path / "ide_outputs" / "session-abc123").resolve()
        assert resolved.is_relative_to(session_root)

    def test_traversal_cannot_reach_a_real_file_outside_the_root(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        # Make the escape target actually exist, so a missing-file result
        # cannot be mistaken for the containment working.
        monkeypatch.chdir(tmp_path)
        secret = tmp_path / "ide_outputs" / "secret.txt"
        secret.parent.mkdir(parents=True, exist_ok=True)
        secret.write_text("do not read me", encoding="utf-8")

        from orchestrator.ide_backend.ide_orchestrator_server import session_file_path

        with pytest.raises(PathTraversalError):
            session_file_path("session-abc123", "../secret.txt")
