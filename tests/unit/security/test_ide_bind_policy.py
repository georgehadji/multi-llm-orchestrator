"""SEC-001a — the IDE service must not be remotely exposed by default.

Regression tests for the bind/CORS policy. Each of these fails against the
pre-fix tree, where `launch.py` and `server.run_ide_server` both defaulted to
``0.0.0.0`` and the FastAPI app was mounted with ``allow_origins=["*"]`` plus
``allow_credentials=True``.
"""

from __future__ import annotations

import pytest

from orchestrator.ide_backend.security import (
    InsecureBindError,
    allowed_origins,
    is_loopback,
    validate_bind_target,
)

pytestmark = pytest.mark.unit


class TestIsLoopback:
    @pytest.mark.parametrize(
        "host",
        ["127.0.0.1", "localhost", "::1", "127.0.0.5", "LOCALHOST"],
    )
    def test_loopback_hosts_recognised(self, host: str) -> None:
        assert is_loopback(host) is True

    @pytest.mark.parametrize(
        "host",
        ["0.0.0.0", "::", "192.168.1.10", "10.0.0.4", "example.com", ""],
    )
    def test_non_loopback_hosts_rejected(self, host: str) -> None:
        assert is_loopback(host) is False


class TestValidateBindTarget:
    def test_loopback_without_auth_is_allowed(self) -> None:
        # The developer default: local only, no auth required.
        validate_bind_target("127.0.0.1", allow_remote=False, auth_required=False)

    def test_remote_bind_requires_explicit_opt_in(self) -> None:
        with pytest.raises(InsecureBindError, match="ORCHESTRATOR_IDE_ALLOW_REMOTE"):
            validate_bind_target("0.0.0.0", allow_remote=False, auth_required=True)

    def test_remote_bind_without_auth_is_refused(self) -> None:
        # This is SEC-001 exactly: reachable from the network, no authentication.
        with pytest.raises(InsecureBindError, match="authentication"):
            validate_bind_target("0.0.0.0", allow_remote=True, auth_required=False)

    def test_remote_bind_with_optin_and_auth_is_allowed(self) -> None:
        validate_bind_target("0.0.0.0", allow_remote=True, auth_required=True)


class TestAllowedOrigins:
    def test_default_origins_are_loopback_only(self, monkeypatch) -> None:
        monkeypatch.delenv("ORCHESTRATOR_IDE_ALLOWED_ORIGINS", raising=False)
        origins = allowed_origins()
        assert origins, "must not be empty — an empty list would disable CORS entirely"
        assert "*" not in origins
        assert all("localhost" in o or "127.0.0.1" in o for o in origins)

    def test_explicit_origins_are_honoured(self, monkeypatch) -> None:
        monkeypatch.setenv(
            "ORCHESTRATOR_IDE_ALLOWED_ORIGINS",
            "https://ide.example.com, https://alt.example.com",
        )
        assert allowed_origins() == [
            "https://ide.example.com",
            "https://alt.example.com",
        ]

    def test_wildcard_origin_is_refused(self, monkeypatch) -> None:
        # Wildcard + credentials is the unsafe combination the audit flagged.
        monkeypatch.setenv("ORCHESTRATOR_IDE_ALLOWED_ORIGINS", "*")
        with pytest.raises(InsecureBindError, match="wildcard"):
            allowed_origins()


class TestServerDefaults:
    def test_run_ide_server_defaults_to_loopback(self) -> None:
        import inspect

        from orchestrator.ide_backend.server import run_ide_server

        default = inspect.signature(run_ide_server).parameters["host"].default
        assert default == "127.0.0.1", "programmatic default must not be 0.0.0.0"

    def test_launcher_defaults_to_loopback(self) -> None:
        import re
        from pathlib import Path

        import orchestrator.ide_backend.launch as launch_mod

        source = Path(launch_mod.__file__).read_text(encoding="utf-8")
        match = re.search(r'add_argument\(\s*"--host",\s*default="([^"]+)"', source)
        assert match, "could not locate the --host argument default"
        assert match.group(1) == "127.0.0.1"

    def test_app_does_not_mount_wildcard_cors(self, monkeypatch) -> None:
        monkeypatch.delenv("ORCHESTRATOR_IDE_ALLOWED_ORIGINS", raising=False)
        from orchestrator.ide_backend.server import create_app

        app = create_app()
        cors = [m for m in app.user_middleware if "CORS" in str(m.cls)]
        assert cors, "CORS middleware should still be configured"
        configured = cors[0].kwargs.get("allow_origins", [])
        assert "*" not in configured
