"""SEC-006 — API-key permissions must be enforced, not merely recorded.

`register_api_key` stored a ``permissions`` list, but `_verify_api_key`
returned a bool and `_require_auth` only asked "does this key exist?". Every
valid key therefore had every capability — execution, cancellation and
supervisor access — whatever it was registered with.
"""

from __future__ import annotations

import pytest

from orchestrator.api_server import APIServer
from orchestrator.domain.security import (
    Permission,
    Principal,
    parse_permissions,
    principal_has_permission,
)
from orchestrator.safety.api_keys import reset_key_store

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _fresh_key_store():
    """`get_key_store()` is process-wide; keys must not leak between tests."""
    reset_key_store()
    yield
    reset_key_store()


class _StubRequest:
    """Minimal stand-in for `aiohttp.web.Request`."""

    def __init__(self, token: str | None = None) -> None:
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._state: dict = {}

    def __setitem__(self, key, value):
        self._state[key] = value

    def __getitem__(self, key):
        return self._state[key]


def _server_with_key(permissions: list[str]) -> tuple[APIServer, str]:
    server = APIServer(auth_required=True)
    raw, _ = server.key_store.issue("test-user", permissions)
    return server, raw


class TestPermissionModel:
    def test_admin_implies_everything(self) -> None:
        admin = Principal("u", "k", frozenset({Permission.ADMIN}))
        for permission in Permission:
            assert admin.has(permission)

    def test_execute_implies_project_cancel(self) -> None:
        # Cancelling a run you started is part of running it; denying this
        # would break pre-existing ["read", "execute"] keys for no gain.
        p = Principal("u", "k", frozenset({Permission.EXECUTE}))
        assert p.has(Permission.PROJECT_CANCEL)

    def test_execute_does_not_imply_supervisor(self) -> None:
        # This is the escalation the finding is about.
        p = Principal("u", "k", frozenset({Permission.EXECUTE, Permission.READ}))
        assert not p.has(Permission.SUPERVISOR_READ)
        assert not p.has(Permission.SUPERVISOR_EXECUTE)

    def test_empty_permissions_grant_nothing(self) -> None:
        p = Principal("u", "k", frozenset())
        assert not any(p.has(permission) for permission in Permission)

    def test_unknown_permission_strings_are_dropped(self) -> None:
        parsed = parse_permissions(["read", "not-a-real-permission", "admin"])
        assert parsed == frozenset({Permission.READ, Permission.ADMIN})

    def test_malformed_permission_payloads_fail_closed(self) -> None:
        for junk in [None, "read", 42, {"read": True}]:
            assert parse_permissions(junk) == frozenset()

    def test_helper_matches_principal_method(self) -> None:
        held = frozenset({Permission.EXECUTE})
        assert principal_has_permission(held, Permission.PROJECT_CANCEL)
        assert not principal_has_permission(held, Permission.ADMIN)


class TestVerifyApiKeyReturnsPrincipal:
    def test_valid_key_yields_principal_with_permissions(self) -> None:
        server, raw = _server_with_key(["read"])
        principal = server._verify_api_key(raw)
        assert isinstance(principal, Principal)
        assert principal.id == "test-user"
        assert principal.permissions == frozenset({Permission.READ})

    def test_invalid_key_yields_none(self) -> None:
        server, _ = _server_with_key(["read"])
        assert server._verify_api_key("orchestrator_wrong") is None


class TestReadOnlyKeyIsConfined:
    """A read-only key must not reach execute, cancel or supervisor routes."""

    def test_read_key_allowed_on_read_routes(self) -> None:
        server, raw = _server_with_key(["read"])
        assert server._require_auth(_StubRequest(raw), Permission.READ) is None

    @pytest.mark.parametrize(
        "permission",
        [
            Permission.EXECUTE,
            Permission.PROJECT_CANCEL,
            Permission.SUPERVISOR_READ,
            Permission.SUPERVISOR_EXECUTE,
            Permission.KEY_REGISTER,
            Permission.ADMIN,
        ],
    )
    def test_read_key_denied_elsewhere(self, permission: Permission) -> None:
        server, raw = _server_with_key(["read"])
        response = server._require_auth(_StubRequest(raw), permission)
        assert response is not None, f"{permission.value} should have been denied"
        assert response.status == 403

    def test_missing_credentials_are_401_not_403(self) -> None:
        server, _ = _server_with_key(["read"])
        response = server._require_auth(_StubRequest(), Permission.READ)
        assert response is not None
        assert response.status == 401

    def test_invalid_key_is_401(self) -> None:
        server, _ = _server_with_key(["read"])
        response = server._require_auth(_StubRequest("orchestrator_bogus"), Permission.READ)
        assert response is not None
        assert response.status == 401

    def test_principal_is_attached_to_the_request(self) -> None:
        server, raw = _server_with_key(["read", "execute"])
        request = _StubRequest(raw)
        assert server._require_auth(request, Permission.EXECUTE) is None
        assert request["principal"].id == "test-user"

    def test_auth_disabled_bypasses_checks(self) -> None:
        # Preserves the documented development behaviour.
        server = APIServer(auth_required=False)
        assert server._require_auth(_StubRequest(), Permission.ADMIN) is None
