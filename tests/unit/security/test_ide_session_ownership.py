"""SEC-001b / T7 — the IDE must authenticate, and sessions must have owners.

Every route under `ide_backend/api/routes.py` took a caller-supplied
``session_id`` and acted on it. No authentication, no ownership: anyone who
could reach the port could read another session's chat history and generated
files, edit them, retarget its tasks, or delete it. Session IDs were
``str(uuid.uuid4())[:8]`` — 32 bits, enumerable in seconds.

T1 closed the network exposure. This closes the authorization hole behind it.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from orchestrator.ide_backend import auth as ide_auth
from orchestrator.ide_backend.server import create_app
from orchestrator.ide_backend.session_manager import (
    get_session_manager,
    reset_session_manager,
)
from orchestrator.safety.api_keys import reset_key_store

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.delenv(ide_auth.AUTH_REQUIRED_ENV, raising=False)
    monkeypatch.setenv("ORCHESTRATOR_IDE_SESSION_HOME", str(tmp_path / "sessions"))
    reset_key_store()
    reset_session_manager()
    get_session_manager(tmp_path / "sessions")
    yield
    reset_session_manager()
    reset_key_store()


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


def _issue(principal_id: str) -> str:
    from orchestrator.domain.security import Permission
    from orchestrator.safety.api_keys import get_key_store

    raw, _ = get_key_store().issue(principal_id, {Permission.READ, Permission.EXECUTE})
    return raw


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


class TestSessionIdentifiers:
    @pytest.mark.asyncio
    async def test_session_ids_are_full_uuid4(self) -> None:
        session = await get_session_manager().create_session(owner_id="alice")
        # str(uuid4())[:8] is 32 bits — enumerable. The whole value, or nothing.
        parsed = uuid.UUID(session.id)
        assert parsed.version == 4
        assert str(parsed) == session.id

    @pytest.mark.asyncio
    async def test_sessions_record_their_owner(self) -> None:
        session = await get_session_manager().create_session(owner_id="alice")
        assert session.owner_id == "alice"

    @pytest.mark.asyncio
    async def test_load_session_refuses_a_traversing_id(self) -> None:
        """`storage_path / f"{session_id}.json"` was caller-controlled."""
        manager = get_session_manager()
        assert await manager.load_session("../../../../etc/passwd") is None
        assert await manager.load_session("..\\..\\evil") is None


class TestOwnershipEnforcement:
    @pytest.mark.asyncio
    async def test_owner_can_read_their_session(self, client: TestClient) -> None:
        token = _issue("alice")
        session = await get_session_manager().create_session(owner_id="alice")

        response = client.get(f"/api/sessions/{session.id}", headers=_headers(token))
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_stranger_cannot_read_someone_elses_session(self, client: TestClient) -> None:
        session = await get_session_manager().create_session(owner_id="alice")
        mallory = _issue("mallory")

        response = client.get(f"/api/sessions/{session.id}", headers=_headers(mallory))
        assert response.status_code == 404, "existence itself should not leak"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("method", "suffix"),
        [
            ("get", ""),
            ("delete", ""),
            ("get", "/files"),
            ("get", "/tasks"),
        ],
    )
    async def test_every_session_scoped_route_checks_ownership(
        self, client: TestClient, method: str, suffix: str
    ) -> None:
        session = await get_session_manager().create_session(owner_id="alice")
        mallory = _issue("mallory")

        call = getattr(client, method)
        response = call(f"/api/sessions/{session.id}{suffix}", headers=_headers(mallory))
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_stranger_cannot_write_a_file(self, client: TestClient) -> None:
        session = await get_session_manager().create_session(owner_id="alice")
        mallory = _issue("mallory")

        response = client.put(
            f"/api/sessions/{session.id}/files/main.py",
            headers=_headers(mallory),
            json={"content": "print('pwned')"},
        )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_stranger_cannot_update_a_task(self, client: TestClient) -> None:
        session = await get_session_manager().create_session(owner_id="alice")
        mallory = _issue("mallory")

        response = client.put(
            f"/api/sessions/{session.id}/tasks/t1",
            headers=_headers(mallory),
            json={"status": "completed"},
        )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_listing_shows_only_your_own_sessions(self, client: TestClient) -> None:
        manager = get_session_manager()
        await manager.create_session(owner_id="alice", project_name="alice-project")
        await manager.create_session(owner_id="mallory", project_name="mallory-project")

        response = client.get("/api/sessions", headers=_headers(_issue("mallory")))
        assert response.status_code == 200
        names = [s["project_name"] for s in response.json()["sessions"]]
        assert names == ["mallory-project"]


class TestAuthRequiredMode:
    def test_no_credentials_is_401_when_auth_is_on(self, client: TestClient, monkeypatch) -> None:
        monkeypatch.setenv(ide_auth.AUTH_REQUIRED_ENV, "true")
        response = client.get("/api/sessions")
        assert response.status_code == 401

    def test_bad_credentials_is_401_when_auth_is_on(self, client: TestClient, monkeypatch) -> None:
        monkeypatch.setenv(ide_auth.AUTH_REQUIRED_ENV, "true")
        response = client.get("/api/sessions", headers=_headers("orchestrator_nope"))
        assert response.status_code == 401

    def test_loopback_dev_still_works_without_a_token(self, client: TestClient) -> None:
        """Auth off is the local default; requests get the local principal."""
        assert client.get("/api/sessions").status_code == 200

    def test_errors_do_not_leak_exception_text(self, client: TestClient, monkeypatch) -> None:
        monkeypatch.setenv(ide_auth.AUTH_REQUIRED_ENV, "true")
        body = client.get("/api/sessions", headers=_headers("orchestrator_nope")).json()
        assert body["detail"] in {"unauthorized", "forbidden", "not_found"}


class TestWebSocketUsesTheSamePath:
    def test_websocket_rejects_an_unauthenticated_client(self, monkeypatch) -> None:
        """The handshake must not be a second, softer auth implementation."""
        monkeypatch.setenv(ide_auth.AUTH_REQUIRED_ENV, "true")
        client = TestClient(create_app())

        with pytest.raises(Exception):  # noqa: B017 - starlette raises on policy close
            with client.websocket_connect(f"/ws/{uuid.uuid4()}"):
                pass

    @pytest.mark.asyncio
    async def test_websocket_rejects_a_session_you_do_not_own(self, monkeypatch) -> None:
        monkeypatch.setenv(ide_auth.AUTH_REQUIRED_ENV, "true")
        session = await get_session_manager().create_session(owner_id="alice")
        mallory = _issue("mallory")
        client = TestClient(create_app())

        with pytest.raises(Exception):  # noqa: B017
            with client.websocket_connect(f"/ws/{session.id}", headers=_headers(mallory)):
                pass
