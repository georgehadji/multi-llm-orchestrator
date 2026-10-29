"""Wave D of the 2026-10-29 audit remediation plan: security hardening."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from orchestrator.security.ws_origin import check_ws_origin

pytestmark = pytest.mark.unit


class _FakeSocket:
    def __init__(self, origin: str | None) -> None:
        self.headers = {} if origin is None else {"origin": origin}
        self.accept = AsyncMock()
        self.close = AsyncMock()


# ── D1: WebSocket Origin check ──────────────────────────────────────────────


class TestCheckWsOrigin:
    def test_no_origin_is_allowed(self) -> None:
        # Browsers always send Origin; a client without one is not a browser page.
        assert check_ws_origin({}) is True

    def test_loopback_origin_allowed_by_default(self) -> None:
        assert check_ws_origin({"origin": "http://127.0.0.1:8888"}) is True
        assert check_ws_origin({"origin": "http://localhost:8888"}) is True

    def test_foreign_origin_refused_by_default(self) -> None:
        assert check_ws_origin({"origin": "https://evil.example"}) is False

    def test_rebinding_name_refused(self) -> None:
        # DNS rebinding: evil.example resolves to 127.0.0.1, but Origin keeps the name.
        assert check_ws_origin({"origin": "http://evil.example:8888"}) is False

    def test_explicit_allowlist_is_exact(self) -> None:
        allowed = ["http://localhost:5173"]
        assert check_ws_origin({"origin": "http://localhost:5173"}, allowed) is True
        assert check_ws_origin({"origin": "http://localhost:9999"}, allowed) is False

    def test_null_origin_refused(self) -> None:
        assert check_ws_origin({"origin": "null"}) is False


class TestDashboardSocketsCheckOrigin:
    @pytest.mark.asyncio
    async def test_chat_socket_closes_foreign_origin_with_1008(self) -> None:
        from orchestrator.dashboard_core.chat_view import handle_chat_websocket

        ws = _FakeSocket("https://evil.example")
        await handle_chat_websocket(ws)
        ws.close.assert_awaited_once_with(code=1008)
        ws.accept.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_event_socket_closes_foreign_origin_with_1008(self) -> None:
        from orchestrator.dashboard_core.core import DashboardCore

        dash = DashboardCore.__new__(DashboardCore)
        dash._websocket_clients = []
        ws = _FakeSocket("https://evil.example")
        await dash.handle_websocket(ws)
        ws.close.assert_awaited_once_with(code=1008)
        ws.accept.assert_not_awaited()
        assert dash._websocket_clients == []


class TestIdeSocketChecksOrigin:
    def test_foreign_origin_closed_with_1008(self, tmp_path, monkeypatch) -> None:
        from orchestrator.ide_backend.server import create_app
        from orchestrator.ide_backend.session_manager import (
            get_session_manager,
            reset_session_manager,
        )

        monkeypatch.setenv("ORCHESTRATOR_IDE_SESSION_HOME", str(tmp_path))
        reset_session_manager()
        get_session_manager(tmp_path)
        try:
            client = TestClient(create_app())
            with pytest.raises(WebSocketDisconnect) as exc:
                with client.websocket_connect(
                    "/ws/new", headers={"Origin": "https://evil.example"}
                ) as ws:
                    ws.receive_json()
            assert exc.value.code == 1008
        finally:
            reset_session_manager()


# ── D2: IDE WebSocket handler registry and server-side session identity ─────


class TestIdeHandlerRegistry:
    def test_register_handler_works_as_decorator(self) -> None:
        from orchestrator.ide_backend.websocket_manager import ConnectionManager

        cm = ConnectionManager()

        @cm.register_handler("x")
        async def handler(data, websocket):
            return None

        assert handler is not None
        assert cm._event_handlers["x"] == [handler]

    def test_handlers_register_once_across_connections(self, tmp_path, monkeypatch) -> None:
        from orchestrator.ide_backend import auth as ide_auth
        from orchestrator.ide_backend.server import create_app
        from orchestrator.ide_backend.session_manager import (
            get_session_manager,
            reset_session_manager,
        )
        from orchestrator.ide_backend.websocket_manager import (
            get_connection_manager,
            reset_connection_manager,
        )

        monkeypatch.delenv(ide_auth.AUTH_REQUIRED_ENV, raising=False)
        monkeypatch.setenv("ORCHESTRATOR_IDE_SESSION_HOME", str(tmp_path))
        reset_session_manager()
        reset_connection_manager()
        get_session_manager(tmp_path)
        try:
            client = TestClient(create_app())
            for _ in range(3):
                with client.websocket_connect("/ws/new") as ws:
                    assert ws.receive_json()["event"] == "session_state"
            handlers = get_connection_manager()._event_handlers
            assert all(len(h) == 1 for h in handlers.values())
            assert "file_update" in handlers
        finally:
            reset_connection_manager()
            reset_session_manager()

    @pytest.mark.asyncio
    async def test_file_update_ignores_payload_session_id(self, tmp_path) -> None:
        from orchestrator.ide_backend.session_manager import SessionManager
        from orchestrator.ide_backend.websocket.handlers import setup_websocket_handlers
        from orchestrator.ide_backend.websocket_manager import ConnectionManager

        sm = SessionManager(tmp_path)
        victim = await sm.create_session(owner_id="victim")
        attacker = await sm.create_session(owner_id="attacker")
        cm = ConnectionManager()
        sm.update_file = AsyncMock(return_value=True)
        setup_websocket_handlers(cm, sm)
        ws = object()
        cm._connection_sessions[ws] = attacker.id
        cm.send_to_session = AsyncMock()

        await cm.handle_event(
            "file_update", {"session_id": victim.id, "path": "a.py", "content": "x"}, ws
        )

        sm.update_file.assert_awaited_once_with(attacker.id, "a.py", "x")
