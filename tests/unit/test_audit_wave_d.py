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
