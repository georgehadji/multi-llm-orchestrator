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


# ── D3: IDE bind policy reads the same auth flag the routes enforce ─────────


class TestIdeBindUsesEnvAuth:
    def test_remote_bind_refused_when_env_auth_off(self, monkeypatch) -> None:
        import inspect

        from orchestrator.ide_backend import auth as ide_auth
        from orchestrator.ide_backend.security import InsecureBindError
        from orchestrator.ide_backend.server import run_ide_server

        monkeypatch.setenv("ORCHESTRATOR_IDE_ALLOW_REMOTE", "true")
        monkeypatch.delenv(ide_auth.AUTH_REQUIRED_ENV, raising=False)
        # The bind check must run before uvicorn is even imported.
        monkeypatch.setitem(__import__("sys").modules, "uvicorn", None)
        # No caller flag can claim auth that the routes do not enforce.
        assert "auth_required" not in inspect.signature(run_ide_server).parameters
        with pytest.raises(InsecureBindError):
            run_ide_server(host="0.0.0.0")


# ── D4: rate limiter keyed on the peer, with bounded state ──────────────────


class TestRateLimiter:
    @pytest.mark.asyncio
    async def test_random_api_key_per_request_is_still_limited(self) -> None:
        import uuid

        from aiohttp.test_utils import TestClient as AioClient
        from aiohttp.test_utils import TestServer

        from orchestrator.api_server import APIServer

        server = APIServer(port=0, auth_required=False, rate_limit=1, rate_window=60)
        client = AioClient(TestServer(server.app))
        await client.start_server()
        try:
            statuses = [
                (await client.get("/health", headers={"X-API-Key": uuid.uuid4().hex})).status
                for _ in range(3)
            ]
            assert statuses[0] == 200
            assert 429 in statuses[1:]
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_fresh_client_gets_a_full_bucket_on_a_fine_clock(self, monkeypatch) -> None:
        import orchestrator.api_server as mod

        # Every call advances the clock, as a high-resolution clock does on Linux.
        clock = [1000.0]

        def tick() -> float:
            clock[0] += 1e-6
            return clock[0]

        monkeypatch.setattr(mod.time, "time", tick)
        limiter = mod.TokenBucketRateLimiter(rate=1, window_seconds=60)
        assert await limiter.is_allowed("peer") is True

    @pytest.mark.asyncio
    async def test_idle_buckets_are_dropped(self, monkeypatch) -> None:
        import orchestrator.api_server as mod

        clock = [1000.0]
        monkeypatch.setattr(mod.time, "time", lambda: clock[0])
        limiter = mod.TokenBucketRateLimiter(rate=5, window_seconds=60)
        for i in range(50):
            await limiter.is_allowed(f"10.0.0.{i}")
        clock[0] += 61
        await limiter.is_allowed("10.0.0.200")
        assert set(limiter._buckets) == {"10.0.0.200"}


# ── D5, D7, D8: api_server input validation ─────────────────────────────────


async def _api_client(**kwargs):
    from aiohttp.test_utils import TestClient as AioClient
    from aiohttp.test_utils import TestServer

    from orchestrator.api_server import APIServer

    kwargs.setdefault("auth_required", False)
    server = APIServer(port=0, rate_limit=1000, **kwargs)
    client = AioClient(TestServer(server.app))
    await client.start_server()
    return client


class TestSpecDirContainment:
    @pytest.mark.asyncio
    async def test_spec_dir_outside_root_is_400_without_echo(self, tmp_path, monkeypatch) -> None:
        root = tmp_path / "root"
        root.mkdir()
        outside = tmp_path / "secret-outside"
        outside.mkdir()
        monkeypatch.chdir(root)
        client = await _api_client()
        try:
            resp = await client.post("/execute/from-speckit", json={"spec_dir": str(outside)})
            assert resp.status == 400
            assert "secret-outside" not in await resp.text()
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_missing_spec_dir_does_not_echo_path(self, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        client = await _api_client()
        try:
            resp = await client.post("/execute/from-speckit", json={"spec_dir": "no-such-dir"})
            assert resp.status == 400
            assert "no-such-dir" not in await resp.text()
        finally:
            await client.close()


class TestCorsPreflight:
    @pytest.mark.asyncio
    async def test_preflight_from_allowed_origin_is_204(self) -> None:
        origin = "http://localhost:3000"
        client = await _api_client(auth_required=True, cors_origins=[origin])
        try:
            resp = await client.options(
                "/execute/project",
                headers={"Origin": origin, "Access-Control-Request-Method": "POST"},
            )
            assert resp.status == 204
            assert resp.headers["Access-Control-Allow-Origin"] == origin
        finally:
            await client.close()


class TestBadTaskType:
    @pytest.mark.asyncio
    async def test_unknown_task_type_is_400(self) -> None:
        client = await _api_client()
        try:
            resp = await client.post(
                "/execute/tasks",
                json={
                    "project_description": "p",
                    "tasks": [{"id": "T1", "type": "not_a_type", "prompt": "x"}],
                },
            )
            assert resp.status == 400
        finally:
            await client.close()


# ── D6: task ids from LLM output never become path components ───────────────


class TestTaskIdTraversal:
    def test_is_safe_id(self) -> None:
        from orchestrator.domain.security import is_safe_id

        assert is_safe_id("task_001")
        assert is_safe_id("T1.a-b")
        for bad in ("../../x", "..", "a/b", "a\b", "", "x*", "c:evil", None, 7):
            assert not is_safe_id(bad), bad

    def _decomposer(self):
        from unittest.mock import MagicMock

        from orchestrator.application.decomposer import Decomposer

        return Decomposer(client=MagicMock(), selector=MagicMock())

    def test_decomposition_drops_traversing_ids(self) -> None:
        import json

        text = json.dumps(
            [
                {"id": "../../x", "type": "code_generation", "prompt": "evil"},
                {"id": "task_001", "type": "code_generation", "prompt": "ok"},
            ]
        )
        tasks = self._decomposer()._parse_decomposition(text)
        assert list(tasks) == ["task_001"]

    def test_partial_repair_drops_traversing_ids(self) -> None:
        tasks = self._decomposer()._repair_partial_tasks(
            [
                {"id": "../../x", "type": "code_generation", "prompt": "evil"},
                {"id": "task_001", "type": "code_generation", "prompt": "ok"},
            ]
        )
        assert list(tasks) == ["task_001"]

    def test_output_writer_keeps_files_inside_out(self, tmp_path) -> None:
        from orchestrator.budget import Budget
        from orchestrator.models import (
            Model,
            ProjectState,
            ProjectStatus,
            Task,
            TaskResult,
            TaskStatus,
            TaskType,
        )
        from orchestrator.output_writer import write_output_dir

        tid = "../../escaped"
        task = Task(id=tid, type=TaskType.WRITING, prompt="p")
        result = TaskResult(
            task_id=tid,
            output="hello",
            score=1.0,
            model_used=Model.GPT_4O_MINI,
            status=TaskStatus.COMPLETED,
        )
        state = ProjectState(
            project_description="d",
            success_criteria="s",
            budget=Budget(max_usd=1.0),
            tasks={tid: task},
            results={tid: result},
            status=ProjectStatus.SUCCESS,
            execution_order=[tid],
        )
        out = tmp_path / "a" / "b" / "out"
        write_output_dir(state, out)
        assert not list((tmp_path / "a").glob("escaped*"))
        assert list(out.glob("escaped_*"))

    @pytest.mark.asyncio
    async def test_checkpoints_refuse_unsafe_ids(self, tmp_path) -> None:
        from orchestrator.checkpoints import CheckpointManager

        manager = CheckpointManager(checkpoint_dir=str(tmp_path / "cp"))
        with pytest.raises(ValueError):
            await manager.save_checkpoint({"a": 1}, "../../x")
        with pytest.raises(ValueError):
            await manager.load_checkpoint("*")
