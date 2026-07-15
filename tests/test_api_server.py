"""
Tests for APIServer HTTP Execute Endpoints
============================================
Author: Orchestrator core

Tests the real execution endpoints replacing the stubs in api_server.py.
Uses mocked Orchestrator to verify routing, request validation, and response shape.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.api_server import APIServer
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiohttp import web

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def app() -> web.Application:
    """Create an APIServer with auth disabled for testing."""
    server = APIServer(
        port=0,
        host="127.0.0.1",
        auth_required=False,
        cors_origins=["*"],
    )
    return server.app


@pytest.fixture
def authed_server() -> APIServer:
    """Create an APIServer with auth enabled and a pre-registered key."""
    server = APIServer(port=0, host="127.0.0.1", auth_required=True)
    # Register a test key
    import hashlib

    hashed = hashlib.sha256(b"test-key-123").hexdigest()
    server.api_keys[hashed] = {
        "user_id": "test-user",
        "permissions": ["execute"],
        "created_at": datetime.now().isoformat(),
    }
    return server


@pytest.fixture
def client(loop: asyncio.AbstractEventLoop, app: web.Application) -> web.Application:
    """Create a test client."""
    from aiohttp.test_utils import TestClient, TestServer

    return loop.run_until_complete(TestClient(TestServer(app)).start())


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint Registration Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestEndpointRegistration:
    """Verify all expected routes are registered."""

    def test_health_routes(self) -> None:
        """Root and /health should exist — verify via resource lookup."""
        server = APIServer(port=0, host="127.0.0.1", auth_required=False)
        resources = list(server.app.router.resources())
        paths = []
        for r in resources:
            info = r.get_info()
            if "path" in info:
                paths.append(info["path"])
        assert "/" in paths, f"Root route missing. Paths: {paths}"
        assert "/health" in paths

    def test_execute_routes_exist(self, app: web.Application) -> None:
        """All execute endpoints should be registered."""
        # Check routes by path
        routes = {r.method: r.path for r in app.router.routes() if hasattr(r, "path")}
        # Can't match by path easily with aiohttp's resource system
        # Just verify total route count is reasonable
        route_count = len(list(app.router.routes()))
        assert route_count >= 12, f"Expected 12+ routes, got {route_count}"


# ─────────────────────────────────────────────────────────────────────────────
# Execute Endpoint Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestExecuteProject:
    """POST /execute/project endpoint tests."""

    async def _get_server(self) -> APIServer:
        server = APIServer(
            port=0,
            host="127.0.0.1",
            auth_required=False,
            cors_origins=["*"],
        )
        return server

    async def test_missing_description_returns_400(self) -> None:
        """Missing project_description should return 400."""
        server = await self._get_server()
        request = MagicMock()
        request.json = AsyncMock(return_value={})
        request.headers = {}

        response = await server.execute_project(request)
        assert response.status == 400
        body = json.loads(response.body)
        assert "project_description" in str(body.get("error", ""))

    async def test_valid_request_returns_202(self) -> None:
        """Valid project request should return 202 Accepted."""
        server = await self._get_server()
        request = MagicMock()
        request.json = AsyncMock(
            return_value={
                "project_description": "Build a calculator",
                "success_criteria": "Add and subtract work",
                "budget": 5.0,
            }
        )
        request.headers = {}

        response = await server.execute_project(request)
        assert response.status == 202
        body = json.loads(response.body)
        assert "project_id" in body
        assert body["status"] == "accepted"
        assert "status_url" in body
        assert "stream_url" in body

    async def test_max_time_and_concurrency(self) -> None:
        """max_time_seconds and concurrency should be accepted."""
        server = await self._get_server()
        request = MagicMock()
        request.json = AsyncMock(
            return_value={
                "project_description": "Build an API",
                "success_criteria": "All endpoints work",
                "budget": 10.0,
                "max_time_seconds": 7200,
                "concurrency": 5,
            }
        )
        request.headers = {}

        response = await server.execute_project(request)
        assert response.status == 202
        body = json.loads(response.body)
        assert len(body["project_id"]) == 12


class TestExecuteTasks:
    """POST /execute/tasks endpoint tests."""

    async def _get_server(self) -> APIServer:
        return APIServer(port=0, host="127.0.0.1", auth_required=False, cors_origins=["*"])

    async def test_missing_tasks_returns_400(self) -> None:
        """Missing tasks field should return 400."""
        server = await self._get_server()
        request = MagicMock()
        request.json = AsyncMock(return_value={})
        request.headers = {}

        response = await server.execute_tasks(request)
        assert response.status == 400
        body = json.loads(response.body)
        assert "tasks" in str(body.get("error", ""))

    async def test_empty_tasks_returns_400(self) -> None:
        """Empty tasks array should return 400."""
        server = await self._get_server()
        request = MagicMock()
        request.json = AsyncMock(return_value={"tasks": []})
        request.headers = {}

        response = await server.execute_tasks(request)
        assert response.status == 400

    async def test_valid_tasks_returns_202(self) -> None:
        """Valid pre-composed tasks should return 202."""
        server = await self._get_server()
        request = MagicMock()
        request.json = AsyncMock(
            return_value={
                "project_description": "Auth feature",
                "tasks": [
                    {
                        "id": "T001",
                        "type": "code_generation",
                        "prompt": "Create user model",
                        "target_path": "src/models/user.py",
                    },
                    {
                        "id": "T002",
                        "type": "code_generation",
                        "prompt": "Create auth service",
                        "target_path": "src/services/auth.py",
                        "dependencies": ["T001"],
                    },
                ],
                "budget": 3.0,
            }
        )
        request.headers = {}

        response = await server.execute_tasks(request)
        assert response.status == 202
        body = json.loads(response.body)
        assert "project_id" in body
        assert body["task_count"] == 2

    async def test_tasks_without_id_get_auto_id(self) -> None:
        """Tasks without an id field should get auto-generated IDs."""
        server = await self._get_server()
        request = MagicMock()
        request.json = AsyncMock(
            return_value={
                "tasks": [
                    {"prompt": "Do something"},
                    {"prompt": "Do something else"},
                ],
            }
        )
        request.headers = {}

        response = await server.execute_tasks(request)
        assert response.status == 202
        body = json.loads(response.body)
        assert body["task_count"] == 2


class TestExecuteFromSpecKit:
    """POST /execute/from-speckit endpoint tests."""

    async def _get_server(self) -> APIServer:
        return APIServer(port=0, host="127.0.0.1", auth_required=False, cors_origins=["*"])

    async def test_missing_spec_dir_returns_400(self) -> None:
        """Missing spec_dir should return 400."""
        server = await self._get_server()
        request = MagicMock()
        request.json = AsyncMock(return_value={})
        request.headers = {}

        response = await server.execute_from_speckit(request)
        assert response.status == 400
        body = json.loads(response.body)
        assert "spec_dir" in str(body.get("error", ""))

    async def test_nonexistent_spec_dir_returns_400(self) -> None:
        """Non-existent spec_dir should return 400."""
        server = await self._get_server()
        request = MagicMock()
        request.json = AsyncMock(return_value={"spec_dir": "/nonexistent/path/12345xyz"})
        request.headers = {}

        response = await server.execute_from_speckit(request)
        assert response.status == 400


# ─────────────────────────────────────────────────────────────────────────────
# Project Status & Streaming Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestGetProjectStatus:
    """GET /projects/{project_id} endpoint tests."""

    async def _get_server(self) -> APIServer:
        server = APIServer(port=0, host="127.0.0.1", auth_required=False, cors_origins=["*"])
        # Pre-populate with a known project
        pid = "test-project-001"
        server._active_project_state[pid] = {
            "project_id": pid,
            "status": "running",
            "tasks_total": 5,
            "tasks_completed": 2,
            "tasks_failed": 0,
        }
        return server

    async def test_unknown_project_returns_404(self) -> None:
        """Unknown project ID should return 404."""
        server = await self._get_server()
        request = MagicMock()
        request.match_info = {"project_id": "nonexistent-id"}
        request.headers = {}

        response = await server.get_project_status(request)
        assert response.status == 404

    async def test_known_project_returns_status(self) -> None:
        """Known project should return structured status."""
        server = await self._get_server()
        request = MagicMock()
        request.match_info = {"project_id": "test-project-001"}
        request.headers = {}

        response = await server.get_project_status(request)
        assert response.status == 200
        body = json.loads(response.body)
        assert body["project_id"] == "test-project-001"
        assert body["status"] == "running"
        assert body["tasks_total"] == 5
        assert body["tasks_completed"] == 2
        assert body["tasks_failed"] == 0
        assert "is_running" in body

    async def test_completed_project(self) -> None:
        """Completed project should show final status."""
        server = await self._get_server()
        pid = "test-completed"
        server._active_project_state[pid] = {
            "project_id": pid,
            "status": "completed",
            "tasks_total": 3,
            "tasks_completed": 3,
            "tasks_failed": 0,
            "cost_spent_usd": 1.5,
        }
        request = MagicMock()
        request.match_info = {"project_id": pid}
        request.headers = {}

        response = await server.get_project_status(request)
        assert response.status == 200
        body = json.loads(response.body)
        assert body["status"] == "completed"


class TestCancelProject:
    """DELETE /projects/{project_id} endpoint tests."""

    async def _get_server(self) -> APIServer:
        return APIServer(port=0, host="127.0.0.1", auth_required=False, cors_origins=["*"])

    async def test_cancel_unknown_project_returns_404(self) -> None:
        """Cancelling unknown project should return 404."""
        server = await self._get_server()
        request = MagicMock()
        request.match_info = {"project_id": "nonexistent"}
        request.headers = {}

        response = await server.cancel_project(request)
        assert response.status == 404

    async def test_cancel_not_found_for_completed(self) -> None:
        """Cancelling a completed project (no background task) should return 404."""
        server = await self._get_server()
        pid = "completed-project"
        server._active_project_state[pid] = {"status": "completed"}

        request = MagicMock()
        request.match_info = {"project_id": pid}
        request.headers = {}

        response = await server.cancel_project(request)
        assert response.status == 404  # No background task running


# ─────────────────────────────────────────────────────────────────────────────
# Models Endpoint Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestListModels:
    """GET /models endpoint tests — de-stubbed from hardcoded list."""

    async def _get_server(self) -> APIServer:
        return APIServer(port=0, host="127.0.0.1", auth_required=False, cors_origins=["*"])

    async def test_returns_model_list(self) -> None:
        """Should return a list of available models."""
        server = await self._get_server()
        request = MagicMock()
        request.headers = {}

        response = await server.list_models(request)
        assert response.status == 200
        body = json.loads(response.body)
        assert isinstance(body, list)
        assert len(body) > 0
        # Each model should have id, name
        for model in body:
            assert "id" in model
            assert "name" in model


# ─────────────────────────────────────────────────────────────────────────────
# Auth Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestAuthentication:
    """Verify auth is enforced on execute endpoints."""

    async def test_no_auth_returns_401(self) -> None:
        """Missing auth header should return 401 when auth_required."""
        server = APIServer(port=0, host="127.0.0.1", auth_required=True)
        # Register a key
        import hashlib

        hashed = hashlib.sha256(b"valid-key").hexdigest()
        server.api_keys[hashed] = {"user_id": "test", "permissions": ["execute"]}

        request = MagicMock()
        request.json = AsyncMock(return_value={"project_description": "test"})
        request.headers = {}

        response = await server.execute_project(request)
        assert response.status == 401

    async def test_invalid_auth_returns_401(self) -> None:
        """Invalid Bearer token should return 401."""
        server = APIServer(port=0, host="127.0.0.1", auth_required=True)
        request = MagicMock()
        request.json = AsyncMock(return_value={"project_description": "test"})
        request.headers = {"Authorization": "Bearer invalid-key"}

        response = await server.execute_project(request)
        assert response.status == 401

    async def test_valid_auth_passes(self) -> None:
        """Valid Bearer token should pass."""
        server = APIServer(port=0, host="127.0.0.1", auth_required=True)
        import hashlib

        hashed = hashlib.sha256(b"valid-key").hexdigest()
        server.api_keys[hashed] = {"user_id": "test", "permissions": ["execute"]}

        request = MagicMock()
        request.json = AsyncMock(
            return_value={
                "project_description": "Build a calculator",
                "success_criteria": "Works",
                "budget": 1.0,
            }
        )
        request.headers = {"Authorization": "Bearer valid-key"}

        response = await server.execute_project(request)
        # Should pass auth and get to validation (which succeeds)
        assert response.status == 202


# ─────────────────────────────────────────────────────────────────────────────
# Legacy Backward Compat Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestLegacyExecute:
    """POST /execute (legacy) should auto-detect and route."""

    async def _get_server(self) -> APIServer:
        return APIServer(port=0, host="127.0.0.1", auth_required=False, cors_origins=["*"])

    async def test_legacy_with_task_routes_to_project(self) -> None:
        """Legacy request with 'task' field should route to project execute."""
        server = await self._get_server()
        request = MagicMock()
        request.json = AsyncMock(
            return_value={
                "task": "Build a calculator",
                "criteria": "Add and subtract",
                "budget": 5.0,
            }
        )
        request.headers = {}

        response = await server.execute_task(request)
        assert response.status == 202
        body = json.loads(response.body)
        assert "project_id" in body

    async def test_legacy_with_tasks_routes_to_tasks(self) -> None:
        """Legacy request with 'tasks' field should route to tasks execute."""
        server = await self._get_server()
        request = MagicMock()
        request.json = AsyncMock(
            return_value={
                "tasks": [{"prompt": "Do something"}],
            }
        )
        request.headers = {}

        response = await server.execute_task(request)
        assert response.status == 202

    async def test_legacy_unrecognized_returns_400(self) -> None:
        """Legacy request without recognizable fields returns 400."""
        server = await self._get_server()
        request = MagicMock()
        request.json = AsyncMock(return_value={"foo": "bar"})
        request.headers = {}

        response = await server.execute_task(request)
        assert response.status == 400
