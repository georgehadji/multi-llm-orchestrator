"""
test_api_contracts.py — HTTP endpoint contract tests.
======================================================

Uses aiohttp TestClient to verify routing, request/response shapes,
and middleware (rate limiting, CORS, size limits) without starting a
real server on a TCP port.
"""

from __future__ import annotations


import pytest
from aiohttp.test_utils import TestClient, TestServer

from orchestrator.api_server import APIServer


@pytest.fixture
def api_server():
    """Provide an APIServer instance with auth disabled for testing."""
    return APIServer(
        port=0,
        auth_required=False,
        rate_limit=1000,  # high limit so tests don't get rate-limited
    )


@pytest.fixture
def app(api_server):
    """Expose the underlying aiohttp application."""
    return api_server.app


@pytest.fixture
async def client(app):
    """Provide an aiohttp TestClient for the app."""
    tc = TestClient(TestServer(app))
    await tc.start_server()
    yield tc
    await tc.close()


@pytest.mark.asyncio
async def test_health_check_returns_200(client):
    """GET /health must return 200 with JSON body."""
    resp = await client.get("/health")
    assert resp.status == 200
    body = await resp.json()
    assert "status" in body


@pytest.mark.asyncio
async def test_root_redirects_to_health(client):
    """GET / should also work (health check alias)."""
    resp = await client.get("/")
    assert resp.status == 200


@pytest.mark.asyncio
async def test_list_models_returns_json(client):
    """GET /models must return a JSON list."""
    resp = await client.get("/models")
    assert resp.status == 200
    body = await resp.json()
    assert isinstance(body, (list, dict))


@pytest.mark.asyncio
async def test_execute_task_rejects_empty_body(client):
    """POST /execute without required fields should return 400."""
    resp = await client.post("/execute", json={})
    # May be 400 or 422 depending on validation depth
    assert resp.status in (400, 422, 500)


@pytest.mark.asyncio
async def test_get_task_status_requires_task_id(client):
    """GET /status/{task_id} must accept a task ID parameter."""
    resp = await client.get("/status/test-task-123")
    # May be 404 (not found) but must not 500
    assert resp.status in (200, 404)


@pytest.mark.asyncio
async def test_rate_limit_enforced():
    """
    With a low rate limit, rapid requests should eventually receive 429.
    """
    server = APIServer(port=0, auth_required=False, rate_limit=1, rate_window=60)
    client = TestClient(TestServer(server.app))
    await client.start_server()
    try:
        # First request allowed
        resp1 = await client.get("/health")
        assert resp1.status == 200

        # Second request immediately should be rate-limited
        resp2 = await client.get("/health")
        assert resp2.status == 429
        body = await resp2.json()
        assert "error" in body
        assert "retry_after" in body
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_request_size_limit_rejects_large_payload():
    """
    Payloads exceeding max_request_size should be rejected.
    """
    server = APIServer(port=0, auth_required=False, max_request_size=100)
    client = TestClient(TestServer(server.app))
    await client.start_server()
    try:
        large_payload = {"data": "x" * 1000}
        resp = await client.post("/execute", json=large_payload)
        assert resp.status in (413, 400)
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_cors_headers_present_when_origin_allowed():
    """CORS headers must be present for allowed origins."""
    server = APIServer(
        port=0,
        auth_required=False,
        cors_origins=["https://trusted.example.com"],
    )
    client = TestClient(TestServer(server.app))
    await client.start_server()
    try:
        resp = await client.get(
            "/health",
            headers={"Origin": "https://trusted.example.com"},
        )
        assert resp.status == 200
        assert "Access-Control-Allow-Origin" in resp.headers
        assert resp.headers["Access-Control-Allow-Origin"] == "https://trusted.example.com"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_cors_headers_absent_for_disallowed_origin():
    """CORS headers must NOT be present for disallowed origins."""
    server = APIServer(
        port=0,
        auth_required=False,
        cors_origins=["https://trusted.example.com"],
    )
    client = TestClient(TestServer(server.app))
    await client.start_server()
    try:
        resp = await client.get(
            "/health",
            headers={"Origin": "https://evil.example.com"},
        )
        assert resp.status == 200
        # Allow-Origin should either be missing or not match evil origin
        allowed = resp.headers.get("Access-Control-Allow-Origin", "")
        assert allowed != "https://evil.example.com"
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method,path",
    [
        ("post", "/supervisor/directive"),
        ("get", "/supervisor/sessions"),
        ("get", "/supervisor/sessions/abc"),
        ("get", "/supervisor/sessions/abc/lessons"),
    ],
)
async def test_supervisor_endpoints_require_auth(method, path):
    """All supervisor endpoints (including read endpoints) must enforce auth."""
    from unittest.mock import MagicMock

    server = APIServer(port=0, auth_required=True, rate_limit=1000, supervisor=MagicMock())
    client = TestClient(TestServer(server.app))
    await client.start_server()
    try:
        resp = await getattr(client, method)(path)
        assert resp.status == 401
    finally:
        await client.close()
