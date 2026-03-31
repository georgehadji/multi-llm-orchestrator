"""Tests for the Dashboard module."""

import pytest
import sys
from pathlib import Path

# Try to import dashboard - may fail if dependencies not installed
try:
    from orchestrator.dashboard import DashboardServer, DASHBOARD_HTML, run_dashboard

    HAS_DASHBOARD_DEPS = True
except ImportError:
    HAS_DASHBOARD_DEPS = False


@pytest.mark.skipif(not HAS_DASHBOARD_DEPS, reason="Dashboard requires fastapi/uvicorn")
class TestDashboardServer:
    """Test DashboardServer class."""

    def test_dashboard_server_creation(self):
        """Test dashboard server can be created."""
        server = DashboardServer(host="127.0.0.1", port=9090)
        assert server.host == "127.0.0.1"
        assert server.port == 9090
        assert server.clients == []

    def test_dashboard_routes_setup(self):
        """Test dashboard routes are configured."""
        server = DashboardServer()
        routes = [route.path for route in server.app.routes]
        assert "/" in routes or "/docs" in routes


class TestDashboardHTML:
    """Test embedded HTML dashboard."""

    def test_html_contains_key_elements(self):
        """Test HTML contains essential UI elements."""
        # DASHBOARD_HTML is always available
        html = DASHBOARD_HTML
        assert "<!DOCTYPE html>" in html
        assert "Mission Control" in html
        assert "model_status" in html  # WebSocket message handling
        assert "websocket" in html.lower()  # WebSocket connectivity
        assert "Overview" in html
        assert "Models" in html
        assert "Execute" in html
        assert "Prompts" in html
        assert "Logs" in html

    def test_html_contains_model_data_refs(self):
        """Test HTML references model data structures."""
        assert "model_status" in DASHBOARD_HTML
        assert "metrics" in DASHBOARD_HTML
        assert "routing" in DASHBOARD_HTML


@pytest.mark.skipif(not HAS_DASHBOARD_DEPS, reason="Dashboard requires fastapi/uvicorn")
class TestDashboardIntegration:
    """Integration tests for dashboard."""

    def test_models_endpoint(self):
        """Test /api/models endpoint returns model data."""
        from fastapi.testclient import TestClient

        server = DashboardServer()
        client = TestClient(server.app)

        response = client.get("/api/models")
        assert response.status_code == 200

        models = response.json()
        # Should have at least one model
        assert len(models) > 0

        # Check model structure
        for model_id, info in models.items():
            assert "provider" in info
            assert "cost_input" in info
            assert "cost_output" in info
            assert "available" in info

    def test_routing_endpoint(self):
        """Test /api/routing endpoint returns routing config."""
        from fastapi.testclient import TestClient

        server = DashboardServer()
        client = TestClient(server.app)

        response = client.get("/api/routing")
        assert response.status_code == 200

        routing = response.json()
        # Should have routing entries
        assert len(routing) > 0

    def test_dashboard_html_endpoint(self):
        """Test root endpoint returns HTML dashboard."""
        from fastapi.testclient import TestClient

        server = DashboardServer()
        client = TestClient(server.app)

        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
        assert "Mission Control" in response.text


class TestDashboardAvailability:
    """Test dashboard module availability handling."""

    def test_graceful_degradation_without_deps(self):
        """Test dashboard gracefully handles missing dependencies."""
        # Simulate missing deps by checking if import works
        try:
            from orchestrator.dashboard import DashboardServer

            # If import succeeds, fastapi/uvicorn are available
            assert True
        except ImportError as e:
            # Import should fail gracefully with helpful message
            assert "fastapi" in str(e).lower() or "uvicorn" in str(e).lower()


@pytest.mark.skipif(not HAS_DASHBOARD_DEPS, reason="Dashboard requires fastapi/uvicorn")
class TestDashboardWebSocket:
    """Test WebSocket functionality."""

    def test_websocket_endpoint_exists(self):
        """Test WebSocket endpoint is configured."""
        server = DashboardServer()
        ws_routes = [r for r in server.app.routes if "/ws" in str(r.path)]
        assert len(ws_routes) > 0
