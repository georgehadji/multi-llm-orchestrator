"""
Audit Wave A — live-by-default one-line fixes
=============================================
Regression tests for docs/plans/2026-10-29-audit-remediation-plan.md, Wave A.

A1: dashboard_core/core.py put the raw ?view= value into the error page HTML (XSS).
A2: the dashboard, command center and MCP server bound 0.0.0.0 by default with no auth,
    so anyone on the network could reach them.
A4: StateManager.close() kept the closed connection on success, so the next call raised
    "Connection closed" and CLI --file runs silently wrote no output.
"""

from __future__ import annotations

import inspect
import sys

import pytest

pytestmark = pytest.mark.unit


async def test_a1_dashboard_error_page_escapes_view_name() -> None:
    from orchestrator.dashboard_core.core import DashboardCore

    page = await DashboardCore().render("<img src=x onerror=alert(1)>")

    assert "<img" not in page
    assert "&lt;img" in page


def test_a2_server_signatures_default_to_loopback() -> None:
    from orchestrator.command_center_server import CommandCenterServer
    from orchestrator.crosscutting.config import OrchestratorSettings
    from orchestrator.dashboard_core.core import DashboardCore, run_dashboard
    from orchestrator.integrations.mcp_server import MCPConfig

    for fn in (DashboardCore.run, run_dashboard, CommandCenterServer.start):
        assert inspect.signature(fn).parameters["host"].default == "127.0.0.1", fn
    assert MCPConfig().host == "127.0.0.1"
    assert OrchestratorSettings.model_fields["mcp_host"].default == "127.0.0.1"


def test_a2_cli_entry_points_default_to_loopback(monkeypatch) -> None:
    import orchestrator.dashboard as dashboard
    import orchestrator.integrations.mcp_server as mcp

    seen: dict[str, str] = {}
    monkeypatch.setattr(sys, "argv", ["prog"])
    monkeypatch.setattr(dashboard, "run_dashboard", lambda host, port: seen.update(dash=host))

    async def fake_run(config):
        seen["mcp"] = config.host

    monkeypatch.setattr(mcp, "run_mcp_server", fake_run)

    from orchestrator.cli_dashboard import main as dashboard_main

    dashboard_main()
    mcp.main()

    assert seen == {"dash": "127.0.0.1", "mcp": "127.0.0.1"}


async def test_a4_state_manager_reconnects_after_close(tmp_path) -> None:
    from orchestrator.infrastructure.state import StateManager

    sm = StateManager(db_path=tmp_path / "state.db")
    await sm._get_conn()
    await sm.close()

    conn = await sm._get_conn()
    async with conn.execute("SELECT 1") as cur:
        assert (await cur.fetchone())[0] == 1
    await sm.close()
