"""
Hunt T15 — integrations/, vcs/, ide_backend/, dashboard_core/, commands/, cli*, entrypoints/
==============================================================================================
Regression tests for the defects fixed in wave T15 of the backend-remainder defect hunt
(docs/hunts/BACKEND_REMAINDER_WAVES_PLAN.md).

C1: entrypoints/chat_cli.py and dashboard_core/chat_view.py both called
    Orchestrator(..., verbose=...) — a keyword argument Orchestrator.__init__ has never
    accepted, crashing both the `orchestrator chat` CLI flow and the dashboard's /ws/chat
    websocket feature the moment either reached the build step.
C2: the `dashboard` console script (orchestrator.cli_dashboard:main) was broken on every
    invocation — orchestrator/dashboard.py aliased run_dashboard to a zero-argument function
    (mission_control.create_view), while cli_dashboard.py called it with host/port/
    open_browser kwargs none of which it accepted.
C3: commands/kanban.py's two lazy imports used the wrong relative-import depth
    (`.kanban.board`/`.kanban.dispatcher` instead of `..kanban.board`/`..kanban.dispatcher`),
    breaking all four kanban subcommands unconditionally.
C4: commands/gateway.py had the identical bug shape for `.gateway.run`, breaking both
    gateway subcommands.
C5: commands/codebase.py's `modify` subcommand accepted a --budget flag but never read it,
    always charging against a hardcoded $10 budget regardless of what the user passed.
C6: commands/nash.py's `backup` subcommand crashed with an unguarded ModuleNotFoundError
    (orchestrator.nash_backup has never existed) instead of failing cleanly.
"""

from __future__ import annotations

import argparse
import inspect
import os
from unittest.mock import AsyncMock, patch

import pytest

pytestmark = pytest.mark.unit


def test_c1_orchestrator_construction_succeeds_without_verbose_kwarg(monkeypatch) -> None:
    """Neither live call site should pass verbose= anymore; construction must succeed."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test-key-for-verification-only")
    from orchestrator.budget import Budget
    from orchestrator.engine import Orchestrator

    orch = Orchestrator(budget=Budget(max_usd=1.0))
    assert orch is not None


def test_c1_call_sites_no_longer_pass_verbose() -> None:
    import inspect as _inspect

    import orchestrator.dashboard_core.chat_view as chat_view
    import orchestrator.entrypoints.chat_cli as chat_cli

    assert "verbose=" not in _inspect.getsource(chat_cli._launch_build)
    assert "verbose=" not in _inspect.getsource(chat_view._run_build)


def test_c2_dashboard_run_dashboard_has_host_port_signature() -> None:
    from orchestrator.dashboard import run_dashboard
    from orchestrator.dashboard_core.core import run_dashboard as canonical

    assert run_dashboard is canonical
    sig = inspect.signature(run_dashboard)
    sig.bind(host="0.0.0.0", port=8080)  # must not raise TypeError


def test_c3_kanban_command_imports_resolve() -> None:
    from orchestrator.commands import kanban

    args = argparse.Namespace(command="list", status=None)
    kanban.execute(args)  # must not raise ModuleNotFoundError


def test_c4_gateway_command_imports_resolve() -> None:
    from orchestrator.commands import gateway

    args = argparse.Namespace(command="status", platforms=None)
    gateway.execute(args)  # must not raise ModuleNotFoundError


@pytest.mark.asyncio
async def test_c5_codebase_modify_budget_flag_is_honored() -> None:
    from orchestrator.commands.codebase import _run_modify

    captured = {}

    class FakeBudget:
        def __init__(self, max_usd):
            captured["max_usd"] = max_usd

    class FakeOrchestrator:
        def __init__(self, budget):
            pass

        async def modify_codebase(self, **kwargs):
            return {}

    with (
        patch("orchestrator.budget.Budget", FakeBudget),
        patch("orchestrator.engine.Orchestrator", FakeOrchestrator),
    ):
        await _run_modify(repo=".", objective="test", dry_run=True, budget=42.0)

    assert captured["max_usd"] == 42.0


def test_c6_nash_backup_fails_cleanly_instead_of_crashing(capsys) -> None:
    from orchestrator.commands.nash import backup

    args = argparse.Namespace(list=True, restore=None, value=False)
    backup(args)  # must not raise ModuleNotFoundError

    out = capsys.readouterr().out
    assert "not implemented" in out.lower()
