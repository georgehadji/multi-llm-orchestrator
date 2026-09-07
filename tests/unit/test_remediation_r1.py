"""
Remediation plan (docs/plans/2026-09-06-outstanding-remediation-plan.md),
Phase R1: mechanical fixes needing no product/architecture decision.

Each test proves the defect for the exact predicted reason (RED against the
pre-fix source) before the fix restores it (GREEN).
"""

from __future__ import annotations

import asyncio
import inspect
import os
from unittest.mock import MagicMock, patch

import click
import pytest

pytestmark = pytest.mark.unit


# ── B1: orchestrator/engine.py — dead _get_active_policies ──────────────────


class TestB1PolicyDeadCode:
    def test_get_active_policies_removed(self):
        """_get_active_policies read self._active_policies, which is
        assigned nowhere in engine.py or container.py, and the method
        itself had zero callers anywhere in the repo — stranded dead code
        from before the RunContext refactor. Removed outright; the still-
        live-but-unread self._run_ctx.active_policies is a separate,
        still-open question (remediation plan B1's larger enforcement-
        wiring decision), not this deletion.
        """
        from orchestrator.engine import Orchestrator

        assert not hasattr(Orchestrator, "_get_active_policies")


# ── C1: orchestrator/commands/codebase.py — modify_codebase honesty ─────────


class TestC1ModifyCodebaseHonestFailure:
    def test_reports_not_implemented_without_traceback(self):
        """Orchestrator has no modify_codebase method; the old broad except
        disguised the resulting AttributeError as a generic "Modification
        failed" message plus a full traceback, on every invocation.
        """
        from orchestrator.commands.codebase import _run_modify

        fake_orch = MagicMock(spec=[])  # no modify_codebase attribute at all
        with patch("orchestrator.engine.Orchestrator", return_value=fake_orch):
            result = asyncio.run(_run_modify(repo=".", objective="x", dry_run=True))

        assert "not implemented" in result.lower()
        assert "Traceback" not in result


# ── C2: orchestrator/operations/diagnostics.py — provider architecture ──────


class TestC2DiagnosticsProviderArchitecture:
    def test_old_check_api_keys_mechanism_was_broken(self):
        """UnifiedClient(model) passed a Model enum where cost_service is
        expected (UnifiedClient.__init__ has no `model` parameter at all —
        it silently became `cost_service`), and .generate() does not exist
        on UnifiedClient (the real method is .call()). Both errors were
        swallowed by the old except clause's narrow string-matching, so
        this check always failed with zero explanation regardless of real
        key validity.
        """
        from orchestrator.infrastructure.llm_client import UnifiedClient

        sig = inspect.signature(UnifiedClient.__init__)
        assert "model" not in sig.parameters
        assert not hasattr(UnifiedClient, "generate")
        assert hasattr(UnifiedClient, "call")

    def test_check_api_keys_reports_missing_keys_clearly(self):
        """With neither OPENROUTER_API_KEY nor DEEPSEEK_API_KEY set, the
        check must report one clear, specific issue instead of silently
        looping through three unreachable per-provider branches.
        """
        from orchestrator.operations.diagnostics import SystemDiagnostic

        env = {k: v for k, v in os.environ.items() if "API_KEY" not in k}
        with patch.dict(os.environ, env, clear=True):
            diag = SystemDiagnostic()
            asyncio.run(diag._check_api_keys())

        assert diag.checks_failed == 1
        assert any("OPENROUTER_API_KEY" in i.description for i in diag.issues)

    def test_check_network_probes_openrouter_not_raw_providers(self):
        """_check_network must probe the hosts UnifiedClient actually calls
        (openrouter.ai / api.deepseek.com / api.x.ai — matching
        _check_environment's already-fixed provider set from hunt T8, C6),
        not api.openai.com / generativelanguage.googleapis.com, which
        UnifiedClient's constructor never even accepts a key for.
        """
        from orchestrator.operations.diagnostics import SystemDiagnostic

        src = inspect.getsource(SystemDiagnostic._check_network)
        assert "openrouter.ai" in src
        assert "api.openai.com" not in src
        assert "generativelanguage.googleapis.com" not in src


# ── C3: orchestrator/cli_nash.py — click.Exit does not exist ────────────────


class TestC3ClickExitTypo:
    def test_click_exit_is_not_a_real_top_level_attribute(self):
        """Every `raise click.Exit(1)` in cli_nash.py (13 sites) referenced
        a top-level `click.Exit` attribute that does not exist in the
        installed click version — AttributeError instead of a clean exit,
        on every error path across nash status/backup/tuning/compare/
        events. The real class is click.exceptions.Exit.
        """
        assert not hasattr(click, "Exit")
        assert hasattr(click.exceptions, "Exit")

    def test_cli_nash_no_longer_references_bare_click_exit(self):
        import orchestrator.cli_nash as cli_nash

        src = inspect.getsource(cli_nash)
        assert "click.Exit(" not in src
        assert "click.exceptions.Exit(" in src

    def test_list_backups_exits_cleanly_without_nash_backup(self):
        """End-to-end proof against the real (missing) nash_backup module:
        _list_backups() must raise click.exceptions.Exit(1) — not a raw
        AttributeError — once it hits the ImportError branch.
        """
        from orchestrator.cli_nash import _list_backups

        with pytest.raises(click.exceptions.Exit) as exc_info:
            asyncio.run(_list_backups())
        assert exc_info.value.exit_code == 1


# ── C4-followup: orchestrator/vcs/service.py — StrEnum needs Python 3.10 ────
# The 3.10/3.11/3.12 matrix added in C4 immediately caught a real bug: this
# module did `from enum import StrEnum` unconditionally, but StrEnum is
# stdlib only from Python 3.11 onward — pyproject.toml declares
# `requires-python = ">=3.10"`, so the package could not even be imported on
# its own declared floor. Reproduced directly: `python3.10 -c "from enum
# import StrEnum"` raises ImportError. Fixed with a version-gated stdlib
# backport (no new dependency) that reproduces StrEnum's real __str__
# behavior exactly — verified identical on python3.10 and python3.11 by
# loading the module under both interpreters directly.


class TestC4FollowupVcsServiceStrEnum:
    def test_check_run_status_behaves_like_a_real_str_enum(self):
        from orchestrator.vcs.service import CheckRunConclusion, CheckRunStatus

        assert isinstance(CheckRunStatus.QUEUED, str)
        assert str(CheckRunStatus.QUEUED) == "queued"
        assert CheckRunStatus.QUEUED == "queued"
        assert f"{CheckRunStatus.QUEUED}" == "queued"
        assert str(CheckRunConclusion.SUCCESS) == "success"

    def test_strenum_import_is_version_gated(self):
        """Guards against reverting to a bare `from enum import StrEnum`
        that breaks Python 3.10 again.
        """
        src = inspect.getsource(__import__("orchestrator.vcs.service", fromlist=["_"]))
        assert "sys.version_info" in src
        assert "from enum import StrEnum" in src  # still used on 3.11+
        assert "class StrEnum(str, Enum)" in src  # the <3.11 backport
