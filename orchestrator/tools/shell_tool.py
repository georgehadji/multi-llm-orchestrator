"""
ShellTool — Execute allowlisted commands without a host shell
==============================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Capability 4 of the Agentic System Implementation Plan.

SEC-003: this tool used to pass ``params["command"]`` straight to
``asyncio.create_subprocess_shell`` with a caller-controlled ``cwd``. The
permission metadata labelled the capability but sandboxed nothing, so any path
that could invoke the tool — a prompt-injected model, a plugin, a remote
request — had arbitrary command execution as the process account.

The policy now is:

* disabled unless ``ORCHESTRATOR_SHELL_TOOL_ENABLED`` is explicitly set;
* argv only, executed with ``create_subprocess_exec`` — never a shell string;
* the executable must be on `DEFAULT_ALLOWED_EXECUTABLES`;
* ``cwd`` is contained under the workspace root;
* output is capped and the process is killed on timeout.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
from pathlib import Path

from orchestrator.safety.secure_execution import PathTraversalError, SecurePath

from .base import Tool, ToolPermission, ToolResult

logger = logging.getLogger("orchestrator.tools.shell_tool")

ENABLED_ENV = "ORCHESTRATOR_SHELL_TOOL_ENABLED"

_TRUTHY = frozenset({"1", "true", "yes", "on"})

#: Build and test tooling only. Deliberately excludes shells (`bash`, `sh`,
#: `cmd`, `powershell`), network clients (`curl`, `wget`, `nc`, `ssh`) and
#: destructive utilities (`rm`, `del`, `mv`) — an allowlist containing a shell
#: or a downloader is not an allowlist.
DEFAULT_ALLOWED_EXECUTABLES: frozenset[str] = frozenset(
    {
        "python",
        "python3",
        "pytest",
        "ruff",
        "black",
        "mypy",
        "node",
        "npm",
        "npx",
        "go",
        "cargo",
        "git",
        "echo",
        "ls",
        "cat",
    }
)

#: Characters that only mean anything to a shell. argv never needs them, and
#: their presence signals the caller expected shell semantics — so refuse
#: rather than silently pass them through as a literal argument.
_SHELL_METACHARACTERS = frozenset(";&|`$><\n\r")

#: Cap captured output so a chatty command cannot exhaust memory.
_MAX_OUTPUT_BYTES = 256 * 1024


def _enabled() -> bool:
    return os.getenv(ENABLED_ENV, "").strip().lower() in _TRUTHY


def _deny(reason: str, message: str) -> ToolResult:
    logger.warning("shell tool denied (%s): %s", reason, message)
    return ToolResult(success=False, output=message, metrics={"denied_reason": reason})


class ShellTool(Tool):
    """Execute an allowlisted command as argv, with timeout and output capture."""

    name = "shell"
    description = "Execute allowlisted build/test commands (no shell, opt-in only)"
    required_permissions = [ToolPermission.SHELL_EXECUTE]

    allowed_executables: frozenset[str] = DEFAULT_ALLOWED_EXECUTABLES

    async def execute(self, params: dict) -> ToolResult:
        argv, error = self._resolve_argv(params)
        if error is not None:
            return error

        if not _enabled():
            return _deny(
                "disabled",
                f"Shell execution is disabled. Set {ENABLED_ENV}=true to enable it.",
            )

        executable = Path(argv[0]).name.lower()
        if executable.endswith(".exe"):
            executable = executable[:-4]
        if executable not in self.allowed_executables:
            return _deny(
                "executable_not_allowed",
                f"Executable {executable!r} is not on the allowlist.",
            )

        try:
            cwd = SecurePath(Path.cwd(), str(params.get("cwd", "."))).resolved
        except PathTraversalError:
            return _deny("cwd_outside_workspace", "cwd escapes the workspace root.")

        timeout = params.get("timeout", 120)
        return await self._run(argv, cwd, timeout)

    def _resolve_argv(self, params: dict) -> tuple[list[str], ToolResult | None]:
        """Normalise `argv`/`command` into an argv list, or return a denial."""
        raw_argv = params.get("argv")
        if raw_argv:
            if not isinstance(raw_argv, (list, tuple)) or not all(
                isinstance(part, str) for part in raw_argv
            ):
                return [], _deny("malformed_argv", "argv must be a list of strings.")
            return list(raw_argv), None

        cmd = params.get("command", "")
        if not cmd:
            # Contract preserved for existing callers and tests.
            return [], ToolResult(success=False, output="No command provided")

        if set(cmd) & _SHELL_METACHARACTERS:
            return [], _deny(
                "shell_metacharacter",
                "Shell metacharacters are not permitted; pass argv instead.",
            )

        try:
            parsed = shlex.split(cmd)
        except ValueError as exc:
            return [], _deny("unparseable_command", f"Could not parse command: {exc}")

        if not parsed:
            return [], ToolResult(success=False, output="No command provided")
        return parsed, None

    async def _run(self, argv: list[str], cwd: Path, timeout: float) -> ToolResult:
        process = None
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout,
            )
            output = (stdout.decode(errors="replace") or stderr.decode(errors="replace"))[
                :_MAX_OUTPUT_BYTES
            ]
            return ToolResult(
                success=process.returncode == 0,
                output=output,
                metrics={"exit_code": process.returncode},
            )
        except TimeoutError:
            # The original left the process running after reporting a timeout.
            if process is not None:
                await self._terminate(process)
            return ToolResult(success=False, output=f"Command timed out ({timeout}s)")
        except FileNotFoundError:
            return _deny("executable_not_found", f"Executable not found: {argv[0]!r}")
        except Exception as exc:
            logger.exception("shell tool failed")
            return ToolResult(success=False, output=str(exc))

    @staticmethod
    async def _terminate(process: asyncio.subprocess.Process) -> None:
        """Kill a timed-out process, escalating if it ignores termination."""
        try:
            process.terminate()
            await asyncio.wait_for(process.wait(), timeout=5)
        except TimeoutError:
            try:
                process.kill()
                await process.wait()
            except ProcessLookupError:
                pass
        except ProcessLookupError:
            pass

    def validate_params(self, params: dict) -> bool:
        return bool(params.get("command") or params.get("argv"))
