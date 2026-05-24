"""
ShellTool — Execute shell commands in isolated environments
=============================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Capability 4 of the Agentic System Implementation Plan.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from .base import Tool, ToolPermission, ToolResult

logger = logging.getLogger("orchestrator.tools.shell_tool")


class ShellTool(Tool):
    """Execute shell commands with timeout and output capture."""

    name = "shell"
    description = "Execute shell commands (install deps, run scripts, etc.)"
    required_permissions = [ToolPermission.SHELL_EXECUTE]

    async def execute(self, params: dict) -> ToolResult:
        cmd = params.get("command", "")
        if not cmd:
            return ToolResult(success=False, output="No command provided")
        cwd = Path(params.get("cwd", "."))
        timeout = params.get("timeout", 120)

        try:
            process = await asyncio.create_subprocess_shell(
                cmd, cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout,
            )
            return ToolResult(
                success=process.returncode == 0,
                output=stdout.decode() or stderr.decode(),
                metrics={"exit_code": process.returncode},
            )
        except asyncio.TimeoutError:
            return ToolResult(success=False, output=f"Command timed out ({timeout}s)")
        except Exception as exc:
            return ToolResult(success=False, output=str(exc))

    def validate_params(self, params: dict) -> bool:
        return bool(params.get("command"))
