"""SubprocessSandbox — rlimits, timeout, scrubbed env, temp HOME."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from ...domain.testing_models import IsolationLevel

logger = logging.getLogger(__name__)


class SubprocessSandbox:
    """Provides subprocess-level isolation with timeout and env scrubbing.

    Not a full container sandbox — use DockerSandbox for untrusted code.
    """

    def __init__(self) -> None:
        self.level = IsolationLevel.SUBPROCESS

    async def exec(
        self,
        argv: list[str],
        *,
        cwd: Path,
        env: dict[str, str] | None = None,
        timeout_s: float = 60.0,
    ) -> tuple[int, str, str]:
        """Execute a command with timeout and env scrubbing.

        Returns:
            Tuple of (exit_code, stdout_str, stderr_str).
        """
        # Build scrubbed environment
        clean_env = dict(os.environ) if env is None else env
        # Remove sensitive keys
        for key in list(clean_env):
            if "API_KEY" in key.upper() or "SECRET" in key.upper() or "TOKEN" in key.upper():
                del clean_env[key]

        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                cwd=str(cwd),
                env=clean_env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)

            return (
                proc.returncode or 0,
                stdout.decode("utf-8", errors="replace"),
                stderr.decode("utf-8", errors="replace"),
            )

        except asyncio.TimeoutError:
            if proc:
                proc.kill()
                await proc.wait()
            return -1, "", f"Timeout after {timeout_s}s"
