"""DockerSandbox — --network=none --read-only --cap-drop=ALL isolation.

Implements the ``SandboxPort`` contract at the Docker tier (F-3). Used for
untrusted model-authored code that requires network denial and kernel-level
containment. Selected automatically when the Docker daemon is reachable
unless ``ORCH_SANDBOX_TIER=subprocess`` is set.

Container flags (per implementation plan §F-3):

* ``--network=none``            — no outbound connections
* ``--read-only``               — no writes to the container filesystem
* ``--cap-drop=ALL``            — no kernel capabilities
* ``--pids-limit=256``          — fork bomb cap
* ``--memory=512m``             — memory bomb cap
* ``--tmpfs /tmp``              — writable scratch, tmpfs-backed (memory only)
* workspace bind-mounted ``:ro`` and used as the working directory
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from ...domain.testing_models import IsolationLevel

logger = logging.getLogger(__name__)

# Docker image used to execute model-authored code. Configurable so
# deployments can pre-pull a pinned-by-digest image (Phase 7 / P-4 pins
# *emitted* images; the sandbox image is an operational knob).
DEFAULT_SANDBOX_IMAGE = "python:3.12-slim"

# Container resource caps (mirror the plan's F-3 table).
_MEMORY_LIMIT = "512m"
_PIDS_LIMIT = "256"


def _docker_available() -> bool:
    """Return True if the Docker CLI responds (cached per process)."""
    if _docker_available._checked:  # type: ignore[attr-defined]
        return _docker_available._result  # type: ignore[attr-defined]
    try:
        import subprocess

        result = subprocess.run(  # nosec B603 - fixed arg list, no shell
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        ok = result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        ok = False
    _docker_available._checked = True  # type: ignore[attr-defined]
    _docker_available._result = ok  # type: ignore[attr-defined]
    return ok


_docker_available._checked = False  # type: ignore[attr-defined]
_docker_available._result = False  # type: ignore[attr-defined]


class DockerSandbox:
    """Docker-tier isolation: network denied, read-only FS, resource caps."""

    level = IsolationLevel.DOCKER

    def __init__(self, image: str | None = None) -> None:
        """Initialize the sandbox.

        Args:
            image: Docker image to run. Defaults to
                ``ORCH_SANDBOX_IMAGE`` or ``python:3.12-slim``.
        """
        self.image = image or os.environ.get("ORCH_SANDBOX_IMAGE", DEFAULT_SANDBOX_IMAGE)

    async def exec(
        self,
        argv: list[str],
        *,
        cwd: Path,
        env: dict[str, str] | None = None,
        timeout_s: float = 60.0,
    ) -> tuple[int, str, str]:
        """Run *argv* inside the container with the workspace mounted read-only.

        Args:
            argv: Command and arguments (runs inside the container).
            cwd: Host directory mounted at ``/workspace`` (read-only).
            env: Optional environment variables forwarded as ``-e`` pairs.
                Sensitive host env is never forwarded — callers must pass an
                explicit, scrubbed dict.
            timeout_s: Maximum wall-clock time.

        Returns:
            Tuple of (exit_code, stdout_str, stderr_str).
        """
        cmd = [
            "docker",
            "run",
            "--rm",
            "--network=none",
            "--read-only",
            "--cap-drop=ALL",
            f"--pids-limit={_PIDS_LIMIT}",
            f"--memory={_MEMORY_LIMIT}",
            "--tmpfs",
            "/tmp:rw,size=64m",  # nosec B108 - container-internal tmpfs mount, not a host path
            "--workdir",
            "/workspace",
            "-v",
            f"{cwd}:/workspace:ro",
        ]
        for key, value in (env or {}).items():
            cmd.extend(["-e", f"{key}={value}"])
        cmd.append(self.image)
        cmd.extend(argv)

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.DEVNULL,
            )
        except FileNotFoundError:
            return 127, "", "docker executable not found"
        except OSError as exc:
            return 126, "", f"Failed to launch docker: {exc}"

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout_s
            )
        except asyncio.TimeoutError:
            # docker run --rm cleans the container up; killing docker CLI
            # terminates the attach, and --rm guarantees no leak.
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()
            return -1, "", f"Timeout after {timeout_s}s"

        stdout = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
        stderr = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""
        return proc.returncode or 0, stdout, stderr
