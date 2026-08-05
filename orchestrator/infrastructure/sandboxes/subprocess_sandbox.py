"""SubprocessSandbox — rlimits, process-group kill, scrubbed env, temp HOME.

Implements the ``SandboxPort`` contract (``orchestrator.domain.ports``) at
the subprocess tier. This is the isolation boundary for every test execution
that does not use Docker (F-3):

* sensitive environment variables are scrubbed (never leaked to model code)
* deterministic pins are set (E-5: PYTHONHASHSEED, TZ, SOURCE_DATE_EPOCH)
* a temporary HOME is provided so model code cannot touch the real home dir
* POSIX rlimits bound address space, process count, and file size
* a timeout kills the whole process group (no zombies, no orphan workers)
* the child inherits no unexpected file descriptors

Limitations (documented, by design): this tier does not provide network
isolation or kernel-level containment — use :class:`DockerSandbox` for
untrusted code that needs those guarantees.
"""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import tempfile
from pathlib import Path

from ...domain.testing_models import IsolationLevel

logger = logging.getLogger(__name__)

# Sensitive env markers scrubbed before any model-authored code executes.
# Expanded per audit finding #6 (PASSWORD / PRIVATE_KEY / DATABASE_URL).
_SENSITIVE_ENV_MARKERS = (
    "API_KEY",
    "SECRET",
    "TOKEN",
    "PASSWORD",
    "PRIVATE_KEY",
    "DATABASE_URL",
)

# Determinism pins (E-5). LC_ALL is POSIX-only; Windows Python ignores it.
_DETERMINISM_PINS = {
    "PYTHONHASHSEED": "0",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONUTF8": "1",
    "SOURCE_DATE_EPOCH": "0",
    "TZ": "UTC",
}

# rlimits applied in the child on POSIX before exec.
_POSIX_RLIMITS = {
    "RLIMIT_AS": 512 * 1024 * 1024,  # 512 MiB address space (memory bomb cap)
    "RLIMIT_NPROC": 256,  # fork bomb cap
    "RLIMIT_FSIZE": 10 * 1024 * 1024,  # 10 MiB file size cap
}


def _child_preexec() -> None:
    """Apply rlimits in the child process (POSIX only, before exec)."""
    try:
        import resource

        for name, limit in _POSIX_RLIMITS.items():
            rlimit = getattr(resource, name, None)
            if rlimit is not None:
                resource.setrlimit(rlimit, (limit, limit))
    except Exception:  # pragma: no cover - resource is stdlib but best-effort
        logger.warning("Failed to apply rlimits in sandbox child", exc_info=True)


def _scrub_env(source: dict[str, str]) -> dict[str, str]:
    """Return *source* minus sensitive keys plus determinism pins."""
    clean = {
        k: v for k, v in source.items() if not any(m in k.upper() for m in _SENSITIVE_ENV_MARKERS)
    }
    clean.update(_DETERMINISM_PINS)
    if platform.system() != "Windows":
        clean["LC_ALL"] = "C.UTF-8"
    return clean


class SubprocessSandbox:
    """Subprocess-tier isolation: rlimits, timeout, scrubbed env, temp HOME."""

    level = IsolationLevel.SUBPROCESS

    def __init__(self) -> None:
        """Initialize the sandbox (no external resources acquired)."""
        self._clean_env: dict[str, str] | None = None

    async def exec(
        self,
        argv: list[str],
        *,
        cwd: Path,
        env: dict[str, str] | None = None,
        timeout_s: float = 60.0,
    ) -> tuple[int, str, str]:
        """Execute *argv* with subprocess-tier isolation.

        Args:
            argv: Command and arguments (list form — never a shell string).
            cwd: Working directory for the child.
            env: Optional base environment; defaults to a scrubbed copy of
                ``os.environ``.
            timeout_s: Maximum wall-clock time; on expiry the whole process
                group is killed.

        Returns:
            Tuple of (exit_code, stdout_str, stderr_str). On timeout the
            exit code is ``-1`` and stderr explains the timeout.
        """
        base_env = dict(os.environ) if not env else dict(env)
        clean_env = _scrub_env(base_env)

        # Temp HOME so model code cannot read/write the real home directory.
        home_dir = tempfile.mkdtemp(prefix="orch-sandbox-home-")
        clean_env["HOME"] = home_dir
        if platform.system() == "Windows":
            clean_env["USERPROFILE"] = home_dir
        clean_env["TMPDIR"] = clean_env["TEMP"] = clean_env["TMP"] = home_dir

        # New process group so a timeout kill takes the whole tree down.
        kwargs: dict = {}
        if platform.system() == "Windows":
            kwargs["creationflags"] = getattr(os, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            kwargs["start_new_session"] = True
            kwargs["preexec_fn"] = _child_preexec

        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                cwd=str(cwd),
                env=clean_env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.DEVNULL,
                **kwargs,
            )
        except FileNotFoundError:
            return 127, "", f"Command not found: {argv[0]}"
        except OSError as exc:
            return 126, "", f"Failed to launch {argv[0]}: {exc}"

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout_s
            )
        except asyncio.TimeoutError:
            await _kill_process_tree(proc)
            return -1, "", f"Timeout after {timeout_s}s"
        finally:
            try:
                import shutil

                shutil.rmtree(home_dir, ignore_errors=True)
            except Exception:  # pragma: no cover - best-effort cleanup
                pass

        stdout = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
        stderr = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""
        return proc.returncode or 0, stdout, stderr


async def _kill_process_tree(proc: asyncio.subprocess.Process) -> None:
    """Kill a subprocess and its whole process group."""
    import signal

    try:
        if platform.system() == "Windows":
            try:
                kill_proc = await asyncio.create_subprocess_exec(
                    "taskkill",
                    "/F",
                    "/T",
                    "/PID",
                    str(proc.pid),
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await asyncio.wait_for(kill_proc.communicate(), timeout=10)
                return
            except (OSError, asyncio.TimeoutError):
                pass  # fall through to direct kill
            proc.kill()
        else:
            sigkill = getattr(signal, "SIGKILL", None)
            if sigkill is not None:
                try:
                    os.killpg(proc.pid, sigkill)
                    return
                except (ProcessLookupError, PermissionError, OSError):
                    pass
            proc.kill()
    except ProcessLookupError:
        pass
    finally:
        try:
            await proc.wait()
        except (ProcessLookupError, OSError):
            pass
