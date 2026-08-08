"""Boot-and-wait helper for live probes (Phase 7, P-3).

Boots a workspace's ``run_command`` as a background subprocess, polls a
health path until it answers, and guarantees teardown on every exit path
(boot failure, probe exception, or normal completion) — a leaked listener
after a probe run is the failure this module exists to prevent.

Subprocess tier only. Docker-tier boot (health-honesty probe) manages its
own container lifecycle directly in ``live_probes.py`` — the two have
almost nothing in common beyond "start something, wait for it".
"""

from __future__ import annotations

import asyncio
import os
import platform
import shlex
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

_PORT_AWARE_TOOLS = {"uvicorn", "flask", "gunicorn"}


def free_port() -> int:
    """Ask the OS for an unused localhost port. Racy under heavy concurrency
    (ponytail: TOCTOU between close() and the child's bind()) — acceptable
    for probe-scale, non-concurrent boots; a collision surfaces as a boot
    failure (VIOLATED with the port-in-use log), never a silent hang.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    finally:
        s.close()


def _augment_argv_with_port(argv: list[str], port: int) -> list[str]:
    if not argv:
        return argv
    tool = Path(argv[0]).stem
    if tool in _PORT_AWARE_TOOLS:
        return [*argv, "--port", str(port)]
    return argv  # skipped: per-framework port injection beyond uvicorn/flask/gunicorn;
    # add a case when a real archetype needs it — PORT env var below covers the rest.


async def _wait_for_http_ok(url: str, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s

    def _get():
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:
                return resp.status == 200
        except urllib.error.HTTPError as exc:
            return exc.code == 200
        except Exception:
            return False

    while time.monotonic() < deadline:
        if await asyncio.to_thread(_get):
            return True
        await asyncio.sleep(0.2)
    return False


async def http_get(url: str, timeout_s: float = 3.0) -> tuple[int | None, str]:
    """GET *url*; never raises. Returns ``(status, body)`` or ``(None, error)``."""

    def _get():
        try:
            with urllib.request.urlopen(url, timeout=timeout_s) as resp:
                return resp.status, resp.read().decode("utf-8", errors="replace")[:1000]
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:1000] if exc.fp else ""
            return exc.code, body
        except Exception as exc:
            return None, str(exc)

    return await asyncio.to_thread(_get)


async def _kill_and_collect(proc: asyncio.subprocess.Process) -> str:
    try:
        proc.terminate()
    except ProcessLookupError:
        return ""
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5.0)
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5.0)
        except asyncio.TimeoutError:
            return "teardown timed out waiting for process exit"
    out = (stdout or b"").decode("utf-8", errors="replace")
    err = (stderr or b"").decode("utf-8", errors="replace")
    return (out[-2000:] + err[-2000:]).strip()


@dataclass
class BootedApp:
    proc: asyncio.subprocess.Process | None
    port: int
    boot_ok: bool
    base_url: str = ""
    log: str = ""
    _torn_down: bool = field(default=False, repr=False)

    async def teardown(self) -> None:
        if self._torn_down:
            return
        self._torn_down = True
        if self.proc is None or self.proc.returncode is not None:
            return
        await _kill_and_collect(self.proc)


async def boot(
    run_command: str,
    cwd: Path,
    env: dict[str, str] | None = None,
    health_path: str = "/health",
    boot_timeout_s: float = 20.0,
) -> BootedApp:
    """Boot *run_command* in the background; wait for *health_path* to answer 200.

    Never raises — a launch failure (bad command, missing interpreter) and a
    boot timeout both degrade to ``BootedApp(boot_ok=False)`` carrying the
    captured log, per the P-3 design: failure to boot is a genuine VIOLATED
    verdict, not an INDETERMINATE one.
    """
    port = free_port()
    # posix=False on Windows: POSIX-mode shlex treats backslash as an escape
    # character and mangles Windows paths (e.g. "C:\Users\...\python.exe").
    argv = _augment_argv_with_port(
        shlex.split(run_command, posix=platform.system() != "Windows"), port
    )
    full_env = {**os.environ, **(env or {}), "PORT": str(port)}

    kwargs: dict = {}
    if platform.system() == "Windows":
        kwargs["creationflags"] = (
            getattr(asyncio.subprocess, "CREATE_NEW_PROCESS_GROUP", 0) or 0x00000200
        )
    else:
        kwargs["start_new_session"] = True

    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(cwd),
            env=full_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            **kwargs,
        )
    except (FileNotFoundError, OSError) as exc:
        return BootedApp(proc=None, port=port, boot_ok=False, log=f"failed to launch: {exc}")

    base_url = f"http://127.0.0.1:{port}"
    boot_ok = await _wait_for_http_ok(f"{base_url}{health_path}", boot_timeout_s)
    if not boot_ok:
        log = await _kill_and_collect(proc)
        return BootedApp(proc=None, port=port, boot_ok=False, base_url=base_url, log=log)
    return BootedApp(proc=proc, port=port, boot_ok=True, base_url=base_url)


__all__ = ["BootedApp", "boot", "free_port", "http_get"]
