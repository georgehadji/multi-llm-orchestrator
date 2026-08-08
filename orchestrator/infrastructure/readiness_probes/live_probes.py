"""Live probes: boot the workspace, observe real runtime behavior (Phase 7, P-3).

Subprocess-tier probes boot ``ctx.profile.run_command`` in the background
via ``_boot.py`` and are teardown-guaranteed regardless of outcome. The
Docker-tier probe manages its own container lifecycle — building and
polling a container's health status has almost nothing in common with
booting a bare subprocess.

A boot failure is ``VIOLATED`` here, never ``INDETERMINATE`` — failure to
boot is a genuine verdict (plan §P-3), unlike a merely unparseable static
artifact.
"""

from __future__ import annotations

import asyncio
import shutil
import time
import uuid

from ...domain.readiness import Evidence, ProbeResult, RequirementOutcome
from ._base import LiveProbe
from ._boot import boot, http_get

_BOOT_TIMEOUT_S = 20.0
_SHUTDOWN_GRACE_S = 5.0
_DOCKER_HEALTHY_TIMEOUT_S = 45.0


def _no_run_command(requirement_id: str) -> RequirementOutcome:
    return RequirementOutcome(
        requirement_id=requirement_id,
        result=ProbeResult.NOT_APPLICABLE,
        evidence=(Evidence(probe=requirement_id, detail="no run_command on ctx.profile"),),
    )


class BootAndServeProbe(LiveProbe):
    """The app boots and answers /health within a bounded timeout."""

    requirement_id = "live.boot-and-serve"

    async def check(self, workspace, ctx) -> RequirementOutcome:
        run_command = getattr(ctx.profile, "run_command", None)
        if not run_command:
            return _no_run_command(self.requirement_id)
        app = await boot(
            run_command, workspace.root, env=workspace.env, boot_timeout_s=_BOOT_TIMEOUT_S
        )
        try:
            if not app.boot_ok:
                return RequirementOutcome(
                    requirement_id=self.requirement_id,
                    result=ProbeResult.VIOLATED,
                    evidence=(
                        Evidence(
                            probe=self.requirement_id,
                            detail=f"did not answer /health within {_BOOT_TIMEOUT_S}s",
                            raw=app.log,
                        ),
                    ),
                )
            return RequirementOutcome(
                requirement_id=self.requirement_id,
                result=ProbeResult.SATISFIED,
                evidence=(
                    Evidence(probe=self.requirement_id, detail=f"healthy at {app.base_url}/health"),
                ),
            )
        finally:
            await app.teardown()


class ReadinessEndpointDistinctProbe(LiveProbe):
    """/ready must exist and must not simply alias /health.

    A generic black-box probe cannot prove /ready reflects *real* dependency
    state for an arbitrary app — that needs a fixture with a controllable
    dependency (see the unit tests). What it can prove for any app: the
    endpoint exists and isn't a copy-pasted /health handler, which is the
    minimum bar for "not vacuous".
    """

    requirement_id = "live.readiness-endpoint-distinct"

    async def check(self, workspace, ctx) -> RequirementOutcome:
        run_command = getattr(ctx.profile, "run_command", None)
        if not run_command:
            return _no_run_command(self.requirement_id)
        app = await boot(
            run_command, workspace.root, env=workspace.env, boot_timeout_s=_BOOT_TIMEOUT_S
        )
        try:
            if not app.boot_ok:
                return RequirementOutcome(
                    requirement_id=self.requirement_id,
                    result=ProbeResult.VIOLATED,
                    evidence=(
                        Evidence(probe=self.requirement_id, detail="app did not boot", raw=app.log),
                    ),
                )
            health_status, health_body = await http_get(f"{app.base_url}/health")
            ready_status, ready_body = await http_get(f"{app.base_url}/ready")
            if ready_status == 404:
                return RequirementOutcome(
                    requirement_id=self.requirement_id,
                    result=ProbeResult.VIOLATED,
                    evidence=(Evidence(probe=self.requirement_id, detail="/ready returns 404"),),
                )
            if (ready_status, ready_body) == (health_status, health_body):
                return RequirementOutcome(
                    requirement_id=self.requirement_id,
                    result=ProbeResult.VIOLATED,
                    evidence=(
                        Evidence(
                            probe=self.requirement_id,
                            detail="/ready response is byte-identical to /health — looks aliased, not a real dependency check",
                        ),
                    ),
                )
            return RequirementOutcome(
                requirement_id=self.requirement_id,
                result=ProbeResult.SATISFIED,
                evidence=(
                    Evidence(
                        probe=self.requirement_id,
                        detail=f"/ready={ready_status}, /health={health_status}",
                    ),
                ),
            )
        finally:
            await app.teardown()


class GracefulShutdownProbe(LiveProbe):
    """SIGTERM during an in-flight request: the request completes and the
    process exits within the grace period, without escalating to SIGKILL.

    Windows note: ``Popen.send_signal(SIGTERM)``/``terminate()`` maps to
    ``TerminateProcess`` there — an immediate hard kill, not a deliverable
    signal a Python app can catch. This probe still runs on Windows but
    will VIOLATE any app, well-behaved or not; treat a Windows verdict as
    INDETERMINATE-in-spirit and trust the POSIX result.
    """

    requirement_id = "live.graceful-shutdown"

    async def check(self, workspace, ctx) -> RequirementOutcome:
        run_command = getattr(ctx.profile, "run_command", None)
        if not run_command:
            return _no_run_command(self.requirement_id)
        app = await boot(
            run_command, workspace.root, env=workspace.env, boot_timeout_s=_BOOT_TIMEOUT_S
        )
        try:
            if not app.boot_ok:
                return RequirementOutcome(
                    requirement_id=self.requirement_id,
                    result=ProbeResult.VIOLATED,
                    evidence=(
                        Evidence(probe=self.requirement_id, detail="app did not boot", raw=app.log),
                    ),
                )
            request_task = asyncio.create_task(http_get(f"{app.base_url}/health"))
            await asyncio.sleep(0.05)
            proc = app.proc
            proc.terminate()
            request_status, _ = await request_task
            start = time.monotonic()
            try:
                await asyncio.wait_for(proc.wait(), timeout=_SHUTDOWN_GRACE_S)
                exited_gracefully = True
            except asyncio.TimeoutError:
                exited_gracefully = False
            elapsed = time.monotonic() - start
            if not exited_gracefully or request_status is None:
                return RequirementOutcome(
                    requirement_id=self.requirement_id,
                    result=ProbeResult.VIOLATED,
                    evidence=(
                        Evidence(
                            probe=self.requirement_id,
                            detail=f"exited_gracefully={exited_gracefully}, in-flight request status={request_status}, elapsed={elapsed:.1f}s",
                        ),
                    ),
                )
            return RequirementOutcome(
                requirement_id=self.requirement_id,
                result=ProbeResult.SATISFIED,
                evidence=(
                    Evidence(
                        probe=self.requirement_id,
                        detail=f"exited in {elapsed:.1f}s, in-flight request completed",
                    ),
                ),
            )
        finally:
            await app.teardown()


class DockerBootsHealthyProbe(LiveProbe):
    """The emitted container reaches Docker's own 'healthy' status.

    Whether the HEALTHCHECK command is *capable* of detecting failure is a
    static property (parseable from the Dockerfile — see
    static_probes.py::DockerfileHealthcheckIsHttpProbe). This probe covers
    the other half: does it actually go healthy under real operation.
    """

    requirement_id = "live.docker-boots-healthy"

    async def check(self, workspace, ctx) -> RequirementOutcome:
        dockerfile = workspace.root / "Dockerfile"
        if not dockerfile.is_file():
            return RequirementOutcome(
                requirement_id=self.requirement_id, result=ProbeResult.NOT_APPLICABLE
            )
        if shutil.which("docker") is None:
            return RequirementOutcome(
                requirement_id=self.requirement_id,
                result=ProbeResult.NOT_APPLICABLE,
                evidence=(Evidence(probe=self.requirement_id, detail="docker CLI not available"),),
            )
        tag = f"orch-readiness-{uuid.uuid4().hex[:12]}"
        name = tag
        build = await asyncio.create_subprocess_exec(
            "docker",
            "build",
            "-t",
            tag,
            str(workspace.root),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        build_out, build_err = await build.communicate()
        if build.returncode != 0:
            return RequirementOutcome(
                requirement_id=self.requirement_id,
                result=ProbeResult.NOT_APPLICABLE,
                evidence=(
                    Evidence(
                        probe=self.requirement_id,
                        detail="docker build failed — cannot characterize this workspace",
                        raw=(build_err or b"").decode("utf-8", errors="replace")[-2000:],
                    ),
                ),
            )
        try:
            run = await asyncio.create_subprocess_exec(
                "docker",
                "run",
                "-d",
                "--name",
                name,
                tag,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            run_out, run_err = await run.communicate()
            if run.returncode != 0:
                return RequirementOutcome(
                    requirement_id=self.requirement_id,
                    result=ProbeResult.VIOLATED,
                    evidence=(
                        Evidence(
                            probe=self.requirement_id,
                            detail="container failed to start",
                            raw=(run_err or b"").decode("utf-8", errors="replace")[-2000:],
                        ),
                    ),
                )
            status = await self._wait_for_health_status(name, _DOCKER_HEALTHY_TIMEOUT_S)
            if status != "healthy":
                logs = await self._container_logs(name)
                return RequirementOutcome(
                    requirement_id=self.requirement_id,
                    result=ProbeResult.VIOLATED,
                    evidence=(
                        Evidence(
                            probe=self.requirement_id,
                            detail=f"never reached healthy (last status: {status!r})",
                            raw=logs,
                        ),
                    ),
                )
            return RequirementOutcome(
                requirement_id=self.requirement_id,
                result=ProbeResult.SATISFIED,
                evidence=(
                    Evidence(probe=self.requirement_id, detail="container reported healthy"),
                ),
            )
        finally:
            await self._teardown_container(name)
            await self._remove_image(tag)

    async def _wait_for_health_status(self, name: str, timeout_s: float) -> str:
        deadline = time.monotonic() + timeout_s
        status = "unknown"
        while time.monotonic() < deadline:
            proc = await asyncio.create_subprocess_exec(
                "docker",
                "inspect",
                "--format={{.State.Health.Status}}",
                name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            out, _ = await proc.communicate()
            status = out.decode("utf-8", errors="replace").strip()
            if status == "healthy":
                return status
            await asyncio.sleep(1.0)
        return status

    async def _container_logs(self, name: str) -> str:
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "logs",
            "--tail",
            "100",
            name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await proc.communicate()
        return (out + err).decode("utf-8", errors="replace")[-2000:]

    async def _teardown_container(self, name: str) -> None:
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "rm",
            "-f",
            name,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.communicate()

    async def _remove_image(self, tag: str) -> None:
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "rmi",
            "-f",
            tag,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.communicate()


LIVE_PROBES = (
    BootAndServeProbe(),
    ReadinessEndpointDistinctProbe(),
    GracefulShutdownProbe(),
    DockerBootsHealthyProbe(),
)

__all__ = [
    "BootAndServeProbe",
    "ReadinessEndpointDistinctProbe",
    "GracefulShutdownProbe",
    "DockerBootsHealthyProbe",
    "LIVE_PROBES",
]
