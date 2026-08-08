"""Docker-tier live probe test (Phase 7, P-3).

Docker-skipped when the daemon isn't reachable, per the plan's own testing
strategy for P-3. Builds a real image from the stdlib fixture app so the
container actually reaches Docker's 'healthy' status through a real HTTP
HEALTHCHECK — the thing DockerBootsHealthyProbe exists to observe.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from orchestrator.domain.ports import ProbeContext
from orchestrator.domain.readiness import AppArchetype, ProbeResult
from orchestrator.domain.testing_models import Workspace
from orchestrator.infrastructure.readiness_probes.live_probes import DockerBootsHealthyProbe

_FIXTURE_SRC = Path(__file__).parent.parent / "fixtures" / "readiness_probes" / "fixture_app.py"

pytestmark = pytest.mark.skipif(shutil.which("docker") is None, reason="docker CLI not available")

CTX = ProbeContext(archetype=AppArchetype.PYTHON_SERVICE)


def _build_context(tmp_path: Path, healthcheck_port: int = 8000) -> Path:
    shutil.copy(_FIXTURE_SRC, tmp_path / "fixture_app.py")
    (tmp_path / "Dockerfile").write_text(
        "FROM python:3.11-slim\n"
        "WORKDIR /app\n"
        "COPY fixture_app.py .\n"
        "ENV PORT=8000\n"
        "EXPOSE 8000\n"
        "HEALTHCHECK --interval=2s --timeout=2s --start-period=2s --retries=3 "
        f"CMD python -c \"import urllib.request; urllib.request.urlopen('http://localhost:{healthcheck_port}/health', timeout=1)\" || exit 1\n"
        'CMD ["python", "fixture_app.py"]\n'
    )
    return tmp_path


@pytest.mark.integration
async def test_container_reaches_healthy(tmp_path):
    root = _build_context(tmp_path)
    outcome = await DockerBootsHealthyProbe().probe(Workspace(root=root, framework="python"), CTX)
    assert outcome.result == ProbeResult.SATISFIED


@pytest.mark.integration
async def test_container_never_healthy_when_healthcheck_targets_wrong_port(tmp_path):
    # HEALTHCHECK dials a port nothing listens on inside the container —
    # connection refused every time, guaranteeing it never reports healthy
    # (a wrong *path* isn't reliable here: the fixture's catch-all handler
    # returns 200 for any unrecognized path, same as a real app's SPA
    # fallback route might).
    root = _build_context(tmp_path, healthcheck_port=8001)
    outcome = await DockerBootsHealthyProbe().probe(Workspace(root=root, framework="python"), CTX)
    assert outcome.result == ProbeResult.VIOLATED


@pytest.mark.unit
async def test_not_applicable_without_dockerfile(tmp_path):
    outcome = await DockerBootsHealthyProbe().probe(
        Workspace(root=tmp_path, framework="python"), CTX
    )
    assert outcome.result == ProbeResult.NOT_APPLICABLE
