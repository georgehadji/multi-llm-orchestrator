"""Boots the emitted service and runs it through P-3's own live probes
(Phase 7, P-6's acceptance bar: "emitted service passes all three P-3 live
probes"). Closes the loop the same way P-4/P-5 closed it against P-2's
static probes.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from orchestrator.domain.ports import ProbeContext
from orchestrator.domain.readiness import AppArchetype, ProbeResult
from orchestrator.domain.testing_models import Workspace
from orchestrator.infrastructure.artifact_emitters.observability_emitter import ObservabilityEmitter
from orchestrator.infrastructure.readiness_probes.live_probes import (
    BootAndServeProbe,
    GracefulShutdownProbe,
    ReadinessEndpointDistinctProbe,
)

pytestmark = pytest.mark.unit


@dataclass
class _Profile:
    project_name: str = "live_probe_app"
    archetype: AppArchetype = AppArchetype.PYTHON_SERVICE


@pytest.fixture
async def emitted_app_dir(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("uvicorn")
    pytest.importorskip("prometheus_client")
    await ObservabilityEmitter().emit(Workspace(root=tmp_path, framework="python"), _Profile())
    return tmp_path


def _ctx() -> ProbeContext:
    run_command = f"{sys.executable} main.py"
    profile = SimpleNamespace(run_command=run_command)
    return ProbeContext(archetype=AppArchetype.PYTHON_SERVICE, profile=profile)


def _ws(root):
    return Workspace(root=root, framework="python", env={"SECRET_KEY": "test-secret"})


_BOOT_RETRY_ATTEMPTS = 3


async def _probe_with_boot_retry(probe, workspace, ctx):
    """A cold ``uvicorn``+``fastapi``+``prometheus_client`` boot on a loaded
    Windows dev box can occasionally brush against the probe's fixed 20s
    budget (observed 15-27s across otherwise-identical runs here) — that is
    scheduler contention, not an application defect. Retry only a
    boot-timeout verdict — ``BootAndServeProbe`` phrases it "did not answer
    /health within ...s" while ``ReadinessEndpointDistinctProbe`` and
    ``GracefulShutdownProbe`` phrase it "app did not boot". Any other
    VIOLATED (wrong content, missing route) is a real finding and must not
    be masked.
    """

    boot_timeout_markers = ("did not boot", "did not answer /health")
    outcome = None
    for attempt in range(_BOOT_RETRY_ATTEMPTS):
        outcome = await probe.probe(workspace, ctx)
        if outcome.result != ProbeResult.VIOLATED:
            return outcome
        if not any(m in e.detail for e in outcome.evidence for m in boot_timeout_markers):
            return outcome
    return outcome


class TestEmittedServicePassesLiveProbes:
    async def test_boot_and_serve(self, emitted_app_dir):
        outcome = await _probe_with_boot_retry(BootAndServeProbe(), _ws(emitted_app_dir), _ctx())
        assert outcome.result == ProbeResult.SATISFIED, outcome.evidence

    async def test_ready_endpoint_is_distinct_from_health(self, emitted_app_dir):
        outcome = await _probe_with_boot_retry(
            ReadinessEndpointDistinctProbe(), _ws(emitted_app_dir), _ctx()
        )
        assert outcome.result == ProbeResult.SATISFIED, outcome.evidence

    @pytest.mark.skipif(
        platform.system() == "Windows",
        reason="SIGTERM is a hard kill on Windows, not a deliverable signal",
    )
    async def test_graceful_shutdown(self, emitted_app_dir):
        outcome = await _probe_with_boot_retry(
            GracefulShutdownProbe(), _ws(emitted_app_dir), _ctx()
        )
        assert outcome.result == ProbeResult.SATISFIED, outcome.evidence
