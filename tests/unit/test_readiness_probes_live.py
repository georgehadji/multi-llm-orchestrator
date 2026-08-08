"""Tests for subprocess-tier live probes (Phase 7, P-3).

Boots the stdlib fixture app at tests/fixtures/readiness_probes/fixture_app.py
under different FIXTURE_MODE values — the satisfying/violating fixtures the
plan's testing strategy calls for, without needing a real framework install.
"""

from __future__ import annotations

import platform
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from orchestrator.domain.ports import ProbeContext
from orchestrator.domain.readiness import AppArchetype, ProbeResult
from orchestrator.domain.testing_models import Workspace
from orchestrator.infrastructure.readiness_probes.live_probes import (
    BootAndServeProbe,
    GracefulShutdownProbe,
    ReadinessEndpointDistinctProbe,
)

_FIXTURE_SRC = Path(__file__).parent.parent / "fixtures" / "readiness_probes" / "fixture_app.py"


@pytest.fixture
def app_dir(tmp_path):
    """Copy the fixture app into tmp_path — avoids shlex-quoting Windows
    paths that contain spaces (this repo's own checkout path does)."""
    dest = tmp_path / "fixture_app.py"
    shutil.copy(_FIXTURE_SRC, dest)
    return tmp_path


def _ctx() -> ProbeContext:
    run_command = f"{sys.executable} fixture_app.py"
    profile = SimpleNamespace(run_command=run_command)
    return ProbeContext(archetype=AppArchetype.PYTHON_SERVICE, profile=profile)


def _ws(root, mode: str = "graceful"):
    return Workspace(root=root, framework="python", env={"FIXTURE_MODE": mode})


@pytest.mark.unit
class TestBootAndServeProbe:
    async def test_not_applicable_without_run_command(self, tmp_path):
        ctx = ProbeContext(archetype=AppArchetype.PYTHON_SERVICE, profile=None)
        outcome = await BootAndServeProbe().probe(_ws(tmp_path), ctx)
        assert outcome.result == ProbeResult.NOT_APPLICABLE

    async def test_satisfied_when_app_boots(self, app_dir):
        outcome = await BootAndServeProbe().probe(_ws(app_dir, "graceful"), _ctx())
        assert outcome.result == ProbeResult.SATISFIED

    async def test_violated_when_command_missing(self, app_dir):
        profile = SimpleNamespace(run_command="definitely-not-a-real-executable-xyz")
        ctx = ProbeContext(archetype=AppArchetype.PYTHON_SERVICE, profile=profile)
        outcome = await BootAndServeProbe().probe(_ws(app_dir, "graceful"), ctx)
        assert outcome.result == ProbeResult.VIOLATED


@pytest.mark.unit
class TestReadinessEndpointDistinctProbe:
    async def test_satisfied_when_distinct(self, app_dir):
        outcome = await ReadinessEndpointDistinctProbe().probe(_ws(app_dir, "graceful"), _ctx())
        assert outcome.result == ProbeResult.SATISFIED

    async def test_violated_when_missing(self, app_dir):
        outcome = await ReadinessEndpointDistinctProbe().probe(_ws(app_dir, "no_ready"), _ctx())
        assert outcome.result == ProbeResult.VIOLATED

    async def test_violated_when_aliased_to_health(self, app_dir):
        outcome = await ReadinessEndpointDistinctProbe().probe(_ws(app_dir, "lying_ready"), _ctx())
        assert outcome.result == ProbeResult.VIOLATED


@pytest.mark.unit
@pytest.mark.skipif(
    platform.system() == "Windows",
    reason="SIGTERM is a hard kill on Windows, not a deliverable signal",
)
class TestGracefulShutdownProbe:
    async def test_satisfied_when_app_handles_sigterm(self, app_dir):
        outcome = await GracefulShutdownProbe().probe(_ws(app_dir, "graceful"), _ctx())
        assert outcome.result == ProbeResult.SATISFIED

    async def test_violated_when_app_ignores_sigterm(self, app_dir):
        outcome = await GracefulShutdownProbe().probe(_ws(app_dir, "stubborn"), _ctx())
        assert outcome.result == ProbeResult.VIOLATED
