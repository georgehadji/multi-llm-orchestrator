"""Tests for supply-chain hardening probes (Phase 7, P-2)."""

from __future__ import annotations

import pytest

from orchestrator.domain.ports import ProbeContext
from orchestrator.domain.readiness import AppArchetype, ProbeResult
from orchestrator.domain.testing_models import Workspace
from orchestrator.infrastructure.readiness_probes.supply_chain_probes import (
    ActionsShaPinnedProbe,
    BaseImageDigestPinnedProbe,
    LockFilePresentProbe,
    SbomPresentProbe,
)

CTX = ProbeContext(archetype=AppArchetype.PYTHON_SERVICE)
SHA = "a" * 40


def _ws(root):
    return Workspace(root=root, framework="python")


@pytest.mark.unit
class TestLockFilePresentProbe:
    async def test_not_applicable_without_pyproject(self, tmp_path):
        outcome = await LockFilePresentProbe().probe(_ws(tmp_path), CTX)
        assert outcome.result == ProbeResult.NOT_APPLICABLE

    async def test_satisfied_with_uv_lock(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
        (tmp_path / "uv.lock").write_text("")
        outcome = await LockFilePresentProbe().probe(_ws(tmp_path), CTX)
        assert outcome.result == ProbeResult.SATISFIED

    async def test_violated_without_lock(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
        outcome = await LockFilePresentProbe().probe(_ws(tmp_path), CTX)
        assert outcome.result == ProbeResult.VIOLATED


@pytest.mark.unit
class TestBaseImageDigestPinnedProbe:
    async def test_not_applicable_without_dockerfile(self, tmp_path):
        outcome = await BaseImageDigestPinnedProbe().probe(_ws(tmp_path), CTX)
        assert outcome.result == ProbeResult.NOT_APPLICABLE

    async def test_satisfied_when_digest_pinned(self, tmp_path):
        (tmp_path / "Dockerfile").write_text(f"FROM python@sha256:{'0' * 64}\n")
        outcome = await BaseImageDigestPinnedProbe().probe(_ws(tmp_path), CTX)
        assert outcome.result == ProbeResult.SATISFIED

    async def test_violated_when_tag_only(self, tmp_path):
        (tmp_path / "Dockerfile").write_text("FROM python:3.12-slim\n")
        outcome = await BaseImageDigestPinnedProbe().probe(_ws(tmp_path), CTX)
        assert outcome.result == ProbeResult.VIOLATED


@pytest.mark.unit
class TestActionsShaPinnedProbe:
    def _write_workflow(self, tmp_path, body):
        wf_dir = tmp_path / ".github" / "workflows"
        wf_dir.mkdir(parents=True)
        (wf_dir / "ci.yml").write_text(body)

    async def test_not_applicable_when_no_workflows(self, tmp_path):
        outcome = await ActionsShaPinnedProbe().probe(_ws(tmp_path), CTX)
        assert outcome.result == ProbeResult.NOT_APPLICABLE

    async def test_satisfied_when_sha_pinned(self, tmp_path):
        self._write_workflow(
            tmp_path,
            f"jobs:\n  x:\n    steps:\n      - uses: actions/checkout@{SHA}\n",
        )
        outcome = await ActionsShaPinnedProbe().probe(_ws(tmp_path), CTX)
        assert outcome.result == ProbeResult.SATISFIED

    async def test_violated_when_tag_pinned(self, tmp_path):
        self._write_workflow(
            tmp_path, "jobs:\n  x:\n    steps:\n      - uses: actions/checkout@v4\n"
        )
        outcome = await ActionsShaPinnedProbe().probe(_ws(tmp_path), CTX)
        assert outcome.result == ProbeResult.VIOLATED

    async def test_indeterminate_when_malformed(self, tmp_path):
        self._write_workflow(tmp_path, "jobs: [unterminated")
        outcome = await ActionsShaPinnedProbe().probe(_ws(tmp_path), CTX)
        assert outcome.result == ProbeResult.INDETERMINATE


@pytest.mark.unit
class TestSbomPresentProbe:
    async def test_violated_when_absent(self, tmp_path):
        outcome = await SbomPresentProbe().probe(_ws(tmp_path), CTX)
        assert outcome.result == ProbeResult.VIOLATED

    async def test_satisfied_when_valid_cyclonedx(self, tmp_path):
        (tmp_path / "sbom.json").write_text('{"bomFormat": "CycloneDX", "components": []}')
        outcome = await SbomPresentProbe().probe(_ws(tmp_path), CTX)
        assert outcome.result == ProbeResult.SATISFIED

    async def test_indeterminate_when_malformed(self, tmp_path):
        (tmp_path / "sbom.json").write_text("{not json")
        outcome = await SbomPresentProbe().probe(_ws(tmp_path), CTX)
        assert outcome.result == ProbeResult.INDETERMINATE
