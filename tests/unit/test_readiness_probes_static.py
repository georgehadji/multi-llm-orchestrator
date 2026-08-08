"""Tests for static CI/pyproject probes (Phase 7, P-2)."""

from __future__ import annotations

import pytest

from orchestrator.domain.ports import ProbeContext
from orchestrator.domain.readiness import AppArchetype, ProbeResult
from orchestrator.domain.testing_models import Workspace
from orchestrator.infrastructure.readiness_probes.static_probes import (
    CiNoUnconditionalBypassProbe,
    CiPermissionsReadOnlyProbe,
    DockerfileHealthcheckIsHttpProbe,
    DockerfileNoMutableTagProbe,
    PyprojectParseableProbe,
)

CTX = ProbeContext(archetype=AppArchetype.PYTHON_SERVICE)


def _ws(root):
    return Workspace(root=root, framework="python")


@pytest.mark.unit
class TestPyprojectParseableProbe:
    async def test_satisfied_when_valid(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\n')
        outcome = await PyprojectParseableProbe().probe(_ws(tmp_path), CTX)
        assert outcome.result == ProbeResult.SATISFIED

    async def test_violated_when_absent(self, tmp_path):
        outcome = await PyprojectParseableProbe().probe(_ws(tmp_path), CTX)
        assert outcome.result == ProbeResult.VIOLATED

    async def test_indeterminate_when_malformed(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("not [ valid toml")
        outcome = await PyprojectParseableProbe().probe(_ws(tmp_path), CTX)
        assert outcome.result == ProbeResult.INDETERMINATE


@pytest.mark.unit
class TestCiPermissionsReadOnlyProbe:
    def _write_workflow(self, tmp_path, body):
        wf_dir = tmp_path / ".github" / "workflows"
        wf_dir.mkdir(parents=True)
        (wf_dir / "ci.yml").write_text(body)

    async def test_not_applicable_when_no_workflows(self, tmp_path):
        outcome = await CiPermissionsReadOnlyProbe().probe(_ws(tmp_path), CTX)
        assert outcome.result == ProbeResult.NOT_APPLICABLE

    async def test_satisfied_when_scoped(self, tmp_path):
        self._write_workflow(tmp_path, "permissions:\n  contents: read\njobs: {}\n")
        outcome = await CiPermissionsReadOnlyProbe().probe(_ws(tmp_path), CTX)
        assert outcome.result == ProbeResult.SATISFIED

    async def test_violated_when_missing(self, tmp_path):
        self._write_workflow(tmp_path, "jobs: {}\n")
        outcome = await CiPermissionsReadOnlyProbe().probe(_ws(tmp_path), CTX)
        assert outcome.result == ProbeResult.VIOLATED

    async def test_violated_when_write_all(self, tmp_path):
        self._write_workflow(tmp_path, "permissions: write-all\njobs: {}\n")
        outcome = await CiPermissionsReadOnlyProbe().probe(_ws(tmp_path), CTX)
        assert outcome.result == ProbeResult.VIOLATED

    async def test_indeterminate_when_malformed(self, tmp_path):
        self._write_workflow(tmp_path, "not: valid: yaml: [")
        outcome = await CiPermissionsReadOnlyProbe().probe(_ws(tmp_path), CTX)
        assert outcome.result == ProbeResult.INDETERMINATE


@pytest.mark.unit
class TestCiNoUnconditionalBypassProbe:
    def _write_workflow(self, tmp_path, body):
        wf_dir = tmp_path / ".github" / "workflows"
        wf_dir.mkdir(parents=True)
        (wf_dir / "ci.yml").write_text(body)

    async def test_satisfied_when_no_bypass(self, tmp_path):
        self._write_workflow(tmp_path, "jobs:\n  x:\n    steps:\n      - run: bandit -r .\n")
        outcome = await CiNoUnconditionalBypassProbe().probe(_ws(tmp_path), CTX)
        assert outcome.result == ProbeResult.SATISFIED

    async def test_violated_when_bypassed(self, tmp_path):
        self._write_workflow(
            tmp_path, "jobs:\n  x:\n    steps:\n      - run: bandit -r . || true\n"
        )
        outcome = await CiNoUnconditionalBypassProbe().probe(_ws(tmp_path), CTX)
        assert outcome.result == ProbeResult.VIOLATED
        assert "bandit" in outcome.evidence[0].detail


@pytest.mark.unit
class TestDockerfileNoMutableTagProbe:
    async def test_not_applicable_when_absent(self, tmp_path):
        outcome = await DockerfileNoMutableTagProbe().probe(_ws(tmp_path), CTX)
        assert outcome.result == ProbeResult.NOT_APPLICABLE

    async def test_satisfied_when_pinned(self, tmp_path):
        (tmp_path / "Dockerfile").write_text("FROM python:3.12.7-slim\n")
        outcome = await DockerfileNoMutableTagProbe().probe(_ws(tmp_path), CTX)
        assert outcome.result == ProbeResult.SATISFIED

    async def test_violated_when_latest(self, tmp_path):
        (tmp_path / "Dockerfile").write_text("FROM python:latest\n")
        outcome = await DockerfileNoMutableTagProbe().probe(_ws(tmp_path), CTX)
        assert outcome.result == ProbeResult.VIOLATED

    async def test_violated_when_no_tag(self, tmp_path):
        (tmp_path / "Dockerfile").write_text("FROM python\n")
        outcome = await DockerfileNoMutableTagProbe().probe(_ws(tmp_path), CTX)
        assert outcome.result == ProbeResult.VIOLATED


@pytest.mark.unit
class TestDockerfileHealthcheckIsHttpProbe:
    async def test_not_applicable_without_dockerfile(self, tmp_path):
        outcome = await DockerfileHealthcheckIsHttpProbe().probe(_ws(tmp_path), CTX)
        assert outcome.result == ProbeResult.NOT_APPLICABLE

    async def test_not_applicable_without_healthcheck_directive(self, tmp_path):
        (tmp_path / "Dockerfile").write_text('FROM python:3.12-slim\nCMD ["python", "app.py"]\n')
        outcome = await DockerfileHealthcheckIsHttpProbe().probe(_ws(tmp_path), CTX)
        assert outcome.result == ProbeResult.NOT_APPLICABLE

    async def test_violated_when_import_only(self, tmp_path):
        (tmp_path / "Dockerfile").write_text(
            'FROM python:3.12-slim\nHEALTHCHECK CMD python -c "import requests" || exit 1\n'
        )
        outcome = await DockerfileHealthcheckIsHttpProbe().probe(_ws(tmp_path), CTX)
        assert outcome.result == ProbeResult.VIOLATED

    async def test_satisfied_when_curl(self, tmp_path):
        (tmp_path / "Dockerfile").write_text(
            "FROM python:3.12-slim\nHEALTHCHECK CMD curl -f http://localhost:8000/health || exit 1\n"
        )
        outcome = await DockerfileHealthcheckIsHttpProbe().probe(_ws(tmp_path), CTX)
        assert outcome.result == ProbeResult.SATISFIED
