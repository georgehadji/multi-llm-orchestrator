"""Tests for the Docker artifact emitter (Phase 7, P-4)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from unittest.mock import patch

import pytest

from orchestrator.domain.readiness import AppArchetype
from orchestrator.domain.testing_models import Workspace
from orchestrator.infrastructure.artifact_emitters._digest import DigestResult
from orchestrator.infrastructure.artifact_emitters._lockfile import LockFileResult
from orchestrator.infrastructure.artifact_emitters.docker_emitter import DockerEmitter


@dataclass
class _Profile:
    project_name: str = "sample_app"
    run_command: str = "python -m uvicorn app:app --host 0.0.0.0"


def _ws(root):
    return Workspace(root=root, framework="python")


@pytest.mark.unit
class TestApplicability:
    def test_applies_to_python_service_and_cli_and_fullstack(self):
        emitter = DockerEmitter()
        assert emitter.applies_to(AppArchetype.PYTHON_SERVICE)
        assert emitter.applies_to(AppArchetype.PYTHON_CLI)
        assert emitter.applies_to(AppArchetype.FULLSTACK)

    def test_does_not_apply_to_web_static(self):
        assert not DockerEmitter().applies_to(AppArchetype.WEB_STATIC)


_RESOLVED_DIGEST = DigestResult(digest="sha256:" + "a" * 64, ok=True, log="")
_UNRESOLVED_DIGEST = DigestResult(digest=None, ok=False, log="docker CLI not available")
_NO_LOCK = LockFileResult(filename=None, tool="none", ok=False, log="no pyproject.toml to lock")
_UV_LOCK = LockFileResult(
    filename="uv.lock",
    tool="uv",
    ok=True,
    log="",
    components=(("requests", "2.32.0"),),
)


@pytest.mark.unit
class TestEmit:
    async def test_writes_dockerfile_dockerignore_and_sbom(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\n')
        with (
            patch(
                "orchestrator.infrastructure.artifact_emitters.docker_emitter.resolve_lockfile",
                return_value=_NO_LOCK,
            ),
            patch(
                "orchestrator.infrastructure.artifact_emitters.docker_emitter.resolve_digest",
                return_value=_UNRESOLVED_DIGEST,
            ),
        ):
            written = await DockerEmitter().emit(_ws(tmp_path), _Profile())

        assert "Dockerfile" in written
        assert ".dockerignore" in written
        assert "sbom.cdx.json" in written
        assert (tmp_path / "Dockerfile").is_file()
        assert (tmp_path / ".dockerignore").is_file()
        assert (tmp_path / "sbom.cdx.json").is_file()

    async def test_digest_pinned_when_resolution_succeeds(self, tmp_path):
        with (
            patch(
                "orchestrator.infrastructure.artifact_emitters.docker_emitter.resolve_lockfile",
                return_value=_NO_LOCK,
            ),
            patch(
                "orchestrator.infrastructure.artifact_emitters.docker_emitter.resolve_digest",
                return_value=_RESOLVED_DIGEST,
            ),
        ):
            await DockerEmitter().emit(_ws(tmp_path), _Profile())

        text = (tmp_path / "Dockerfile").read_text(encoding="utf-8")
        assert f"FROM python:3.12-slim@{_RESOLVED_DIGEST.digest}" in text

    async def test_digest_resolution_failure_degrades_to_marker_never_a_bare_mutable_tag(
        self, tmp_path
    ):
        with (
            patch(
                "orchestrator.infrastructure.artifact_emitters.docker_emitter.resolve_lockfile",
                return_value=_NO_LOCK,
            ),
            patch(
                "orchestrator.infrastructure.artifact_emitters.docker_emitter.resolve_digest",
                return_value=_UNRESOLVED_DIGEST,
            ),
        ):
            await DockerEmitter().emit(_ws(tmp_path), _Profile())

        text = (tmp_path / "Dockerfile").read_text(encoding="utf-8")
        assert "DIGEST-INDETERMINATE" in text
        assert "@sha256:" not in text
        # the marker documents *why* — never a bare, unexplained mutable tag
        assert "docker CLI not available" in text

    async def test_run_command_used_as_exec_form_cmd(self, tmp_path):
        with (
            patch(
                "orchestrator.infrastructure.artifact_emitters.docker_emitter.resolve_lockfile",
                return_value=_NO_LOCK,
            ),
            patch(
                "orchestrator.infrastructure.artifact_emitters.docker_emitter.resolve_digest",
                return_value=_UNRESOLVED_DIGEST,
            ),
        ):
            await DockerEmitter().emit(
                _ws(tmp_path), _Profile(run_command="python app.py --port 9")
            )

        text = (tmp_path / "Dockerfile").read_text(encoding="utf-8")
        assert 'CMD ["python", "app.py", "--port", "9"]' in text

    async def test_default_run_command_when_profile_has_none(self, tmp_path):
        with (
            patch(
                "orchestrator.infrastructure.artifact_emitters.docker_emitter.resolve_lockfile",
                return_value=_NO_LOCK,
            ),
            patch(
                "orchestrator.infrastructure.artifact_emitters.docker_emitter.resolve_digest",
                return_value=_UNRESOLVED_DIGEST,
            ),
        ):
            await DockerEmitter().emit(_ws(tmp_path), None)

        text = (tmp_path / "Dockerfile").read_text(encoding="utf-8")
        assert 'CMD ["python", "main.py"]' in text

    async def test_healthcheck_is_http_and_user_is_nonroot(self, tmp_path):
        with (
            patch(
                "orchestrator.infrastructure.artifact_emitters.docker_emitter.resolve_lockfile",
                return_value=_NO_LOCK,
            ),
            patch(
                "orchestrator.infrastructure.artifact_emitters.docker_emitter.resolve_digest",
                return_value=_UNRESOLVED_DIGEST,
            ),
        ):
            await DockerEmitter().emit(_ws(tmp_path), _Profile())

        text = (tmp_path / "Dockerfile").read_text(encoding="utf-8")
        assert "HEALTHCHECK" in text
        assert "curl -f http://localhost:8000/health" in text
        assert "USER appuser" in text

    async def test_lockfile_written_when_resolved(self, tmp_path):
        with (
            patch(
                "orchestrator.infrastructure.artifact_emitters.docker_emitter.resolve_lockfile",
                return_value=_UV_LOCK,
            ),
            patch(
                "orchestrator.infrastructure.artifact_emitters.docker_emitter.resolve_digest",
                return_value=_UNRESOLVED_DIGEST,
            ),
        ):
            written = await DockerEmitter().emit(_ws(tmp_path), _Profile())

        assert "uv.lock" in written
        text = (tmp_path / "Dockerfile").read_text(encoding="utf-8")
        assert "uv sync --frozen" in text

    async def test_no_lockfile_entry_when_resolution_failed(self, tmp_path):
        with (
            patch(
                "orchestrator.infrastructure.artifact_emitters.docker_emitter.resolve_lockfile",
                return_value=_NO_LOCK,
            ),
            patch(
                "orchestrator.infrastructure.artifact_emitters.docker_emitter.resolve_digest",
                return_value=_UNRESOLVED_DIGEST,
            ),
        ):
            written = await DockerEmitter().emit(_ws(tmp_path), _Profile())

        assert "uv.lock" not in written
        assert "requirements.txt" not in written


@pytest.mark.unit
class TestSbomShape:
    async def test_sbom_is_valid_cyclonedx_with_components(self, tmp_path):
        with (
            patch(
                "orchestrator.infrastructure.artifact_emitters.docker_emitter.resolve_lockfile",
                return_value=_UV_LOCK,
            ),
            patch(
                "orchestrator.infrastructure.artifact_emitters.docker_emitter.resolve_digest",
                return_value=_UNRESOLVED_DIGEST,
            ),
        ):
            await DockerEmitter().emit(_ws(tmp_path), _Profile())

        data = json.loads((tmp_path / "sbom.cdx.json").read_text(encoding="utf-8"))
        assert data["bomFormat"] == "CycloneDX"
        assert data["components"][0]["name"] == "requests"
        assert data["components"][0]["version"] == "2.32.0"


@pytest.mark.unit
class TestPassesSupplyChainProbes:
    """Acceptance criterion (plan §P-4): 'P-2 supply-chain probes pass on emitted output.'"""

    async def test_base_image_digest_pinned_probe_satisfied_when_resolved(self, tmp_path):
        from orchestrator.domain.readiness import ProbeResult
        from orchestrator.domain.ports import ProbeContext
        from orchestrator.infrastructure.readiness_probes.supply_chain_probes import (
            BaseImageDigestPinnedProbe,
        )

        with (
            patch(
                "orchestrator.infrastructure.artifact_emitters.docker_emitter.resolve_lockfile",
                return_value=_NO_LOCK,
            ),
            patch(
                "orchestrator.infrastructure.artifact_emitters.docker_emitter.resolve_digest",
                return_value=_RESOLVED_DIGEST,
            ),
        ):
            await DockerEmitter().emit(_ws(tmp_path), _Profile())

        ctx = ProbeContext(archetype=AppArchetype.PYTHON_SERVICE)
        outcome = await BaseImageDigestPinnedProbe().probe(_ws(tmp_path), ctx)
        assert outcome.result == ProbeResult.SATISFIED

    async def test_sbom_present_probe_satisfied(self, tmp_path):
        from orchestrator.domain.readiness import ProbeResult
        from orchestrator.domain.ports import ProbeContext
        from orchestrator.infrastructure.readiness_probes.supply_chain_probes import (
            SbomPresentProbe,
        )

        with (
            patch(
                "orchestrator.infrastructure.artifact_emitters.docker_emitter.resolve_lockfile",
                return_value=_NO_LOCK,
            ),
            patch(
                "orchestrator.infrastructure.artifact_emitters.docker_emitter.resolve_digest",
                return_value=_UNRESOLVED_DIGEST,
            ),
        ):
            await DockerEmitter().emit(_ws(tmp_path), _Profile())

        ctx = ProbeContext(archetype=AppArchetype.PYTHON_SERVICE)
        outcome = await SbomPresentProbe().probe(_ws(tmp_path), ctx)
        assert outcome.result == ProbeResult.SATISFIED

    async def test_lock_file_present_probe_satisfied_when_uv_lock_resolved(self, tmp_path):
        from orchestrator.domain.readiness import ProbeResult
        from orchestrator.domain.ports import ProbeContext
        from orchestrator.infrastructure.readiness_probes.supply_chain_probes import (
            LockFilePresentProbe,
        )

        (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\n')
        (tmp_path / "uv.lock").write_text('[[package]]\nname = "requests"\nversion = "2.32.0"\n')
        with (
            patch(
                "orchestrator.infrastructure.artifact_emitters.docker_emitter.resolve_lockfile",
                return_value=_UV_LOCK,
            ),
            patch(
                "orchestrator.infrastructure.artifact_emitters.docker_emitter.resolve_digest",
                return_value=_UNRESOLVED_DIGEST,
            ),
        ):
            await DockerEmitter().emit(_ws(tmp_path), _Profile())

        ctx = ProbeContext(archetype=AppArchetype.PYTHON_SERVICE)
        outcome = await LockFilePresentProbe().probe(_ws(tmp_path), ctx)
        assert outcome.result == ProbeResult.SATISFIED
