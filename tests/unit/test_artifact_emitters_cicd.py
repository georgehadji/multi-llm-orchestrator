"""Tests for the CI/CD artifact emitter (Phase 7, P-5)."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import patch

import pytest
import yaml

from orchestrator.domain.readiness import AppArchetype
from orchestrator.domain.testing_models import Workspace
from orchestrator.infrastructure.artifact_emitters._action_pins import ActionPinResult
from orchestrator.infrastructure.artifact_emitters.cicd_emitter import CicdEmitter

_RESOLVED = ActionPinResult(sha="a" * 40, ok=True, log="")
_UNRESOLVED = ActionPinResult(sha=None, ok=False, log="git not available on PATH")


@dataclass
class _Profile:
    project_name: str = "sample_app"


def _ws(root):
    return Workspace(root=root, framework="python")


def _patched(resolved: bool):
    return patch(
        "orchestrator.infrastructure.artifact_emitters.cicd_emitter.resolve_action_sha",
        return_value=_RESOLVED if resolved else _UNRESOLVED,
    )


@pytest.mark.unit
class TestApplicability:
    def test_applies_to_python_service_cli_library_and_fullstack(self):
        emitter = CicdEmitter()
        assert emitter.applies_to(AppArchetype.PYTHON_SERVICE)
        assert emitter.applies_to(AppArchetype.PYTHON_CLI)
        assert emitter.applies_to(AppArchetype.LIBRARY)
        assert emitter.applies_to(AppArchetype.FULLSTACK)

    def test_does_not_apply_to_web_static(self):
        assert not CicdEmitter().applies_to(AppArchetype.WEB_STATIC)


@pytest.mark.unit
class TestEmit:
    async def test_writes_workflow_file_that_parses_as_yaml(self, tmp_path):
        with _patched(resolved=True):
            written = await CicdEmitter().emit(_ws(tmp_path), _Profile())

        assert ".github/workflows/ci.yml" in written
        path = tmp_path / ".github" / "workflows" / "ci.yml"
        assert path.is_file()
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(data, dict)
        assert "jobs" in data

    async def test_permissions_block_is_read_only_by_default(self, tmp_path):
        with _patched(resolved=True):
            await CicdEmitter().emit(_ws(tmp_path), _Profile())

        data = yaml.safe_load(
            (tmp_path / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        )
        assert data["permissions"] == {"contents": "read"}

    async def test_concurrency_group_present(self, tmp_path):
        with _patched(resolved=True):
            await CicdEmitter().emit(_ws(tmp_path), _Profile())

        data = yaml.safe_load(
            (tmp_path / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        )
        assert data["concurrency"]["cancel-in-progress"] is True
        assert "group" in data["concurrency"]

    async def test_no_bare_or_true_on_any_security_step(self, tmp_path):
        with _patched(resolved=True):
            await CicdEmitter().emit(_ws(tmp_path), _Profile())

        text = (tmp_path / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        assert "|| true" not in text

    async def test_bandit_pip_audit_and_trivy_jobs_present(self, tmp_path):
        with _patched(resolved=True):
            await CicdEmitter().emit(_ws(tmp_path), _Profile())

        data = yaml.safe_load(
            (tmp_path / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        )
        jobs = data["jobs"]
        assert "bandit" in str(jobs["security"])
        assert "pip-audit" in str(jobs["dependency-audit"])
        assert "trivy-action" in str(jobs["trivy"])

    async def test_every_uses_is_sha_pinned_when_resolution_succeeds(self, tmp_path):
        with _patched(resolved=True):
            await CicdEmitter().emit(_ws(tmp_path), _Profile())

        text = (tmp_path / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        data = yaml.safe_load(text)

        def find_uses(node):
            if isinstance(node, dict):
                for key, value in node.items():
                    if key == "uses" and isinstance(value, str):
                        yield value
                    else:
                        yield from find_uses(value)
            elif isinstance(node, list):
                for item in node:
                    yield from find_uses(item)

        for uses in find_uses(data):
            assert "@" + "a" * 40 in uses, uses

    async def test_sha_resolution_failure_degrades_to_marker_keeping_the_ref_runnable(
        self, tmp_path
    ):
        with _patched(resolved=False):
            await CicdEmitter().emit(_ws(tmp_path), _Profile())

        text = (tmp_path / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        assert "ACTION-SHA-INDETERMINATE" in text
        # still a valid, runnable tag ref — never a fabricated SHA
        assert "actions/checkout@v4" in text
        # yaml must still parse even in the degraded case
        assert isinstance(yaml.safe_load(text), dict)

    async def test_cov_fail_under_inherited_from_generated_projects_pyproject(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text(
            "[tool.coverage.report]\nfail_under = 42\n", encoding="utf-8"
        )
        with _patched(resolved=True):
            await CicdEmitter().emit(_ws(tmp_path), _Profile())

        text = (tmp_path / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        assert "--cov-fail-under=42" in text

    async def test_no_cov_fail_under_flag_when_pyproject_has_no_floor(self, tmp_path):
        with _patched(resolved=True):
            await CicdEmitter().emit(_ws(tmp_path), _Profile())

        text = (tmp_path / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        assert "--cov-fail-under" not in text


@pytest.mark.unit
class TestPassesCiProbes:
    """Acceptance criterion (plan §P-5): 'P-2 CI probes pass on emitted output.'"""

    async def test_ci_permissions_read_only_probe_satisfied(self, tmp_path):
        from orchestrator.domain.ports import ProbeContext
        from orchestrator.domain.readiness import ProbeResult
        from orchestrator.infrastructure.readiness_probes.static_probes import (
            CiPermissionsReadOnlyProbe,
        )

        with _patched(resolved=True):
            await CicdEmitter().emit(_ws(tmp_path), _Profile())

        ctx = ProbeContext(archetype=AppArchetype.PYTHON_SERVICE)
        outcome = await CiPermissionsReadOnlyProbe().probe(_ws(tmp_path), ctx)
        assert outcome.result == ProbeResult.SATISFIED

    async def test_ci_no_security_bypass_probe_satisfied(self, tmp_path):
        from orchestrator.domain.ports import ProbeContext
        from orchestrator.domain.readiness import ProbeResult
        from orchestrator.infrastructure.readiness_probes.static_probes import (
            CiNoUnconditionalBypassProbe,
        )

        with _patched(resolved=True):
            await CicdEmitter().emit(_ws(tmp_path), _Profile())

        ctx = ProbeContext(archetype=AppArchetype.PYTHON_SERVICE)
        outcome = await CiNoUnconditionalBypassProbe().probe(_ws(tmp_path), ctx)
        assert outcome.result == ProbeResult.SATISFIED

    async def test_actions_sha_pinned_probe_satisfied_when_resolution_succeeds(self, tmp_path):
        from orchestrator.domain.ports import ProbeContext
        from orchestrator.domain.readiness import ProbeResult
        from orchestrator.infrastructure.readiness_probes.supply_chain_probes import (
            ActionsShaPinnedProbe,
        )

        with _patched(resolved=True):
            await CicdEmitter().emit(_ws(tmp_path), _Profile())

        ctx = ProbeContext(archetype=AppArchetype.PYTHON_SERVICE)
        outcome = await ActionsShaPinnedProbe().probe(_ws(tmp_path), ctx)
        assert outcome.result == ProbeResult.SATISFIED
