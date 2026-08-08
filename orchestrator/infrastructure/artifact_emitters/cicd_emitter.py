"""CI/CD artifact emitter (Phase 7, P-5).

Wires up ``generators/cicd_generator.py`` — it already emitted
``permissions: contents: read`` but was never called from anywhere. This
emitter is the single owner of the GitHub Actions workflow: it replaces
``assembler.py``'s inline template and adds what P-5 requires on top —
a bandit step with no ``|| true``, a pip-audit job, a trivy scan (deferred
from P-4), ``concurrency:`` grouping, SHA-pinned actions (with an explicit,
greppable marker when resolution is unavailable — never a silently-mutable
tag), and a ``--cov-fail-under`` inherited from the target project's own
``pyproject.toml``.
"""

from __future__ import annotations

import asyncio
import tomllib
from pathlib import Path
from typing import Any

from ...domain.readiness import AppArchetype
from ...domain.testing_models import Workspace
from ...generators.cicd_generator import CICDPipelineBuilder
from ._action_pins import resolve_action_sha

_ACTIONS: tuple[tuple[str, str], ...] = (
    ("actions/checkout", "v4"),
    ("actions/setup-python", "v5"),
    ("codecov/codecov-action", "v3"),
    ("aquasecurity/trivy-action", "0.24.0"),
    ("docker/setup-buildx-action", "v3"),
    ("docker/login-action", "v3"),
    ("docker/build-push-action", "v5"),
)

_PYTHON_VERSIONS = ("3.10", "3.11", "3.12")


def _resolve_cov_floor(root: Path) -> int | None:
    pyproject = root / "pyproject.toml"
    if not pyproject.is_file():
        return None
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return None
    value = data.get("tool", {}).get("coverage", {}).get("report", {}).get("fail_under")
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _resolve_action_pins() -> dict[str, str]:
    pins: dict[str, str] = {}
    for owner_repo, ref in _ACTIONS:
        result = resolve_action_sha(owner_repo, ref)
        key = f"{owner_repo}@{ref}"
        if result.ok:
            pins[key] = f"{owner_repo}@{result.sha}  # {ref}"
        else:
            pins[key] = f"{owner_repo}@{ref}  # ACTION-SHA-INDETERMINATE: {result.log}"
    return pins


def _docker_steps() -> list[dict[str, Any]]:
    return [
        {"uses": "docker/setup-buildx-action@v3"},
        {
            "uses": "docker/login-action@v3",
            "with": {
                "registry": "ghcr.io",
                "username": "${{ github.actor }}",
                "password": "${{ secrets.GITHUB_TOKEN }}",
            },
        },
        {
            "uses": "docker/build-push-action@v5",
            "with": {
                "context": ".",
                "push": "true",
                "tags": "ghcr.io/${{ github.repository }}:${{ github.sha }}",
                "cache-from": "type=gha",
                "cache-to": "type=gha,mode=max",
            },
        },
    ]


def _release_steps() -> list[dict[str, Any]]:
    return [
        {"name": "Set up Python", "run": "python --version"},
        {"name": "Install build dependencies", "run": "pip install build twine"},
        {"name": "Build package", "run": "python -m build"},
        {
            "name": "Publish to PyPI",
            "run": (
                "TWINE_USERNAME=__token__ "
                "TWINE_PASSWORD=${{ secrets.PYPI_API_TOKEN }} "
                "twine upload dist/*"
            ),
        },
    ]


class CicdEmitter:
    """Emits a single GitHub Actions workflow: ``.github/workflows/ci.yml``."""

    name = "cicd"

    def applies_to(self, archetype: AppArchetype) -> bool:
        return archetype in (
            AppArchetype.PYTHON_SERVICE,
            AppArchetype.PYTHON_CLI,
            AppArchetype.LIBRARY,
            AppArchetype.FULLSTACK,
        )

    async def emit(self, workspace: Workspace, profile: Any) -> list[str]:
        cov_floor = await asyncio.to_thread(_resolve_cov_floor, workspace.root)
        pins = await asyncio.to_thread(_resolve_action_pins)

        content = self._build_workflow(cov_floor, pins)

        workflow_dir = workspace.root / ".github" / "workflows"
        workflow_dir.mkdir(parents=True, exist_ok=True)
        workflow_path = workflow_dir / "ci.yml"
        workflow_path.write_text(content, encoding="utf-8")

        return [workflow_path.relative_to(workspace.root).as_posix()]

    def _build_workflow(self, cov_floor: int | None, pins: dict[str, str]) -> str:
        builder = (
            CICDPipelineBuilder()
            .for_github_actions()
            .with_name("CI/CD")
            .with_branches(["main"])
            .with_concurrency_group("${{ github.workflow }}-${{ github.ref }}")
            .with_action_pins(pins)
        )
        if cov_floor is not None:
            builder = builder.with_coverage_floor(cov_floor)

        builder = (
            builder.add_job("lint", ["python"], "lint")
            .add_job("security", ["python"], "security_scan")
            .add_job("dependency-audit", ["python"], "dependency_audit")
            .add_job("trivy", [], "vulnerability_scan")
            .add_job(
                "test",
                ["python"],
                "test",
                matrix={"python-version": list(_PYTHON_VERSIONS)},
            )
            .add_job(
                "docker",
                needs=["lint", "test"],
                job_type="deploy",
                if_condition="github.event_name == 'push' && github.ref == 'refs/heads/main'",
                permissions={"contents": "read", "packages": "write"},
                custom_steps=_docker_steps(),
            )
            .add_job(
                "release",
                needs=["lint", "test"],
                job_type="release",
                if_condition="github.event_name == 'release'",
                custom_steps=_release_steps(),
            )
        )
        return builder.build()


__all__ = ["CicdEmitter"]
