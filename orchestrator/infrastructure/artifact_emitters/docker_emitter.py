"""Docker artifact emitter (Phase 7, P-4).

Single owner of Dockerfile emission. Wraps
``generators/docker_generator.py``'s ``DockerfileBuilder`` (already gives a
non-root user and an HTTP healthcheck) instead of reimplementing it, and
adds what P-4 requires on top: a resolved lock file, a digest-pinned base
image (with an explicit, greppable marker when resolution is unavailable —
never a silently-mutable tag), and a CycloneDX SBOM.

``assembler.py``'s inline Dockerfile f-string and ``verifier.py``'s minimal
fallback both delegate here now — one emitter owns the artifact, per the
plan's consolidation requirement.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from ...domain.readiness import AppArchetype
from ...domain.testing_models import Workspace
from ...generators.docker_generator import DockerfileBuilder
from ._digest import DigestResult, resolve_digest
from ._lockfile import resolve_lockfile
from ._sbom import build_cyclonedx_sbom, write_sbom

_BASE_IMAGE = "python:3.12-slim"
_DEFAULT_RUN_COMMAND = "python main.py"

_DOCKERIGNORE = """.git
.gitignore
__pycache__/
*.py[cod]
*.egg-info/
.venv/
venv/
.pytest_cache/
.mypy_cache/
.ruff_cache/
tests/
docs/
*.md
.env
"""


class DockerEmitter:
    """Emits Dockerfile, .dockerignore, a resolved lock file, and an SBOM."""

    name = "docker"

    def applies_to(self, archetype: AppArchetype) -> bool:
        return archetype in (
            AppArchetype.PYTHON_SERVICE,
            AppArchetype.PYTHON_CLI,
            AppArchetype.FULLSTACK,
        )

    async def emit(self, workspace: Workspace, profile: Any) -> list[str]:
        project_name = getattr(profile, "project_name", None) or workspace.root.name
        run_command = getattr(profile, "run_command", None) or _DEFAULT_RUN_COMMAND

        lock_result = await asyncio.to_thread(resolve_lockfile, workspace.root)
        digest_result = await asyncio.to_thread(resolve_digest, _BASE_IMAGE)

        dockerfile_path = workspace.root / "Dockerfile"
        dockerfile_path.write_text(
            self._build_dockerfile(run_command, lock_result, digest_result), encoding="utf-8"
        )

        dockerignore_path = workspace.root / ".dockerignore"
        dockerignore_path.write_text(_DOCKERIGNORE, encoding="utf-8")

        sbom = build_cyclonedx_sbom(project_name, "0.0.0", lock_result.components)
        sbom_path = write_sbom(workspace.root, sbom)

        written = [dockerfile_path, dockerignore_path, sbom_path]
        if lock_result.ok and lock_result.filename:
            written.append(workspace.root / lock_result.filename)
        return [str(p.relative_to(workspace.root)) for p in written]

    def _build_dockerfile(self, run_command: str, lock_result, digest_result: DigestResult) -> str:
        builder = (
            DockerfileBuilder().for_python().add_env("PYTHONUNBUFFERED", "1").add_workdir("/app")
        )
        if lock_result.filename == "uv.lock":
            builder = builder.copy("pyproject.toml uv.lock", ".").run(
                "pip install --no-cache-dir uv && uv sync --frozen --no-dev"
            )
        else:
            builder = builder.copy_requirements().run(
                "pip install --no-cache-dir -r requirements.txt"
            )
        content = (
            builder.copy_all()
            .expose(8000)
            .add_healthcheck()
            .add_nonroot_user()
            .cmd(run_command)
            .build()
        )
        return self._pin_base_image(content, digest_result)

    def _pin_base_image(self, content: str, digest_result: DigestResult) -> str:
        base_line = f"FROM {_BASE_IMAGE}"
        if digest_result.ok and digest_result.digest:
            return content.replace(base_line, f"{base_line}@{digest_result.digest}", 1)
        marker = (
            f"# DIGEST-INDETERMINATE: unable to resolve a sha256 digest for {_BASE_IMAGE} "
            f"({digest_result.log or 'resolution unavailable'}); pin manually before production use"
        )
        return content.replace(base_line, f"{marker}\n{base_line}", 1)


__all__ = ["DockerEmitter"]
