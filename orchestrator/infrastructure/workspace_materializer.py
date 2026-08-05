"""
WorkspaceMaterializer — materialize a Workspace domain object to a real directory.
=================================================================================

Writes source files, test files, and manifest to a temp directory,
preparing the workspace for test execution.

Usage:
    materializer = WorkspaceMaterializer()
    ws = await materializer.materialize(
        artifact="def add(a, b): return a + b",
        test_code="def test_add(): assert add(1, 2) == 3",
        framework="pytest",
        extra_files={"requirements.txt": "pytest"},
    )
    # ws.root / "main.py" now exists
    # ws.root / "test_main.py" now exists
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from ..domain.testing_models import Workspace

logger = logging.getLogger(__name__)

_MAIN_FILENAMES: dict[str, str] = {
    "pytest": "main.py",
    "jest": "main.test.js",
    "vitest": "main.test.ts",
    "mocha": "main.test.js",
    "go": "main_test.go",
    "cargo": "src/lib.rs",
}


class WorkspaceMaterializer:
    """Materialize a Workspace domain object to a real temp directory."""

    def __init__(self, base_dir: str | Path | None = None) -> None:
        self._base_dir = Path(base_dir) if base_dir else None

    async def materialize(
        self,
        artifact: str = "",
        test_code: str = "",
        framework: str = "pytest",
        *,
        extra_files: dict[str, str] | None = None,
        source_files: dict[str, str] | None = None,
        test_files: dict[str, str] | None = None,
        env: dict[str, str] | None = None,
    ) -> Workspace:
        """Create a temp directory with source and test files.

        Args:
            artifact: Main source code (written to main.py or equivalent).
            test_code: Test code (written to test_main.py or equivalent).
            framework: Test framework identifier.
            extra_files: Additional files to write (filename -> content).
            source_files: Explicit source file dict (path -> content).
            test_files: Explicit test file dict (path -> content).
            env: Environment variables for the workspace.

        Returns:
            A Workspace domain object describing the materialized tree.
        """
        root = Path(tempfile.mkdtemp(dir=self._base_dir))
        logger.info("Materialized workspace at %s", root)

        written_sources: list[Path] = []
        written_tests: list[Path] = []
        manifest: Path | None = None

        # Write extra files (requirements.txt, package.json, etc.)
        for filename, content in (extra_files or {}).items():
            target = (root / filename).resolve()
            # Audit fix #4: path traversal guard — no file may escape the
            # materialized workspace root (e.g. filename="../../etc/passwd").
            if root.resolve() not in target.parents and target != root.resolve():
                raise ValueError(f"Path traversal rejected in workspace file name: {filename!r}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            if filename in (
                "pyproject.toml",
                "requirements.txt",
                "setup.py",
                "package.json",
                "go.mod",
                "Cargo.toml",
            ):
                manifest = target

        # Write source files (from dict or artifact string)
        if source_files:
            for filepath, content in source_files.items():
                target = (root / filepath).resolve()
                if root.resolve() not in target.parents and target != root.resolve():
                    raise ValueError(f"Path traversal rejected in source file name: {filepath!r}")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
                written_sources.append(target)
        elif artifact:
            main_name = _MAIN_FILENAMES.get(framework, "main.py")
            target = root / main_name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(artifact, encoding="utf-8")
            written_sources.append(target)

        # Write test files (from dict or test_code string)
        if test_files:
            for filepath, content in test_files.items():
                target = (root / filepath).resolve()
                if root.resolve() not in target.parents and target != root.resolve():
                    raise ValueError(f"Path traversal rejected in test file name: {filepath!r}")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
                written_tests.append(target)
        elif test_code:
            test_name = f"test_{_MAIN_FILENAMES.get(framework, 'main.py')}"
            target = root / test_name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(test_code, encoding="utf-8")
            written_tests.append(target)

        # Auto-detect manifest
        if manifest is None:
            for candidate in (
                "pyproject.toml",
                "requirements.txt",
                "setup.py",
                "package.json",
                "go.mod",
                "Cargo.toml",
            ):
                p = root / candidate
                if p.exists():
                    manifest = p
                    break

        return Workspace(
            root=root,
            framework=framework,
            source_files=tuple(written_sources),
            test_files=tuple(written_tests),
            manifest=manifest,
            env=env or {},
        )

    async def cleanup(self, workspace: Workspace) -> None:
        """Remove a materialized workspace directory."""
        import shutil

        if workspace.root.exists():
            shutil.rmtree(str(workspace.root), ignore_errors=True)
            logger.info("Cleaned up workspace at %s", workspace.root)
