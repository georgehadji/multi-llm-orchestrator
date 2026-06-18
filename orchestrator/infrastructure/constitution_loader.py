"""
Constitution Loader — Per-Project Authority Constraints
=========================================================
CodeWhale Phase 3 — Infrastructure adapter.

Loads .orchestrator/constitution.json and exposes enforcement helpers
for the write paths, critique cycle, and codebase modifications.

Architecture:
    Infrastructure layer — satisfies no domain port directly (Constitution is
    a value object, not a protocol), but provides loading + caching + enforcement.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..domain.constitution import ProjectConstitution

logger = logging.getLogger(__name__)


class ConstitutionLoader:
    """Loads, caches, and applies project constitution constraints.

    Usage:
        loader = ConstitutionLoader(project_root=".")
        constitution = loader.get()  # auto-discovers .orchestrator/constitution.json
        if constitution.is_path_protected("src/core/config.py"):
            # skip or warn
    """

    def __init__(
        self,
        project_root: str | Path | None = None,
        constitution_path: str | Path | None = None,
    ):
        self._project_root = Path(project_root).resolve() if project_root else Path.cwd().resolve()
        self._explicit_path = Path(constitution_path) if constitution_path else None
        self._constitution: ProjectConstitution | None = None
        self._loaded = False

    def get(self, reload: bool = False) -> ProjectConstitution:
        """Get the project constitution (cached after first load).

        Args:
            reload: If True, force re-read from disk.

        Returns:
            ProjectConstitution (empty = no restrictions).
        """
        if not self._loaded or reload:
            if self._explicit_path:
                self._constitution = ProjectConstitution.from_file(self._explicit_path)
            else:
                self._constitution = ProjectConstitution.discover(self._project_root)
            self._loaded = True

            if self._constitution.protect_paths:
                logger.info(
                    "Constitution: %d protected paths, require_tests=%s, %d validators",
                    len(self._constitution.protect_paths),
                    self._constitution.require_tests,
                    len(self._constitution.required_validators),
                )
        return self._constitution or ProjectConstitution()

    def check_file_write(self, path: str | Path, content: str = "") -> tuple[bool, str]:
        """Check if a file write is allowed by the constitution.

        Args:
            path: File path being written (relative or absolute).
            content: File content (for size check).

        Returns:
            (allowed: bool, reason: str)
        """
        constitution = self.get()
        path_str = str(path)

        # Check protected paths
        if constitution.is_path_protected(path_str):
            return False, f"Path '{path_str}' is protected by project constitution"

        # Check max file size
        if constitution.max_file_size_bytes > 0 and len(content) > constitution.max_file_size_bytes:
            return False, (
                f"File size {len(content)} bytes exceeds constitution limit "
                f"of {constitution.max_file_size_bytes} bytes"
            )

        return True, ""

    def check_import(self, import_name: str) -> tuple[bool, str]:
        """Check if an import is forbidden.

        Returns:
            (allowed: bool, reason: str)
        """
        constitution = self.get()
        if constitution.is_import_forbidden(import_name):
            return False, f"Import '{import_name}' is forbidden by project constitution"
        return True, ""

    def get_required_validators(self) -> list[str]:
        """Return the list of validators that must run for every task."""
        return self.get().required_validators
