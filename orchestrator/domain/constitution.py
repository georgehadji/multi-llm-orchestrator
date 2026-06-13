"""
Project Constitution — Per-Project Authority Constraints
==========================================================
CodeWhale Phase 3 implementation.

Declares what a project allows, forbids, and requires — loaded from
.orchestrator/constitution.json at the project root.

Architecture:
    Domain layer — pure value objects, no infrastructure imports.
    Loaded by infrastructure.constitution_loader.ConstitutionLoader.
"""

from __future__ import annotations

import fnmatch
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ProjectConstitution:
    """Per-project authority constraints.

    These are loaded from `.orchestrator/constitution.json` at the
    project root. Empty defaults mean "no restriction" — the project
    is governed only by global settings.

    Example .orchestrator/constitution.json:
    ```json
    {
      "protect_paths": ["src/domain/**", "tests/**", "Makefile"],
      "require_review_above": 0.80,
      "require_tests": true,
      "required_validators": ["python_syntax", "ruff"],
      "forbidden_imports": ["requests", "subprocess"],
      "max_file_size_bytes": 500000
    }
    ```
    """

    # Glob patterns for files that must NEVER be modified or deleted
    protect_paths: list[str] = field(default_factory=list)

    # Minimum critique score to skip human review (0.0 = always require)
    require_review_above: float = 0.0

    # If True, CODE_GEN tasks must also produce test files
    require_tests: bool = False

    # Hard validators to run on every task (appended to task.hard_validators)
    required_validators: list[str] = field(default_factory=list)

    # Packages that must not appear in generated imports
    forbidden_imports: list[str] = field(default_factory=list)

    # Maximum allowed size for any generated file (bytes), 0 = unlimited
    max_file_size_bytes: int = 0

    def is_path_protected(self, path: str) -> bool:
        """Check if a file path matches any protected glob pattern."""
        for pattern in self.protect_paths:
            if fnmatch.fnmatch(path, pattern):
                return True
        return False

    def is_import_forbidden(self, import_name: str) -> bool:
        """Check if an import is on the forbidden list."""
        return import_name in self.forbidden_imports

    def validate(self) -> list[str]:
        """Validate constitution rules for internal consistency.

        Returns a list of warnings (empty = valid).
        """
        warnings: list[str] = []
        if self.require_review_above < 0 or self.require_review_above > 1:
            warnings.append("require_review_above must be between 0.0 and 1.0")
        if self.max_file_size_bytes < 0:
            warnings.append("max_file_size_bytes must be >= 0")
        return warnings

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProjectConstitution:
        """Create from a dictionary (parsed JSON).

        Casts numeric fields with error recovery — invalid values revert to
        defaults rather than raising. Logs a warning for each invalid field.
        """
        # Safe numeric conversion
        try:
            review_above = float(data.get("require_review_above", 0.0))
        except (ValueError, TypeError):
            logger.warning(
                "Invalid require_review_above=%r, defaulting to 0.0",
                data.get("require_review_above"),
            )
            review_above = 0.0

        try:
            max_size = int(data.get("max_file_size_bytes", 0))
        except (ValueError, TypeError):
            logger.warning(
                "Invalid max_file_size_bytes=%r, defaulting to 0",
                data.get("max_file_size_bytes"),
            )
            max_size = 0

        return cls(
            protect_paths=data.get("protect_paths", []),
            require_review_above=review_above,
            require_tests=bool(data.get("require_tests", False)),
            required_validators=data.get("required_validators", []),
            forbidden_imports=data.get("forbidden_imports", []),
            max_file_size_bytes=max_size,
        )

    @classmethod
    def from_file(cls, path: str | Path) -> ProjectConstitution:
        """Load from a JSON file."""
        path = Path(path)
        if not path.exists():
            logger.debug("No constitution file at %s — using empty defaults", path)
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            constitution = cls.from_dict(data)
            warnings = constitution.validate()
            if warnings:
                for w in warnings:
                    logger.warning("Constitution warning in %s: %s", path, w)
            logger.info("Loaded constitution from %s (%d rules)", path, len(data))
            return constitution
        except json.JSONDecodeError as e:
            logger.error("Failed to parse constitution at %s: %s", path, e)
            return cls()
        except Exception as e:
            logger.error("Failed to load constitution from %s: %s", path, e)
            return cls()

    @classmethod
    def discover(
        cls,
        project_root: str | Path | None = None,
        max_depth: int = 10,
    ) -> ProjectConstitution:
        """Auto-discover constitution.json in the project directory tree.

        Searches upward from current directory (or project_root if given)
        for `.orchestrator/constitution.json`.

        Stops at Path.home() or after max_depth parents to avoid walking
        the entire filesystem or accidentally picking up a parent project's
        constitution.

        Returns empty constitution (no restrictions) if not found.
        """
        search_start = Path(project_root).resolve() if project_root else Path.cwd().resolve()
        home = Path.home().resolve()

        # Walk up the directory tree with limits
        parents = [search_start] + list(search_start.parents)
        for i, parent in enumerate(parents):
            if i >= max_depth:
                break
            if parent == home.parent:
                # Don't look above home directory
                break
            candidate = parent / ".orchestrator" / "constitution.json"
            if candidate.exists():
                return cls.from_file(candidate)

        return cls()
