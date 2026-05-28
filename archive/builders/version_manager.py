"""
VersionManager — First-class versioning for generated code.
==============================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Part of Category 2, Phase V6 (v0-inspired): Auto-version every code change
with diff, revert, and chain navigation. Inspired by v0's version system.

Usage:
    vman = VersionManager(project_dir)
    version = await vman.capture("Added auth module", files=["auth.py"])
    diff = await vman.diff("v1", "v2")
    await vman.revert_to("v1")
"""

from __future__ import annotations

import difflib
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CodeVersion:
    """A single version snapshot of generated code."""

    version_id: str  # e.g., "v1", "v2", ...
    description: str
    timestamp: float = field(default_factory=time.time)
    files: dict[str, str] = field(default_factory=dict)  # filename -> sha256
    diff_from_previous: str = ""  # unified diff from previous version
    model_used: str = ""
    task_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "version_id": self.version_id,
            "description": self.description,
            "timestamp": self.timestamp,
            "files": self.files,
            "diff_from_previous": self.diff_from_previous,
            "model_used": self.model_used,
            "task_id": self.task_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CodeVersion:
        return cls(
            version_id=d["version_id"],
            description=d.get("description", ""),
            timestamp=d.get("timestamp", time.time()),
            files=d.get("files", {}),
            diff_from_previous=d.get("diff_from_previous", ""),
            model_used=d.get("model_used", ""),
            task_id=d.get("task_id", ""),
        )


class VersionManager:
    """Manages a chain of code versions with diff and revert support."""

    def __init__(self, project_dir: str | Path, versions_dir: str | None = None):
        self.project_dir = Path(project_dir)
        self.versions_dir = Path(versions_dir or (self.project_dir / ".versions"))
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self._versions: list[CodeVersion] = []
        self._load_versions()

    def _load_versions(self) -> None:
        """Load version chain from disk."""
        chain_file = self.versions_dir / "version_chain.json"
        if chain_file.exists():
            try:
                data = json.loads(chain_file.read_text(encoding="utf-8"))
                self._versions = [CodeVersion.from_dict(d) for d in data]
            except Exception as e:
                logger.warning(f"Failed to load version chain: {e}")

    def _save_versions(self) -> None:
        """Persist version chain to disk."""
        chain_file = self.versions_dir / "version_chain.json"
        chain_file.write_text(
            json.dumps(
                [v.to_dict() for v in self._versions],
                indent=2,
            ),
            encoding="utf-8",
        )

    def _hash_file(self, path: Path) -> str:
        """Hash a file's contents."""
        if not path.exists():
            return ""
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    def _generate_diff(self, old_version: CodeVersion, new_files: dict[str, str]) -> str:
        """Generate unified diff from previous version to current files."""
        diff_parts = []

        for filename, new_hash in new_files.items():
            new_path = self.project_dir / filename
            old_path = self.versions_dir / f"{old_version.version_id}" / filename

            if not new_path.exists():
                continue

            new_content = new_path.read_text(encoding="utf-8", errors="replace")
            old_content = (
                old_path.read_text(encoding="utf-8", errors="replace") if old_path.exists() else ""
            )

            diff = difflib.unified_diff(
                old_content.splitlines(keepends=True),
                new_content.splitlines(keepends=True),
                fromfile=f"a/{filename}",
                tofile=f"b/{filename}",
            )
            diff_text = "".join(diff)
            if diff_text:
                diff_parts.append(diff_text)

        return "\n".join(diff_parts)

    async def capture(
        self,
        description: str,
        files: list[str] | None = None,
        model_used: str = "",
        task_id: str = "",
    ) -> CodeVersion:
        """Capture the current state as a new version.

        Args:
            description: What changed in this version
            files: Specific files to capture (None = all tracked files)
            model_used: Model that generated this version
            task_id: Task that produced this version

        Returns:
            CodeVersion with diff from previous version
        """
        version_num = len(self._versions) + 1
        version_id = f"v{version_num}"

        # Hash current files
        files_to_capture = files or []
        if not files_to_capture:
            # Capture all files in project dir (excluding .versions/)
            for f in self.project_dir.rglob("*"):
                if f.is_file() and ".versions" not in str(f):
                    files_to_capture.append(str(f.relative_to(self.project_dir)))

        file_hashes: dict[str, str] = {}
        for filename in files_to_capture:
            fpath = self.project_dir / filename
            file_hashes[filename] = self._hash_file(fpath)

        # Generate diff from previous version
        diff = ""
        if self._versions:
            prev = self._versions[-1]
            diff = self._generate_diff(prev, file_hashes)

            # Save current files for future diffs
            save_dir = self.versions_dir / version_id
            save_dir.mkdir(parents=True, exist_ok=True)
            for filename, fhash in file_hashes.items():
                src = self.project_dir / filename
                dst = save_dir / filename
                if src.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    dst.write_bytes(src.read_bytes())

        version = CodeVersion(
            version_id=version_id,
            description=description,
            files=file_hashes,
            diff_from_previous=diff,
            model_used=model_used,
            task_id=task_id,
        )

        self._versions.append(version)
        self._save_versions()

        logger.info(f"Version {version_id} captured: {description}")
        return version

    async def diff(self, version_a: str, version_b: str) -> str:
        """Generate a diff between two versions.

        Args:
            version_a: First version ID (e.g., "v1")
            version_b: Second version ID (e.g., "v3")

        Returns:
            Unified diff string
        """
        va = self.get_version(version_a)
        vb = self.get_version(version_b)
        if not va or not vb:
            return ""

        # Use saved files for diff
        diff_parts = []
        all_files = set(va.files.keys()) | set(vb.files.keys())

        for filename in sorted(all_files):
            path_a = self.versions_dir / va.version_id / filename
            path_b = self.versions_dir / vb.version_id / filename

            content_a = (
                path_a.read_text(encoding="utf-8", errors="replace") if path_a.exists() else ""
            )
            content_b = (
                path_b.read_text(encoding="utf-8", errors="replace") if path_b.exists() else ""
            )

            diff = difflib.unified_diff(
                content_a.splitlines(keepends=True),
                content_b.splitlines(keepends=True),
                fromfile=f"{va.version_id}/{filename}",
                tofile=f"{vb.version_id}/{filename}",
            )
            diff_text = "".join(diff)
            if diff_text:
                diff_parts.append(diff_text)

        return "\n".join(diff_parts)

    async def revert_to(self, version_id: str) -> bool:
        """Revert working files to a specific version.

        Args:
            version_id: Version to revert to

        Returns:
            True if successful
        """
        version = self.get_version(version_id)
        if not version:
            logger.warning(f"Version {version_id} not found")
            return False

        save_dir = self.versions_dir / version_id
        if not save_dir.exists():
            logger.warning(f"Saved files for {version_id} not found")
            return False

        for filename in version.files:
            src = save_dir / filename
            dst = self.project_dir / filename
            if src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(src.read_bytes())

        logger.info(f"Reverted to {version_id}")
        return True

    def get_version(self, version_id: str) -> CodeVersion | None:
        """Get a specific version by ID."""
        for v in self._versions:
            if v.version_id == version_id:
                return v
        return None

    @property
    def latest(self) -> CodeVersion | None:
        """Get the latest version."""
        return self._versions[-1] if self._versions else None

    @property
    def version_count(self) -> int:
        return len(self._versions)

    def chain(self) -> list[dict[str, str]]:
        """Return the version chain for navigation."""
        return [
            {
                "id": v.version_id,
                "description": v.description,
                "model": v.model_used,
                "files_changed": len(v.files),
            }
            for v in self._versions
        ]