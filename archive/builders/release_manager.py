"""
ReleaseManager - Semantic versioning and release management.
==============================================================
Author: Georgios-Chrysovalantis Chatzivantsidis
Part of Category 2, Phase R2 (Retool-inspired).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging
import time
import re

logger = logging.getLogger(__name__)


@dataclass
class Release:
    version: str
    status: str = "draft"  # draft, published, archived
    created_at: float = 0.0
    published_at: float = 0.0
    description: str = ""
    changes: list = field(default_factory=list)
    author: str = ""
    commit_hash: str = ""

    def to_dict(self):
        return {
            "version": self.version,
            "status": self.status,
            "created_at": self.created_at,
            "published_at": self.published_at,
            "description": self.description,
            "changes": self.changes,
            "author": self.author,
            "commit_hash": self.commit_hash,
        }


class ReleaseManager:
    """Manages semantic versioning and release lifecycle."""

    def __init__(self, project_dir="."):
        self._dir = Path(project_dir) / ".releases"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._releases: list[Release] = []
        self._load()

    def _load(self):
        fp = self._dir / "releases.json"
        if fp.exists():
            try:
                self._releases = [Release(**d) for d in json.loads(fp.read_text(encoding="utf-8"))]
            except Exception:
                pass

    def _save(self):
        (self._dir / "releases.json").write_text(
            json.dumps([r.to_dict() for r in self._releases], indent=2), encoding="utf-8"
        )

    @property
    def current_version(self):
        return self._releases[-1].version if self._releases else "0.1.0"

    def bump(self, level="patch"):
        """Bump version: major, minor, or patch."""
        parts = [int(x) for x in self.current_version.split(".")]
        if level == "major":
            parts[0] += 1
            parts[1] = 0
            parts[2] = 0
        elif level == "minor":
            parts[1] += 1
            parts[2] = 0
        else:
            parts[2] += 1
        return ".".join(str(p) for p in parts)

    def create_release(self, version=None, level="patch", description="", changes=None):
        """Create a new release in draft status."""
        ver = version or self.bump(level)
        rel = Release(
            version=ver,
            status="draft",
            created_at=time.time(),
            description=description,
            changes=changes or [],
            author="AI Orchestrator",
        )
        self._releases.append(rel)
        self._save()
        return rel

    def publish(self, version):
        """Publish a draft release."""
        for r in self._releases:
            if r.version == version and r.status == "draft":
                r.status = "published"
                r.published_at = time.time()
                self._save()
                return True
        return False

    def generate_changelog(self):
        """Generate changelog from all releases."""
        lines = ["# Changelog", ""]
        for r in reversed(self._releases):
            status = f" ({r.status})" if r.status != "published" else ""
            lines.append(
                f"## {r.version}{status} ({time.strftime('%Y-%m-%d', time.localtime(r.created_at or time.time()))})"
            )
            if r.description:
                lines.append(f"{r.description}")
            for c in r.changes:
                lines.append(f"- {c}")
            lines.append("")
        return "\n".join(lines)

    def diff_releases(self, v1, v2):
        """Show what changed between two releases."""
        r1 = next((r for r in self._releases if r.version == v1), None)
        r2 = next((r for r in self._releases if r.version == v2), None)
        if not r1 or not r2:
            return f"Releases not found: {v1}, {v2}"
        lines = [f"## Diff: {v1} -> {v2}", ""]
        if r1.changes:
            lines.append(f"### Removed in {v1}")
            for c in r1.changes:
                lines.append(f"- {c}")
        if r2.changes:
            lines.append(f"### Added in {v2}")
            for c in r2.changes:
                lines.append(f"- {c}")
        return "\n".join(lines)