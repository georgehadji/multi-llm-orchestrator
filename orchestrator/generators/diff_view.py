"""
DiffViewProvider - Side-by-side diffs and checkpoint timeline.
===============================================================
Author: Georgios-Chrysovalantis Chatzivantsidis
Part of Category 7, Phase U3 (UI): Diff view data provider.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import difflib
import json
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class DiffEntry:
    filename: str
    old_content: str = ""
    new_content: str = ""
    unified_diff: str = ""
    added_lines: int = 0
    removed_lines: int = 0
    timestamp: float = 0.0

    def to_dict(self):
        return {
            "filename": self.filename,
            "added": self.added_lines,
            "removed": self.removed_lines,
            "diff": self.unified_diff[:5000],
            "timestamp": self.timestamp,
        }


@dataclass
class TimelinePoint:
    label: str
    timestamp: float = 0.0
    type: str = "checkpoint"  # checkpoint, version, restore, snapshot
    summary: str = ""
    files: list = field(default_factory=list)

    def to_dict(self):
        return {
            "label": self.label,
            "timestamp": self.timestamp,
            "type": self.type,
            "summary": self.summary,
            "files": self.files,
        }


class DiffViewProvider:
    """Provides diff and timeline data for the diff view panel."""

    def __init__(self, project_dir="."):
        self._dir = Path(project_dir)
        self._timeline: list[TimelinePoint] = []
        self._load()

    def _load(self):
        fp = self._dir / ".diff_history.json"
        if fp.exists():
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
                self._timeline = [TimelinePoint(**t) for t in data.get("timeline", [])]
            except Exception:
                pass

    def _save(self):
        (self._dir / ".diff_history.json").write_text(
            json.dumps({"timeline": [t.to_dict() for t in self._timeline[-50:]]}, indent=2),
            encoding="utf-8",
        )

    def record_checkpoint(self, label, files=None):
        tp = TimelinePoint(
            label=label,
            timestamp=time.time(),
            type="checkpoint",
            summary=f"Checkpoint: {label}",
            files=files or [],
        )
        self._timeline.append(tp)
        self._save()
        return tp

    def record_version(self, version_id, files=None):
        tp = TimelinePoint(
            label=f"v{version_id}",
            timestamp=time.time(),
            type="version",
            summary=f"Version {version_id}",
            files=files or [],
        )
        self._timeline.append(tp)
        self._save()
        return tp

    def diff_files(self, old_content, new_content, filename=""):
        """Generate a side-by-side diff between two file contents."""
        old_lines = old_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)
        diff = difflib.unified_diff(
            old_lines, new_lines, fromfile=f"a/{filename}", tofile=f"b/{filename}"
        )
        unified = "".join(diff)
        added = sum(
            1 for l in unified.split(chr(10)) if l.startswith("+") and not l.startswith("+++")
        )
        removed = sum(
            1 for l in unified.split(chr(10)) if l.startswith("-") and not l.startswith("---")
        )
        return DiffEntry(
            filename=filename,
            old_content=old_content,
            new_content=new_content,
            unified_diff=unified,
            added_lines=added,
            removed_lines=removed,
            timestamp=time.time(),
        )

    def get_timeline(self):
        """Get timeline for the diff view timeline navigation."""
        return [t.to_dict() for t in self._timeline]

    def diff_snapshot(self, file_path, older, newer):
        """Diff two snapshots/versions of a file."""
        fp = self._dir / file_path
        current = fp.read_text(encoding="utf-8") if fp.exists() else ""
        return self.diff_files(older or "", newer or current, file_path)