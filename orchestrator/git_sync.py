"""
TwoWayGitSync - Pull external changes back into AI context.
=============================================================
Author: Georgios-Chrysovalantis Chatzivantsidis
Part of Category 8, Phase N6 (Newly-inspired).
"""

from __future__ import annotations
import subprocess
import logging
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SyncEvent:
    timestamp: float = 0.0
    type: str = ""  # "push", "pull", "merge", "conflict"
    files_changed: list = field(default_factory=list)
    summary: str = ""


class TwoWayGitSync:
    """Syncs external git changes back into the AI context."""

    def __init__(self, repo_dir="."):
        self.repo_dir = Path(repo_dir)
        self._history: list[SyncEvent] = []
        self._last_sync_hash: str = ""
        self._load()

    def _load(self):
        fp = self.repo_dir / ".git_sync.json"
        if fp.exists():
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
                self._last_sync_hash = data.get("last_sync_hash", "")
                self._history = [SyncEvent(**e) for e in data.get("history", [])]
            except Exception:
                pass

    def _save(self):
        (self.repo_dir / ".git_sync.json").write_text(
            json.dumps(
                {
                    "last_sync_hash": self._last_sync_hash,
                    "history": [
                        {
                            "timestamp": e.timestamp,
                            "type": e.type,
                            "files_changed": e.files_changed,
                            "summary": e.summary,
                        }
                        for e in self._history
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def current_hash(self):
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(self.repo_dir),
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.stdout.strip()
        except Exception:
            return ""

    def pull(self):
        """Pull external changes from remote and return summary."""
        try:
            result = subprocess.run(
                ["git", "pull", "--ff-only"],
                cwd=str(self.repo_dir),
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                return SyncEvent(
                    timestamp=time.time(),
                    type="conflict",
                    summary=f"Pull failed: {result.stderr[:200]}",
                )
            return self._detect_changes()
        except Exception as e:
            return SyncEvent(timestamp=time.time(), type="error", summary=str(e))

    def _detect_changes(self):
        """Detect what changed since last sync."""
        prev = self._last_sync_hash or "HEAD~1"
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", prev, "HEAD"],
                cwd=str(self.repo_dir),
                capture_output=True,
                text=True,
                timeout=10,
            )
            files = [f for f in result.stdout.strip().split("\n") if f]
            self._last_sync_hash = self.current_hash()
            event = SyncEvent(
                timestamp=time.time(),
                type="pull",
                files_changed=files,
                summary=f"{len(files)} files changed",
            )
            self._history.append(event)
            self._save()
            return event
        except Exception as e:
            return SyncEvent(summary=str(e))

    def get_changed_content(self, files=None, max_chars=5000):
        """Get content of changed files for AI context injection."""
        prev = self._last_sync_hash or "HEAD~1"
        content = {}
        # Distinguish None (auto-detect changed files from git) from an explicit
        # empty list (caller says "no files" → return no content).
        if files is None:
            try:
                result = subprocess.run(
                    ["git", "diff", "--name-only", prev, "HEAD"],
                    cwd=str(self.repo_dir),
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                targets = [f for f in result.stdout.strip().split("\n") if f]
            except Exception:
                targets = []
        else:
            targets = files
        for f in targets:
            fp = self.repo_dir / f
            if fp.exists():
                text = fp.read_text(encoding="utf-8")
                content[f] = text[:max_chars]
        return content

    def build_context(self, files=None):
        """Build AI context string from pulled changes."""
        content = self.get_changed_content(files)
        if not content:
            return "No changes to sync."
        parts = ["## External Changes (via git sync)", ""]
        for f, text in content.items():
            parts.append(f"### {f}")
            parts.append(f"```\n{text[:2000]}\n```")
            parts.append("")
        return "\n".join(parts)

    def history(self):
        return [
            {
                "timestamp": e.timestamp,
                "type": e.type,
                "files": len(e.files_changed),
                "summary": e.summary,
            }
            for e in self._history
        ]
