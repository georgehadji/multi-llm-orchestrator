"""
TargetLockFiles - AI scope control: focus and lock files.
============================================================
Author: Georgios-Chrysovalantis Chatzivantsidis
Part of Category 12, Phase W2 (Bolt.new-inspired).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import fnmatch
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class ScopeConfig:
    target_files: list = field(default_factory=list)  # Files AI CAN modify
    locked_files: list = field(default_factory=list)  # Files AI CANNOT modify
    locked_dirs: list = field(default_factory=list)  # Directories AI CANNOT touch
    target_patterns: list = field(default_factory=list)  # Glob patterns for target
    lock_patterns: list = field(default_factory=list)  # Glob patterns for lock

    def to_dict(self):
        return {
            "target_files": self.target_files,
            "locked_files": self.locked_files,
            "locked_dirs": self.locked_dirs,
            "target_patterns": self.target_patterns,
            "lock_patterns": self.lock_patterns,
        }


class FileScopeManager:
    """Controls which files AI can and cannot modify."""

    def __init__(self, config_dir="."):
        self._dir = Path(config_dir)
        self.config = ScopeConfig()
        self._load()

    def _load(self):
        fp = self._dir / ".ai_scope.json"
        if fp.exists():
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
                self.config = ScopeConfig(**data)
            except Exception:
                pass

    def save(self):
        (self._dir / ".ai_scope.json").write_text(
            json.dumps(self.config.to_dict(), indent=2), encoding="utf-8"
        )

    def target(self, files):
        """Set target files — AI can only modify these."""
        self.config.target_files = files if isinstance(files, list) else [files]
        self.save()

    def lock(self, files):
        """Lock files — AI cannot modify these."""
        self.config.locked_files = files if isinstance(files, list) else [files]
        self.save()

    def lock_dir(self, dirs):
        self.config.locked_dirs = dirs if isinstance(dirs, list) else [dirs]
        self.save()

    def can_modify(self, filepath):
        """Check if AI can modify this file."""
        path = str(filepath)
        # Check locked files/dirs first
        for locked in self.config.locked_files:
            if path.endswith(locked) or Path(path).name == locked:
                return False
        for locked_dir in self.config.locked_dirs:
            if locked_dir in path:
                return False
        for pattern in self.config.lock_patterns:
            if fnmatch.fnmatch(Path(path).name, pattern):
                return False
        # If targets specified, must be in target list
        if self.config.target_files:
            for target in self.config.target_files:
                if path.endswith(target) or Path(path).name == target:
                    return True
            for pattern in self.config.target_patterns:
                if fnmatch.fnmatch(Path(path).name, pattern):
                    return True
            return False
        return True

    def filter_files(self, files):
        """Filter a list of files, returning only those AI can modify."""
        return [f for f in files if self.can_modify(f)]

    def scoped_files(self, root="."):
        """List all files in scope (that AI can modify)."""
        root_path = Path(root)
        result = []
        for f in root_path.rglob("*"):
            if f.is_file():
                rel = str(f.relative_to(root_path))
                if self.can_modify(rel):
                    result.append(rel)
        return result