"""
CrossProjectReferencer - Reference code across projects.
===========================================================
Author: Georgios-Chrysovalantis Chatzivantsidis
Part of Category 3, Phase 9 (Lovable-inspired): @ProjectName reuse.
"""

from __future__ import annotations
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ProjectReference:
    name: str
    path: str
    files: dict = field(default_factory=dict)
    summary: str = ""

    def to_dict(self):
        return {"name": self.name, "path": self.path, "files": self.files, "summary": self.summary}

    @classmethod
    def from_dict(cls, d):
        return cls(
            name=d["name"],
            path=d.get("path", ""),
            files=d.get("files", {}),
            summary=d.get("summary", ""),
        )


class CrossProjectReferencer:
    """Resolves @ProjectName references to reuse code across projects."""

    def __init__(self, projects_root="outputs"):
        self._root = Path(projects_root)
        self._projects: dict[str, ProjectReference] = {}
        self._load()

    def _load(self):
        fp = self._root / ".project_index.json"
        if fp.exists():
            try:
                for d in json.loads(fp.read_text(encoding="utf-8")):
                    ref = ProjectReference.from_dict(d)
                    self._projects[ref.name] = ref
            except Exception:
                pass

    def _save(self):
        self._root.mkdir(parents=True, exist_ok=True)
        (self._root / ".project_index.json").write_text(
            json.dumps([p.to_dict() for p in self._projects.values()], indent=2), encoding="utf-8"
        )

    def index(self, name, project_dir, summary=""):
        """Register a project for cross-referencing."""
        path = Path(project_dir)
        if not path.exists():
            return None
        files = {}
        for f in path.rglob("*.py"):
            if f.is_file() and f.stat().st_size < 100000:
                files[str(f.relative_to(path))] = f.read_text(encoding="utf-8")[:5000]
        ref = ProjectReference(name=name, path=str(path), files=files, summary=summary)
        self._projects[name] = ref
        self._save()
        logger.info(f"Indexed project '{name}' with {len(files)} files")
        return ref

    def resolve(self, text):
        """Resolve @ProjectName references in text to file content.

        Replaces @ProjectName/path/to/file.py with the actual file content.
        """
        pattern = r"@(\w+)(?:/([^\s]+))?"

        def replacer(match):
            name = match.group(1)
            file_path = match.group(2)
            ref = self._projects.get(name)
            if not ref:
                return match.group(0)
            if file_path and file_path in ref.files:
                return f"```python\n{ref.files[file_path]}\n```"
            elif ref.summary:
                return f"/* Project {name}: {ref.summary} */"
            return match.group(0)

        return re.sub(pattern, replacer, text)

    def find_file(self, project_name, filename):
        """Find a specific file in a referenced project."""
        ref = self._projects.get(project_name)
        if not ref:
            return None
        path = ref.path
        fp = Path(path) / filename
        return fp.read_text(encoding="utf-8") if fp.exists() else None

    def list_projects(self):
        return [
            {"name": p.name, "path": p.path, "files": len(p.files), "summary": p.summary}
            for p in self._projects.values()
        ]
