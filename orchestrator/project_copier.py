"""
CopyProject - Duplicate project for safe experimentation.
===========================================================
Author: Georgios-Chrysovalantis Chatzivantsidis
Part of Category 2, Phase D5 (Dyad-inspired).
"""

from __future__ import annotations
import shutil, time, logging, json
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ProjectCopy:
    copy_id: str
    original_dir: str
    copy_dir: str
    created_at: float = 0.0
    description: str = ""
    experiments: list = field(default_factory=list)

    def to_dict(self):
        return {
            "copy_id": self.copy_id,
            "original": self.original_dir,
            "copy": self.copy_dir,
            "created_at": self.created_at,
            "description": self.description,
            "experiments": self.experiments,
        }


class ProjectCopier:
    """Creates and manages project copies for safe experimentation."""

    def __init__(self, workspace="."):
        self.workspace = Path(workspace)
        self._copies: list[ProjectCopy] = []
        self._copies_dir = self.workspace / ".project_copies"
        self._copies_dir.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self):
        fp = self._copies_dir / "copies.json"
        if fp.exists():
            try:
                self._copies = [
                    ProjectCopy(**d) for d in json.loads(fp.read_text(encoding="utf-8"))
                ]
            except Exception:
                pass

    def _save(self):
        (self._copies_dir / "copies.json").write_text(
            json.dumps([c.to_dict() for c in self._copies], indent=2), encoding="utf-8"
        )

    def duplicate(self, project_dir, description=""):
        """Create a complete copy of a project for experimentation."""
        src = Path(project_dir)
        if not src.exists():
            return None
        copy_id = f"copy_{int(time.time())}"
        dst = self._copies_dir / copy_id
        shutil.copytree(
            src,
            dst,
            ignore=shutil.ignore_patterns(
                ".git", "__pycache__", "*.pyc", ".project_copies", "node_modules", ".venv", "venv"
            ),
        )

        cp = ProjectCopy(
            copy_id=copy_id,
            original_dir=str(src),
            copy_dir=str(dst),
            created_at=time.time(),
            description=description,
        )
        self._copies.append(cp)
        self._save()
        logger.info(f"Project duplicated to {dst}")
        return cp

    def diff(self, copy_id):
        """Compare a copy against its original."""
        cp = next((c for c in self._copies if c.copy_id == copy_id), None)
        if not cp:
            return {}
        import difflib, os

        diff = {}
        orig = Path(cp.original_dir)
        dup = Path(cp.copy_dir)
        for root, dirs, files in os.walk(str(dup)):
            for f in files:
                rel = Path(root).relative_to(dup) / f
                dup_file = dup / rel
                orig_file = orig / rel
                dup_content = dup_file.read_text(encoding="utf-8") if dup_file.exists() else ""
                orig_content = orig_file.read_text(encoding="utf-8") if orig_file.exists() else ""
                if dup_content != orig_content:
                    d = difflib.unified_diff(
                        orig_content.splitlines(),
                        dup_content.splitlines(),
                        fromfile=f"original/{rel}",
                        tofile=f"copy/{rel}",
                    )
                    diff[str(rel)] = "\n".join(d)
        return diff

    def merge_back(self, copy_id, files=None):
        """Merge changes from a copy back to the original."""
        cp = next((c for c in self._copies if c.copy_id == copy_id), None)
        if not cp:
            return 0
        orig = Path(cp.original_dir)
        dup = Path(cp.copy_dir)
        merged = 0
        for f in dup.rglob("*"):
            if f.is_file():
                rel = f.relative_to(dup)
                if files and str(rel) not in files:
                    continue
                orig_file = orig / rel
                orig_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, orig_file)
                merged += 1
        logger.info(f"Merged {merged} files back to {cp.original_dir}")
        return merged

    def list_copies(self):
        return [
            {
                "id": c.copy_id,
                "original": c.original_dir,
                "description": c.description,
                "created": time.strftime("%H:%M", time.localtime(c.created_at)),
            }
            for c in self._copies
        ]
