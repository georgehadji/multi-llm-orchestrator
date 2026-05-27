"""
ProjectSiteSeparation - Dev workspace vs published site.
==========================================================
Author: Georgios-Chrysovalantis Chatzivantsidis
Part of Category 2, Phase W4 (Bolt.new-inspired).
"""

from __future__ import annotations
import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PublishState:
    version: str = "0.1.0"
    published_at: float = 0.0
    published_dir: str = ""
    files_count: int = 0
    description: str = ""

    def to_dict(self):
        return {
            "version": self.version,
            "published_at": self.published_at,
            "published_dir": self.published_dir,
            "files_count": self.files_count,
            "description": self.description,
        }


class SiteManager:
    """Separates dev workspace from published site with explicit publish step."""

    def __init__(self, project_dir=".", site_dir=None):
        self.project_dir = Path(project_dir)
        self._dev_dir = self.project_dir / ".dev"
        self._site_dir = self.project_dir / (site_dir or "site")
        self._dev_dir.mkdir(parents=True, exist_ok=True)
        self._history: list[PublishState] = []
        self._load()

    def _load(self):
        fp = self._dev_dir / "publish_history.json"
        if fp.exists():
            try:
                self._history = [
                    PublishState(**d) for d in json.loads(fp.read_text(encoding="utf-8"))
                ]
            except Exception:
                pass

    def _save(self):
        (self._dev_dir / "publish_history.json").write_text(
            json.dumps([h.to_dict() for h in self._history], indent=2), encoding="utf-8"
        )

    @property
    def is_published(self):
        return len(self._history) > 0

    @property
    def latest_version(self):
        return self._history[-1].version if self._history else "0.0.0"

    @property
    def unpublished_changes(self):
        """Check if dev workspace has changes since last publish."""
        if not self._history:
            return True
        last = self._history[-1]
        last_dir = Path(last.published_dir) if last.published_dir else None
        if not last_dir or not last_dir.exists():
            return True
        import hashlib

        for f in self._dev_dir.rglob("*"):
            if f.is_file():
                rel = f.relative_to(self._dev_dir)
                site_file = last_dir / rel
                if not site_file.exists():
                    return True
                if (
                    hashlib.sha256(f.read_bytes()).hexdigest()
                    != hashlib.sha256(site_file.read_bytes()).hexdigest()
                ):
                    return True
        return False

    def publish(self, version=None, description=""):
        """Publish the dev workspace to the site directory."""
        if version is None:
            parts = self.latest_version.split(".")
            parts[-1] = str(int(parts[-1]) + 1)
            version = ".".join(parts)

        pub_dir = self._site_dir / f"v{version}"
        if pub_dir.exists():
            shutil.rmtree(pub_dir)
        shutil.copytree(
            self._dev_dir,
            pub_dir,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git", ".dev", "node_modules"),
        )

        files_count = sum(1 for _ in pub_dir.rglob("*") if _.is_file())
        ps = PublishState(
            version=version,
            published_at=time.time(),
            published_dir=str(pub_dir),
            files_count=files_count,
            description=description,
        )
        self._history.append(ps)
        self._save()
        logger.info(f"Published v{version} ({files_count} files)")
        return ps

    def rollback_publish(self, version):
        """Rollback site to a previous published version."""
        pub_dir = self._site_dir / f"v{version}"
        if not pub_dir.exists():
            return False
        if self._dev_dir.exists():
            shutil.rmtree(self._dev_dir)
        shutil.copytree(pub_dir, self._dev_dir)
        logger.info(f"Rolled back to v{version}")
        return True

    def history(self):
        return [h.to_dict() for h in self._history]
