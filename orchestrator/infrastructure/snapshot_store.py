"""
Snapshot Store — Content-preserving workspace snapshots
=========================================================
CodeWhale Phase 2 implementation.

Stores actual file contents (not just hashes) enabling true workspace rollback.

Two backends:
    GitSnapshotStore  (primary) — Uses a shadow git repo. Gives git log, diff,
                       checkout for free. Best when git is available.
    TarSnapshotStore  (fallback) — Uses tarfile + SHA-256 manifest. Works
                       anywhere Python does.

Architecture:
    Infrastructure layer adapter — satisfies domain.ports.SnapshotPort.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
import subprocess
import tarfile
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..domain.ports import SnapshotPort

logger = logging.getLogger(__name__)


# ── Default location ─────────────────────────────────────────────────────────


def default_snapshot_dir() -> str:
    """Return default snapshot storage directory."""
    return str(Path.home() / ".orchestrator_cache" / "snapshots")


# ── Git-backed snapshot store (primary) ──────────────────────────────────────


class GitSnapshotStore(SnapshotPort):
    """Content-preserving snapshots stored as git commits in a shadow repo.

    Each snapshot is a git commit. Restoring checks out the commit's tree
    into the target directory. Diff uses native `git diff`.

    The shadow repo is a full working tree under storage_dir/worktree/
    with .git at storage_dir/.git/.
    """

    def __init__(self, storage_dir: str | None = None):
        self._storage = Path(storage_dir or default_snapshot_dir())
        self._work_dir = self._storage / "worktree"
        self._git_dir = self._storage / ".git"
        self._meta_dir = self._storage / "meta"
        self._meta_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ───────────────────────────────────────────────────────────

    async def create(
        self,
        label: str,
        source_dir: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Snapshot source_dir into a git commit.

        Returns the commit SHA (abbreviated, 12 chars).
        """
        source = Path(source_dir)
        if not source.exists():
            raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

        await self._ensure_repo()

        # Sync source into worktree (rsync-style copy)
        await asyncio.to_thread(self._sync_to_worktree, source)

        # Git add + commit
        commit_hash = await self._git_commit(label)

        # Save metadata alongside
        await self._save_meta(commit_hash, label, metadata)

        logger.info("Snapshot '%s' created: %s (%s)", label, commit_hash, source_dir)
        return commit_hash

    async def restore(self, snapshot_id: str, target_dir: str) -> bool:
        """Restore snapshot file contents into target_dir."""
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)

        await self._ensure_repo()

        # Check out the commit into the worktree
        if not await self._git_checkout(snapshot_id):
            logger.warning("Snapshot '%s' not found for restore", snapshot_id)
            return False

        # Sync worktree to target
        await asyncio.to_thread(self._sync_to_target, target)

        logger.info("Restored snapshot '%s' -> %s", snapshot_id, target_dir)
        return True

    async def list_snapshots(self) -> list[dict[str, Any]]:
        """List all snapshots with metadata."""
        await self._ensure_repo()
        snapshots = await self._git_log()

        # Enrich with metadata
        for snap in snapshots:
            sid = snap["id"]
            meta_path = self._meta_dir / f"{sid}.json"
            if meta_path.exists():
                try:
                    snap["metadata"] = json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception:
                    snap["metadata"] = {}

        return snapshots

    async def diff(self, snapshot_a: str, snapshot_b: str) -> dict[str, Any]:
        """Compare two snapshots and return rich diff."""
        await self._ensure_repo()
        raw = await self._git_diff(snapshot_a, snapshot_b)
        return self._parse_diff(raw)

    async def delete(self, snapshot_id: str) -> bool:
        """Remove a snapshot. Note: git doesn't delete easily — this tags it."""
        await self._ensure_repo()
        try:
            # Tag as deleted so we can still recover if needed
            await self._run_git("tag", f"deleted/{snapshot_id}", snapshot_id)
            # Remove metadata file
            meta_path = self._meta_dir / f"{snapshot_id}.json"
            if meta_path.exists():
                meta_path.unlink()
            logger.info("Snapshot '%s' marked as deleted", snapshot_id)
            return True
        except Exception as e:
            logger.warning("Failed to delete snapshot '%s': %s", snapshot_id, e)
            return False

    # ── Internal helpers ─────────────────────────────────────────────────────

    async def _run_git(self, *args: str, timeout: int = 60) -> subprocess.CompletedProcess:
        """Run a git command non-blocking via asyncio.to_thread."""
        return await asyncio.to_thread(
            subprocess.run,
            ["git", *args],
            cwd=str(self._work_dir),
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    async def _ensure_repo(self) -> None:
        """Initialize shadow git repo if not already initialized."""
        self._work_dir.mkdir(parents=True, exist_ok=True)
        if not self._git_dir.exists():
            await self._run_git("init")
            await self._run_git("config", "user.name", "orchestrator-snapshot")
            await self._run_git("config", "user.email", "snapshot@orchestrator.local")
            logger.info("Initialized snapshot git repo at %s", self._work_dir)

    def _sync_to_worktree(self, source: Path) -> None:
        """Copy source directory contents into the worktree."""
        for item in source.iterdir():
            dest = self._work_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)

    def _sync_to_target(self, target: Path) -> None:
        """Copy worktree contents into target directory."""
        for item in self._work_dir.iterdir():
            if item.name == ".git":
                continue
            dest = target / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)

    async def _git_commit(self, label: str) -> str:
        """Git add all + commit with label as message. Returns abbreviated SHA."""
        await self._run_git("add", "-A")

        # Only commit if there are changes
        status = await self._run_git("status", "--porcelain")
        if not status.stdout.strip():
            # No changes — return HEAD SHA
            result = await self._run_git("rev-parse", "--short", "HEAD")
            return result.stdout.strip() or "empty"

        await self._run_git("commit", "-m", label, "--allow-empty")
        sha_result = await self._run_git("rev-parse", "--short", "HEAD")
        return sha_result.stdout.strip()

    async def _git_checkout(self, snapshot_id: str) -> bool:
        """Check out a snapshot commit. Returns False if not found."""
        # First check if the ref exists
        result = await self._run_git("cat-file", "-t", snapshot_id)
        if result.returncode != 0:
            return False

        # Reset worktree to that commit
        await self._run_git("checkout", "--force", snapshot_id)
        return True

    async def _git_log(self) -> list[dict[str, Any]]:
        """Parse git log into structured snapshots list."""
        result = await self._run_git("log", "--format=%H|%h|%s|%ct", "--max-count=100")
        if result.returncode != 0 or not result.stdout.strip():
            return []

        snapshots = []
        for line in result.stdout.strip().splitlines():
            parts = line.split("|", 3)
            if len(parts) >= 4:
                snapshots.append(
                    {
                        "id": parts[1],
                        "label": parts[2],
                        "timestamp": int(parts[3]),
                        "full_id": parts[0],
                    }
                )
        return snapshots

    async def _git_diff(self, a: str, b: str) -> str:
        """Run git diff between two refs."""
        result = await self._run_git("diff", "--unified=3", a, b)
        return result.stdout

    @staticmethod
    def _parse_diff(raw: str) -> dict[str, Any]:
        """Parse unified diff into structured format."""
        added: list[str] = []
        removed: list[str] = []
        modified: list[str] = []
        file_diffs: dict[str, str] = {}

        current_file = ""
        for line in raw.splitlines():
            if line.startswith("--- a/"):
                continue
            if line.startswith("+++ b/"):
                parts = line.split("/", 1)
                current_file = parts[1] if len(parts) > 1 else ""
                continue
            if line.startswith("diff --git"):
                continue
            if line.startswith("new file mode"):
                if current_file:
                    added.append(current_file)
                continue
            if line.startswith("deleted file mode"):
                if current_file:
                    removed.append(current_file)
                continue
            if line.startswith("--- /dev/null"):
                if current_file and current_file not in added:
                    added.append(current_file)
                continue
            if line.startswith("--- ") and current_file:
                # File existed before — it's a modification
                if current_file not in added and current_file not in removed:
                    modified.append(current_file)

        # Collect the actual diffs per file
        if raw:
            # Simple heuristic: split on "diff --git" markers
            chunks = raw.split("diff --git ")
            for chunk in chunks[1:]:  # skip preamble
                lines = chunk.splitlines()
                if lines:
                    # extract filename from first line
                    fn_parts = lines[0].split(" b/", 1)
                    fn = fn_parts[-1] if len(fn_parts) > 1 else "unknown"
                    file_diffs[fn] = "diff --git " + chunk

        return {
            "added_files": sorted(set(added)),
            "removed_files": sorted(set(removed)),
            "modified_files": sorted(set(modified)),
            "file_diffs": file_diffs,
        }

    async def _save_meta(self, snapshot_id: str, label: str, metadata: dict | None) -> None:
        """Persist metadata alongside the snapshot."""
        meta = {
            "snapshot_id": snapshot_id,
            "label": label,
            "timestamp": int(time.time()),
            "metadata": metadata or {},
        }
        meta_path = self._meta_dir / f"{snapshot_id}.json"
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


# ── Tar-based snapshot store (fallback when git unavailable) ─────────────────


class TarSnapshotStore(SnapshotPort):
    """Fallback snapshot store using tarfile + SHA-256 manifest.

    Each snapshot is a .tar.xz file with an accompanying manifest.json.
    """

    def __init__(self, storage_dir: str | None = None):
        self._storage = Path(storage_dir or default_snapshot_dir() + "_tar")
        self._storage.mkdir(parents=True, exist_ok=True)

    async def create(
        self,
        label: str,
        source_dir: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        source = Path(source_dir)
        if not source.exists():
            raise FileNotFoundError(f"Source dir not found: {source_dir}")

        # Generate unique id from content hash + timestamp
        content_hash = self._hash_dir(source)
        snapshot_id = f"{content_hash[:12]}-{int(time.time())}"
        archive_path = self._storage / f"{snapshot_id}.tar.xz"

        # Create tar archive with xz compression
        with tarfile.open(archive_path, "w:xz") as tar:
            for item in source.rglob("*"):
                if item.is_file():
                    tar.add(item, arcname=item.relative_to(source))

        # Write manifest
        manifest = {
            "snapshot_id": snapshot_id,
            "label": label,
            "timestamp": int(time.time()),
            "content_hash": content_hash,
            "file_count": len(list(source.rglob("*"))),
            "metadata": metadata or {},
        }
        manifest_path = self._storage / f"{snapshot_id}.manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        logger.info(
            "TarSnapshot '%s' created: %s (%s files)", label, snapshot_id, manifest["file_count"]
        )
        return snapshot_id

    async def restore(self, snapshot_id: str, target_dir: str) -> bool:
        archive_path = self._storage / f"{snapshot_id}.tar.xz"
        if not archive_path.exists():
            logger.warning("TarSnapshot '%s' not found", snapshot_id)
            return False

        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)

        def _safe_members(members):
            for m in members:
                dest = os.path.normpath(os.path.join(str(target), m.name))
                if not dest.startswith(os.path.normpath(str(target))):
                    raise ValueError(f"Path traversal blocked: {m.name}")
                yield m

        with tarfile.open(archive_path, "r:xz") as tar:
            tar.extractall(path=str(target), members=_safe_members(tar.getmembers()))

        logger.info("Restored TarSnapshot '%s' -> %s", snapshot_id, target_dir)
        return True

    async def list_snapshots(self) -> list[dict[str, Any]]:
        snapshots = []
        for mf in sorted(self._storage.glob("*.manifest.json"), reverse=True):
            try:
                data = json.loads(mf.read_text(encoding="utf-8"))
                snapshots.append(data)
            except Exception:
                pass
        return snapshots

    async def diff(self, snapshot_a: str, snapshot_b: str) -> dict[str, Any]:
        # For tar, we restore both to temp dirs and compare
        with (
            tempfile.TemporaryDirectory(prefix="snap_diff_a_") as tmp_a,
            tempfile.TemporaryDirectory(prefix="snap_diff_b_") as tmp_b,
        ):
            ok_a = await self.restore(snapshot_a, tmp_a)
            ok_b = await self.restore(snapshot_b, tmp_b)
            if not ok_a or not ok_b:
                return {
                    "error": "One or both snapshots not found",
                    "added_files": [],
                    "removed_files": [],
                    "modified_files": [],
                    "file_diffs": {},
                }

            return self._diff_dirs(Path(tmp_a), Path(tmp_b))

    async def delete(self, snapshot_id: str) -> bool:
        deleted = False
        for p in self._storage.glob(f"{snapshot_id}.*"):
            p.unlink()
            deleted = True
        return deleted

    # ── Internal helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _hash_dir(path: Path) -> str:
        """Compute a combined SHA-256 for all files in a directory."""
        hasher = hashlib.sha256()
        for f in sorted(path.rglob("*")):
            if f.is_file():
                hasher.update(f.read_bytes())
        return hasher.hexdigest()

    @staticmethod
    def _diff_dirs(a: Path, b: Path) -> dict[str, Any]:
        """Compare two directories by file listing and content hashes."""
        files_a = {
            f.relative_to(a): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in a.rglob("*")
            if f.is_file()
        }
        files_b = {
            f.relative_to(b): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in b.rglob("*")
            if f.is_file()
        }

        set_a = set(files_a.keys())
        set_b = set(files_b.keys())

        added = [str(p) for p in sorted(set_b - set_a)]
        removed = [str(p) for p in sorted(set_a - set_b)]
        modified = [str(p) for p in sorted(set_a & set_b) if files_a[p] != files_b[p]]

        return {
            "added_files": added,
            "removed_files": removed,
            "modified_files": modified,
            "file_diffs": {},
        }
