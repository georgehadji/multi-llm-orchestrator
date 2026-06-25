"""
WorktreeManager — per-task git worktree isolation for parallel sub-agents.

FIX-2 (Loop Engineering §VI.E): prevents the tangled loop where concurrent
sub-agents write to the same directory and produce unresolvable merge conflicts.

Strategy:
  - Git repo present → git worktree add per task (true isolation)
  - Not a git repo    → per-task temp directory (fallback, still isolated)

On success:  worktree/tempdir removed.
On failure:  quarantined under <base_dir>/quarantine/<task_id> for inspection.
Stale cap:   enforced at acquire time via MAX_LIVE env var (default 20).
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

logger = logging.getLogger("orchestrator.vcs.worktree_manager")

_MAX_LIVE = int(os.getenv("ORCH_WORKTREE_MAX_LIVE", "20"))
_GIT_TIMEOUT = int(os.getenv("ORCH_WORKTREE_GIT_TIMEOUT", "30"))


class WorktreeError(RuntimeError):
    """Raised when a worktree operation cannot complete."""


class WorktreeManager:
    """Manages per-task isolated working directories for parallel sub-agents.

    Usage (preferred — async context manager):
        async with mgr.acquire(task_id) as wt_path:
            # wt_path is an isolated directory; changes are local to task
            ...
        # on exit: cleaned up (success) or quarantined (exception)
    """

    def __init__(
        self,
        base_dir: Path | str | None = None,
        repo_root: Path | str | None = None,
    ) -> None:
        self._base = Path(base_dir) if base_dir else Path(tempfile.gettempdir()) / "orch-worktrees"
        self._repo_root = Path(repo_root) if repo_root else Path.cwd()
        self._base.mkdir(parents=True, exist_ok=True)
        self._live: set[str] = set()

    # ── Public API ───────────────────────────────────────────────────────────

    def is_git_repo(self) -> bool:
        return (self._repo_root / ".git").exists()

    @property
    def quarantine_dir(self) -> Path:
        q = self._base / "_quarantine"
        q.mkdir(parents=True, exist_ok=True)
        return q

    def _worktree_path(self, task_id: str) -> Path:
        safe = task_id.replace("/", "_").replace("\\", "_")
        return self._base / safe

    @asynccontextmanager
    async def acquire(self, task_id: str) -> AsyncIterator[Path]:
        """Yield an isolated working directory for *task_id*.

        On normal exit: directory removed.
        On exception:   directory moved to quarantine for inspection.
        """
        if len(self._live) >= _MAX_LIVE:
            raise WorktreeError(
                f"WorktreeManager: live worktree cap ({_MAX_LIVE}) reached. "
                "Set ORCH_WORKTREE_MAX_LIVE to increase."
            )

        wt_path = self._worktree_path(task_id)
        self._live.add(task_id)
        success = False

        try:
            if self.is_git_repo():
                await self._git_create(task_id, wt_path)
            else:
                await self._tempdir_create(wt_path)

            yield wt_path
            success = True

        except WorktreeError:
            raise
        except Exception:
            raise
        finally:
            self._live.discard(task_id)
            await self._cleanup(task_id, wt_path, success=success)

    # ── Git worktree helpers ─────────────────────────────────────────────────

    async def _git_create(self, task_id: str, wt_path: Path) -> None:
        if wt_path.exists():
            logger.warning("Worktree path already exists, removing stale: %s", wt_path)
            await self._run_git("worktree", "remove", "--force", str(wt_path))

        branch = f"wt/{task_id.replace('/', '_')}"
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: self._run_git_sync("worktree", "add", "-b", branch, str(wt_path), "HEAD"),
        )
        logger.debug("git worktree created: %s (branch %s)", wt_path, branch)

    async def _run_git(self, *args: str) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: self._run_git_sync(*args))

    def _run_git_sync(self, *args: str) -> None:
        cmd = ["git", "-C", str(self._repo_root), *args]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=_GIT_TIMEOUT)
        if result.returncode != 0:
            raise WorktreeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")

    # ── Temp dir helpers ─────────────────────────────────────────────────────

    async def _tempdir_create(self, wt_path: Path) -> None:
        wt_path.mkdir(parents=True, exist_ok=True)
        logger.debug("temp worktree created (non-git fallback): %s", wt_path)

    # ── Cleanup ──────────────────────────────────────────────────────────────

    async def _cleanup(self, task_id: str, wt_path: Path, *, success: bool) -> None:
        if not wt_path.exists():
            return

        if success:
            await self._remove(task_id, wt_path)
        else:
            await self._quarantine(task_id, wt_path)

    async def _remove(self, task_id: str, wt_path: Path) -> None:
        loop = asyncio.get_running_loop()
        try:
            if self.is_git_repo():
                await loop.run_in_executor(
                    None,
                    lambda: self._run_git_sync("worktree", "remove", "--force", str(wt_path)),
                )
            else:
                await loop.run_in_executor(None, lambda: shutil.rmtree(wt_path, ignore_errors=True))
            logger.debug("worktree removed: %s", wt_path)
        except Exception as exc:
            logger.warning("Failed to remove worktree %s: %s", wt_path, exc)

    async def _quarantine(self, task_id: str, wt_path: Path) -> None:
        dest = self.quarantine_dir / task_id.replace("/", "_")
        loop = asyncio.get_running_loop()
        try:
            if dest.exists():
                await loop.run_in_executor(None, lambda: shutil.rmtree(dest, ignore_errors=True))
            await loop.run_in_executor(None, lambda: shutil.copytree(wt_path, dest))
            await loop.run_in_executor(None, lambda: shutil.rmtree(wt_path, ignore_errors=True))
            logger.warning(
                "worktree quarantined after failure — inspect at %s", dest
            )
        except Exception as exc:
            logger.error("Failed to quarantine worktree %s: %s", wt_path, exc)
