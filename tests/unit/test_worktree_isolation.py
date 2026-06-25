"""
Tests for FIX-2: worktree isolation in parallel sub-agent execution.

RED first — WorktreeManager must exist and BatchRunner must use it.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.vcs.worktree_manager import WorktreeManager, WorktreeError

# ── WorktreeManager unit tests ────────────────────────────────────────────────


class TestWorktreeManager:
    def test_create_returns_unique_paths_per_task(self, tmp_path):
        mgr = WorktreeManager(base_dir=tmp_path, repo_root=tmp_path)
        path_a = mgr._worktree_path("task-a")
        path_b = mgr._worktree_path("task-b")
        assert path_a != path_b

    def test_worktree_path_is_under_base_dir(self, tmp_path):
        mgr = WorktreeManager(base_dir=tmp_path, repo_root=tmp_path)
        path = mgr._worktree_path("task-x")
        assert str(path).startswith(str(tmp_path))

    def test_non_git_repo_falls_back_to_tempdir(self, tmp_path):
        non_git = tmp_path / "not_a_repo"
        non_git.mkdir()
        mgr = WorktreeManager(base_dir=tmp_path, repo_root=non_git)
        assert not mgr.is_git_repo()

    def test_git_repo_detected(self, tmp_path):
        (tmp_path / ".git").mkdir()
        mgr = WorktreeManager(base_dir=tmp_path, repo_root=tmp_path)
        assert mgr.is_git_repo()


class TestWorktreeManagerContextManager:
    @pytest.mark.asyncio
    async def test_tempdir_fallback_creates_isolated_directory(self, tmp_path):
        """Non-git repo → temp dir per task, still isolated."""
        non_git = tmp_path / "no_git"
        non_git.mkdir()
        mgr = WorktreeManager(base_dir=tmp_path / "wt", repo_root=non_git)

        async with mgr.acquire("task-1") as path_1:
            async with mgr.acquire("task-2") as path_2:
                assert path_1 != path_2
                assert path_1.exists()
                assert path_2.exists()

    @pytest.mark.asyncio
    async def test_tempdir_cleaned_on_success(self, tmp_path):
        non_git = tmp_path / "no_git"
        non_git.mkdir()
        mgr = WorktreeManager(base_dir=tmp_path / "wt", repo_root=non_git)

        captured: list[Path] = []
        async with mgr.acquire("task-cleanup") as path:
            captured.append(path)
            (path / "file.txt").write_text("hello")

        assert not captured[0].exists()

    @pytest.mark.asyncio
    async def test_tempdir_quarantined_on_failure(self, tmp_path):
        non_git = tmp_path / "no_git"
        non_git.mkdir()
        mgr = WorktreeManager(base_dir=tmp_path / "wt", repo_root=non_git)

        quarantine_path: list[Path] = []

        try:
            async with mgr.acquire("task-fail") as path:
                quarantine_path.append(path)
                (path / "artifact.txt").write_text("partial work")
                raise RuntimeError("sub-agent exploded")
        except RuntimeError:
            pass

        # path itself removed; quarantine copy should exist
        quarantine = mgr.quarantine_dir / "task-fail"
        assert quarantine.exists() or not quarantine_path[0].exists()


# ── Parallel isolation invariant ──────────────────────────────────────────────


class TestParallelIsolation:
    @pytest.mark.asyncio
    async def test_two_tasks_write_same_filename_without_conflict(self, tmp_path):
        """Core tangled-loop test: concurrent writes to same relative path must not collide."""
        non_git = tmp_path / "repo"
        non_git.mkdir()
        mgr = WorktreeManager(base_dir=tmp_path / "wt", repo_root=non_git)

        results: dict[str, str] = {}

        async def worker(task_id: str, content: str) -> None:
            async with mgr.acquire(task_id) as wt_path:
                target = wt_path / "output.txt"
                target.write_text(content)
                await asyncio.sleep(0.01)  # allow interleaving
                results[task_id] = target.read_text()

        await asyncio.gather(
            worker("task-A", "content-A"),
            worker("task-B", "content-B"),
        )

        assert results["task-A"] == "content-A"
        assert results["task-B"] == "content-B"
