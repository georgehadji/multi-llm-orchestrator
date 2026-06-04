"""
CachePathProvider — Single source of truth for orchestrator cache paths
=========================================================================
Replaces 21+ hardcoded ``Path.home() / ".orchestrator_cache"`` references
with a single injectable provider.  All modules that need cache storage
should receive a ``CachePathProvider`` instance rather than constructing
paths ad-hoc.

Usage:
    from orchestrator.infrastructure.path_provider import CachePathProvider

    paths = CachePathProvider()
    state_db = paths.db("state.db")
    # -> ~/.orchestrator_cache/state.db
"""

from __future__ import annotations

from pathlib import Path


class CachePathProvider:
    """Canonical cache-directory provider for the orchestrator.

    All path construction flows through this class so that:
    - The cache root can be overridden in tests (set ``ORCH_CACHE_HOME``).
    - Migration to XDG or platform-appropriate directories is a one-line change.
    - Disk-space and permission checks can be centralized.
    """

    def __init__(self, root: str | Path | None = None) -> None:
        import os

        if root is not None:
            self._root = Path(root)
        else:
            env = os.environ.get("ORCH_CACHE_HOME", "")
            self._root = Path(env) if env else Path.home() / ".orchestrator_cache"

    # ── root ────────────────────────────────────────────────────────────────

    @property
    def root(self) -> Path:
        """Return the cache root directory (created on first access)."""
        self._root.mkdir(parents=True, exist_ok=True)
        return self._root

    # ── database paths ──────────────────────────────────────────────────────

    def db(self, name: str) -> Path:
        """Return ``root / name``, ensuring the parent directory exists."""
        self.root  # trigger mkdir
        return self._root / name

    @property
    def state_db(self) -> Path:
        return self.db("state.db")

    @property
    def cache_db(self) -> Path:
        return self.db("cache.db")

    @property
    def cache_l2_db(self) -> Path:
        return self.db("cache_l2.db")

    @property
    def secure_cache_db(self) -> Path:
        return self.db("secure_cache.db")

    @property
    def kanban_db(self) -> Path:
        return self.db("kanban.db")

    @property
    def patterns_db(self) -> Path:
        return self.db("patterns.db")

    @property
    def telemetry_db(self) -> Path:
        return self.db("telemetry.db")

    @property
    def trajectories_db(self) -> Path:
        return self.db("trajectories.db")

    @property
    def skills_db(self) -> Path:
        return self.db("skills.db")

    @property
    def events_db(self) -> Path:
        return self.db("events.db")

    @property
    def budget_db(self) -> Path:
        return self.db("budget.db")

    # ── sub-directories ─────────────────────────────────────────────────────

    def subdir(self, name: str) -> Path:
        """Return ``root / name``, creating it if needed."""
        p = self._root / name
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def automations_dir(self) -> Path:
        return self.subdir("automations")

    @property
    def rollouts_dir(self) -> Path:
        return self.subdir("rollouts")

    @property
    def hitl_dir(self) -> Path:
        return self.subdir("hitl")

    @property
    def transfer_learning_dir(self) -> Path:
        return self.subdir("transfer_learning")

    @property
    def memory_dir(self) -> Path:
        return self.subdir("memory")

    @property
    def meta_dir(self) -> Path:
        return self.subdir("meta_v2")

    @property
    def archive_dir(self) -> Path:
        return self.subdir("archive")

    @property
    def sessions_dir(self) -> Path:
        return self.subdir("sessions")

    @property
    def restore_points_dir(self) -> Path:
        return self.subdir("restore_points")

    @property
    def checkpoints_dir(self) -> Path:
        return self.subdir("checkpoints")
