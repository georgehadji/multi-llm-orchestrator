"""
Infrastructure adapter for SQLite persistence used by SkillStore.

Extracts the ``aiosqlite`` dependency from ``application.skill_store``
so the application layer depends only on ``domain.ports.SkillStorePort``,
not on concrete database libraries.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("orchestrator.infrastructure.skill_store_adapter")


class SkillDbAdapter:
    """Wraps aiosqlite connections for skill/trajectory persistence.

    Provides the same CRUD methods as the old SkillStore,
    but from the infrastructure layer.
    """

    def __init__(
        self,
        traj_path: Path | None = None,
        skill_path: Path | None = None,
    ) -> None:
        self._traj_path = traj_path or Path.home() / ".orchestrator_cache" / "trajectories.db"
        self._skill_path = skill_path or Path.home() / ".orchestrator_cache" / "skills.db"
        self._traj_db: Any = None  # aiosqlite.Connection
        self._skill_db: Any = None  # aiosqlite.Connection

    async def connect(self) -> None:
        """Open both databases and ensure schemas exist."""
        import aiosqlite as _aio

        self._traj_path.parent.mkdir(parents=True, exist_ok=True)
        self._skill_path.parent.mkdir(parents=True, exist_ok=True)

        self._traj_db = await _aio.connect(self._traj_path)
        self._traj_db.row_factory = _aio.Row
        await self._traj_db.executescript(
            """
            CREATE TABLE IF NOT EXISTS trajectories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                task_type TEXT NOT NULL,
                prompt TEXT,
                output TEXT,
                score REAL,
                critique_text TEXT,
                model_used TEXT,
                cost_usd REAL,
                recorded_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_trajectories_type
                ON trajectories (task_type, recorded_at DESC);
        """
        )
        await self._traj_db.commit()

        self._skill_db = await _aio.connect(self._skill_path)
        self._skill_db.row_factory = _aio.Row
        await self._skill_db.executescript(
            """
            CREATE TABLE IF NOT EXISTS skills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_type TEXT NOT NULL,
                skill_doc TEXT NOT NULL,
                score REAL,
                epoch INTEGER NOT NULL DEFAULT 0,
                is_best INTEGER NOT NULL DEFAULT 1,
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_skills_type_best
                ON skills (task_type, is_best, epoch DESC);
            CREATE TABLE IF NOT EXISTS skill_patches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_type TEXT NOT NULL,
                epoch INTEGER NOT NULL,
                patch_op TEXT NOT NULL,
                anchor TEXT NOT NULL DEFAULT '',
                content TEXT NOT NULL DEFAULT '',
                token_cost INTEGER NOT NULL DEFAULT 0,
                accepted INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS negative_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_type TEXT NOT NULL,
                patches_json TEXT NOT NULL,
                rejection_reason TEXT NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_nfb_type_time
                ON negative_feedback (task_type, created_at DESC);
        """
        )
        await self._skill_db.commit()
        logger.debug(
            "SkillDbAdapter connected: traj=%s skill=%s", self._traj_path, self._skill_path
        )

    async def close(self) -> None:
        """Close both databases."""
        for db in (self._traj_db, self._skill_db):
            if db is not None:
                await db.close()
        logger.debug("SkillDbAdapter closed")

    async def execute_traj(self, sql: str, params: tuple = ()) -> Any:
        """Execute on the trajectories database and return cursor."""
        assert self._traj_db is not None, "SkillDbAdapter not connected"
        cur = await self._traj_db.execute(sql, params)
        return cur

    async def execute_skill(self, sql: str, params: tuple = ()) -> Any:
        """Execute on the skills database and return cursor."""
        assert self._skill_db is not None, "SkillDbAdapter not connected"
        cur = await self._skill_db.execute(sql, params)
        return cur

    async def commit_traj(self) -> None:
        """Commit the trajectories database."""
        if self._traj_db is not None:
            await self._traj_db.commit()

    async def commit_skill(self) -> None:
        """Commit the skills database."""
        if self._skill_db is not None:
            await self._skill_db.commit()

    async def fetchone_traj(self, sql: str, params: tuple = ()) -> Any:
        """Execute and fetch one row from trajectories."""
        cur = await self.execute_traj(sql, params)
        return await cur.fetchone()

    async def fetchall_traj(self, sql: str, params: tuple = ()) -> list[Any]:
        """Execute and fetch all rows from trajectories."""
        cur = await self.execute_traj(sql, params)
        return await cur.fetchall()

    async def fetchall_skill(self, sql: str, params: tuple = ()) -> list[Any]:
        """Execute and fetch all rows from skills."""
        cur = await self.execute_skill(sql, params)
        return await cur.fetchall()

    async def fetchone_skill(self, sql: str, params: tuple = ()) -> Any:
        """Execute and fetch one row from skills."""
        cur = await self.execute_skill(sql, params)
        return await cur.fetchone()

    async def executemany_skill(self, sql: str, params_list: list[tuple]) -> None:
        """Execute executemany on skills database."""
        assert self._skill_db is not None
        await self._skill_db.executemany(sql, params_list)
