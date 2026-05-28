"""
SkillStore — async SQLite persistence for SkillOpt
====================================================
Two databases:

  ~/.orchestrator_cache/trajectories.db — task execution observations
  ~/.orchestrator_cache/skills.db       — skill documents, patch history,
                                          and negative-feedback buffer

Design follows the PatternStore / TelemetryStore append-only pattern:
- Writes are INSERT-only (no UPDATE on history tables)
- ``skills`` table marks previous rows is_best=0 before inserting the new best
- ``negative_feedback`` accumulates rejected patches for the optimizer to learn from

Implements domain.ports.SkillStorePort.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import aiosqlite

from ..models import TaskType
from ..models_skill import SkillPatch, Trajectory

logger = logging.getLogger("orchestrator.skill_store")

_DEFAULT_TRAJ_PATH = Path.home() / ".orchestrator_cache" / "trajectories.db"
_DEFAULT_SKILL_PATH = Path.home() / ".orchestrator_cache" / "skills.db"

# ─────────────────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────────────────

_TRAJ_SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS trajectories (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id      TEXT NOT NULL,
    task_type    TEXT NOT NULL,
    prompt       TEXT NOT NULL,
    output       TEXT NOT NULL,
    score        REAL NOT NULL,
    critique_text TEXT NOT NULL DEFAULT '',
    model_used   TEXT NOT NULL DEFAULT '',
    cost_usd     REAL NOT NULL DEFAULT 0.0,
    recorded_at  REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_traj_type_score
    ON trajectories (task_type, score DESC);

CREATE INDEX IF NOT EXISTS idx_traj_type_time
    ON trajectories (task_type, recorded_at DESC);
"""

_SKILL_SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS skills (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    task_type  TEXT NOT NULL,
    skill_doc  TEXT NOT NULL,
    score      REAL NOT NULL,
    epoch      INTEGER NOT NULL DEFAULT 0,
    is_best    INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_skills_type_best
    ON skills (task_type, is_best, epoch DESC);

CREATE TABLE IF NOT EXISTS skill_patches (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    task_type  TEXT NOT NULL,
    epoch      INTEGER NOT NULL,
    patch_op   TEXT NOT NULL,
    anchor     TEXT NOT NULL DEFAULT '',
    content    TEXT NOT NULL DEFAULT '',
    token_cost INTEGER NOT NULL DEFAULT 0,
    accepted   INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS negative_feedback (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    task_type        TEXT NOT NULL,
    patches_json     TEXT NOT NULL,
    rejection_reason TEXT NOT NULL,
    created_at       REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_nfb_type_time
    ON negative_feedback (task_type, created_at DESC);
"""


# ─────────────────────────────────────────────────────────────────────────────
# SkillStore
# ─────────────────────────────────────────────────────────────────────────────


class SkillStore:
    """SQLite-backed implementation of SkillStorePort.

    Call ``await store.connect()`` before use; ``await store.close()`` when done.
    Both databases are opened lazily on first use if connect() is not called
    explicitly (convenience for unit tests).
    """

    def __init__(
        self,
        traj_path: Path = _DEFAULT_TRAJ_PATH,
        skill_path: Path = _DEFAULT_SKILL_PATH,
    ) -> None:
        self._traj_path = traj_path
        self._skill_path = skill_path
        self._traj_db: aiosqlite.Connection | None = None
        self._skill_db: aiosqlite.Connection | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open both databases and ensure schemas exist."""
        self._traj_path.parent.mkdir(parents=True, exist_ok=True)
        self._skill_path.parent.mkdir(parents=True, exist_ok=True)

        self._traj_db = await aiosqlite.connect(self._traj_path)
        self._traj_db.row_factory = aiosqlite.Row
        await self._traj_db.executescript(_TRAJ_SCHEMA)
        await self._traj_db.commit()

        self._skill_db = await aiosqlite.connect(self._skill_path)
        self._skill_db.row_factory = aiosqlite.Row
        await self._skill_db.executescript(_SKILL_SCHEMA)
        await self._skill_db.commit()

        logger.debug("SkillStore connected: traj=%s skill=%s", self._traj_path, self._skill_path)

    async def _ensure_connected(self) -> None:
        if self._traj_db is None or self._skill_db is None:
            await self.connect()

    async def close(self) -> None:
        if self._traj_db is not None:
            await self._traj_db.close()
            self._traj_db = None
        if self._skill_db is not None:
            await self._skill_db.close()
            self._skill_db = None

    # ------------------------------------------------------------------
    # Trajectories
    # ------------------------------------------------------------------

    async def save_trajectory(self, t: Trajectory) -> None:
        """Append one trajectory observation."""
        await self._ensure_connected()
        assert self._traj_db is not None
        await self._traj_db.execute(
            """
            INSERT INTO trajectories
                (task_id, task_type, prompt, output, score, critique_text,
                 model_used, cost_usd, recorded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                t.task_id,
                t.task_type.value,
                t.prompt[:4000],   # cap very long prompts
                t.output[:8000],   # cap very long outputs
                t.score,
                t.critique_text[:2000],
                t.model_used,
                t.cost_usd,
                t.recorded_at,
            ),
        )
        await self._traj_db.commit()

    async def load_trajectories(self, task_type: TaskType, limit: int = 50) -> list[Trajectory]:
        """Return the most recent *limit* trajectories for *task_type*, oldest-first."""
        await self._ensure_connected()
        assert self._traj_db is not None
        async with self._traj_db.execute(
            """
            SELECT task_id, task_type, prompt, output, score, critique_text,
                   model_used, cost_usd, recorded_at
            FROM   trajectories
            WHERE  task_type = ?
            ORDER  BY recorded_at DESC
            LIMIT  ?
            """,
            (task_type.value, limit),
        ) as cursor:
            rows = await cursor.fetchall()

        # Reverse so caller receives oldest-first (natural training order)
        return [
            Trajectory(
                task_id=row["task_id"],
                task_type=task_type,
                prompt=row["prompt"],
                output=row["output"],
                score=row["score"],
                critique_text=row["critique_text"],
                model_used=row["model_used"],
                cost_usd=row["cost_usd"],
                recorded_at=row["recorded_at"],
            )
            for row in reversed(rows)
        ]

    # ------------------------------------------------------------------
    # Skills
    # ------------------------------------------------------------------

    async def save_skill(
        self, task_type: TaskType, skill_doc: str, score: float, epoch: int
    ) -> None:
        """Persist a new best skill and demote the previous best row."""
        await self._ensure_connected()
        assert self._skill_db is not None
        now = time.time()
        # Demote previous best
        await self._skill_db.execute(
            "UPDATE skills SET is_best = 0 WHERE task_type = ? AND is_best = 1",
            (task_type.value,),
        )
        # Insert new best
        await self._skill_db.execute(
            """
            INSERT INTO skills (task_type, skill_doc, score, epoch, is_best, created_at)
            VALUES (?, ?, ?, ?, 1, ?)
            """,
            (task_type.value, skill_doc, score, epoch, now),
        )
        await self._skill_db.commit()
        logger.info("SkillStore: saved skill %s epoch=%d score=%.3f", task_type.value, epoch, score)

    async def load_best_skill(
        self, task_type: TaskType
    ) -> tuple[str, float, int] | None:
        """Return (skill_doc, score, epoch) for the current best skill, or None."""
        await self._ensure_connected()
        assert self._skill_db is not None
        async with self._skill_db.execute(
            """
            SELECT skill_doc, score, epoch
            FROM   skills
            WHERE  task_type = ? AND is_best = 1
            ORDER  BY epoch DESC
            LIMIT  1
            """,
            (task_type.value,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return row["skill_doc"], row["score"], row["epoch"]

    # ------------------------------------------------------------------
    # Skill patches (audit trail)
    # ------------------------------------------------------------------

    async def save_patches(
        self, task_type: TaskType, epoch: int, patches: list[SkillPatch], accepted: bool
    ) -> None:
        """Record each patch in the audit trail."""
        await self._ensure_connected()
        assert self._skill_db is not None
        now = time.time()
        await self._skill_db.executemany(
            """
            INSERT INTO skill_patches
                (task_type, epoch, patch_op, anchor, content, token_cost, accepted, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    task_type.value,
                    epoch,
                    p.op,
                    p.anchor,
                    p.content,
                    p.token_cost,
                    int(accepted),
                    now,
                )
                for p in patches
            ],
        )
        await self._skill_db.commit()

    # ------------------------------------------------------------------
    # Negative-feedback buffer
    # ------------------------------------------------------------------

    async def save_negative_feedback(
        self, task_type: TaskType, patches: list[SkillPatch], reason: str
    ) -> None:
        """Store a rejected patch batch so the optimizer can avoid repeating it."""
        await self._ensure_connected()
        assert self._skill_db is not None
        patches_json = json.dumps([asdict(p) for p in patches])
        await self._skill_db.execute(
            """
            INSERT INTO negative_feedback (task_type, patches_json, rejection_reason, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (task_type.value, patches_json, reason, time.time()),
        )
        await self._skill_db.commit()

    async def load_negative_feedback(
        self, task_type: TaskType, limit: int = 20
    ) -> list[dict]:
        """Return the most recent *limit* rejected patch batches."""
        await self._ensure_connected()
        assert self._skill_db is not None
        async with self._skill_db.execute(
            """
            SELECT patches_json, rejection_reason, created_at
            FROM   negative_feedback
            WHERE  task_type = ?
            ORDER  BY created_at DESC
            LIMIT  ?
            """,
            (task_type.value, limit),
        ) as cursor:
            rows = await cursor.fetchall()
        return [
            {
                "patches": json.loads(row["patches_json"]),
                "rejection_reason": row["rejection_reason"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]
