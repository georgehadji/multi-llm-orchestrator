"""
SkillStore — SkillOpt persistence (adapter-injected)
======================================================
Uses an injected ``SkillDbAdapter`` for SQLite operations. The adapter lives
in ``infrastructure.skill_store_adapter``, keeping this module free of direct
``aiosqlite`` imports (satisfying the application-no-concrete-infra contract).

Implements domain.ports.SkillStorePort.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from typing import Any

from ..models import TaskType
from ..models_skill import SkillPatch, Trajectory

logger = logging.getLogger("orchestrator.skill_store")


class SkillStore:
    """SkillOpt persistence — delegates SQLite I/O to SkillDbAdapter.

    Accepts an initialized adapter; callers must ``await adapter.connect()``
    before passing it in, and ``await adapter.close()`` after use.

    Implements domain.ports.SkillStorePort.
    """

    def __init__(self, db: Any) -> None:
        self._db = db

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Delegate to adapter."""
        await self._db.connect()

    async def close(self) -> None:
        """Delegate to adapter."""
        await self._db.close()

    # ------------------------------------------------------------------
    # Trajectories
    # ------------------------------------------------------------------

    async def save_trajectory(self, t: Trajectory) -> None:
        """Append one trajectory observation."""
        await self._db.execute_traj(
            """
            INSERT INTO trajectories
                (task_id, task_type, prompt, output, score, critique_text,
                 model_used, cost_usd, recorded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                t.task_id,
                t.task_type.value,
                t.prompt[:4000],
                t.output[:8000],
                t.score,
                t.critique_text[:2000],
                t.model_used,
                t.cost_usd,
                t.recorded_at,
            ),
        )
        await self._db.commit_traj()

    async def load_trajectories(self, task_type: TaskType, limit: int = 50) -> list[Trajectory]:
        """Return the most recent *limit* trajectories, oldest-first."""
        rows = await self._db.fetchall_traj(
            """
            SELECT task_id, task_type, prompt, output, score, critique_text,
                   model_used, cost_usd, recorded_at
            FROM   trajectories
            WHERE  task_type = ?
            ORDER  BY recorded_at DESC
            LIMIT  ?
            """,
            (task_type.value, limit),
        )
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
        now = time.time()
        await self._db.execute_skill(
            "UPDATE skills SET is_best = 0 WHERE task_type = ? AND is_best = 1",
            (task_type.value,),
        )
        await self._db.execute_skill(
            """
            INSERT INTO skills (task_type, skill_doc, score, epoch, is_best, created_at)
            VALUES (?, ?, ?, ?, 1, ?)
            """,
            (task_type.value, skill_doc, score, epoch, now),
        )
        await self._db.commit_skill()
        logger.info("SkillStore: saved skill %s epoch=%d score=%.3f", task_type.value, epoch, score)

    async def load_best_skill(self, task_type: TaskType) -> tuple[str, float, int] | None:
        """Return (skill_doc, score, epoch) for the current best skill, or None."""
        row = await self._db.fetchone_skill(
            """
            SELECT skill_doc, score, epoch
            FROM   skills
            WHERE  task_type = ? AND is_best = 1
            ORDER  BY epoch DESC
            LIMIT  1
            """,
            (task_type.value,),
        )
        if row is None:
            return None
        return row["skill_doc"], row["score"], row["epoch"]

    async def save_patches(
        self, task_type: TaskType, epoch: int, patches: list[SkillPatch], accepted: bool
    ) -> None:
        """Record each patch in the audit trail."""
        now = time.time()
        params_list = [
            (task_type.value, epoch, p.op, p.anchor, p.content, p.token_cost, int(accepted), now)
            for p in patches
        ]
        await self._db.executemany_skill(
            """
            INSERT INTO skill_patches
                (task_type, epoch, patch_op, anchor, content, token_cost, accepted, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            params_list,
        )
        await self._db.commit_skill()

    # ------------------------------------------------------------------
    # Negative-feedback buffer
    # ------------------------------------------------------------------

    async def save_negative_feedback(
        self, task_type: TaskType, patches: list[SkillPatch], reason: str
    ) -> None:
        """Store a rejected patch batch so the optimizer can avoid repeating it."""
        patches_json = json.dumps([asdict(p) for p in patches])
        await self._db.execute_skill(
            """
            INSERT INTO negative_feedback (task_type, patches_json, rejection_reason, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (task_type.value, patches_json, reason, time.time()),
        )
        await self._db.commit_skill()

    async def load_negative_feedback(
        self, task_type: TaskType, limit: int = 20
    ) -> list[dict]:
        """Return the most recent *limit* rejected patch batches."""
        rows = await self._db.fetchall_skill(
            """
            SELECT patches_json, rejection_reason, created_at
            FROM   negative_feedback
            WHERE  task_type = ?
            ORDER  BY created_at DESC
            LIMIT  ?
            """,
            (task_type.value, limit),
        )
        return [
            {
                "patches": json.loads(row["patches_json"]),
                "rejection_reason": row["rejection_reason"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]
