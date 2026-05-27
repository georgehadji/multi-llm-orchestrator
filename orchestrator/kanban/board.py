"""
Kanban Board — SQLite-Backed Multi-Project Work Queue
========================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

SQLite-backed persistent work queue for long-running orchestration
projects. Supports enqueue, claim, complete, and fail operations.

Isolation model (from Hermes Agent's Kanban):
- Board is the hard boundary — workers are spawned with the board ID
  pinned in their env so they can't see other boards.
- After failure_limit consecutive non-success attempts (default: 2),
  the dispatcher auto-blocks the task to prevent spin loops.

Integration: Started via `orchestrator kanban start` or run alongside
the gateway. KanbanBoard is the data store; KanbanDispatcher (in
dispatcher.py) is the background loop that monitors and assigns work.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("orchestrator.kanban.board")

# Default database path
_DEFAULT_DB_PATH = Path.home() / ".orchestrator_cache" / "kanban.db"

# Maximum consecutive failures before auto-blocking
_FAILURE_LIMIT: int = 2


# ─────────────────────────────────────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS kanban_tasks (
    task_id       TEXT PRIMARY KEY,
    project_spec  TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'todo',
    assignee      TEXT,
    created_at    REAL NOT NULL,
    claimed_at    REAL,
    completed_at  REAL,
    priority      INTEGER NOT NULL DEFAULT 0,
    failure_count INTEGER NOT NULL DEFAULT 0,
    error_log     TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_kanban_status
    ON kanban_tasks(status, priority);
"""


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class KanbanTask:
    """A single kanban task.

    Attributes:
        task_id: Unique task identifier.
        project_spec: JSON string with project description and criteria.
        status: ``todo`` | ``claimed`` | ``running`` | ``completed`` | ``failed`` | ``blocked``
        assignee: Worker identifier who claimed this task.
        created_at: Unix timestamp of creation.
        claimed_at: Unix timestamp when claimed.
        completed_at: Unix timestamp when completed/failed.
        priority: Higher = more important. Default 0.
        failure_count: Number of consecutive non-success outcomes.
        error_log: Accumulated error messages.
    """

    task_id: str
    project_spec: str
    status: str = "todo"
    assignee: str = ""
    created_at: float = field(default_factory=time.time)
    claimed_at: float = 0.0
    completed_at: float = 0.0
    priority: int = 0
    failure_count: int = 0
    error_log: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# KanbanBoard
# ─────────────────────────────────────────────────────────────────────────────


class KanbanBoard:
    """SQLite-backed multi-project work queue.

    Usage:
        board = KanbanBoard()
        task_id = await board.enqueue({"description": "Build API"})
        task = await board.claim_next("worker-1")
        await board.complete(task_id, result={})
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = Path(db_path) if db_path is not None else _DEFAULT_DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        try:
            conn = sqlite3.connect(str(self._db_path))
            conn.executescript(_SCHEMA)
            conn.commit()
            conn.close()
        except sqlite3.Error as exc:
            logger.error("Failed to init kanban schema: %s", exc)

    # ── Write API ───────────────────────────────────────────────────────────

    async def enqueue(
        self,
        project_spec: dict[str, Any] | str,
        priority: int = 0,
    ) -> str:
        """Add a new task to the queue.

        Args:
            project_spec: Project description dict or JSON string.
            priority: Priority (higher = more important).

        Returns:
            The new task_id.
        """
        spec_str = json.dumps(project_spec) if isinstance(project_spec, dict) else project_spec
        task_id = f"kanban_{uuid.uuid4().hex[:8]}"

        async with self._lock:
            try:
                conn = sqlite3.connect(str(self._db_path))
                conn.execute(
                    """INSERT INTO kanban_tasks
                       (task_id, project_spec, status, priority, created_at)
                       VALUES (?, ?, 'todo', ?, ?)""",
                    (task_id, spec_str, priority, time.time()),
                )
                conn.commit()
                conn.close()
                logger.info("Kanban: enqueued %s (priority=%d)", task_id, priority)
                return task_id
            except sqlite3.Error as exc:
                logger.error("Kanban: enqueue failed: %s", exc)
                raise

    async def claim_next(self, assignee: str) -> KanbanTask | None:
        """Atomically claim the highest-priority available task.

        Concurrency-safe: uses ``UPDATE ... LIMIT 1`` to claim atomically.

        Args:
            assignee: Worker identifier.

        Returns:
            A KanbanTask if one was available, None otherwise.
        """
        async with self._lock:
            try:
                conn = sqlite3.connect(str(self._db_path))
                conn.row_factory = sqlite3.Row

                # Atomic claim: find and update in one step
                row = conn.execute(
                    """UPDATE kanban_tasks
                       SET status = 'claimed',
                           assignee = ?,
                           claimed_at = ?
                       WHERE task_id = (
                           SELECT task_id FROM kanban_tasks
                           WHERE status = 'todo'
                           ORDER BY priority DESC, created_at ASC
                           LIMIT 1
                       )
                       RETURNING *""",
                    (assignee, time.time()),
                ).fetchone()
                conn.commit()
                conn.close()

                if row is None:
                    return None

                return KanbanTask(
                    task_id=row["task_id"],
                    project_spec=row["project_spec"],
                    status=row["status"],
                    assignee=row["assignee"] or "",
                    created_at=row["created_at"],
                    claimed_at=row["claimed_at"] or 0.0,
                    completed_at=row["completed_at"] or 0.0,
                    priority=row["priority"],
                    failure_count=row["failure_count"],
                    error_log=row["error_log"] or "",
                )
            except sqlite3.Error as exc:
                logger.error("Kanban: claim failed: %s", exc)
                return None

    async def complete(
        self,
        task_id: str,
        result: dict[str, Any] | None = None,
    ) -> bool:
        """Mark a task as completed.

        Args:
            task_id: Task ID to complete.
            result: Optional result data (stored in error_log).

        Returns:
            True if the task was updated.
        """
        return await self._update_status(task_id, "completed", result=result)

    async def fail(
        self,
        task_id: str,
        error: str = "",
    ) -> bool:
        """Mark a task as failed.

        After ``_FAILURE_LIMIT`` consecutive failures, the task is
        auto-blocked to prevent spin loops.

        Args:
            task_id: Task ID to fail.
            error: Error description.

        Returns:
            True if the task was updated.
        """
        async with self._lock:
            try:
                conn = sqlite3.connect(str(self._db_path))
                row = conn.execute(
                    "SELECT failure_count FROM kanban_tasks WHERE task_id = ?",
                    (task_id,),
                ).fetchone()

                if row is None:
                    conn.close()
                    return False

                new_count = row[0] + 1
                new_status = "blocked" if new_count >= _FAILURE_LIMIT else "failed"
                new_error = f"[{new_count}] {error}\n" if error else ""

                conn.execute(
                    """UPDATE kanban_tasks
                       SET status = ?, failure_count = ?,
                           error_log = error_log || ?,
                           completed_at = ?
                       WHERE task_id = ?""",
                    (new_status, new_count, new_error, time.time(), task_id),
                )
                conn.commit()
                conn.close()

                if new_status == "blocked":
                    logger.warning(
                        "Kanban: %s blocked after %d failures",
                        task_id,
                        new_count,
                    )

                return True
            except sqlite3.Error as exc:
                logger.error("Kanban: fail update failed: %s", exc)
                return False

    # ── Read API ────────────────────────────────────────────────────────────

    async def list_tasks(
        self,
        status: str | None = None,
        limit: int = 50,
    ) -> list[KanbanTask]:
        """List tasks, optionally filtered by status.

        Args:
            status: Filter by status (``"todo"``, ``"claimed"``, etc.)
                or None for all.
            limit: Maximum results.

        Returns:
            List of KanbanTask objects.
        """
        try:
            conn = sqlite3.connect(str(self._db_path))
            conn.row_factory = sqlite3.Row

            if status:
                rows = conn.execute(
                    """SELECT * FROM kanban_tasks
                       WHERE status = ?
                       ORDER BY priority DESC, created_at DESC
                       LIMIT ?""",
                    (status, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM kanban_tasks ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            conn.close()

            return [
                KanbanTask(
                    task_id=r["task_id"],
                    project_spec=r["project_spec"],
                    status=r["status"],
                    assignee=r["assignee"] or "",
                    created_at=r["created_at"],
                    claimed_at=r["claimed_at"] or 0.0,
                    completed_at=r["completed_at"] or 0.0,
                    priority=r["priority"],
                    failure_count=r["failure_count"],
                    error_log=r["error_log"] or "",
                )
                for r in rows
            ]
        except sqlite3.Error as exc:
            logger.debug("Kanban: list failed: %s", exc)
            return []

    async def get_stats(self) -> dict[str, Any]:
        """Return aggregate statistics about the board."""
        try:
            conn = sqlite3.connect(str(self._db_path))
            total = conn.execute("SELECT COUNT(*) FROM kanban_tasks").fetchone()[0]
            todo = conn.execute(
                "SELECT COUNT(*) FROM kanban_tasks WHERE status = 'todo'"
            ).fetchone()[0]
            claimed = conn.execute(
                "SELECT COUNT(*) FROM kanban_tasks WHERE status = 'claimed'"
            ).fetchone()[0]
            completed = conn.execute(
                "SELECT COUNT(*) FROM kanban_tasks WHERE status = 'completed'"
            ).fetchone()[0]
            failed = conn.execute(
                "SELECT COUNT(*) FROM kanban_tasks WHERE status = 'failed'"
            ).fetchone()[0]
            blocked = conn.execute(
                "SELECT COUNT(*) FROM kanban_tasks WHERE status = 'blocked'"
            ).fetchone()[0]
            conn.close()
            return {
                "total": total,
                "todo": todo,
                "claimed": claimed,
                "completed": completed,
                "failed": failed,
                "blocked": blocked,
            }
        except sqlite3.Error:
            return {
                "total": 0,
                "todo": 0,
                "claimed": 0,
                "completed": 0,
                "failed": 0,
                "blocked": 0,
            }

    # ── Internal ────────────────────────────────────────────────────────────

    async def _update_status(
        self,
        task_id: str,
        status: str,
        result: dict[str, Any] | None = None,
    ) -> bool:
        """Update a task's status."""
        async with self._lock:
            try:
                conn = sqlite3.connect(str(self._db_path))
                result_str = json.dumps(result) if result else ""
                conn.execute(
                    "UPDATE kanban_tasks SET status = ?, error_log = ?, "
                    "completed_at = ? WHERE task_id = ?",
                    (status, result_str, time.time(), task_id),
                )
                changed = conn.total_changes > 0
                conn.commit()
                conn.close()
                return changed
            except sqlite3.Error:
                return False
