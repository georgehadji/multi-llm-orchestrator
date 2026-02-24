"""
State Persistence — async SQLite-backed project state for crash recovery
========================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis
Saves full project state after each task completion.
Enables resume from last checkpoint on HALT/crash.

Serialization: JSON (not pickle) — safe for untrusted DB files,
human-readable, and grep-able for debugging.

FIX #10: Migrated from sync sqlite3 to async aiosqlite for consistency
         with cache.py. All public methods are now async.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Optional

import aiosqlite

from .models import (
    Budget, Model, ProjectState, ProjectStatus,
    Task, TaskResult, TaskStatus, TaskType,
)

logger = logging.getLogger("orchestrator.state")

DEFAULT_STATE_PATH = Path.home() / ".orchestrator_cache" / "state.db"


# ─────────────────────────────────────────────
# JSON serializers / deserializers
# ─────────────────────────────────────────────

def _budget_to_dict(b: Budget) -> dict:
    return {
        "max_usd": b.max_usd,
        "max_time_seconds": b.max_time_seconds,
        "spent_usd": b.spent_usd,
        "start_time": b.start_time,
        "phase_spent": b.phase_spent,
    }


def _budget_from_dict(d: dict) -> Budget:
    b = Budget(
        max_usd=d["max_usd"],
        max_time_seconds=d["max_time_seconds"],
        spent_usd=d["spent_usd"],
        start_time=d["start_time"],
    )
    b.phase_spent = d.get("phase_spent", b.phase_spent)
    return b


def _task_to_dict(t: Task) -> dict:
    return {
        "id": t.id,
        "type": t.type.value,
        "prompt": t.prompt,
        "context": t.context,
        "dependencies": t.dependencies,
        "acceptance_threshold": t.acceptance_threshold,
        "max_iterations": t.max_iterations,
        "max_output_tokens": t.max_output_tokens,
        "status": t.status.value,
        "hard_validators": t.hard_validators,
    }


def _task_from_dict(d: dict) -> Task:
    t = Task(
        id=d["id"],
        type=TaskType(d["type"]),
        prompt=d["prompt"],
        context=d.get("context", ""),
        dependencies=d.get("dependencies", []),
        hard_validators=d.get("hard_validators", []),
    )
    t.acceptance_threshold = d["acceptance_threshold"]
    t.max_iterations = d["max_iterations"]
    t.max_output_tokens = d["max_output_tokens"]
    t.status = TaskStatus(d["status"])
    return t


def _result_to_dict(r: TaskResult) -> dict:
    return {
        "task_id": r.task_id,
        "output": r.output,
        "score": r.score,
        "model_used": r.model_used.value,
        "reviewer_model": r.reviewer_model.value if r.reviewer_model else None,
        "tokens_used": r.tokens_used,
        "iterations": r.iterations,
        "cost_usd": r.cost_usd,
        "status": r.status.value,
        "critique": r.critique,
        "deterministic_check_passed": r.deterministic_check_passed,
        "degraded_fallback_count": r.degraded_fallback_count,
    }


def _result_from_dict(d: dict) -> TaskResult:
    reviewer = d.get("reviewer_model")
    return TaskResult(
        task_id=d["task_id"],
        output=d["output"],
        score=d["score"],
        model_used=Model(d["model_used"]),
        reviewer_model=Model(reviewer) if reviewer else None,
        tokens_used=d.get("tokens_used", {"input": 0, "output": 0}),
        iterations=d.get("iterations", 0),
        cost_usd=d.get("cost_usd", 0.0),
        status=TaskStatus(d.get("status", "completed")),
        critique=d.get("critique", ""),
        deterministic_check_passed=d.get("deterministic_check_passed", True),
        degraded_fallback_count=d.get("degraded_fallback_count", 0),
    )


def _state_to_dict(state: ProjectState) -> dict:
    return {
        "project_description": state.project_description,
        "success_criteria": state.success_criteria,
        "budget": _budget_to_dict(state.budget),
        "tasks": {tid: _task_to_dict(t) for tid, t in state.tasks.items()},
        "results": {tid: _result_to_dict(r) for tid, r in state.results.items()},
        "api_health": state.api_health,
        "status": state.status.value,
        "execution_order": state.execution_order,
    }


def _state_from_dict(d: dict) -> ProjectState:
    return ProjectState(
        project_description=d["project_description"],
        success_criteria=d["success_criteria"],
        budget=_budget_from_dict(d["budget"]),
        tasks={tid: _task_from_dict(t) for tid, t in d.get("tasks", {}).items()},
        results={tid: _result_from_dict(r) for tid, r in d.get("results", {}).items()},
        api_health=d.get("api_health", {}),
        status=ProjectStatus(d.get("status", "PARTIAL_SUCCESS")),
        execution_order=d.get("execution_order", []),
    )


# ─────────────────────────────────────────────
# StateManager (async)
# ─────────────────────────────────────────────

class StateManager:
    """
    FIX #10: Async aiosqlite-backed state manager.
    Persistent connection with one-time schema init.
    """

    def __init__(self, db_path: Path = DEFAULT_STATE_PATH):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = str(db_path)
        self._conn: Optional[aiosqlite.Connection] = None
        self._lock: Optional[asyncio.Lock] = None  # lazy — created inside event loop

    async def _get_conn(self) -> aiosqlite.Connection:
        if self._lock is None:
            self._lock = asyncio.Lock()
        if self._conn is None:
            async with self._lock:
                if self._conn is None:
                    self._conn = await aiosqlite.connect(self._db_path)
                    await self._conn.execute("PRAGMA journal_mode=WAL")
                    await self._conn.executescript("""
                        CREATE TABLE IF NOT EXISTS projects (
                            project_id TEXT PRIMARY KEY,
                            state      TEXT NOT NULL,
                            status     TEXT NOT NULL,
                            created_at REAL NOT NULL,
                            updated_at REAL NOT NULL
                        );
                        CREATE TABLE IF NOT EXISTS checkpoints (
                            id         INTEGER PRIMARY KEY AUTOINCREMENT,
                            project_id TEXT NOT NULL,
                            task_id    TEXT NOT NULL,
                            state      TEXT NOT NULL,
                            created_at REAL NOT NULL,
                            FOREIGN KEY (project_id) REFERENCES projects(project_id)
                        );
                    """)
                    await self._conn.commit()
        return self._conn

    async def save_project(self, project_id: str, state: ProjectState):
        now = time.time()
        blob = json.dumps(_state_to_dict(state))
        db = await self._get_conn()
        await db.execute(
            """INSERT OR REPLACE INTO projects
               (project_id, state, status, created_at, updated_at)
               VALUES (?, ?, ?, COALESCE(
                   (SELECT created_at FROM projects WHERE project_id = ?), ?
               ), ?)""",
            (project_id, blob, state.status.value, project_id, now, now)
        )
        await db.commit()

    async def save_checkpoint(self, project_id: str, task_id: str,
                              state: ProjectState):
        blob = json.dumps(_state_to_dict(state))
        db = await self._get_conn()
        await db.execute(
            "INSERT INTO checkpoints (project_id, task_id, state, created_at) "
            "VALUES (?, ?, ?, ?)",
            (project_id, task_id, blob, time.time())
        )
        await db.commit()
        logger.info(f"Checkpoint saved: project={project_id}, task={task_id}")

    async def load_project(self, project_id: str) -> Optional[ProjectState]:
        db = await self._get_conn()
        async with db.execute(
            "SELECT state FROM projects WHERE project_id = ?", (project_id,)
        ) as cursor:
            row = await cursor.fetchone()
        if row:
            return _state_from_dict(json.loads(row[0]))
        return None

    async def load_latest_checkpoint(self, project_id: str) -> Optional[ProjectState]:
        db = await self._get_conn()
        async with db.execute(
            """SELECT state FROM checkpoints
               WHERE project_id = ? ORDER BY created_at DESC LIMIT 1""",
            (project_id,)
        ) as cursor:
            row = await cursor.fetchone()
        if row:
            return _state_from_dict(json.loads(row[0]))
        return None

    async def list_projects(self) -> list[dict]:
        db = await self._get_conn()
        async with db.execute(
            "SELECT project_id, status, created_at, updated_at "
            "FROM projects ORDER BY updated_at DESC"
        ) as cursor:
            rows = await cursor.fetchall()
        return [
            {"project_id": r[0], "status": r[1],
             "created_at": r[2], "updated_at": r[3]}
            for r in rows
        ]

    async def delete_project(self, project_id: str):
        db = await self._get_conn()
        await db.execute(
            "DELETE FROM checkpoints WHERE project_id = ?", (project_id,)
        )
        await db.execute(
            "DELETE FROM projects WHERE project_id = ?", (project_id,)
        )
        await db.commit()

    async def close(self):
        """Close the aiosqlite connection gracefully before the event loop shuts down."""
        if self._conn is not None:
            try:
                await self._conn.close()
                # Yield control so the aiosqlite background thread can finish
                # its final callbacks before asyncio.run() closes the loop.
                await asyncio.sleep(0)
            except Exception:
                pass
            finally:
                self._conn = None


# ─────────────────────────────────────────────
# Migration functions (Task 2: Database Persistence)
# ─────────────────────────────────────────────

def migrate_add_resume_fields(db_path: str | Path) -> bool:
    """
    Add project_description and keywords_json columns to projects table.

    Parameters:
    -----------
    db_path : str | Path
        Path to the SQLite database file

    Returns:
    --------
    bool
        True if migration succeeded, False otherwise

    Notes:
    ------
    - Idempotent: safe to call multiple times (checks if columns exist)
    - Handles empty databases gracefully
    - Existing projects get NULL for new columns
    """
    try:
        import sqlite3

        db_path = Path(db_path) if isinstance(db_path, str) else db_path
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Check if project_description column exists
        cursor.execute("PRAGMA table_info(projects)")
        columns = {row[1] for row in cursor.fetchall()}

        # Add project_description if it doesn't exist
        if "project_description" not in columns:
            cursor.execute(
                "ALTER TABLE projects ADD COLUMN project_description TEXT"
            )

        # Add keywords_json if it doesn't exist
        if "keywords_json" not in columns:
            cursor.execute(
                "ALTER TABLE projects ADD COLUMN keywords_json TEXT"
            )

        conn.commit()
        conn.close()
        return True

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return False


def extract_and_store_keywords(description: str | None) -> str | None:
    """
    Extract keywords from description and return as JSON string.

    Parameters:
    -----------
    description : str | None
        Project description to extract keywords from

    Returns:
    --------
    str | None
        JSON array string (e.g., '["api", "rest"]') or None if description is None/empty
    """
    if not description:
        return None

    from .resume_detector import _extract_keywords

    keywords = _extract_keywords(description)

    if not keywords:
        return None

    return json.dumps(keywords)
