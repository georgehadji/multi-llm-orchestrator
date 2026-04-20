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

from .budget import Budget
from .models import (
    AttemptRecord,
    Model,
    ProjectState,
    ProjectStatus,
    Task,
    TaskResult,
    TaskStatus,
    TaskType,
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
        # FIX-RESUME-001: Include original_start_time for elapsed time calculation when resuming
        "original_start_time": b.original_start_time,
        "phase_spent": b.phase_spent,
    }


def _budget_from_dict(d: dict) -> Budget:
    b = Budget(
        max_usd=d["max_usd"],
        max_time_seconds=d["max_time_seconds"],
        spent_usd=d["spent_usd"],
        start_time=d["start_time"],
    )
    # FIX-RESUME-001: Restore original_start_time for elapsed time calculation
    b.original_start_time = d.get("original_start_time", d["start_time"])
    b.phase_spent = d.get("phase_spent", b.phase_spent)
    return b


def _task_to_dict(t: Task) -> dict:
    """Serialize Task to dict - includes all App Builder fields (BUG-001 FIX)."""
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
        # BUG-001 FIX: Added missing App Builder fields
        "target_path": t.target_path,
        "module_name": t.module_name,
        "tech_context": t.tech_context,
    }


def _task_from_dict(d: dict) -> Task:
    """Deserialize dict to Task - restores App Builder fields (BUG-001 FIX)."""
    t = Task(
        id=d["id"],
        type=TaskType(d["type"]),
        prompt=d["prompt"],
        context=d.get("context", ""),
        dependencies=d.get("dependencies", []),
        hard_validators=d.get("hard_validators", []),
        # BUG-001 FIX: Restore App Builder fields with defaults for backward compat
        target_path=d.get("target_path", ""),
        module_name=d.get("module_name", ""),
        tech_context=d.get("tech_context", ""),
    )
    # Use .get() with defaults for backward compatibility
    t.acceptance_threshold = d.get("acceptance_threshold", 0.85)
    t.max_iterations = d.get("max_iterations", 3)
    t.max_output_tokens = d.get("max_output_tokens", 4096)
    t.status = TaskStatus(d.get("status", "pending"))
    return t


def _attempt_to_dict(a: AttemptRecord) -> dict:
    return {
        "attempt_num": a.attempt_num,
        "model_used": a.model_used,
        "output_snippet": a.output_snippet,
        "failure_reason": a.failure_reason,
        "validators_failed": a.validators_failed,
    }


def _attempt_from_dict(d: dict) -> AttemptRecord:
    return AttemptRecord(
        attempt_num=d.get("attempt_num", 1),
        model_used=d.get("model_used", ""),
        output_snippet=d.get("output_snippet", ""),
        failure_reason=d.get("failure_reason", ""),
        validators_failed=d.get("validators_failed", []),
    )


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
        "attempt_history": [_attempt_to_dict(a) for a in r.attempt_history],
    }


def _result_from_dict(d: dict) -> TaskResult:
    reviewer = d.get("reviewer_model")

    # Handle unknown models (e.g., removed models like kimi-k2.5)
    def _safe_model(model_value: str) -> Model:
        try:
            return Model(model_value)
        except ValueError:
            # Return GPT_4O_MINI as fallback for unknown models
            return Model.GPT_4O_MINI

    return TaskResult(
        task_id=d["task_id"],
        output=d["output"],
        score=d["score"],
        model_used=_safe_model(d["model_used"]),
        reviewer_model=_safe_model(reviewer) if reviewer else None,
        tokens_used=d.get("tokens_used", {"input": 0, "output": 0}),
        iterations=d.get("iterations", 0),
        cost_usd=d.get("cost_usd", 0.0),
        status=TaskStatus(d.get("status", "completed")),
        critique=d.get("critique", ""),
        deterministic_check_passed=d.get("deterministic_check_passed", True),
        degraded_fallback_count=d.get("degraded_fallback_count", 0),
        attempt_history=[_attempt_from_dict(a) for a in d.get("attempt_history", [])],
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

    # Connection timeout in seconds (prevents infinite hangs)
    _CONN_TIMEOUT: float = 10.0

    def __init__(self, db_path: Path = DEFAULT_STATE_PATH):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = str(db_path)
        self._conn: Optional[aiosqlite.Connection] = None
        self._lock: Optional[asyncio.Lock] = None  # lazy — created inside event loop

    async def _get_conn(self) -> aiosqlite.Connection:
        """
        Get or create persistent connection with proper error handling.

        BUG-DBCONN-001 FIX: Connection initialization is protected with
        try/except to ensure proper cleanup on error.

        FIX STATE-001: Added timeout to prevent infinite hangs on DB init.
        """
        if self._lock is None:
            self._lock = asyncio.Lock()
        if self._conn is None:
            async with self._lock:
                if self._conn is None:
                    # Use a local `conn` during the entire init sequence.
                    # self._conn is intentionally kept None until *after* the
                    # migration completes so that a concurrent caller whose outer
                    # `if self._conn is None` check races ahead sees None, enters
                    # the lock, and blocks — instead of receiving a half-ready
                    # connection that is missing the migration columns.
                    conn = None
                    try:
                        # FIX STATE-001: Timeout prevents infinite hang
                        conn = await asyncio.wait_for(
                            aiosqlite.connect(self._db_path), timeout=self._CONN_TIMEOUT
                        )
                        # WAL mode for better concurrency
                        await conn.execute("PRAGMA journal_mode=WAL")
                        # CRITICAL FIX: synchronous=FULL for durability on power failure
                        # This ensures data is written to disk before commit returns
                        await conn.execute("PRAGMA synchronous=FULL")
                        # Additional durability settings
                        await conn.execute(
                            "PRAGMA wal_autocheckpoint=0"
                        )  # Manual checkpoint control
                        await conn.executescript("""
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
                        await conn.commit()
                        # BUG-NEW-003 FIX: migrate_add_resume_fields uses synchronous
                        # sqlite3.connect(), which would block the event loop if called
                        # directly inside this async context.  Offload it to the default
                        # thread-pool executor so the loop stays responsive.
                        # NOTE: self._conn stays None here so no concurrent caller can
                        # bypass the lock and use the DB before migration finishes.
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(None, migrate_add_resume_fields, self._db_path)
                        # Migration complete — only now expose the connection.
                        self._conn = conn
                    except asyncio.TimeoutError:
                        logger.error("State DB connection timed out after %ds", self._CONN_TIMEOUT)
                        if conn is not None:
                            try:
                                await conn.close()
                            except Exception:
                                pass
                        raise
                    except aiosqlite.Error as e:
                        logger.error("Failed to initialize state connection: %s", e)
                        if conn is not None:
                            try:
                                await conn.close()
                            except Exception:
                                pass
                        raise
                    except Exception as e:
                        logger.error("Unexpected error during state init: %s", e)
                        if conn is not None:
                            try:
                                await conn.close()
                            except Exception:
                                pass
                        raise
        return self._conn

    async def save_project(self, project_id: str, state: ProjectState):
        now = time.time()
        blob = json.dumps(_state_to_dict(state))

        # Extract and store resume metadata for auto-resume detection
        keywords_json = extract_and_store_keywords(state.project_description)

        db = await self._get_conn()
        await db.execute(
            """INSERT OR REPLACE INTO projects
               (project_id, state, status, created_at, updated_at, project_description, keywords_json)
               VALUES (?, ?, ?, COALESCE(
                   (SELECT created_at FROM projects WHERE project_id = ?), ?
               ), ?, ?, ?)""",
            (
                project_id,
                blob,
                state.status.value,
                project_id,
                now,
                now,
                state.project_description,
                keywords_json,
            ),
        )
        await db.commit()
        # CRITICAL FIX: Checkpoint WAL after project save for durability
        await self._checkpoint_wal()

    async def save_checkpoint(self, project_id: str, task_id: str, state: ProjectState):
        blob = json.dumps(_state_to_dict(state))
        db = await self._get_conn()
        await db.execute(
            "INSERT INTO checkpoints (project_id, task_id, state, created_at) "
            "VALUES (?, ?, ?, ?)",
            (project_id, task_id, blob, time.time()),
        )
        await db.commit()
        # CRITICAL FIX: Checkpoint WAL to main database after critical writes
        # This ensures data survives power failure
        await self._checkpoint_wal()
        logger.info(f"Checkpoint saved: project={project_id}, task={task_id}")

    async def _checkpoint_wal(self):
        """Checkpoint WAL to main database file for durability."""
        try:
            db = await self._get_conn()
            await db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except Exception as e:
            logger.warning(f"WAL checkpoint failed: {e}")

    async def load_project(self, project_id: str) -> Optional[ProjectState]:
        db = await self._get_conn()
        async with db.execute(
            "SELECT state FROM projects WHERE project_id = ?", (project_id,)
        ) as cursor:
            row = await cursor.fetchone()
        if row:
            return self._deserialize_state(row[0], context=f"project={project_id}")
        return None

    async def load_latest_checkpoint(self, project_id: str) -> Optional[ProjectState]:
        db = await self._get_conn()
        async with db.execute(
            """SELECT state FROM checkpoints
               WHERE project_id = ? ORDER BY created_at DESC LIMIT 1""",
            (project_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row:
            return self._deserialize_state(row[0], context=f"checkpoint project={project_id}")
        return None

    def _deserialize_state(self, blob: str, context: str = "") -> Optional[ProjectState]:
        """
        Safely deserialize a persisted ProjectState blob.

        Returns None (and logs a warning) instead of raising on corrupt data,
        so callers can skip a damaged resume rather than crashing.
        """
        try:
            data = json.loads(blob)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Corrupt JSON in state store (%s): %s — skipping resume", context, exc)
            return None
        try:
            return _state_from_dict(data)
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning(
                "Schema mismatch in persisted state (%s): %s — skipping resume", context, exc
            )
            return None

    async def list_projects(self) -> list[dict]:
        db = await self._get_conn()
        async with db.execute(
            "SELECT project_id, status, created_at, updated_at "
            "FROM projects ORDER BY updated_at DESC"
        ) as cursor:
            rows = await cursor.fetchall()
        return [
            {"project_id": r[0], "status": r[1], "created_at": r[2], "updated_at": r[3]}
            for r in rows
        ]

    async def delete_project(self, project_id: str):
        db = await self._get_conn()
        await db.execute("DELETE FROM checkpoints WHERE project_id = ?", (project_id,))
        await db.execute("DELETE FROM projects WHERE project_id = ?", (project_id,))
        await db.commit()

    async def find_resumable(self, keywords: list[str]) -> list[dict]:
        """Find projects with matching keywords that are resumable.

        Returns list of dicts with: project_id, description, keywords, status, updated_at
        Only returns PARTIAL_SUCCESS or IN_PROGRESS projects.
        ``updated_at`` is a Unix timestamp float from the DB.
        """
        db = await self._get_conn()

        # Build a query that matches projects with any of the provided keywords
        # We'll do a text-based search in the keywords_json column
        if keywords:
            keyword_conditions = " OR ".join(["keywords_json LIKE ?"] * len(keywords))
            query_params = [f'%"{kw}"%' for kw in keywords]

            query = f"""SELECT project_id, project_description, keywords_json, status, updated_at
                       FROM projects
                       WHERE status IN ('PARTIAL_SUCCESS', 'IN_PROGRESS')
                       AND keywords_json IS NOT NULL
                       AND ({keyword_conditions})
                       ORDER BY updated_at DESC
                       LIMIT 50"""

            async with db.execute(query, query_params) as cursor:
                rows = await cursor.fetchall()
        else:
            # If no keywords provided, return all resumable projects
            async with db.execute(
                """SELECT project_id, project_description, keywords_json, status, updated_at
                   FROM projects
                   WHERE status IN ('PARTIAL_SUCCESS', 'IN_PROGRESS')
                   AND keywords_json IS NOT NULL
                   ORDER BY updated_at DESC
                   LIMIT 50""",
            ) as cursor:
                rows = await cursor.fetchall()

        result = []
        for row in rows:
            project_id, desc, kw_json, status, updated_at = row
            try:
                keywords_list = json.loads(kw_json) if kw_json else []
            except json.JSONDecodeError:
                keywords_list = []
            result.append(
                {
                    "project_id": project_id,
                    "description": desc or "",
                    "keywords": keywords_list,
                    "status": status,
                    "updated_at": updated_at or 0.0,
                }
            )
        return result

    async def _has_resume_columns(self) -> bool:
        """Check if resume columns exist in projects table.

        Returns:
            True if project_description and keywords_json columns exist
        """
        db = await self._get_conn()
        async with db.execute("PRAGMA table_info(projects)") as cursor:
            rows = await cursor.fetchall()
            columns = {row[1] for row in rows}
            return "project_description" in columns and "keywords_json" in columns

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
            cursor.execute("ALTER TABLE projects ADD COLUMN project_description TEXT")

        # Add keywords_json if it doesn't exist
        if "keywords_json" not in columns:
            cursor.execute("ALTER TABLE projects ADD COLUMN keywords_json TEXT")

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

    return json.dumps(sorted(keywords))
