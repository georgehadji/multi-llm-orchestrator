"""
SupervisorStore — persistent session + lesson storage.

SQLite-backed, WAL-enabled, append-only lessons.  Secrets are redacted from
``signal`` and ``detail`` before persistence.

Author: Georgios-Chrysovalantis Chatzivantsidis
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from orchestrator.infrastructure.path_provider import CachePathProvider

if TYPE_CHECKING:
    import aiosqlite

from .models import Lesson, SupervisorSession

logger = logging.getLogger("orchestrator.supervisor.store")

_DEFAULT_PATH = CachePathProvider().db("supervisor.db")

_SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    status TEXT NOT NULL,
    summary TEXT NOT NULL DEFAULT '',
    directive_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS lessons (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    task_type TEXT NOT NULL,
    kind TEXT NOT NULL,
    signal TEXT NOT NULL,
    detail TEXT NOT NULL,
    created_at REAL NOT NULL,
    FOREIGN KEY(session_id) REFERENCES sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_lessons_tasktype ON lessons(task_type);
CREATE INDEX IF NOT EXISTS idx_lessons_created ON lessons(created_at);
CREATE INDEX IF NOT EXISTS idx_lessons_session ON lessons(session_id);
"""

_SECRET_PATTERNS = [
    re.compile(r"(api[_-]?key\s*[:=]\s*)[\"']?[\w\-]{10,}[\"']?", re.IGNORECASE),
    re.compile(r"(token\s*[:=]\s*)[\"']?[\w\-]{10,}[\"']?", re.IGNORECASE),
    re.compile(r"(secret\s*[:=]\s*)[\"']?[\w\-]{10,}[\"']?", re.IGNORECASE),
    re.compile(r"(password\s*[:=]\s*)[\"']?[^\s\"']+[\"']?", re.IGNORECASE),
    re.compile(r"(bearer\s+)\"?[\w\-]{10,}\"?", re.IGNORECASE),
    re.compile(r"(sk-[a-zA-Z0-9]{20,})", re.IGNORECASE),
]

_MAX_DETAIL_LEN = 8000
_MAX_SIGNAL_LEN = 1000
_MAX_SUMMARY_LEN = 4000


def _sanitize(text: str) -> str:
    """Redact likely secrets before persisting."""
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(r"\1[REDACTED]", text)
    return text


def _new_id() -> str:
    return uuid.uuid4().hex[:16]


def _row_to_lesson(row: Any) -> Lesson:
    return Lesson(
        id=row["id"],
        session_id=row["session_id"],
        project_id=row["project_id"],
        task_type=row["task_type"],
        kind=row["kind"],
        signal=row["signal"],
        detail=row["detail"],
        created_at=row["created_at"],
    )


def _row_to_session(row: Any) -> SupervisorSession:
    return SupervisorSession(
        id=row["id"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        status=row["status"],
        summary=row["summary"],
        directive_count=row["directive_count"],
    )


class SupervisorStore:
    """SQLite-backed store for supervisor sessions and lessons."""

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or _DEFAULT_PATH
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        """Open the database and ensure the schema exists."""
        import aiosqlite as _aio

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await _aio.connect(self._db_path)
        self._db.row_factory = _aio.Row
        await self._db.executescript(_SCHEMA)
        await self._db.commit()
        logger.debug("SupervisorStore connected: %s", self._db_path)

    async def _ensure_connected(self) -> None:
        if self._db is None:
            await self.connect()

    async def close(self) -> None:
        """Close the database connection."""
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def create_session(self) -> SupervisorSession:
        """Create and persist a new supervisor session."""
        await self._ensure_connected()
        assert self._db is not None
        now = time.time()
        session = SupervisorSession(
            id=_new_id(),
            created_at=now,
            updated_at=now,
            status="idle",
            summary="",
            directive_count=0,
        )
        await self._db.execute(
            """
            INSERT INTO sessions (id, created_at, updated_at, status, summary, directive_count)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                session.id,
                session.created_at,
                session.updated_at,
                session.status,
                session.summary,
                session.directive_count,
            ),
        )
        await self._db.commit()
        return session

    async def touch_session(self, session_id: str) -> None:
        """Bump the updated_at timestamp of a session."""
        await self._ensure_connected()
        assert self._db is not None
        await self._db.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?",
            (time.time(), session_id),
        )
        await self._db.commit()

    async def set_summary(
        self,
        session_id: str,
        summary: str,
        status: str,
    ) -> None:
        """Update session summary and status."""
        await self._ensure_connected()
        assert self._db is not None
        summary = _sanitize(summary[:_MAX_SUMMARY_LEN])
        await self._db.execute(
            "UPDATE sessions SET summary = ?, status = ?, updated_at = ? WHERE id = ?",
            (summary, status, time.time(), session_id),
        )
        await self._db.commit()

    async def increment_directive_count(self, session_id: str) -> None:
        """Increment the number of directives handled in this session."""
        await self._ensure_connected()
        assert self._db is not None
        await self._db.execute(
            """
            UPDATE sessions
            SET directive_count = directive_count + 1, updated_at = ?
            WHERE id = ?
            """,
            (time.time(), session_id),
        )
        await self._db.commit()

    async def record_lesson(self, lesson: Lesson) -> None:
        """Persist one lesson, sanitising its text first."""
        await self._ensure_connected()
        assert self._db is not None
        detail = _sanitize(lesson.detail[:_MAX_DETAIL_LEN])
        signal = _sanitize(lesson.signal[:_MAX_SIGNAL_LEN])
        await self._db.execute(
            """
            INSERT INTO lessons
                (id, session_id, project_id, task_type, kind, signal, detail, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                lesson.id,
                lesson.session_id,
                lesson.project_id,
                lesson.task_type,
                lesson.kind,
                signal,
                detail,
                lesson.created_at,
            ),
        )
        await self._db.commit()

    async def recent_lessons(
        self,
        task_type: str | None = None,
        limit: int = 5,
    ) -> list[Lesson]:
        """Return the most recent lessons, optionally filtered by task type."""
        await self._ensure_connected()
        assert self._db is not None
        if task_type:
            async with self._db.execute(
                """
                SELECT id, session_id, project_id, task_type, kind, signal, detail, created_at
                FROM lessons
                WHERE task_type = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (task_type, limit),
            ) as cursor:
                rows = await cursor.fetchall()
        else:
            async with self._db.execute(
                """
                SELECT id, session_id, project_id, task_type, kind, signal, detail, created_at
                FROM lessons
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ) as cursor:
                rows = await cursor.fetchall()
        return [_row_to_lesson(row) for row in reversed(rows)]  # type: ignore[call-overload]

    async def search_lessons(self, text: str, limit: int = 10) -> list[Lesson]:
        """Full-text-ish search over signal and detail (FTS in Phase 3)."""
        await self._ensure_connected()
        assert self._db is not None
        pattern = f"%{_sanitize(text[:200])}%"
        async with self._db.execute(
            """
            SELECT id, session_id, project_id, task_type, kind, signal, detail, created_at
            FROM lessons
            WHERE signal LIKE ? OR detail LIKE ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (pattern, pattern, limit),
        ) as cursor:
            rows = await cursor.fetchall()
        return [_row_to_lesson(row) for row in rows]

    async def get_session(self, session_id: str) -> SupervisorSession | None:
        """Load a single session by id."""
        await self._ensure_connected()
        assert self._db is not None
        async with self._db.execute(
            """
            SELECT id, created_at, updated_at, status, summary, directive_count
            FROM sessions
            WHERE id = ?
            """,
            (session_id,),
        ) as cursor:
            row = await cursor.fetchone()
        return _row_to_session(row) if row else None

    async def list_sessions(self, limit: int = 50) -> list[SupervisorSession]:
        """List sessions by most recently updated."""
        await self._ensure_connected()
        assert self._db is not None
        async with self._db.execute(
            """
            SELECT id, created_at, updated_at, status, summary, directive_count
            FROM sessions
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (limit,),
        ) as cursor:
            rows = await cursor.fetchall()
        return [_row_to_session(row) for row in rows]
