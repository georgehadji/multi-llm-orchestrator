"""
Pattern Store — SQLite-Backed Persistent Pattern Library
===========================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Append-only SQLite store for reusable code patterns. Follows the
TelemetryStore design:
- INSERT-only writes (never UPDATE/DELETE active records)
- Implicit dedup by generated_code_hash (same code = same pattern)
- Usage tracking: reuse_count, last_reused_at, avg_score_on_reuse
- Archive: sets status='archived' + archived_at (never deletes)
- Provenance gating: curator only touches provenance="agent" patterns

Schema:
  patterns          — one row per unique pattern (active or archived)
  pattern_artifacts — full prompt text + generated code for each pattern

Integration: Populated by PatternExtractor, consumed by PatternCurator
and PatternInjector. PatternStore is injected into Orchestrator via
Orchestrator.__init__().
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..models import TaskType as _TaskType

if TYPE_CHECKING:
    from .extractor import ExtractedPattern

logger = logging.getLogger("orchestrator.pattern_learner.store")

# Default database path
_DEFAULT_DB_PATH = Path.home() / ".orchestrator_cache" / "patterns.db"


# ─────────────────────────────────────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS patterns (
    pattern_id         TEXT PRIMARY KEY,
    task_type          TEXT NOT NULL,
    prompt_fingerprint TEXT NOT NULL,
    generated_code_hash TEXT NOT NULL UNIQUE,
    quality_score      REAL NOT NULL,
    model_used         TEXT NOT NULL DEFAULT '',
    provenance         TEXT NOT NULL DEFAULT 'agent',
    status             TEXT NOT NULL DEFAULT 'active',
    reuse_count        INTEGER NOT NULL DEFAULT 0,
    last_reused_at     REAL,
    avg_score_on_reuse REAL,
    created_at         REAL NOT NULL,
    archived_at        REAL
);

CREATE TABLE IF NOT EXISTS pattern_artifacts (
    pattern_id       TEXT NOT NULL REFERENCES patterns(pattern_id),
    prompt_text      TEXT NOT NULL,
    generated_code   TEXT NOT NULL,
    critique_text    TEXT DEFAULT '',
    revision_context TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_patterns_type
    ON patterns(task_type, status);

CREATE INDEX IF NOT EXISTS idx_patterns_fingerprint
    ON patterns(prompt_fingerprint, status);
"""


# ─────────────────────────────────────────────────────────────────────────────
# PatternStore
# ─────────────────────────────────────────────────────────────────────────────


class PatternStore:
    """Persistent pattern library with usage tracking.

    Usage:
        store = PatternStore()
        await store.insert(pattern, prompt="...", code="...")
        similar = await store.find_similar("code_generation", prompt)
        count = await store.archive_stale(days=60)
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = Path(db_path) if db_path is not None else _DEFAULT_DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        """Create tables and indexes on first use."""
        try:
            conn = sqlite3.connect(str(self._db_path))
            conn.executescript(_SCHEMA)
            conn.commit()
            conn.close()
        except sqlite3.Error as exc:
            logger.error("Failed to initialize pattern store schema: %s", exc)

    # ── Write API ───────────────────────────────────────────────────────────

    async def insert(
        self,
        pattern: "ExtractedPattern",
        prompt_text: str = "",
        generated_code: str = "",
        critique_text: str = "",
    ) -> bool:
        """Insert a new pattern. Returns True if inserted, False if duplicate.

        Deduplication is by ``generated_code_hash`` — identical code
        from different prompts produces only one entry.
        """
        async with self._lock:
            try:
                conn = sqlite3.connect(str(self._db_path))
                try:
                    conn.execute(
                        """INSERT OR IGNORE INTO patterns
                           (pattern_id, task_type, prompt_fingerprint,
                            generated_code_hash, quality_score, model_used,
                            provenance, status, created_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?)""",
                        (
                            pattern.pattern_id,
                            pattern.task_type,
                            pattern.prompt_fingerprint,
                            pattern.generated_code_hash,
                            pattern.quality_score,
                            pattern.model_used,
                            pattern.provenance,
                            pattern.created_at,
                        ),
                    )
                    if conn.total_changes == 0:
                        return False  # duplicate

                    if prompt_text or generated_code:
                        conn.execute(
                            """INSERT INTO pattern_artifacts
                               (pattern_id, prompt_text, generated_code, critique_text)
                               VALUES (?, ?, ?, ?)""",
                            (pattern.pattern_id, prompt_text, generated_code, critique_text),
                        )
                    conn.commit()
                    return True
                finally:
                    conn.close()
            except sqlite3.Error as exc:
                logger.error("Failed to insert pattern %s: %s", pattern.pattern_id, exc)
                return False

    async def record_reuse(self, pattern_id: str, score: float) -> None:
        """Update usage tracking for a reused pattern.

        Atomically increments reuse_count, sets last_reused_at, and
        updates avg_score_on_reuse via weighted average.
        """
        async with self._lock:
            try:
                conn = sqlite3.connect(str(self._db_path))
                try:
                    row = conn.execute(
                        "SELECT reuse_count, avg_score_on_reuse FROM patterns "
                        "WHERE pattern_id = ? AND status = 'active'",
                        (pattern_id,),
                    ).fetchone()

                    if row is None:
                        return

                    old_count, old_avg = row
                    new_count = old_count + 1
                    new_avg = ((old_avg or 0.0) * (old_count) + score) / new_count

                    conn.execute(
                        "UPDATE patterns SET reuse_count = ?, last_reused_at = ?, "
                        "avg_score_on_reuse = ? WHERE pattern_id = ?",
                        (new_count, time.time(), round(new_avg, 4), pattern_id),
                    )
                    conn.commit()
                finally:
                    conn.close()
            except sqlite3.Error as exc:
                logger.warning("Failed to record reuse for %s: %s", pattern_id, exc)

    async def archive_stale(self, days: int = 60) -> int:
        """Archive patterns unused for ``days``.

        Never deletes — sets status='archived' + archived_at timestamp.
        Only touches provenance="agent" patterns.
        Returns count of patterns archived.
        """
        cutoff = time.time() - days * 86400
        async with self._lock:
            try:
                conn = sqlite3.connect(str(self._db_path))
                try:
                    cursor = conn.execute(
                        "UPDATE patterns SET status = 'archived', archived_at = ? "
                        "WHERE status = 'active' "
                        "AND provenance = 'agent' "
                        "AND (last_reused_at IS NULL OR last_reused_at < ?) "
                        "AND created_at < ?",
                        (time.time(), cutoff, cutoff),
                    )
                    conn.commit()
                    count = cursor.rowcount
                    if count:
                        logger.info("Archived %d stale pattern(s)", count)
                    return count
                finally:
                    conn.close()
            except sqlite3.Error as exc:
                logger.warning("Failed to archive stale patterns: %s", exc)
                return 0

    # ── Read API ────────────────────────────────────────────────────────────

    async def find_similar(
        self,
        task_type: str,
        prompt: str,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        """Find active patterns with similar task type.

        Currently matches by task_type prefix. Future versions will
        compute prompt similarity via fingerprint prefix matching.

        Args:
            task_type: Task type string to match.
            prompt: Prompt text (reserved for future similarity matching).
            limit: Maximum patterns to return.

        Returns:
            List of dicts with keys: pattern_id, task_type, quality_score,
            reuse_count, generated_code (first 1500 chars).
        """
        try:
            conn = sqlite3.connect(str(self._db_path))
            conn.row_factory = sqlite3.Row
            try:
                rows = conn.execute(
                    """SELECT p.pattern_id, p.task_type, p.quality_score,
                              p.model_used, p.reuse_count, p.avg_score_on_reuse,
                              a.generated_code
                       FROM patterns p
                       LEFT JOIN pattern_artifacts a ON p.pattern_id = a.pattern_id
                       WHERE p.status = 'active'
                       AND p.task_type = ?
                       ORDER BY p.quality_score DESC
                       LIMIT ?""",
                    (task_type, limit),
                ).fetchall()

                return [
                    {
                        "pattern_id": row["pattern_id"],
                        "task_type": row["task_type"],
                        "quality_score": row["quality_score"],
                        "model_used": row["model_used"],
                        "reuse_count": row["reuse_count"],
                        "avg_score_on_reuse": row["avg_score_on_reuse"],
                        "generated_code": (row["generated_code"] or "")[:1500],
                    }
                    for row in rows
                ]
            finally:
                conn.close()
        except sqlite3.Error as exc:
            logger.debug("find_similar query failed: %s", exc)
            return []

    async def get_stats(self) -> dict[str, Any]:
        """Return aggregate statistics about the pattern library."""
        try:
            conn = sqlite3.connect(str(self._db_path))
            try:
                total = conn.execute("SELECT COUNT(*) FROM patterns").fetchone()[0]
                active = conn.execute(
                    "SELECT COUNT(*) FROM patterns WHERE status = 'active'"
                ).fetchone()[0]
                archived = conn.execute(
                    "SELECT COUNT(*) FROM patterns WHERE status = 'archived'"
                ).fetchone()[0]
                total_reuses = conn.execute(
                    "SELECT COALESCE(SUM(reuse_count), 0) FROM patterns"
                ).fetchone()[0]
                return {
                    "total_patterns": total,
                    "active_patterns": active,
                    "archived_patterns": archived,
                    "total_reuses": total_reuses,
                }
            finally:
                conn.close()
        except sqlite3.Error:
            return {
                "total_patterns": 0,
                "active_patterns": 0,
                "archived_patterns": 0,
                "total_reuses": 0,
            }

    async def get_active_patterns_by_type(self) -> dict[str, list[dict[str, Any]]]:
        """Return all active patterns grouped by task_type.

        Used by PatternCurator for duplicate detection.

        Returns:
            Dict mapping task_type → list of pattern dicts with keys:
            pattern_id, quality_score, prompt_text (first 500 chars).
        """
        try:
            conn = sqlite3.connect(str(self._db_path))
            conn.row_factory = sqlite3.Row
            try:
                rows = conn.execute("""SELECT p.pattern_id, p.task_type, p.quality_score,
                              a.prompt_text
                       FROM patterns p
                       LEFT JOIN pattern_artifacts a ON p.pattern_id = a.pattern_id
                       WHERE p.status = 'active' AND p.provenance = 'agent'
                       ORDER BY p.task_type, p.quality_score DESC""").fetchall()

                groups: dict[str, list[dict[str, Any]]] = {}
                for row in rows:
                    task_type = row["task_type"]
                    if task_type not in groups:
                        groups[task_type] = []
                    groups[task_type].append(
                        {
                            "pattern_id": row["pattern_id"],
                            "quality_score": row["quality_score"],
                            "prompt_text": (row["prompt_text"] or "")[:500],
                        }
                    )
                return groups
            finally:
                conn.close()
        except sqlite3.Error as exc:
            logger.debug("get_active_patterns_by_type query failed: %s", exc)
            return {}

    async def archive_single(self, pattern_id: str) -> bool:
        """Archive a single pattern by ID.

        Only touches provenance="agent" patterns. Sets status='archived'
        + archived_at timestamp. Never deletes.

        Returns:
            True if archived, False if not found or not eligible.
        """
        async with self._lock:
            try:
                conn = sqlite3.connect(str(self._db_path))
                try:
                    cursor = conn.execute(
                        "UPDATE patterns SET status = 'archived', archived_at = ? "
                        "WHERE pattern_id = ? "
                        "AND status = 'active' "
                        "AND provenance = 'agent'",
                        (time.time(), pattern_id),
                    )
                    conn.commit()
                    return cursor.rowcount > 0
                finally:
                    conn.close()
            except sqlite3.Error as exc:
                logger.warning("Failed to archive pattern %s: %s", pattern_id, exc)
                return False
