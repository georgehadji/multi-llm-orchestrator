"""
TelemetryStore — persistent cross-run learning for the multi-LLM orchestrator.
===============================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis
Plan:   docs/plans/2026-02-25-learn-and-show-design.md

Owns all reads and writes to ~/.orchestrator_cache/telemetry.db (SQLite).
No other module touches the DB directly.

Design:
  - Append-only writes (INSERT only, never UPDATE/DELETE)
  - All writes are fire-and-forget (asyncio.create_task) — never block hot path
  - Reads aggregate historical snapshots per (model, task_type) pair
  - Warm-start thresholds: COLD <10, WARM 10-49, HOT ≥50 total calls

Schema:
  model_snapshots   — one row per ModelProfile snapshot after each run
  routing_events    — one row per TaskResult after each task
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import aiosqlite

from .models import Model, TaskType, TaskResult
from .policy import ModelProfile

logger = logging.getLogger("orchestrator.telemetry_store")

_DEFAULT_DB_PATH = Path.home() / ".orchestrator_cache" / "telemetry.db"

# Warm-start thresholds (plan: learn-and-show-design.md)
_COLD_THRESHOLD = 10   # < 10: cold, use defaults
_HOT_THRESHOLD  = 50   # ≥ 50: hot, use 100% historical

# Prevent division-by-zero in value_score
_EPSILON = 0.0001


# ─────────────────────────────────────────────────────────────────────────────
# Return types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HistoricalProfile:
    """Aggregated historical data for a (model, task_type) pair."""
    model: Model
    task_type: TaskType
    quality_score: float
    trust_factor: float
    avg_latency_ms: float
    latency_p95_ms: float
    success_rate: float
    avg_cost_usd: float
    call_count: int


@dataclass
class ModelRanking:
    """Per-model aggregate ranking across all task types in the window."""
    model: Model
    value_score: float     # quality_score / (avg_cost_usd + ε)
    call_count: int
    avg_cost_usd: float
    quality_score: float
    confidence: str        # "HOT" | "WARM" | "COLD"
    trend: float = 0.0     # % change in value_score vs prior window (future)


@dataclass
class Recommendation:
    """Advisory routing recommendation derived from historical data."""
    message: str
    estimated_savings_per_month: Optional[float] = None


# ─────────────────────────────────────────────────────────────────────────────
# TelemetryStore
# ─────────────────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS model_snapshots (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id           TEXT    NOT NULL,
    model                TEXT    NOT NULL,
    task_type            TEXT    NOT NULL,
    quality_score        REAL    NOT NULL,
    trust_factor         REAL    NOT NULL,
    avg_latency_ms       REAL    NOT NULL,
    latency_p95_ms       REAL    NOT NULL,
    success_rate         REAL    NOT NULL,
    avg_cost_usd         REAL    NOT NULL,
    call_count           INTEGER NOT NULL,
    failure_count        INTEGER NOT NULL,
    validator_fail_count INTEGER NOT NULL,
    recorded_at          REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS routing_events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id   TEXT    NOT NULL,
    task_id      TEXT    NOT NULL,
    task_type    TEXT    NOT NULL,
    model_chosen TEXT    NOT NULL,
    reviewer     TEXT,
    score        REAL    NOT NULL,
    cost_usd     REAL    NOT NULL,
    iterations   INTEGER NOT NULL,
    det_passed   INTEGER NOT NULL,
    recorded_at  REAL    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_snapshots_model
    ON model_snapshots(model, task_type, recorded_at);

CREATE INDEX IF NOT EXISTS idx_routing_model
    ON routing_events(model_chosen, task_type, recorded_at);

CREATE TABLE IF NOT EXISTS pending_writes (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    payload    TEXT    NOT NULL,
    created_at REAL    NOT NULL
);
"""


class TelemetryStore:
    """
    Persistent cross-run telemetry store backed by SQLite.

    Parameters
    ----------
    db_path:
        Path to the SQLite file. Defaults to ~/.orchestrator_cache/telemetry.db.
        Pass a tmp_path in tests to avoid touching the real store.
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._db_path = Path(db_path) if db_path is not None else _DEFAULT_DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialised = False

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _ensure_schema(self) -> None:
        """Create tables and indexes on first use."""
        if self._initialised:
            return
        async with aiosqlite.connect(self._db_path) as db:
            await db.executescript(_SCHEMA)
            await db.commit()
        self._initialised = True

    # ── Write API (fire-and-forget safe) ──────────────────────────────────────

    async def record_snapshot(
        self,
        project_id: str,
        model: Model,
        task_type: TaskType,
        profile: ModelProfile,
    ) -> None:
        """
        Persist a ModelProfile snapshot after a run completes.

        Should be called for every profile that had ≥1 call this run.
        Writes are always INSERT-only (append-only log).
        """
        await self._ensure_schema()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT INTO model_snapshots
                    (project_id, model, task_type, quality_score, trust_factor,
                     avg_latency_ms, latency_p95_ms, success_rate, avg_cost_usd,
                     call_count, failure_count, validator_fail_count, recorded_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    project_id,
                    model.value,
                    task_type.value,
                    profile.quality_score,
                    profile.trust_factor,
                    profile.avg_latency_ms,
                    profile.latency_p95_ms,
                    profile.success_rate,
                    profile.avg_cost_usd,
                    profile.call_count,
                    profile.failure_count,
                    profile.validator_fail_count,
                    time.time(),
                ),
            )
            await db.commit()

    async def record_routing_event(
        self,
        project_id: str,
        task_id: str,
        task_type: TaskType,
        result: TaskResult,
    ) -> None:
        """
        Persist one routing event after a task completes.

        Parameters
        ----------
        task_type:
            The TaskType of the task (from Task.type — not stored on TaskResult).
        """
        await self._ensure_schema()
        reviewer_val = result.reviewer_model.value if result.reviewer_model else None
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT INTO routing_events
                    (project_id, task_id, task_type, model_chosen, reviewer,
                     score, cost_usd, iterations, det_passed, recorded_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    project_id,
                    task_id,
                    task_type.value,
                    result.model_used.value,
                    reviewer_val,
                    result.score,
                    result.cost_usd,
                    result.iterations,
                    1 if result.deterministic_check_passed else 0,
                    time.time(),
                ),
            )
            await db.commit()

    # ── Write-ahead queue (WAL) ───────────────────────────────────────────────
    #
    # Problem: asyncio.create_task() fire-and-forget writes are cancelled when
    # the event loop closes, silently losing telemetry data.
    #
    # Fix: enqueue_snapshot() writes the intent to pending_writes *synchronously*
    # before any fire-and-forget task.  drain_queue() processes the queue
    # atomically and is called at warm-start time so a subsequent startup can
    # recover orphaned writes from a prior crashed session.

    async def enqueue_snapshot(
        self,
        project_id: str,
        model: Model,
        task_type: TaskType,
        profile: ModelProfile,
    ) -> None:
        """
        Persist a write intent to pending_writes synchronously.

        Call this instead of record_snapshot() for fire-and-forget paths.
        The row is guaranteed to be durable before this coroutine returns.
        A subsequent call to drain_queue() (e.g. at warm-start) will commit
        the data to model_snapshots even if the process crashed in between.
        """
        await self._ensure_schema()
        payload = json.dumps({
            "project_id": project_id,
            "model":      model.value,
            "task_type":  task_type.value,
            "profile": {
                "quality_score":        profile.quality_score,
                "trust_factor":         profile.trust_factor,
                "avg_latency_ms":       profile.avg_latency_ms,
                "latency_p95_ms":       profile.latency_p95_ms,
                "success_rate":         profile.success_rate,
                "avg_cost_usd":         profile.avg_cost_usd,
                "call_count":           profile.call_count,
                "failure_count":        profile.failure_count,
                "validator_fail_count": profile.validator_fail_count,
            },
        })
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT INTO pending_writes (payload, created_at) VALUES (?, ?)",
                (payload, time.time()),
            )
            await db.commit()

    async def drain_queue(self) -> int:
        """
        Drain all pending_writes into model_snapshots atomically.

        Idempotent: safe to call multiple times; rows that have already been
        moved are deleted from pending_writes so a second call is a no-op.

        Returns the number of rows drained (0 when the queue is empty).
        Called at warm-start time so orphaned writes from a prior crashed
        session are included in this run's routing decisions.
        """
        await self._ensure_schema()
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT id, payload FROM pending_writes ORDER BY id"
            ) as cur:
                rows = await cur.fetchall()

            if not rows:
                return 0

            ids_to_delete: list[int] = []
            for row_id, payload_str in rows:
                try:
                    payload     = json.loads(payload_str)
                    p           = payload["profile"]
                    await db.execute(
                        """
                        INSERT INTO model_snapshots
                            (project_id, model, task_type, quality_score, trust_factor,
                             avg_latency_ms, latency_p95_ms, success_rate, avg_cost_usd,
                             call_count, failure_count, validator_fail_count, recorded_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            payload.get("project_id", ""),
                            payload["model"],
                            payload["task_type"],
                            p["quality_score"],
                            p["trust_factor"],
                            p["avg_latency_ms"],
                            p["latency_p95_ms"],
                            p["success_rate"],
                            p["avg_cost_usd"],
                            p["call_count"],
                            p["failure_count"],
                            p["validator_fail_count"],
                            time.time(),
                        ),
                    )
                except (json.JSONDecodeError, KeyError) as exc:
                    logger.warning(
                        "drain_queue: skipping malformed pending_write id=%d: %s",
                        row_id, exc,
                    )
                finally:
                    # Always remove processed rows — even malformed ones should
                    # not re-poison the queue on the next startup.
                    ids_to_delete.append(row_id)

            if ids_to_delete:
                placeholders = ",".join("?" * len(ids_to_delete))
                await db.execute(
                    f"DELETE FROM pending_writes WHERE id IN ({placeholders})",
                    ids_to_delete,
                )
            await db.commit()

        return len(ids_to_delete)

    # ── Read API ──────────────────────────────────────────────────────────────

    async def load_historical_profile(
        self,
        model: Model,
        task_type: TaskType,
        days: int = 90,
    ) -> Optional[HistoricalProfile]:
        """
        Load aggregated historical data for a (model, task_type) pair.

        Returns None when total call_count < 10 (cold start — use defaults).
        Returns a HistoricalProfile when ≥ 10 calls exist.

        The caller (Orchestrator.__init__) applies the blend ratio:
          WARM (10-49): 40% historical / 60% default
          HOT  (≥ 50):  100% historical
        """
        await self._ensure_schema()
        cutoff = time.time() - days * 86400

        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                """
                SELECT
                    AVG(quality_score),
                    AVG(trust_factor),
                    AVG(avg_latency_ms),
                    AVG(latency_p95_ms),
                    AVG(success_rate),
                    AVG(avg_cost_usd),
                    SUM(call_count)
                FROM model_snapshots
                WHERE model = ? AND task_type = ? AND recorded_at >= ?
                """,
                (model.value, task_type.value, cutoff),
            ) as cur:
                row = await cur.fetchone()

        if row is None or row[0] is None:
            return None

        total_calls = int(row[6] or 0)
        if total_calls < _COLD_THRESHOLD:
            return None

        return HistoricalProfile(
            model=model,
            task_type=task_type,
            quality_score=row[0],
            trust_factor=row[1],
            avg_latency_ms=row[2],
            latency_p95_ms=row[3],
            success_rate=row[4],
            avg_cost_usd=row[5],
            call_count=total_calls,
        )

    async def model_rankings(self, days: int = 30) -> list[ModelRanking]:
        """
        Return all models ranked by value_score (quality / cost) descending.

        Aggregates across all task types within the time window.
        """
        await self._ensure_schema()
        cutoff = time.time() - days * 86400

        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                """
                SELECT
                    model,
                    AVG(quality_score)  AS quality,
                    AVG(avg_cost_usd)   AS cost,
                    SUM(call_count)     AS calls
                FROM model_snapshots
                WHERE recorded_at >= ?
                GROUP BY model
                ORDER BY (AVG(quality_score) / (AVG(avg_cost_usd) + ?)) DESC
                """,
                (cutoff, _EPSILON),
            ) as cur:
                rows = await cur.fetchall()

        rankings: list[ModelRanking] = []
        for row in rows:
            model_val, quality, cost, calls = row
            try:
                model = Model(model_val)
            except ValueError:
                continue
            value_score = quality / (cost + _EPSILON)
            confidence = _confidence(int(calls or 0))
            rankings.append(ModelRanking(
                model=model,
                value_score=round(value_score, 4),
                call_count=int(calls or 0),
                avg_cost_usd=round(cost, 6),
                quality_score=round(quality, 4),
                confidence=confidence,
            ))

        return rankings

    async def task_type_leaders(self, days: int = 30) -> dict[TaskType, ModelRanking]:
        """
        Return the best model per task type, keyed by TaskType.

        "Best" = highest value_score (quality / cost) within the window.
        """
        await self._ensure_schema()
        cutoff = time.time() - days * 86400

        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                """
                SELECT
                    task_type,
                    model,
                    AVG(quality_score)  AS quality,
                    AVG(avg_cost_usd)   AS cost,
                    SUM(call_count)     AS calls
                FROM model_snapshots
                WHERE recorded_at >= ?
                GROUP BY task_type, model
                """,
                (cutoff,),
            ) as cur:
                rows = await cur.fetchall()

        # Group by task_type, keep the best value_score per type
        best: dict[TaskType, ModelRanking] = {}
        for row in rows:
            task_type_val, model_val, quality, cost, calls = row
            try:
                task_type = TaskType(task_type_val)
                model = Model(model_val)
            except ValueError:
                continue

            value_score = quality / (cost + _EPSILON)
            ranking = ModelRanking(
                model=model,
                value_score=round(value_score, 4),
                call_count=int(calls or 0),
                avg_cost_usd=round(cost, 6),
                quality_score=round(quality, 4),
                confidence=_confidence(int(calls or 0)),
            )
            if task_type not in best or value_score > best[task_type].value_score:
                best[task_type] = ranking

        return best

    async def recommendations(self, days: int = 30) -> list[Recommendation]:
        """
        Generate advisory routing recommendations from historical data.

        Current heuristic: for each task type, if a cheaper model has equivalent
        or better quality than the most-used model, recommend switching.

        Only fires recommendations for HOT models (≥50 calls) to avoid
        noise from cold/warm data.
        """
        rankings = await self.model_rankings(days=days)
        leaders = await self.task_type_leaders(days=days)

        recs: list[Recommendation] = []

        # Per-task-type: find cheaper equivalent to the leader
        for task_type, leader in leaders.items():
            if leader.confidence == "COLD":
                continue  # insufficient data

            # Among HOT models for this task type, look for a cheaper equivalent
            # We need per-task rankings — re-query task-type specific data
            await self._ensure_schema()
            cutoff = time.time() - days * 86400
            async with aiosqlite.connect(self._db_path) as db:
                async with db.execute(
                    """
                    SELECT model, AVG(quality_score) AS q, AVG(avg_cost_usd) AS c,
                           SUM(call_count) AS calls
                    FROM model_snapshots
                    WHERE task_type = ? AND recorded_at >= ?
                    GROUP BY model
                    HAVING SUM(call_count) >= ?
                    ORDER BY AVG(avg_cost_usd) ASC
                    """,
                    (task_type.value, cutoff, _COLD_THRESHOLD),
                ) as cur:
                    task_rows = await cur.fetchall()

            for row in task_rows:
                model_val, quality, cost, calls = row
                if model_val == leader.model.value:
                    continue
                # Equivalent quality = within 2% of leader
                quality_equivalent = quality >= leader.quality_score * 0.98
                significantly_cheaper = cost < leader.avg_cost_usd * 0.7  # 30%+ cheaper
                if quality_equivalent and significantly_cheaper:
                    try:
                        alt_model = Model(model_val)
                    except ValueError:
                        continue
                    savings_ratio = (leader.avg_cost_usd - cost) / (leader.avg_cost_usd + _EPSILON)
                    recs.append(Recommendation(
                        message=(
                            f"Route {task_type.value} tasks to {alt_model.value} instead of "
                            f"{leader.model.value} — same quality ({quality:.2f} vs "
                            f"{leader.quality_score:.2f}), saves ~{savings_ratio:.0%} per call."
                        ),
                        estimated_savings_per_month=None,
                    ))

        # Cross-task: flag models with declining value vs cheaper equivalent in rankings
        hot_rankings = [r for r in rankings if r.confidence == "HOT"]
        for i, expensive in enumerate(hot_rankings):
            for cheaper in hot_rankings:
                if cheaper.model == expensive.model:
                    continue
                if (cheaper.avg_cost_usd < expensive.avg_cost_usd * 0.5
                        and cheaper.quality_score >= expensive.quality_score * 0.98
                        and cheaper.value_score > expensive.value_score * 1.5):
                    recs.append(Recommendation(
                        message=(
                            f"{cheaper.model.value} has equivalent quality to "
                            f"{expensive.model.value} at {cheaper.avg_cost_usd / (expensive.avg_cost_usd + _EPSILON):.0%} "
                            f"of the cost — consider routing more traffic to it."
                        ),
                    ))
                    break  # one rec per expensive model

        return recs


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _confidence(call_count: int) -> str:
    if call_count >= _HOT_THRESHOLD:
        return "HOT"
    if call_count >= _COLD_THRESHOLD:
        return "WARM"
    return "COLD"
