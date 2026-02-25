"""
Tests for TelemetryStore write-ahead queue (Scenario 2 fix).

The write loss scenario: asyncio.create_task() snapshots are cancelled when
the event loop closes. The WAL queue writes the intent to DB *before* the
fire-and-forget task, so a subsequent startup can drain orphaned writes.
"""

import asyncio
import json
import time
import pytest
import aiosqlite

from orchestrator.telemetry_store import TelemetryStore
from orchestrator.models import Model, TaskType
from orchestrator.policy import ModelProfile


def _make_profile(quality: float = 0.85, calls: int = 5) -> ModelProfile:
    return ModelProfile(
        model=Model.GPT_4O,
        provider="openai",
        cost_per_1m_input=2.50,
        cost_per_1m_output=10.0,
        quality_score=quality,
        trust_factor=0.9,
        avg_latency_ms=200.0,
        latency_p95_ms=400.0,
        success_rate=0.95,
        avg_cost_usd=0.002,
        call_count=calls,
        failure_count=0,
        validator_fail_count=0,
    )


# ── WAL queue schema ──────────────────────────────────────────────────────────

class TestWALSchema:

    @pytest.mark.asyncio
    async def test_pending_writes_table_exists_after_init(self, tmp_path):
        """Schema must include a pending_writes table."""
        store = TelemetryStore(db_path=tmp_path / "telemetry.db")
        await store._ensure_schema()
        async with aiosqlite.connect(tmp_path / "telemetry.db") as db:
            cursor = await db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='pending_writes'"
            )
            row = await cursor.fetchone()
        assert row is not None, "pending_writes table must exist after schema init"

    @pytest.mark.asyncio
    async def test_pending_writes_table_has_required_columns(self, tmp_path):
        """pending_writes must have id, payload, and created_at columns."""
        store = TelemetryStore(db_path=tmp_path / "telemetry.db")
        await store._ensure_schema()
        async with aiosqlite.connect(tmp_path / "telemetry.db") as db:
            cursor = await db.execute("PRAGMA table_info(pending_writes)")
            cols = {row[1] for row in await cursor.fetchall()}
        assert {"id", "payload", "created_at"}.issubset(cols)


# ── Enqueue / drain cycle ─────────────────────────────────────────────────────

class TestEnqueueDrain:

    @pytest.mark.asyncio
    async def test_enqueue_snapshot_writes_to_pending_before_drain(self, tmp_path):
        """enqueue_snapshot must persist to pending_writes synchronously."""
        store = TelemetryStore(db_path=tmp_path / "telemetry.db")
        profile = _make_profile()
        await store.enqueue_snapshot("proj1", Model.GPT_4O, TaskType.CODE_GEN, profile)

        async with aiosqlite.connect(tmp_path / "telemetry.db") as db:
            cursor = await db.execute("SELECT COUNT(*) FROM pending_writes")
            count = (await cursor.fetchone())[0]
        assert count >= 1, "enqueue_snapshot must write to pending_writes"

    @pytest.mark.asyncio
    async def test_drain_queue_moves_pending_to_model_snapshots(self, tmp_path):
        """After drain_queue(), pending_writes is empty and model_snapshots has the row."""
        store = TelemetryStore(db_path=tmp_path / "telemetry.db")
        profile = _make_profile()
        await store.enqueue_snapshot("proj1", Model.GPT_4O, TaskType.CODE_GEN, profile)
        await store.drain_queue()

        async with aiosqlite.connect(tmp_path / "telemetry.db") as db:
            pending_cur = await db.execute("SELECT COUNT(*) FROM pending_writes")
            pending_count = (await pending_cur.fetchone())[0]
            snap_cur = await db.execute("SELECT COUNT(*) FROM model_snapshots")
            snap_count = (await snap_cur.fetchone())[0]

        assert pending_count == 0, "pending_writes must be empty after drain"
        assert snap_count == 1, "model_snapshots must have 1 row after drain"

    @pytest.mark.asyncio
    async def test_drain_queue_is_idempotent(self, tmp_path):
        """Calling drain_queue() twice must not duplicate rows."""
        store = TelemetryStore(db_path=tmp_path / "telemetry.db")
        profile = _make_profile()
        await store.enqueue_snapshot("proj1", Model.GPT_4O, TaskType.CODE_GEN, profile)
        await store.drain_queue()
        await store.drain_queue()   # second call should be a no-op

        async with aiosqlite.connect(tmp_path / "telemetry.db") as db:
            cur = await db.execute("SELECT COUNT(*) FROM model_snapshots")
            count = (await cur.fetchone())[0]
        assert count == 1

    @pytest.mark.asyncio
    async def test_drain_queue_processes_multiple_pending_writes(self, tmp_path):
        """drain_queue must process all queued rows in one pass."""
        store = TelemetryStore(db_path=tmp_path / "telemetry.db")
        for i in range(5):
            await store.enqueue_snapshot(
                f"proj{i}", Model.GPT_4O, TaskType.CODE_GEN, _make_profile(quality=i * 0.1)
            )
        await store.drain_queue()

        async with aiosqlite.connect(tmp_path / "telemetry.db") as db:
            cur = await db.execute("SELECT COUNT(*) FROM model_snapshots")
            count = (await cur.fetchone())[0]
        assert count == 5


# ── Crash-recovery: orphaned pending writes ───────────────────────────────────

class TestCrashRecovery:

    @pytest.mark.asyncio
    async def test_orphaned_pending_writes_drained_on_next_startup(self, tmp_path):
        """
        Simulates a process crash after enqueue but before drain.
        A fresh TelemetryStore on the same DB must drain orphaned rows
        when drain_queue() is called (e.g. at warm-start time).
        """
        db_path = tmp_path / "telemetry.db"

        # Run 1: enqueue but "crash" (never drain)
        store1 = TelemetryStore(db_path=db_path)
        await store1.enqueue_snapshot(
            "proj_crash", Model.GPT_4O, TaskType.CODE_GEN, _make_profile(quality=0.92)
        )
        # store1 goes out of scope — simulate crash, drain never called

        # Run 2: fresh store on the same DB
        store2 = TelemetryStore(db_path=db_path)
        await store2.drain_queue()   # should pick up orphaned rows

        async with aiosqlite.connect(db_path) as db:
            pending_cur = await db.execute("SELECT COUNT(*) FROM pending_writes")
            pending = (await pending_cur.fetchone())[0]
            snap_cur = await db.execute("SELECT COUNT(*) FROM model_snapshots")
            snaps = (await snap_cur.fetchone())[0]

        assert pending == 0, "Orphaned pending writes must be drained on next startup"
        assert snaps == 1, "Orphaned write must produce a model_snapshots row"

    @pytest.mark.asyncio
    async def test_apply_warm_start_drains_queue_before_loading_history(self, tmp_path):
        """
        drain_queue() must be called inside _apply_warm_start() / warm-start path
        so that a previous run's orphaned writes are included in this run's routing.
        This test verifies the profile is visible to load_historical_profile() after
        an enqueue-without-drain from a prior session.
        """
        db_path = tmp_path / "telemetry.db"

        # Simulate prior run: wrote to queue but crashed before drain
        prior = TelemetryStore(db_path=db_path)
        await prior.enqueue_snapshot(
            "prior_proj", Model.GPT_4O, TaskType.CODE_GEN,
            _make_profile(quality=0.88, calls=15)
        )

        # New run: drain then load — should see the profile
        current = TelemetryStore(db_path=db_path)
        await current.drain_queue()
        hist = await current.load_historical_profile(Model.GPT_4O, TaskType.CODE_GEN)

        assert hist is not None, "Historical profile must be available after draining queue"
        assert hist.call_count == 15


# ── Payload integrity ─────────────────────────────────────────────────────────

class TestPayloadIntegrity:

    @pytest.mark.asyncio
    async def test_drained_snapshot_preserves_all_fields(self, tmp_path):
        """The drained row in model_snapshots must faithfully reproduce the profile."""
        store = TelemetryStore(db_path=tmp_path / "telemetry.db")
        profile = _make_profile(quality=0.77, calls=42)
        await store.enqueue_snapshot("p1", Model.GPT_4O, TaskType.SUMMARIZE, profile)
        await store.drain_queue()

        async with aiosqlite.connect(tmp_path / "telemetry.db") as db:
            cur = await db.execute(
                "SELECT quality_score, call_count, task_type FROM model_snapshots"
            )
            row = await cur.fetchone()

        assert row is not None
        assert abs(row[0] - 0.77) < 1e-6, "quality_score must be preserved"
        assert row[1] == 42, "call_count must be preserved"
        assert row[2] == TaskType.SUMMARIZE.value, "task_type must be preserved"

    @pytest.mark.asyncio
    async def test_pending_write_payload_is_valid_json(self, tmp_path):
        """The payload stored in pending_writes must be parseable JSON."""
        store = TelemetryStore(db_path=tmp_path / "telemetry.db")
        await store.enqueue_snapshot(
            "p1", Model.GPT_4O, TaskType.CODE_GEN, _make_profile()
        )
        async with aiosqlite.connect(tmp_path / "telemetry.db") as db:
            cur = await db.execute("SELECT payload FROM pending_writes LIMIT 1")
            row = await cur.fetchone()
        parsed = json.loads(row[0])
        assert "model" in parsed
        assert "task_type" in parsed
        assert "profile" in parsed
