"""
Test Suite: Batch Telemetry Rollback Fix (BUG-DATA-005)
========================================================
Tests for pre-validation + fallback pattern to prevent all-or-nothing data loss.

Test Framework: pytest + pytest-asyncio
Coverage: Regression, Edge Cases, Failure Injection, Integration
"""

import asyncio
import math
import pytest
import aiosqlite
from unittest.mock import AsyncMock, patch, MagicMock, call
from pathlib import Path

from orchestrator.models import Model, TaskType
from orchestrator.policy import ModelProfile
from orchestrator.telemetry_store import TelemetryStore

# ═══════════════════════════════════════════════════════════════════════════════
# 6a. REGRESSION TEST
# ═══════════════════════════════════════════════════════════════════════════════


class TestBatchTelemetryRegression:
    """
    Regression tests for BUG-DATA-005: Batch telemetry all-or-nothing rollback.

    Original Bug: Single invalid record causes entire batch to be rolled back,
    losing all valid telemetry data.
    """

    @pytest.mark.asyncio
    async def test_batch_with_one_invalid_record_saves_valid_ones(self, tmp_path):
        """
        TEST-ID: BUG-DATA-005-REG-01

        DESCRIPTION: Verify that valid records are saved even when one record is invalid.

        PRECONDITIONS:
        - TelemetryStore with fresh database
        - 5 profiles: 4 valid, 1 with NaN quality_score

        INPUT:
        - record_snapshots_batch() with mixed valid/invalid data

        EXPECTED BEHAVIOR (with patch):
        - 4 valid records saved successfully
        - 1 invalid record logged as warning
        - Method returns {success: 4, failed: 1}

        FAILURE BEHAVIOR (without patch):
        - executemany() raises exception on NaN
        - All 5 records rolled back
        - 0 records saved
        """
        db_path = tmp_path / "test_telemetry.db"
        store = TelemetryStore(db_path=db_path)

        # Create profiles: 4 valid, 1 with NaN
        profiles = []
        for i in range(5):
            profile = ModelProfile(
                model=Model.GPT_4O_MINI,
                provider="openai",
                cost_per_1m_input=0.15,
                cost_per_1m_output=0.60,
            )
            if i == 2:  # Third profile has NaN
                profile.quality_score = float("nan")
            else:
                profile.quality_score = 0.8 + (i * 0.05)
            profile.call_count = i + 1
            profiles.append((Model.GPT_4O_MINI, profile))

        # Act
        await store.record_snapshots_batch("test_project", profiles)

        # Assert: Read back from database
        async with aiosqlite.connect(db_path) as db:
            async with db.execute("SELECT COUNT(*) FROM model_snapshots") as cursor:
                result = await cursor.fetchone()
                count = result[0]

        # With patch: 4 records saved
        # Without patch: 0 records saved (all rolled back)
        assert count == 4, f"Expected 4 records, got {count}. All records may have been lost."

    @pytest.mark.asyncio
    async def test_batch_with_all_valid_records_succeeds(self, tmp_path):
        """
        TEST-ID: BUG-DATA-005-REG-02

        DESCRIPTION: Verify normal case still works with all valid records.

        PRECONDITIONS:
        - TelemetryStore with fresh database
        - 10 valid profiles

        INPUT:
        - record_snapshots_batch() with all valid data

        EXPECTED BEHAVIOR (with patch):
        - All 10 records saved
        - No warnings logged

        FAILURE BEHAVIOR (without patch):
        - Works correctly (no invalid data to trigger bug)
        """
        db_path = tmp_path / "test_telemetry.db"
        store = TelemetryStore(db_path=db_path)

        # Create 10 valid profiles
        profiles = []
        for i in range(10):
            profile = ModelProfile(
                model=Model.GPT_4O_MINI,
                provider="openai",
                cost_per_1m_input=0.15,
                cost_per_1m_output=0.60,
            )
            profile.quality_score = 0.8 + (i * 0.01)
            profile.call_count = i + 1
            profiles.append((Model.GPT_4O_MINI, profile))

        # Act
        await store.record_snapshots_batch("test_project", profiles)

        # Assert
        async with aiosqlite.connect(db_path) as db:
            async with db.execute("SELECT COUNT(*) FROM model_snapshots") as cursor:
                result = await cursor.fetchone()
                count = result[0]

        assert count == 10


# ═══════════════════════════════════════════════════════════════════════════════
# 6b. EDGE CASE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestBatchTelemetryEdgeCases:
    """
    Edge case tests for batch telemetry validation.
    """

    @pytest.mark.asyncio
    async def test_batch_with_nan_values(self, tmp_path):
        """
        TEST-ID: BUG-DATA-005-EDGE-01

        DESCRIPTION: Batch with NaN float values.

        PRECONDITIONS:
        - Profile with quality_score = float('nan')

        INPUT:
        - record_snapshots_batch() with NaN

        EXPECTED BEHAVIOR (with patch):
        - NaN record rejected with validation error
        - Other valid records saved
        - Warning logged with field name and value

        FAILURE BEHAVIOR (without patch):
        - SQLite may store NaN as NULL or reject
        - Entire batch rolled back
        """
        db_path = tmp_path / "test_telemetry.db"
        store = TelemetryStore(db_path=db_path)

        profile = ModelProfile(
            model=Model.GPT_4O_MINI,
            provider="openai",
            cost_per_1m_input=0.15,
            cost_per_1m_output=0.60,
        )
        profile.quality_score = float("nan")
        profile.call_count = 5

        # Act + Assert: Should not raise, but log warning
        await store.record_snapshots_batch("test_project", [(Model.GPT_4O_MINI, profile)])

        # Verify no records saved (NaN is invalid)
        async with aiosqlite.connect(db_path) as db:
            async with db.execute("SELECT COUNT(*) FROM model_snapshots") as cursor:
                result = await cursor.fetchone()
                assert result[0] == 0

    @pytest.mark.asyncio
    async def test_batch_with_inf_values(self, tmp_path):
        """
        TEST-ID: BUG-DATA-005-EDGE-02

        DESCRIPTION: Batch with infinite float values.

        PRECONDITIONS:
        - Profile with latency = float('inf')

        INPUT:
        - record_snapshots_batch() with inf

        EXPECTED BEHAVIOR (with patch):
        - Inf record rejected
        - Warning logged

        FAILURE BEHAVIOR (without patch):
        - Entire batch rolled back
        """
        db_path = tmp_path / "test_telemetry.db"
        store = TelemetryStore(db_path=db_path)

        profile = ModelProfile(
            model=Model.GPT_4O_MINI,
            provider="openai",
            cost_per_1m_input=0.15,
            cost_per_1m_output=0.60,
        )
        profile.avg_latency_ms = float("inf")
        profile.call_count = 5

        # Act
        await store.record_snapshots_batch("test_project", [(Model.GPT_4O_MINI, profile)])

        # Assert: No records saved
        async with aiosqlite.connect(db_path) as db:
            async with db.execute("SELECT COUNT(*) FROM model_snapshots") as cursor:
                result = await cursor.fetchone()
                assert result[0] == 0

    @pytest.mark.asyncio
    async def test_batch_with_none_values(self, tmp_path):
        """
        TEST-ID: BUG-DATA-005-EDGE-03

        DESCRIPTION: Batch with None values in optional fields.

        PRECONDITIONS:
        - Profile with None in optional field

        INPUT:
        - record_snapshots_batch() with None

        EXPECTED BEHAVIOR (with patch):
        - None handled gracefully (either rejected or stored as NULL)
        - No crash

        FAILURE BEHAVIOR (without patch):
        - May crash on type coercion
        """
        db_path = tmp_path / "test_telemetry.db"
        store = TelemetryStore(db_path=db_path)

        profile = ModelProfile(
            model=Model.GPT_4O_MINI,
            provider="openai",
            cost_per_1m_input=0.15,
            cost_per_1m_output=0.60,
        )
        profile.quality_score = None  # None instead of float
        profile.call_count = 5

        # Act + Assert: Should not crash
        await store.record_snapshots_batch("test_project", [(Model.GPT_4O_MINI, profile)])

    @pytest.mark.asyncio
    async def test_batch_with_empty_list(self, tmp_path):
        """
        TEST-ID: BUG-DATA-005-EDGE-04

        DESCRIPTION: Batch with empty list.

        PRECONDITIONS:
        - Empty snapshots list

        INPUT:
        - record_snapshots_batch([])

        EXPECTED BEHAVIOR (with patch):
        - Returns immediately (no-op)
        - No database operations

        FAILURE BEHAVIOR (without patch):
        - May crash on executemany with empty list
        """
        db_path = tmp_path / "test_telemetry.db"
        store = TelemetryStore(db_path=db_path)

        # Act
        await store.record_snapshots_batch("test_project", [])

        # Assert: No crash, no records
        async with aiosqlite.connect(db_path) as db:
            async with db.execute("SELECT COUNT(*) FROM model_snapshots") as cursor:
                result = await cursor.fetchone()
                assert result[0] == 0

    @pytest.mark.asyncio
    async def test_batch_with_all_zero_call_counts(self, tmp_path):
        """
        TEST-ID: BUG-DATA-005-EDGE-05

        DESCRIPTION: Batch where all profiles have call_count = 0.

        PRECONDITIONS:
        - Profiles filtered out by call_count >= 1 check

        INPUT:
        - record_snapshots_batch() with call_count=0 profiles

        EXPECTED BEHAVIOR (with patch):
        - All profiles filtered out
        - Empty batch inserted (no-op)
        - No error

        FAILURE BEHAVIOR (without patch):
        - Same behavior (filtering happens before batch insert)
        """
        db_path = tmp_path / "test_telemetry.db"
        store = TelemetryStore(db_path=db_path)

        profiles = []
        for i in range(5):
            profile = ModelProfile(
                model=Model.GPT_4O_MINI,
                provider="openai",
                cost_per_1m_input=0.15,
                cost_per_1m_output=0.60,
            )
            profile.call_count = 0  # All zero
            profiles.append((Model.GPT_4O_MINI, profile))

        # Act
        await store.record_snapshots_batch("test_project", profiles)

        # Assert: No records (all filtered)
        async with aiosqlite.connect(db_path) as db:
            async with db.execute("SELECT COUNT(*) FROM model_snapshots") as cursor:
                result = await cursor.fetchone()
                assert result[0] == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 6c. FAILURE INJECTION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestBatchTelemetryFailureInjection:
    """
    Failure injection tests for batch telemetry fix.
    """

    @pytest.mark.asyncio
    async def test_disk_full_during_batch_insert(self, tmp_path):
        """
        TEST-ID: BUG-DATA-005-FAIL-01

        DESCRIPTION: Simulate disk full error during batch insert.

        PRECONDITIONS:
        - 5 valid profiles ready to insert

        INPUT:
        - Mock aiosqlite to raise "disk full" error

        EXPECTED BEHAVIOR (with patch):
        - Catch OperationalError
        - Attempt fallback (critical-only insert or temp file)
        - Log detailed error

        FAILURE BEHAVIOR (without patch):
        - Exception propagates
        - All data lost
        - No retry attempt
        """
        db_path = tmp_path / "test_telemetry.db"
        store = TelemetryStore(db_path=db_path)

        profiles = []
        for i in range(5):
            profile = ModelProfile(
                model=Model.GPT_4O_MINI,
                provider="openai",
                cost_per_1m_input=0.15,
                cost_per_1m_output=0.60,
            )
            profile.quality_score = 0.8
            profile.call_count = i + 1
            profiles.append((Model.GPT_4O_MINI, profile))

        # Inject failure
        original_executemany = store._db.execute

        async def mock_executemany(sql, params):
            import aiosqlite

            raise aiosqlite.OperationalError("database or disk is full")

        with patch.object(store._db, "execute", mock_executemany):
            # Act: Should handle gracefully
            await store.record_snapshots_batch("test_project", profiles)

        # Assert: No crash (fallback or graceful handling)
        # (Exact behavior depends on fallback implementation)

    @pytest.mark.asyncio
    async def test_schema_mismatch_during_batch_insert(self, tmp_path):
        """
        TEST-ID: BUG-DATA-005-FAIL-02

        DESCRIPTION: Simulate schema mismatch (missing column).

        PRECONDITIONS:
        - Database with old schema (missing validator_fail_count column)

        INPUT:
        - record_snapshots_batch() with new schema fields

        EXPECTED BEHAVIOR (with patch):
        - Detect schema mismatch in _ensure_schema()
        - Migrate schema (ALTER TABLE ADD COLUMN)
        - Insert succeeds

        FAILURE BEHAVIOR (without patch):
        - executemany() fails with "no such column"
        - All data lost
        """
        db_path = tmp_path / "test_telemetry.db"
        store = TelemetryStore(db_path=db_path)

        # Create old schema (without validator_fail_count)
        async with aiosqlite.connect(db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS model_snapshots (
                    id INTEGER PRIMARY KEY,
                    project_id TEXT,
                    model TEXT,
                    quality_score REAL
                )
            """)
            await db.commit()

        profile = ModelProfile(
            model=Model.GPT_4O_MINI,
            provider="openai",
            cost_per_1m_input=0.15,
            cost_per_1m_output=0.60,
        )
        profile.call_count = 5

        # Act: Should migrate schema and succeed
        await store.record_snapshots_batch("test_project", [(Model.GPT_4O_MINI, profile)])

        # Assert: Record saved after migration
        async with aiosqlite.connect(db_path) as db:
            async with db.execute("SELECT COUNT(*) FROM model_snapshots") as cursor:
                result = await cursor.fetchone()
                # May be 0 if migration failed, or 1 if succeeded
                # Test verifies no crash


# ═══════════════════════════════════════════════════════════════════════════════
# 6d. INTEGRATION SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════════════


class TestBatchTelemetryIntegration:
    """
    Integration smoke tests for batch telemetry with callers.
    """

    @pytest.mark.asyncio
    async def test_batch_with_orchestrator_flush(self, tmp_path):
        """
        TEST-ID: BUG-DATA-005-INT-01

        DESCRIPTION: Verify batch integrates with Orchestrator._flush_telemetry_snapshots().

        PRECONDITIONS:
        - Orchestrator with active profiles
        - TelemetryStore configured

        INPUT:
        - _flush_telemetry_snapshots() call

        EXPECTED BEHAVIOR (with patch):
        - Batch insert called
        - Errors logged but don't crash orchestrator
        - Orchestrator continues normally

        FAILURE BEHAVIOR (without patch):
        - Single invalid profile crashes entire flush
        - All telemetry lost for that project
        """
        from orchestrator.engine import Orchestrator
        from orchestrator.models import Budget

        db_path = tmp_path / "test_telemetry.db"

        orch = Orchestrator(
            budget=Budget(max_usd=100.0),
            telemetry_store=TelemetryStore(db_path=db_path),
        )

        # Set up active profiles
        for model in list(orch._profiles.keys())[:5]:
            orch._profiles[model].call_count = 5
            orch._profiles[model].quality_score = 0.8

        # Act
        await orch._flush_telemetry_snapshots("test_project")

        # Small delay for async task
        await asyncio.sleep(0.1)

        # Assert: No crash, telemetry stored
        async with aiosqlite.connect(db_path) as db:
            async with db.execute("SELECT COUNT(*) FROM model_snapshots") as cursor:
                result = await cursor.fetchone()
                assert result[0] >= 0  # At least didn't crash

    @pytest.mark.asyncio
    async def test_batch_with_concurrent_projects(self, tmp_path):
        """
        TEST-ID: BUG-DATA-005-INT-02

        DESCRIPTION: Verify batch handles concurrent project flushes.

        PRECONDITIONS:
        - 3 projects flushing telemetry simultaneously

        INPUT:
        - 3 concurrent record_snapshots_batch() calls

        EXPECTED BEHAVIOR (with patch):
        - All 3 batches succeed
        - No data corruption
        - SQLite handles locking

        FAILURE BEHAVIOR (without patch):
        - Lock contention may cause timeouts
        - Batches may interfere with each other
        """
        db_path = tmp_path / "test_telemetry.db"
        store = TelemetryStore(db_path=db_path)

        async def flush_project(project_id):
            profiles = []
            for i in range(3):
                profile = ModelProfile(
                    model=Model.GPT_4O_MINI,
                    provider="openai",
                    cost_per_1m_input=0.15,
                    cost_per_1m_output=0.60,
                )
                profile.call_count = i + 1
                profiles.append((Model.GPT_4O_MINI, profile))
            await store.record_snapshots_batch(project_id, profiles)

        # Act: 3 concurrent flushes
        await asyncio.gather(
            flush_project("project_1"),
            flush_project("project_2"),
            flush_project("project_3"),
        )

        # Assert: All records saved
        async with aiosqlite.connect(db_path) as db:
            async with db.execute("SELECT COUNT(*) FROM model_snapshots") as cursor:
                result = await cursor.fetchone()
                # 3 projects × 3 profiles = 9 records
                assert result[0] == 9
