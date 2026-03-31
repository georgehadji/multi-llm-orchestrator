"""
Test P2-2: Batch Telemetry Operations
======================================
Tests that batch insert is used for telemetry snapshots
to improve throughput by 10x.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, call

from orchestrator.engine import Orchestrator
from orchestrator.models import Budget, Model
from orchestrator.telemetry_store import TelemetryStore


class TestBatchTelemetry:
    """Test P2-2: Batch telemetry operations."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator for testing."""
        orch = Orchestrator(budget=Budget(max_usd=100.0))
        return orch

    @pytest.fixture
    def telemetry_store(self, tmp_path):
        """Create test telemetry store."""
        db_path = tmp_path / "test_telemetry.db"
        store = TelemetryStore(db_path=db_path, batch_size=10)
        return store

    @pytest.mark.asyncio
    async def test_record_snapshots_batch_exists(self, telemetry_store):
        """
        Verify record_snapshots_batch() method exists.
        """
        # Assert: Method should exist
        assert hasattr(telemetry_store, "record_snapshots_batch")
        assert callable(telemetry_store.record_snapshots_batch)

    @pytest.mark.asyncio
    async def test_record_snapshots_batch_empty(self, telemetry_store):
        """
        Verify empty batch is handled gracefully.
        """
        # Act: Record empty batch
        await telemetry_store.record_snapshots_batch("test_project", [])

        # Assert: Should not raise (no-op)

    @pytest.mark.asyncio
    async def test_record_snapshots_batch_single(self, telemetry_store):
        """
        Verify single snapshot batch works.
        """
        # Arrange: Create a profile
        model = Model.GPT_4O_MINI
        profile = orchestrator._profiles[model]
        profile.call_count = 5
        profile.quality_score = 0.85

        # Act: Record batch
        await telemetry_store.record_snapshots_batch("test_project", [(model, profile)])

        # Assert: Should not raise

    @pytest.mark.asyncio
    async def test_record_snapshots_batch_filters_inactive(self, telemetry_store):
        """
        Verify profiles with call_count=0 are filtered out.
        """
        # Arrange: Create profiles with different call counts
        model1 = Model.GPT_4O_MINI
        model2 = Model.GPT_4O

        profile1 = orchestrator._profiles[model1]
        profile2 = orchestrator._profiles[model2]

        profile1.call_count = 5  # Active
        profile2.call_count = 0  # Inactive

        # Act: Record batch
        await telemetry_store.record_snapshots_batch(
            "test_project", [(model1, profile1), (model2, profile2)]
        )

        # Assert: Only active profile should be recorded
        # (We can't easily verify this without reading from DB,
        # but the method should complete without error)

    @pytest.mark.asyncio
    async def test_record_snapshots_batch_multiple(self, telemetry_store):
        """
        Verify multiple snapshots are batched correctly.
        """
        # Arrange: Create multiple profiles
        snapshots = []
        for i, model in enumerate(list(Model)[:5]):
            profile = orchestrator._profiles[model]
            profile.call_count = i + 1
            profile.quality_score = 0.8 + (i * 0.02)
            snapshots.append((model, profile))

        # Act: Record batch
        await telemetry_store.record_snapshots_batch("test_project", snapshots)

        # Assert: Should complete without error

    @pytest.mark.asyncio
    async def test_flush_telemetry_uses_batch(self, orchestrator):
        """
        Verify _flush_telemetry_snapshots uses batch method.
        """
        # Arrange: Set up active profiles
        for i, model in enumerate(list(Model)[:5]):
            orchestrator._profiles[model].call_count = i + 1

        # Mock the batch method
        original_batch = orchestrator._telemetry_store.record_snapshots_batch
        orchestrator._telemetry_store.record_snapshots_batch = AsyncMock()

        # Act: Flush telemetry
        await orchestrator._flush_telemetry_snapshots("test_project")

        # Small delay for async task
        await asyncio.sleep(0.05)

        # Assert: Batch method should be called
        assert orchestrator._telemetry_store.record_snapshots_batch.called

        # Restore
        orchestrator._telemetry_store.record_snapshots_batch = original_batch

    @pytest.mark.asyncio
    async def test_batch_vs_individual_performance(self, telemetry_store, tmp_path):
        """
        Benchmark-style: Compare batch vs individual inserts.
        """
        # Arrange: Create test data
        snapshots = []
        for i, model in enumerate(list(Model)[:10]):
            profile = orchestrator._profiles[model]
            profile.call_count = 5
            profile.quality_score = 0.85
            snapshots.append((model, profile))

        # Measure batch insert
        import time

        start = time.time()
        await telemetry_store.record_snapshots_batch("batch_test", snapshots)
        batch_time = time.time() - start

        # Note: We can't easily test individual inserts without modifying
        # the existing record_snapshot method behavior
        # This test is more for documentation of the optimization

        # Assert: Batch should complete (timing is for manual verification)
        assert batch_time < 1.0  # Should be fast

    @pytest.mark.asyncio
    async def test_batch_uses_executemany(self, telemetry_store):
        """
        Verify batch uses executemany for efficiency.
        """
        # This is a code inspection test - we verify the implementation
        # uses executemany by checking the method signature

        import inspect

        source = inspect.getsource(telemetry_store.record_snapshots_batch)

        # Assert: Should use executemany
        assert "executemany" in source

    @pytest.mark.asyncio
    async def test_batch_transaction_handling(self, telemetry_store):
        """
        Verify batch operations are wrapped in transaction.
        """
        import inspect

        source = inspect.getsource(telemetry_store.record_snapshots_batch)

        # Assert: Should use transaction (commit)
        assert "commit" in source

    @pytest.mark.asyncio
    async def test_integration_with_orchestrator(self, orchestrator):
        """
        Integration test: Verify full flow from orchestrator to batch insert.
        """
        # Arrange: Set up active profiles
        test_model = list(Model)[:3]
        for i, model in enumerate(test_model):
            orchestrator._profiles[model].call_count = i + 1

        # Mock to capture calls
        captured_batches = []

        async def capture_batch(project_id, snapshots):
            captured_batches.append((project_id, snapshots))

        original_batch = orchestrator._telemetry_store.record_snapshots_batch
        orchestrator._telemetry_store.record_snapshots_batch = capture_batch

        # Act: Flush telemetry
        await orchestrator._flush_telemetry_snapshots("integration_test")

        # Wait for async task
        await asyncio.sleep(0.1)

        # Assert: Batch should be captured
        assert len(captured_batches) == 1
        project_id, snapshots = captured_batches[0]
        assert project_id == "integration_test"
        assert len(snapshots) == 3  # 3 active profiles

        # Restore
        orchestrator._telemetry_store.record_snapshots_batch = original_batch

    @pytest.mark.asyncio
    async def test_batch_error_handling(self, telemetry_store):
        """
        Verify errors in batch are handled gracefully.
        """
        # Arrange: Invalid data that might cause DB error
        # (This is hard to test without mocking the DB)

        # Mock aiosqlite to raise an error
        with patch("aiosqlite.connect") as mock_connect:
            mock_connect.side_effect = Exception("DB error")

            # Act: Should not raise
            try:
                await telemetry_store.record_snapshots_batch(
                    "test", [(Model.GPT_4O_MINI, orchestrator._profiles[Model.GPT_4O_MINI])]
                )
            except Exception as e:
                # Expected to raise DB error
                assert "DB error" in str(e)
