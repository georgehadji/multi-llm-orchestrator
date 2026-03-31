"""
Reliability Regression Tests
============================
Tests for bug fixes:
- BUG-RACE-001: AdaptiveRouter thread-safety
- BUG-RACE-002: Orchestrator.results concurrent access
- BUG-SHUTDOWN-001: Fire-and-forget task tracking
- BUG-DBCONN-001: Database connection error handling
- BUG-EVENTLOOP-001: Event loop shutdown ordering

These tests prevent regression of critical reliability fixes.
"""

import asyncio
import pytest
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, MagicMock, patch

from orchestrator.adaptive_router import AdaptiveRouter, ModelState
from orchestrator.models import Model, TaskType, Task, TaskResult, TaskStatus, Budget
from orchestrator.telemetry_store import TelemetryStore
from orchestrator.state import StateManager

# ═══════════════════════════════════════════════════════════════════════════════
# BUG-RACE-001: AdaptiveRouter Thread-Safety Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestAdaptiveRouterThreadSafety:
    """Test thread-safety of AdaptiveRouter under concurrent access."""

    def test_concurrent_record_timeout(self):
        """Multiple threads calling record_timeout should not corrupt state."""
        router = AdaptiveRouter(timeout_threshold=100)
        num_threads = 50
        calls_per_thread = 100

        def timeout_worker():
            for _ in range(calls_per_thread):
                router.record_timeout(Model.GPT_4O)

        threads = [threading.Thread(target=timeout_worker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Count should be exactly num_threads * calls_per_thread
        expected = num_threads * calls_per_thread
        assert router._timeout_counts[Model.GPT_4O] == expected

    def test_concurrent_timeout_and_success(self):
        """Mixed timeout/success calls should maintain consistent state."""
        router = AdaptiveRouter(timeout_threshold=10)

        def timeout_worker():
            for _ in range(50):
                router.record_timeout(Model.GPT_4O)

        def success_worker():
            for _ in range(50):
                router.record_success(Model.GPT_4O)

        threads = []
        for _ in range(10):
            threads.append(threading.Thread(target=timeout_worker))
            threads.append(threading.Thread(target=success_worker))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should not crash, state should be valid
        state = router.get_state(Model.GPT_4O)
        assert state in [ModelState.HEALTHY, ModelState.DEGRADED]

    def test_concurrent_state_reads(self):
        """Multiple concurrent state reads should all return valid states."""
        router = AdaptiveRouter(timeout_threshold=3)

        # Put model in DEGRADED state
        for _ in range(3):
            router.record_timeout(Model.GPT_4O)

        states = []
        lock = threading.Lock()

        def read_state():
            state = router.get_state(Model.GPT_4O)
            with lock:
                states.append(state)

        threads = [threading.Thread(target=read_state) for _ in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All reads should return valid ModelState
        assert len(states) == 100
        assert all(
            s in [ModelState.HEALTHY, ModelState.DEGRADED, ModelState.DISABLED] for s in states
        )

    def test_cooldown_boundary_race(self):
        """Test state transitions at cooldown boundary."""
        router = AdaptiveRouter(timeout_threshold=3, cooldown_seconds=0.1)

        # Trigger DEGRADED state
        for _ in range(3):
            router.record_timeout(Model.GPT_4O)

        assert router.get_state(Model.GPT_4O) == ModelState.DEGRADED

        # Wait for cooldown
        time.sleep(0.15)

        # Multiple threads should all see HEALTHY
        results = []

        def check_and_reset():
            state = router.get_state(Model.GPT_4O)
            results.append(state)

        threads = [threading.Thread(target=check_and_reset) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should see HEALTHY (cooldown elapsed, state reset)
        assert all(s == ModelState.HEALTHY for s in results)

    def test_record_auth_failure_threading(self):
        """Concurrent auth failures should correctly disable model."""
        router = AdaptiveRouter()

        def auth_failure_worker():
            for _ in range(10):
                router.record_auth_failure(Model.GPT_4O)

        threads = [threading.Thread(target=auth_failure_worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Model should be disabled
        assert Model.GPT_4O in router._disabled
        assert router.get_state(Model.GPT_4O) == ModelState.DISABLED

    def test_preferred_model_concurrent(self):
        """Concurrent preferred_model calls should return consistent results."""
        router = AdaptiveRouter()

        # Set up latencies
        router.record_latency(Model.GPT_4O, 100)
        router.record_latency(Model.GPT_4O_MINI, 50)
        router.record_latency(Model.GEMINI_FLASH, 75)

        results = []
        lock = threading.Lock()

        def get_preferred():
            preferred = router.preferred_model(
                [
                    Model.GPT_4O,
                    Model.GPT_4O_MINI,
                    Model.GEMINI_FLASH,
                ]
            )
            with lock:
                results.append(preferred)

        threads = [threading.Thread(target=get_preferred) for _ in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should return GPT_4O_MINI (lowest latency)
        assert all(r == Model.GPT_4O_MINI for r in results)


class TestAdaptiveRouterEdgeCases:
    """Edge case tests for AdaptiveRouter."""

    def test_timeout_threshold_boundary(self):
        """Test behavior exactly at timeout threshold."""
        router = AdaptiveRouter(timeout_threshold=3)

        # Just below threshold
        router.record_timeout(Model.GPT_4O)
        router.record_timeout(Model.GPT_4O)
        assert router.get_state(Model.GPT_4O) == ModelState.HEALTHY

        # At threshold
        router.record_timeout(Model.GPT_4O)
        assert router.get_state(Model.GPT_4O) == ModelState.DEGRADED

    def test_success_resets_timeout_count(self):
        """Success should reset timeout count."""
        router = AdaptiveRouter(timeout_threshold=3)

        router.record_timeout(Model.GPT_4O)
        router.record_timeout(Model.GPT_4O)
        router.record_success(Model.GPT_4O)
        router.record_timeout(Model.GPT_4O)

        # Should not be DEGRADED (count was reset)
        assert router.get_state(Model.GPT_4O) == ModelState.HEALTHY
        assert router._timeout_counts[Model.GPT_4O] == 1

    def test_disabled_model_ignores_timeouts(self):
        """Disabled model should ignore subsequent timeout calls."""
        router = AdaptiveRouter()

        router.record_auth_failure(Model.GPT_4O)
        assert router.get_state(Model.GPT_4O) == ModelState.DISABLED

        # These should be ignored
        for _ in range(100):
            router.record_timeout(Model.GPT_4O)

        # Should still be disabled, not crash
        assert router.get_state(Model.GPT_4O) == ModelState.DISABLED


# ═══════════════════════════════════════════════════════════════════════════════
# BUG-DBCONN-001: Database Connection Error Handling Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestTelemetryStoreErrorHandling:
    """Test TelemetryStore error handling for database connections."""

    @pytest.mark.asyncio
    async def test_ensure_schema_idempotent(self, tmp_path):
        """Schema initialization should be idempotent."""
        db_path = tmp_path / "telemetry.db"
        store = TelemetryStore(db_path=db_path)

        # Call multiple times
        await store._ensure_schema()
        assert store._initialised == True

        await store._ensure_schema()
        assert store._initialised == True

        # Should not raise
        await store._ensure_schema()
        assert store._initialised == True

    @pytest.mark.asyncio
    async def test_ensure_schema_handles_corruption(self, tmp_path):
        """Schema initialization should handle errors gracefully."""
        db_path = tmp_path / "telemetry.db"
        store = TelemetryStore(db_path=db_path)

        # Write garbage to the DB file to simulate corruption
        db_path.write_text("not a valid sqlite database")

        # Should handle the error and retry on next call
        with pytest.raises(Exception):
            await store._ensure_schema()

        # _initialised should be False after failure
        assert store._initialised == False


class TestStateManagerErrorHandling:
    """Test StateManager error handling for database connections."""

    @pytest.mark.asyncio
    async def test_get_conn_failure_cleanup(self, tmp_path):
        """Connection failure should clean up partial state."""
        db_path = tmp_path / "state.db"
        state_mgr = StateManager(db_path=db_path)

        # Mock aiosqlite to fail
        async def mock_connect_fail(path):
            import aiosqlite

            raise aiosqlite.Error("Simulated connection failure")

        with patch("orchestrator.state.aiosqlite.connect", mock_connect_fail):
            with pytest.raises(Exception):
                await state_mgr._get_conn()

            # Connection should be None after failure
            assert state_mgr._conn is None


# ═══════════════════════════════════════════════════════════════════════════════
# BUG-SHUTDOWN-001 & BUG-EVENTLOOP-001: Shutdown Ordering Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestOrchestratorShutdownOrdering:
    """Test proper shutdown ordering for Orchestrator."""

    @pytest.mark.asyncio
    async def test_background_tasks_tracked(self):
        """Background tasks should be tracked for shutdown."""
        from orchestrator.engine import Orchestrator

        orch = Orchestrator()

        # Initially no background tasks
        assert len(orch._background_tasks) == 0

        # Simulate creating a background task
        async def dummy_task():
            await asyncio.sleep(0.01)
            return "done"

        task = asyncio.create_task(dummy_task())
        orch._background_tasks.add(task)
        task.add_done_callback(orch._background_tasks.discard)

        assert len(orch._background_tasks) == 1

        # Wait for task to complete
        await task

        # Task should be removed from set
        assert len(orch._background_tasks) == 0

    @pytest.mark.asyncio
    async def test_aexit_waits_for_background_tasks(self):
        """__aexit__ should wait for background tasks to complete."""
        from orchestrator.engine import Orchestrator

        orch = Orchestrator()

        # Track if task completed
        task_completed = False

        async def slow_task():
            nonlocal task_completed
            await asyncio.sleep(0.1)
            task_completed = True

        # Add background task
        task = asyncio.create_task(slow_task())
        orch._background_tasks.add(task)
        task.add_done_callback(orch._background_tasks.discard)

        # Exit context manager (should wait for task)
        async with orch:
            pass

        # Task should have completed
        assert task_completed == True

    @pytest.mark.asyncio
    async def test_aexit_timeout_for_slow_tasks(self):
        """__aexit__ should timeout for very slow tasks."""
        from orchestrator.engine import Orchestrator

        orch = Orchestrator()

        # Track if task completed
        task_completed = False

        async def very_slow_task():
            nonlocal task_completed
            await asyncio.sleep(10)  # Longer than 5s timeout
            task_completed = True

        # Add background task
        task = asyncio.create_task(very_slow_task())
        orch._background_tasks.add(task)
        task.add_done_callback(orch._background_tasks.discard)

        # Exit context manager (should timeout after 5s)
        start = time.time()
        async with orch:
            pass
        elapsed = time.time() - start

        # Should have timed out (around 5s, not 10s)
        assert elapsed < 7.0  # Allow some margin

        # Task should not have completed
        assert task_completed == False


# ═══════════════════════════════════════════════════════════════════════════════
# BUG-RACE-002: Orchestrator.results Concurrent Access Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestOrchestratorResultsConcurrency:
    """Test concurrent access to Orchestrator.results dict."""

    @pytest.mark.asyncio
    async def test_concurrent_results_writes(self):
        """Concurrent writes to results dict should be safe."""
        from orchestrator.engine import Orchestrator

        orch = Orchestrator()

        async def write_result(task_id: str, score: float):
            result = TaskResult(
                task_id=task_id,
                output=f"output_{task_id}",
                score=score,
                model_used=Model.GPT_4O_MINI,
                status=TaskStatus.COMPLETED,
            )
            async with orch._results_lock:
                orch.results[task_id] = result

        # Write concurrently
        tasks = [write_result(f"task_{i}", i * 0.1) for i in range(50)]
        await asyncio.gather(*tasks)

        # All results should be present
        assert len(orch.results) == 50
        for i in range(50):
            assert f"task_{i}" in orch.results
            assert orch.results[f"task_{i}"].score == i * 0.1

    @pytest.mark.asyncio
    async def test_concurrent_results_read_write(self):
        """Concurrent reads and writes should be safe."""
        from orchestrator.engine import Orchestrator

        orch = Orchestrator()

        async def writer(task_id: str):
            for i in range(10):
                result = TaskResult(
                    task_id=task_id,
                    output=f"output_{i}",
                    score=i * 0.1,
                    model_used=Model.GPT_4O_MINI,
                    status=TaskStatus.COMPLETED,
                )
                async with orch._results_lock:
                    orch.results[task_id] = result
                await asyncio.sleep(0.001)

        async def reader():
            for _ in range(100):
                async with orch._results_lock:
                    _ = dict(orch.results)  # Copy while holding lock
                await asyncio.sleep(0.001)

        # Run writers and readers concurrently
        writers = [writer(f"task_{i}") for i in range(5)]
        readers = [reader() for _ in range(10)]

        await asyncio.gather(*writers, *readers)

        # Should not crash
        assert len(orch.results) == 5


# ═══════════════════════════════════════════════════════════════════════════════
# Integration Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestReliabilityIntegration:
    """Integration tests for reliability fixes."""

    @pytest.mark.asyncio
    async def test_full_lifecycle_no_leaks(self, tmp_path):
        """Full orchestrator lifecycle should not leak resources."""
        from orchestrator.engine import Orchestrator
        from orchestrator.models import Budget

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # Create orchestrator with temp paths
        orch = Orchestrator(
            budget=Budget(max_usd=1.0, max_time_seconds=60),
        )

        # Simulate some background tasks
        async def dummy_bg():
            await asyncio.sleep(0.01)

        for _ in range(10):
            task = asyncio.create_task(dummy_bg())
            orch._background_tasks.add(task)
            task.add_done_callback(orch._background_tasks.discard)

        # Wait a bit for tasks to complete
        await asyncio.sleep(0.1)

        # Clean shutdown
        await orch.close()

        # All background tasks should be cleaned up
        assert len(orch._background_tasks) == 0

    @pytest.mark.asyncio
    async def test_router_and_orchestrator_concurrent(self):
        """Router and orchestrator should work correctly under concurrency."""
        from orchestrator.engine import Orchestrator
        from orchestrator.models import Budget

        orch = Orchestrator(budget=Budget(max_usd=1.0))

        # Simulate concurrent router updates from multiple "tasks"
        def update_router():
            for _ in range(50):
                orch._adaptive_router.record_timeout(Model.GPT_4O)
                orch._adaptive_router.record_success(Model.GPT_4O)
                orch._adaptive_router.get_state(Model.GPT_4O)

        threads = [threading.Thread(target=update_router) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should not crash, router should be in valid state
        state = orch._adaptive_router.get_state(Model.GPT_4O)
        assert state in [ModelState.HEALTHY, ModelState.DEGRADED, ModelState.DISABLED]

        await orch.close()
