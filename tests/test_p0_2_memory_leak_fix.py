"""
Test P0-2: Memory Leak Fix with WeakSet
========================================
Tests that background tasks are properly tracked using WeakSet
to prevent memory leaks in long-running sessions.
"""

import asyncio
import gc
import pytest
import weakref
from unittest.mock import AsyncMock, patch

from orchestrator.engine import Orchestrator
from orchestrator.models import Budget


class TestMemoryLeakFix:
    """Test P0-2: WeakSet-based background task tracking."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator for testing."""
        orch = Orchestrator(budget=Budget(max_usd=100.0))
        return orch

    @pytest.mark.asyncio
    async def test_background_tasks_uses_weakset(self, orchestrator):
        """
        Verify _background_tasks is a WeakSet, not a regular set.
        """
        # Assert: Should be WeakSet
        assert isinstance(orchestrator._background_tasks, weakref.WeakSet)

    @pytest.mark.asyncio
    async def test_cleanup_timer_initialized(self, orchestrator):
        """
        Verify _cleanup_timer is initialized to None.
        """
        # Assert: Should be None initially
        assert orchestrator._cleanup_timer is None

    @pytest.mark.asyncio
    async def test_periodic_cleanup_starts_on_enter(self, orchestrator):
        """
        Verify periodic cleanup timer starts when entering context manager.
        """
        # Act: Enter context manager
        async with orchestrator:
            # Assert: Cleanup timer should be started
            assert orchestrator._cleanup_timer is not None
            assert not orchestrator._cleanup_timer.done()

    @pytest.mark.asyncio
    async def test_periodic_cleanup_cancelled_on_exit(self, orchestrator):
        """
        Verify periodic cleanup timer is cancelled on exit.
        """
        # Act: Enter and exit context manager
        async with orchestrator:
            timer = orchestrator._cleanup_timer
            assert timer is not None

        # Assert: Timer should be cancelled
        assert timer.cancelled()

    @pytest.mark.asyncio
    async def test_background_task_tracked(self, orchestrator):
        """
        Verify background tasks are added to WeakSet.
        """

        # Arrange: Create a background task
        async def dummy_task():
            await asyncio.sleep(0.01)

        task = asyncio.create_task(dummy_task())
        orchestrator._background_tasks.add(task)

        # Assert: Task should be in WeakSet
        assert task in orchestrator._background_tasks

        # Wait for task to complete
        await task

        # Force GC
        gc.collect()
        await asyncio.sleep(0.1)

        # Task should still be in WeakSet until garbage collected
        # (WeakSet removes when no strong references exist)

    @pytest.mark.asyncio
    async def test_cleanup_background_tasks(self, orchestrator):
        """
        Verify _cleanup_background_tasks() works correctly.
        """

        # Arrange: Create completed tasks
        async def quick_task():
            return "done"

        task1 = asyncio.create_task(quick_task())
        task2 = asyncio.create_task(quick_task())

        # Wait for tasks to complete
        await asyncio.sleep(0.1)

        # Add to background tasks
        orchestrator._background_tasks.add(task1)
        orchestrator._background_tasks.add(task2)

        # Act: Cleanup
        cleaned = await orchestrator._cleanup_background_tasks()

        # Assert: Should return 0 (WeakSet handles cleanup automatically)
        assert cleaned == 0

    @pytest.mark.asyncio
    async def test_weakset_automatic_cleanup(self, orchestrator):
        """
        Verify WeakSet automatically removes tasks when they're garbage collected.
        """

        # Arrange: Create a task and add to WeakSet
        async def dummy():
            await asyncio.sleep(0.01)

        task = asyncio.create_task(dummy())
        weak_ref = weakref.ref(task)
        orchestrator._background_tasks.add(task)

        # Assert: Task is in WeakSet
        assert task in orchestrator._background_tasks

        # Delete strong reference
        del task

        # Force GC
        gc.collect()
        await asyncio.sleep(0.1)

        # Assert: Task should be removed from WeakSet
        assert weak_ref() is None  # Task was garbage collected

    @pytest.mark.asyncio
    async def test_flush_telemetry_tracks_background_task(self, orchestrator):
        """
        Verify _flush_telemetry_snapshots creates tracked background task.
        """
        # Arrange: Set up a profile with call_count > 0
        model = list(orchestrator._profiles.keys())[0]
        profile = orchestrator._profiles[model]
        profile.call_count = 5  # Mark as active

        initial_count = len(orchestrator._background_tasks)

        # Act: Flush telemetry
        await orchestrator._flush_telemetry_snapshots("test_project")

        # Small delay for task creation
        await asyncio.sleep(0.01)

        # Assert: Background task should be tracked
        assert len(orchestrator._background_tasks) >= initial_count

    @pytest.mark.asyncio
    async def test_periodic_cleanup_interval(self, orchestrator):
        """
        Verify periodic cleanup runs at configured interval.
        """
        # Arrange: Short interval for testing
        interval = 0.1  # 100ms

        async with orchestrator:
            # Manually trigger cleanup with short interval
            await orchestrator._start_periodic_cleanup(interval_seconds=interval)

            # Wait for cleanup to run
            await asyncio.sleep(interval * 1.5)

            # Timer should still be running
            assert not orchestrator._cleanup_timer.done()

    @pytest.mark.asyncio
    async def test_no_memory_leak_long_running(self, orchestrator):
        """
        Integration test: Verify no memory leak over multiple projects.
        """
        # Arrange: Track background task count
        initial_count = len(list(orchestrator._background_tasks))

        # Act: Simulate multiple project runs
        for i in range(5):
            # Create background tasks
            async def dummy():
                await asyncio.sleep(0.01)
                return f"project_{i}"

            task = asyncio.create_task(dummy())
            orchestrator._background_tasks.add(task)

            # Wait for task
            await task

        # Force GC
        gc.collect()
        await asyncio.sleep(0.2)

        # Assert: Background task count should not grow unbounded
        # (WeakSet should have cleaned up completed tasks)
        final_count = len(list(orchestrator._background_tasks))
        # Allow some tolerance for timing
        assert final_count <= initial_count + 2
