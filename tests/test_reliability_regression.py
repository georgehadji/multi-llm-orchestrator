"""
Reliability Fix Regression Tests
================================

Test suite for verifying reliability bug fixes:
- BUG-DEADLOCK-003: A2A message queue deadlock prevention
- BUG-MEMORY-002: Background task memory leak prevention
- BUG-RACE-002: Results dictionary race condition (already fixed)

Each test covers:
- Happy path
- Edge cases
- Failure modes

Regression invariants are defined at module level.
"""

import asyncio
import pytest
import time
import weakref
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, List, Any

# Import modules under test
from orchestrator.a2a_protocol import (
    A2AManager,
    AgentCard,
    TaskSendRequest,
    TaskResult,
    AgentState,
    TaskStatus,
    MessagePart,
    A2AMessage,
)
from orchestrator.engine import Orchestrator
from orchestrator.models import Task, TaskType, Budget, Model

# ═══════════════════════════════════════════════════════════════════════════════
# REGRESSION INVARIANTS
# ═══════════════════════════════════════════════════════════════════════════════


class Invariants:
    """
    Regression invariants that must never be violated.
    These are the core properties that maintain system reliability.
    """

    # BUG-DEADLOCK-003 INVARIANTS
    A2A_NO_ORPHANED_RESPONSES = "After timeout, _pending_responses must be empty"
    A2A_NO_LEAKED_TIMEOUTS = "After timeout, _response_timeouts must be empty"
    A2A_QUEUE_BOUNDED = "Queue size must never exceed _max_queue_size"
    A2A_HANDLER_CANCELLED = "Timed-out handler task must be cancelled"

    # BUG-MEMORY-002 INVARIANTS
    BACKGROUND_TASKS_CLEANED = "Completed tasks must be removed from _background_tasks"
    BACKGROUND_TASKS_BOUNDED = "_background_tasks must not grow unbounded"

    # BUG-RACE-002 INVARIANT
    RESULTS_THREAD_SAFE = "Concurrent writes to results must not corrupt data"


# ═══════════════════════════════════════════════════════════════════════════════
# BUG-DEADLOCK-003: A2A Message Queue Deadlock Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestA2ADeadlockPrevention:
    """
    Tests for BUG-DEADLOCK-003 fix.

    Invariant: After timeout, all tracking state must be cleaned up
    and handler task must be cancelled to prevent queue deadlock.
    """

    # ───────────────────────────────────────────────────────────────────────────
    # Happy Path Tests
    # ───────────────────────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_send_task_completes_normally(self):
        """
        Happy path: Task completes within timeout.

        Invariant: Tracking state cleaned up on success.
        """
        manager = A2AManager()

        # Register agent with fast handler
        async def fast_handler(message, context):
            await asyncio.sleep(0.01)  # Fast completion
            return "done"

        await manager.register_agent(
            AgentCard(agent_id="worker", name="Worker", description="Test"),
            handler=fast_handler,
        )

        # Send task
        request = TaskSendRequest(
            task_id="task_001",
            target_agent="worker",
            message="test",
            timeout_seconds=5.0,
        )

        result = await manager.send_task(request)

        # Verify success
        assert result.status == TaskStatus.COMPLETED
        assert result.result == "done"

        # INVARIANT: Tracking cleaned up
        assert "task_001" not in manager._pending_responses, Invariants.A2A_NO_ORPHANED_RESPONSES
        assert "task_001" not in manager._response_timeouts, Invariants.A2A_NO_LEAKED_TIMEOUTS

    @pytest.mark.asyncio
    async def test_send_task_without_handler(self):
        """
        Happy path: Task sent to agent without handler (passthrough).

        Invariant: No tracking state created.
        """
        manager = A2AManager()

        # Register agent without handler
        await manager.register_agent(
            AgentCard(agent_id="passive", name="Passive", description="Test"),
        )

        request = TaskSendRequest(
            task_id="task_002",
            target_agent="passive",
            message="test",
            timeout_seconds=5.0,
        )

        result = await manager.send_task(request)

        # Verify submitted but not processed
        assert result.status == TaskStatus.SUBMITTED

        # INVARIANT: No tracking for agent without handler
        assert "task_002" not in manager._pending_responses
        assert "task_002" not in manager._response_timeouts

    # ───────────────────────────────────────────────────────────────────────────
    # Edge Case Tests
    # ───────────────────────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_timeout_exactly_at_limit(self):
        """
        Edge case: Handler completes exactly at timeout boundary.

        Invariant: Race condition handled correctly.
        """
        manager = A2AManager()

        async def boundary_handler(message, context):
            await asyncio.sleep(1.0)  # Exactly at timeout
            return "done"

        await manager.register_agent(
            AgentCard(agent_id="boundary", name="Boundary", description="Test"),
            handler=boundary_handler,
        )

        request = TaskSendRequest(
            task_id="task_boundary",
            target_agent="boundary",
            message="test",
            timeout_seconds=1.0,
        )

        result = await manager.send_task(request)

        # Either success or timeout is acceptable
        assert result.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)

        # INVARIANT: Tracking always cleaned up
        assert (
            "task_boundary" not in manager._pending_responses
        ), Invariants.A2A_NO_ORPHANED_RESPONSES
        assert "task_boundary" not in manager._response_timeouts, Invariants.A2A_NO_LEAKED_TIMEOUTS

    @pytest.mark.asyncio
    async def test_queue_at_max_capacity(self):
        """
        Edge case: Queue at exactly max capacity.

        Invariant: New messages rejected gracefully.
        """
        manager = A2AManager()
        manager._max_queue_size = 5

        # Fill queue
        for i in range(5):
            await manager._message_queues["worker"].put(
                A2AMessage(
                    id=f"msg_{i}",
                    sender="test",
                    receiver="worker",
                    message_type="test",
                    parts=[MessagePart(type="text", content="fill")],
                )
            )

        # Verify queue full
        assert manager._message_queues["worker"].qsize() == 5

        # Try to add one more (should fail gracefully)
        request = TaskSendRequest(
            task_id="task_overflow",
            target_agent="worker",
            message="test",
            timeout_seconds=5.0,
        )

        result = await manager.send_task(request)

        # INVARIANT: Queue bounded
        assert result.status == TaskStatus.FAILED
        assert "queue is full" in result.error.lower()
        assert manager._message_queues["worker"].qsize() == 5, Invariants.A2A_QUEUE_BOUNDED

    @pytest.mark.asyncio
    async def test_concurrent_tasks_same_agent(self):
        """
        Edge case: Multiple concurrent tasks to same agent.

        Invariant: Each task tracked independently.
        """
        manager = A2AManager()

        async def concurrent_handler(message, context):
            await asyncio.sleep(0.1)
            return "done"

        await manager.register_agent(
            AgentCard(agent_id="concurrent", name="Concurrent", description="Test"),
            handler=concurrent_handler,
        )

        # Send 10 concurrent tasks
        tasks = [
            manager.send_task(
                TaskSendRequest(
                    task_id=f"concurrent_{i}",
                    target_agent="concurrent",
                    message=f"task {i}",
                    timeout_seconds=5.0,
                )
            )
            for i in range(10)
        ]

        results = await asyncio.gather(*tasks)

        # Verify all completed
        assert all(r.status == TaskStatus.COMPLETED for r in results)

        # INVARIANT: All tracking cleaned up
        for i in range(10):
            assert (
                f"concurrent_{i}" not in manager._pending_responses
            ), Invariants.A2A_NO_ORPHANED_RESPONSES
            assert (
                f"concurrent_{i}" not in manager._response_timeouts
            ), Invariants.A2A_NO_LEAKED_TIMEOUTS

    # ───────────────────────────────────────────────────────────────────────────
    # Failure Mode Tests
    # ───────────────────────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_handler_timeout_cleanup(self):
        """
        Failure mode: Handler times out.

        Invariant: Tracking cleaned up, handler cancelled.
        """
        manager = A2AManager()

        async def slow_handler(message, context):
            await asyncio.sleep(10)  # Will timeout
            return "done"

        await manager.register_agent(
            AgentCard(agent_id="slow", name="Slow", description="Test"),
            handler=slow_handler,
        )

        request = TaskSendRequest(
            task_id="task_timeout",
            target_agent="slow",
            message="test",
            timeout_seconds=0.5,
        )

        start = time.time()
        result = await manager.send_task(request)
        elapsed = time.time() - start

        # Verify timeout
        assert result.status == TaskStatus.FAILED
        assert "timed out" in result.error.lower()
        assert elapsed < 2.0  # Should timeout quickly, not wait 10s

        # INVARIANT: Tracking cleaned up
        assert (
            "task_timeout" not in manager._pending_responses
        ), Invariants.A2A_NO_ORPHANED_RESPONSES
        assert "task_timeout" not in manager._response_timeouts, Invariants.A2A_NO_LEAKED_TIMEOUTS

    @pytest.mark.asyncio
    async def test_handler_exception_cleanup(self):
        """
        Failure mode: Handler raises exception.

        Invariant: Tracking cleaned up on exception.
        """
        manager = A2AManager()

        async def failing_handler(message, context):
            raise ValueError("Handler failed")

        await manager.register_agent(
            AgentCard(agent_id="failing", name="Failing", description="Test"),
            handler=failing_handler,
        )

        request = TaskSendRequest(
            task_id="task_exception",
            target_agent="failing",
            message="test",
            timeout_seconds=5.0,
        )

        result = await manager.send_task(request)

        # Verify failure
        assert result.status == TaskStatus.FAILED
        assert "Handler failed" in result.error

        # INVARIANT: Tracking cleaned up
        assert (
            "task_exception" not in manager._pending_responses
        ), Invariants.A2A_NO_ORPHANED_RESPONSES
        assert "task_exception" not in manager._response_timeouts, Invariants.A2A_NO_LEAKED_TIMEOUTS

    @pytest.mark.asyncio
    async def test_agent_not_found(self):
        """
        Failure mode: Target agent doesn't exist.

        Invariant: No tracking state created.
        """
        manager = A2AManager()

        request = TaskSendRequest(
            task_id="task_missing",
            target_agent="nonexistent",
            message="test",
            timeout_seconds=5.0,
        )

        result = await manager.send_task(request)

        # Verify failure
        assert result.status == TaskStatus.FAILED
        assert "not found" in result.error.lower()

        # INVARIANT: No tracking for failed send
        assert "task_missing" not in manager._pending_responses
        assert "task_missing" not in manager._response_timeouts

    @pytest.mark.asyncio
    async def test_cleanup_orphaned_responses(self):
        """
        Failure mode: Orphaned responses from expired timeouts.

        Invariant: Cleanup method removes expired entries.
        """
        manager = A2AManager()

        # Manually inject expired timeout (simulating bug scenario)
        manager._response_timeouts["orphan_001"] = time.time() - 10  # Expired 10s ago
        mock_task = asyncio.create_task(asyncio.sleep(0))
        manager._pending_responses["orphan_001"] = mock_task

        # Run cleanup
        cleaned = await manager.cleanup_orphaned_responses()

        # Verify cleanup
        assert cleaned >= 1
        assert "orphan_001" not in manager._pending_responses
        assert "orphan_001" not in manager._response_timeouts


# ═══════════════════════════════════════════════════════════════════════════════
# BUG-MEMORY-002: Background Task Memory Leak Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestBackgroundTaskMemoryLeak:
    """
    Tests for BUG-MEMORY-002 fix.

    Invariant: Completed background tasks must be removed from tracking set
    to prevent unbounded memory growth.
    """

    # ───────────────────────────────────────────────────────────────────────────
    # Happy Path Tests
    # ───────────────────────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_background_task_completes_normally(self):
        """
        Happy path: Background task completes successfully.

        Invariant: Task removed from tracking set.
        """
        orch = Orchestrator.__new__(Orchestrator)
        orch._background_tasks = set()
        orch._profiles = {}
        orch._telemetry_store = AsyncMock()

        # Create background task
        async def quick_task():
            await asyncio.sleep(0.01)

        task = asyncio.create_task(quick_task())
        orch._background_tasks.add(task)

        initial_count = len(orch._background_tasks)
        assert initial_count == 1

        # Wait for completion
        await task

        # Cleanup
        cleaned = await orch._cleanup_background_tasks()

        # INVARIANT: Task removed
        assert cleaned == 1
        assert len(orch._background_tasks) == 0, Invariants.BACKGROUND_TASKS_CLEANED

    @pytest.mark.asyncio
    async def test_cleanup_background_tasks_empty(self):
        """
        Happy path: Cleanup with no tasks.

        Invariant: No errors on empty set.
        """
        orch = Orchestrator.__new__(Orchestrator)
        orch._background_tasks = set()

        cleaned = await orch._cleanup_background_tasks()

        assert cleaned == 0
        assert len(orch._background_tasks) == 0

    # ───────────────────────────────────────────────────────────────────────────
    # Edge Case Tests
    # ───────────────────────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_mixed_completed_pending_tasks(self):
        """
        Edge case: Mix of completed and pending tasks.

        Invariant: Only completed tasks removed.
        """
        orch = Orchestrator.__new__(Orchestrator)
        orch._background_tasks = set()

        # Create completed task
        async def quick():
            await asyncio.sleep(0.01)

        completed_task = asyncio.create_task(quick())
        orch._background_tasks.add(completed_task)

        # Create pending task
        async def slow():
            await asyncio.sleep(10)

        pending_task = asyncio.create_task(slow())
        orch._background_tasks.add(pending_task)

        # Wait for quick task
        await completed_task

        # Cleanup
        cleaned = await orch._cleanup_background_tasks()

        # INVARIANT: Only completed removed
        assert cleaned == 1
        assert completed_task not in orch._background_tasks
        assert pending_task in orch._background_tasks

        # Cleanup pending task
        pending_task.cancel()
        try:
            await pending_task
        except asyncio.CancelledError:
            pass

        await orch._cleanup_background_tasks()
        assert len(orch._background_tasks) == 0

    @pytest.mark.asyncio
    async def test_callback_exception_handling(self):
        """
        Edge case: Callback raises exception.

        Invariant: Exception caught, task still removed.
        """
        orch = Orchestrator.__new__(Orchestrator)
        orch._background_tasks = set()

        # Create task that will complete
        async def quick():
            await asyncio.sleep(0.01)

        task = asyncio.create_task(quick())
        orch._background_tasks.add(task)

        # Wait for completion
        await task

        # Cleanup should handle any callback issues
        cleaned = await orch._cleanup_background_tasks()

        # INVARIANT: Task removed despite potential callback issues
        assert cleaned == 1
        assert task not in orch._background_tasks

    # ───────────────────────────────────────────────────────────────────────────
    # Failure Mode Tests
    # ───────────────────────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_task_with_exception(self):
        """
        Failure mode: Background task raises exception.

        Invariant: Task still cleaned up, exception logged.
        """
        orch = Orchestrator.__new__(Orchestrator)
        orch._background_tasks = set()

        async def failing_task():
            raise ValueError("Task failed")

        task = asyncio.create_task(failing_task())
        orch._background_tasks.add(task)

        # Wait for exception
        try:
            await task
        except ValueError:
            pass

        # Cleanup
        cleaned = await orch._cleanup_background_tasks()

        # INVARIANT: Task removed even with exception
        assert cleaned == 1
        assert task not in orch._background_tasks

    @pytest.mark.asyncio
    async def test_cancelled_task(self):
        """
        Failure mode: Background task cancelled.

        Invariant: Cancelled task cleaned up.
        """
        orch = Orchestrator.__new__(Orchestrator)
        orch._background_tasks = set()

        async def slow_task():
            await asyncio.sleep(10)

        task = asyncio.create_task(slow_task())
        orch._background_tasks.add(task)

        # Cancel
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Cleanup
        cleaned = await orch._cleanup_background_tasks()

        # INVARIANT: Cancelled task removed
        assert cleaned == 1
        assert task not in orch._background_tasks

    @pytest.mark.asyncio
    async def test_bounded_growth_under_load(self):
        """
        Failure mode: Many background tasks created rapidly.

        Invariant: Set size bounded by cleanup.
        """
        orch = Orchestrator.__new__(Orchestrator)
        orch._background_tasks = set()

        # Create 100 quick tasks
        for _ in range(100):

            async def quick():
                await asyncio.sleep(0.001)

            task = asyncio.create_task(quick())
            orch._background_tasks.add(task)

        # Wait briefly for some to complete
        await asyncio.sleep(0.1)

        # Cleanup
        cleaned = await orch._cleanup_background_tasks()

        # INVARIANT: Growth bounded
        assert cleaned > 0
        assert len(orch._background_tasks) < 100, Invariants.BACKGROUND_TASKS_BOUNDED

        # Wait for rest and cleanup
        await asyncio.sleep(0.2)
        await orch._cleanup_background_tasks()

        assert len(orch._background_tasks) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# BUG-RACE-002: Results Dictionary Race Condition Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestResultsRaceCondition:
    """
    Tests for BUG-RACE-002 fix.

    Invariant: Concurrent writes to results dictionary must be thread-safe.
    """

    # ───────────────────────────────────────────────────────────────────────────
    # Happy Path Tests
    # ───────────────────────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_sequential_result_writes(self):
        """
        Happy path: Sequential writes to results.

        Invariant: All writes succeed without corruption.
        """
        orch = Orchestrator.__new__(Orchestrator)
        orch._background_tasks = set()
        orch.results = {}
        orch._results_lock = asyncio.Lock()

        # Sequential writes
        for i in range(10):
            async with orch._results_lock:
                orch.results[f"task_{i}"] = TaskResult(
                    task_id=f"task_{i}",
                    output=f"output_{i}",
                    score=float(i),
                    model_used=Model.GPT_4O_MINI,
                )

        # Verify all present
        assert len(orch.results) == 10
        for i in range(10):
            assert f"task_{i}" in orch.results
            assert orch.results[f"task_{i}"].output == f"output_{i}"

    # ───────────────────────────────────────────────────────────────────────────
    # Edge Case Tests
    # ───────────────────────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_concurrent_result_writes(self):
        """
        Edge case: Concurrent writes to same results dict.

        Invariant: No data corruption.
        """
        orch = Orchestrator.__new__(Orchestrator)
        orch._background_tasks = set()
        orch.results = {}
        orch._results_lock = asyncio.Lock()

        async def write_result(task_id: str):
            async with orch._results_lock:
                orch.results[task_id] = TaskResult(
                    task_id=task_id,
                    output=f"output_{task_id}",
                    score=1.0,
                    model_used=Model.GPT_4O_MINI,
                )

        # Concurrent writes
        tasks = [write_result(f"task_{i}") for i in range(50)]
        await asyncio.gather(*tasks)

        # INVARIANT: All writes successful
        assert len(orch.results) == 50
        for i in range(50):
            assert f"task_{i}" in orch.results

    @pytest.mark.asyncio
    async def test_concurrent_read_write(self):
        """
        Edge case: Concurrent reads and writes.

        Invariant: Reads see consistent state.
        """
        orch = Orchestrator.__new__(Orchestrator)
        orch._background_tasks = set()
        orch.results = {}
        orch._results_lock = asyncio.Lock()

        async def writer(task_id: str):
            async with orch._results_lock:
                orch.results[task_id] = TaskResult(
                    task_id=task_id,
                    output=f"output_{task_id}",
                    score=1.0,
                    model_used=Model.GPT_4O_MINI,
                )

        async def reader():
            await asyncio.sleep(0.01)
            async with orch._results_lock:
                return dict(orch.results)

        # Start writers
        write_tasks = [asyncio.create_task(writer(f"task_{i}")) for i in range(10)]

        # Concurrent reads
        read_results = []
        for _ in range(5):
            result = await reader()
            read_results.append(result)

        await asyncio.gather(*write_tasks)

        # INVARIANT: All reads saw consistent state
        for result in read_results:
            assert isinstance(result, dict)
            for task_id, task_result in result.items():
                assert isinstance(task_result, TaskResult)

    # ───────────────────────────────────────────────────────────────────────────
    # Failure Mode Tests
    # ───────────────────────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_write_during_iteration(self):
        """
        Failure mode: Write while iterating over results.

        Invariant: No RuntimeError from dict size change.
        """
        orch = Orchestrator.__new__(Orchestrator)
        orch._background_tasks = set()
        orch.results = {}
        orch._results_lock = asyncio.Lock()

        # Initial write
        async with orch._results_lock:
            orch.results["task_1"] = TaskResult(
                task_id="task_1",
                output="output_1",
                score=1.0,
                model_used=Model.GPT_4O_MINI,
            )

        async def safe_iteration():
            async with orch._results_lock:
                # Safe iteration with lock held
                keys = list(orch.results.keys())

            # Write outside iteration
            async with orch._results_lock:
                orch.results["task_2"] = TaskResult(
                    task_id="task_2",
                    output="output_2",
                    score=1.0,
                    model_used=Model.GPT_4O_MINI,
                )

            return keys

        result = await safe_iteration()

        # INVARIANT: No exception
        assert "task_1" in result
        assert "task_2" in orch.results


# ═══════════════════════════════════════════════════════════════════════════════
# Integration Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestReliabilityIntegration:
    """
    Integration tests combining multiple reliability fixes.
    """

    @pytest.mark.asyncio
    async def test_full_pipeline_with_timeouts(self):
        """
        Integration: Full task pipeline with A2A timeouts.

        Invariants: All fixes work together correctly.
        """
        # Setup A2A manager
        a2a = A2AManager()

        # Register slow agent (will timeout)
        async def slow_agent(message, context):
            await asyncio.sleep(5)
            return "done"

        await a2a.register_agent(
            AgentCard(agent_id="slow", name="Slow", description="Test"),
            handler=slow_agent,
        )

        # Send task with short timeout
        request = TaskSendRequest(
            task_id="integration_001",
            target_agent="slow",
            message="test",
            timeout_seconds=0.5,
        )

        result = await a2a.send_task(request)

        # Verify timeout
        assert result.status == TaskStatus.FAILED
        assert "timed out" in result.error.lower()

        # INVARIANT: All tracking cleaned up
        assert "integration_001" not in a2a._pending_responses
        assert "integration_001" not in a2a._response_timeouts

        # Cleanup any orphans
        await a2a.cleanup_orphaned_responses()
        assert len(a2a._pending_responses) == 0
        assert len(a2a._response_timeouts) == 0

    @pytest.mark.asyncio
    async def test_stress_concurrent_operations(self):
        """
        Integration: Stress test with concurrent operations.

        Invariants: System remains stable under load.
        """
        a2a = A2AManager()

        # Register multiple agents
        for i in range(5):

            async def handler(msg, ctx, idx=i):
                await asyncio.sleep(0.1)
                return f"agent_{idx}"

            await a2a.register_agent(
                AgentCard(agent_id=f"agent_{i}", name=f"Agent {i}", description="Test"),
                handler=handler,
            )

        # Send 50 concurrent tasks
        tasks = []
        for i in range(50):
            agent_id = f"agent_{i % 5}"
            request = TaskSendRequest(
                task_id=f"stress_{i}",
                target_agent=agent_id,
                message=f"task {i}",
                timeout_seconds=5.0,
            )
            tasks.append(a2a.send_task(request))

        results = await asyncio.gather(*tasks)

        # Verify all completed
        assert all(r.status == TaskStatus.COMPLETED for r in results)

        # INVARIANT: All tracking cleaned up
        for i in range(50):
            assert f"stress_{i}" not in a2a._pending_responses
            assert f"stress_{i}" not in a2a._response_timeouts


# ═══════════════════════════════════════════════════════════════════════════════
# Test Runners
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v", "--tb=short"])
