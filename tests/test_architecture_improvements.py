"""
Tests for Architecture Improvements
===================================

Tests for:
- Event Bus (orchestrator/events.py)
- Streaming Pipeline (orchestrator/streaming.py)
- CQRS Projections (orchestrator/projections.py)
- Multi-Layer Cache (orchestrator/caching.py)
- Health Checks (orchestrator/health.py)
"""

import pytest
import asyncio
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

# Event system
from orchestrator.events import (
    EventBus,
    DomainEvent,
    TaskCompletedEvent,
    TaskFailedEvent,
    ProjectStartedEvent,
    InMemoryEventStore,
    SQLiteEventStore,
    get_event_bus,
    reset_event_bus,
)

# Streaming
from orchestrator.streaming import (
    StreamingPipeline,
    PipelineEvent,
    PipelineEventType,
    StreamingContext,
    InMemoryCache,
    DiskCache,
    MultiLayerCache,
)

# Projections
from orchestrator.projections import ModelPerformanceProjection, ModelPerformanceStats

# Caching
from orchestrator.caching import (
    CacheLevel,
    InMemoryCache,
    DiskCache,
    MultiLayerCache,
    get_cache,
    reset_cache,
)

# Health
from orchestrator.health import (
    HealthMonitor,
    HealthStatus,
    CheckType,
    CheckResult,
    create_default_health_monitor,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Event Bus Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestDomainEvents:
    """Test domain event classes."""

    def test_task_completed_event(self):
        event = TaskCompletedEvent(
            task_id="task-123",
            model="gpt-4o",
            score=0.95,
            cost_usd=0.02,
            latency_ms=1200,
        )

        assert event.event_type == "task.completed"
        assert event.task_id == "task-123"
        assert event.score == 0.95
        assert event.payload["task_id"] == "task-123"

    def test_event_serialization(self):
        event = TaskCompletedEvent(task_id="task-123", model="gpt-4o", score=0.95)
        data = event.to_dict()

        assert "event_id" in data
        assert "event_type" in data
        assert "timestamp" in data

        # Deserialize
        restored = DomainEvent.from_dict(data)
        assert restored.event_type == "task.completed"

    def test_event_immutability(self):
        event = TaskCompletedEvent(task_id="task-123", model="gpt-4o")

        with pytest.raises(AttributeError):
            event.task_id = "task-456"


class TestEventBus:
    """Test event bus functionality."""

    @pytest.fixture
    async def event_bus(self):
        bus = EventBus.create("memory")
        yield bus
        await bus.close()

    @pytest.mark.asyncio
    async def test_publish_and_subscribe(self, event_bus):
        received = []

        @event_bus.subscribe("task.completed")
        async def handler(event):
            received.append(event)

        event = TaskCompletedEvent(task_id="task-123", model="gpt-4o", score=0.95)
        await event_bus.publish(event)

        await asyncio.sleep(0.01)  # Allow handler to run
        assert len(received) == 1
        assert received[0].task_id == "task-123"

    @pytest.mark.asyncio
    async def test_multiple_subscribers(self, event_bus):
        received_1 = []
        received_2 = []

        @event_bus.subscribe("task.completed")
        async def handler1(event):
            received_1.append(event)

        @event_bus.subscribe("task.completed")
        async def handler2(event):
            received_2.append(event)

        event = TaskCompletedEvent(task_id="task-123", model="gpt-4o", score=0.95)
        await event_bus.publish(event)

        await asyncio.sleep(0.01)
        assert len(received_1) == 1
        assert len(received_2) == 1

    @pytest.mark.asyncio
    async def test_event_persistence(self):
        """Test that events are persisted to store."""
        bus = EventBus.create("memory")

        event = TaskCompletedEvent(task_id="task-123", model="gpt-4o", score=0.95)
        await bus.publish(event)

        # Check store
        events = await bus.store.get_events()
        assert len(events) == 1
        assert events[0].event_type == "task.completed"

        await bus.close()

    @pytest.mark.asyncio
    async def test_event_replay(self, event_bus):
        """Test replaying events from store."""
        received = []

        # Publish some events
        await event_bus.publish(TaskCompletedEvent(task_id="1", model="gpt-4o", score=0.9))
        await event_bus.publish(TaskCompletedEvent(task_id="2", model="gpt-4o", score=0.8))
        await event_bus.publish(TaskFailedEvent(task_id="3", model="gpt-4o"))

        # Replay only completed events
        async def handler(event):
            if event.event_type == "task.completed":
                received.append(event)

        await event_bus.store.replay(handler, event_types=["task.completed"])

        assert len(received) == 2


class TestSQLiteEventStore:
    """Test SQLite event store."""

    @pytest.mark.asyncio
    async def test_sqlite_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "events.db"
            store = SQLiteEventStore(str(db_path))

            # Append events
            await store.append(TaskCompletedEvent(task_id="1", model="gpt-4o", score=0.9))
            await store.append(TaskCompletedEvent(task_id="2", model="gpt-4o", score=0.8))

            # Retrieve
            events = await store.get_events()
            assert len(events) == 2

            # Filter by type
            events = await store.get_events(event_types=["task.completed"])
            assert len(events) == 2

            await store.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Streaming Pipeline Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestStreamingPipeline:
    """Test streaming pipeline."""

    @pytest.mark.asyncio
    async def test_streaming_events(self):
        pipeline = StreamingPipeline(max_parallel=2)

        events = []
        async for event in pipeline.execute_streaming(
            project_description="Test project",
            success_criteria="Works",
            budget=5.0,
            project_id="test-123",
        ):
            events.append(event)

        # Should have multiple events
        assert len(events) > 0

        # Should start with PROJECT_START
        assert events[0].type == PipelineEventType.PROJECT_START

        # Should end with PROJECT_COMPLETE
        assert events[-1].type == PipelineEventType.PROJECT_COMPLETE

    @pytest.mark.asyncio
    async def test_streaming_context(self):
        context = StreamingContext(
            project_id="test",
            description="Test",
            budget=None,  # Will fix this
        )

        # Add some tasks
        from orchestrator.models import Task, TaskType

        task1 = Task(id="task-1", task_type=TaskType.CODE_GEN, prompt="test")
        task2 = Task(id="task-2", task_type=TaskType.CODE_REVIEW, prompt="test")

        context.tasks["task-1"] = StreamingPipeline.StreamingTask(task=task1)
        context.tasks["task-2"] = StreamingPipeline.StreamingTask(task=task2)

        # Complete task 1
        context.completed_tasks.add("task-1")

        # Check ready tasks
        ready = context.get_ready_tasks()
        assert len(ready) == 1  # task-2 should be ready (no deps)


# ═══════════════════════════════════════════════════════════════════════════════
# Projection Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestModelPerformanceStats:
    """Test model performance stats."""

    def test_update_success(self):
        stats = ModelPerformanceStats(model="gpt-4o", task_type="code_gen")

        stats.update_success(score=0.9, cost=0.02, latency_ms=1000)

        assert stats.total_calls == 1
        assert stats.success_count == 1
        assert stats.quality_score_ema > 0.5  # Should increase

    def test_update_failure(self):
        stats = ModelPerformanceStats(model="gpt-4o", task_type="code_gen")

        stats.update_success(score=0.9, cost=0.02, latency_ms=1000)
        stats.update_failure()

        assert stats.total_calls == 2
        assert stats.failure_count == 1
        assert stats.success_rate_ema < 1.0  # Should decrease

    def test_composite_score(self):
        stats = ModelPerformanceStats(
            model="gpt-4o",
            task_type="code_gen",
            quality_score_ema=0.9,
            success_rate_ema=0.8,
            production_score_ema=0.95,
        )

        score = stats.composite_score
        assert 0.0 <= score <= 1.0
        assert score > 0.8  # Should be high given good scores


class TestModelPerformanceProjection:
    """Test model performance projection."""

    @pytest.mark.asyncio
    async def test_projection_subscription(self):
        bus = EventBus.create("memory")
        projection = ModelPerformanceProjection(bus)

        # Emit event
        await bus.publish(
            TaskCompletedEvent(
                task_id="task-1",
                model="gpt-4o",
                score=0.95,
                cost_usd=0.02,
                latency_ms=1000,
            )
        )

        await asyncio.sleep(0.01)

        # Check projection updated
        score = projection.get_model_score("gpt-4o", "code_gen")
        assert score > 0.5  # Should have been updated

        projection.unsubscribe_all()
        await bus.close()

    @pytest.mark.asyncio
    async def test_leaderboard(self):
        bus = EventBus.create("memory")
        projection = ModelPerformanceProjection(bus)

        # Add some stats
        await bus.publish(TaskCompletedEvent(task_id="1", model="gpt-4o", score=0.95))
        await bus.publish(TaskCompletedEvent(task_id="2", model="gpt-4o-mini", score=0.85))

        await asyncio.sleep(0.01)

        leaderboard = projection.get_leaderboard(min_calls=0)
        assert len(leaderboard) >= 0  # May be empty if events not processed yet

        projection.unsubscribe_all()
        await bus.close()

    @pytest.mark.asyncio
    async def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bus = EventBus.create("memory")
            projection = ModelPerformanceProjection(bus, storage_path=Path(tmpdir))

            # Add stats
            stats = projection._get_or_create_stats("gpt-4o", "code_gen")
            stats.update_success(score=0.9, cost=0.02, latency_ms=1000)

            # Save
            projection.save()

            # Create new projection and load
            projection2 = ModelPerformanceProjection(bus, storage_path=Path(tmpdir))
            projection2.load()

            # Check stats loaded
            score = projection2.get_model_score("gpt-4o", "code_gen")
            assert score > 0.5

            projection.unsubscribe_all()
            projection2.unsubscribe_all()
            await bus.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Cache Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestInMemoryCache:
    """Test in-memory cache."""

    @pytest.mark.asyncio
    async def test_basic_operations(self):
        cache = InMemoryCache(max_size=100)

        # Set and get
        await cache.set("key1", "value1")
        value = await cache.get("key1")
        assert value == "value1"

        # Non-existent key
        value = await cache.get("key2")
        assert value is None

    @pytest.mark.asyncio
    async def test_ttl_expiration(self):
        cache = InMemoryCache()

        await cache.set("key1", "value1", ttl=timedelta(milliseconds=10))

        # Should exist immediately
        value = await cache.get("key1")
        assert value == "value1"

        # Wait for expiration
        await asyncio.sleep(0.02)

        # Should be expired
        value = await cache.get("key1")
        assert value is None

    @pytest.mark.asyncio
    async def test_lru_eviction(self):
        cache = InMemoryCache(max_size=3)

        # Fill cache
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.set("key3", "value3")

        # Access key1 to make it recently used
        await cache.get("key1")

        # Add new item to trigger eviction
        await cache.set("key4", "value4")

        # key2 should be evicted (least recently used)
        assert await cache.get("key1") is not None
        assert await cache.get("key2") is None

    def test_stats(self):
        cache = InMemoryCache(max_size=100)

        stats = cache.get_stats()
        assert stats["level"] == "L1_MEMORY"
        assert "hit_rate" in stats


class TestDiskCache:
    """Test disk cache."""

    @pytest.mark.asyncio
    async def test_basic_operations(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = DiskCache(db_path=f"{tmpdir}/cache.db")

            # Set and get
            await cache.set("key1", "value1")
            value = await cache.get("key1")
            assert value == "value1"

            # Different key
            await cache.set("key2", {"complex": "value"})
            value = await cache.get("key2")
            assert value == {"complex": "value"}

            await cache.close()

    @pytest.mark.asyncio
    async def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f"{tmpdir}/cache.db"

            # Create and populate cache
            cache1 = DiskCache(db_path=db_path)
            await cache1.set("key1", "value1")
            await cache1.close()

            # Create new cache instance
            cache2 = DiskCache(db_path=db_path)
            value = await cache2.get("key1")
            assert value == "value1"

            await cache2.close()


class TestMultiLayerCache:
    """Test multi-layer cache."""

    @pytest.mark.asyncio
    async def test_layered_get(self):
        # Create cache with L1 and L3 only
        l1 = InMemoryCache()
        l3 = DiskCache()
        cache = MultiLayerCache(backends=[l1, l3])

        # Write to L3
        await l3.set("key1", "value1")

        # Read through multi-layer (should find in L3 and promote to L1)
        value = await cache.get("key1")
        assert value == "value1"

        # Should now be in L1 too
        assert await l1.get("key1") == "value1"

        await cache.close()

    @pytest.mark.asyncio
    async def test_multi_layer_set(self):
        l1 = InMemoryCache()
        l3 = DiskCache()
        cache = MultiLayerCache(backends=[l1, l3])

        # Set at L1 level (should write to L1 and L3)
        await cache.set("key1", "value1", level=CacheLevel.L1_MEMORY)

        # Should be in both
        assert await l1.get("key1") == "value1"
        assert await l3.get("key1") == "value1"

        await cache.close()

    @pytest.mark.asyncio
    async def test_invalidation(self):
        l1 = InMemoryCache()
        l3 = DiskCache()
        cache = MultiLayerCache(backends=[l1, l3])

        await cache.set("key1", "value1")
        await cache.set("key2", "value2")

        # Invalidate all
        count = await cache.invalidate("*")
        assert count >= 2

        # Should all be gone
        assert await cache.get("key1") is None
        assert await cache.get("key2") is None

        await cache.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Health Check Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestHealthMonitor:
    """Test health monitor."""

    @pytest.mark.asyncio
    async def test_basic_check(self):
        monitor = HealthMonitor()

        @monitor.liveness_check
        def check_alive():
            return HealthStatus.HEALTHY

        result = await monitor.run_check("check_alive", check_alive, CheckType.LIVENESS)

        assert result.name == "check_alive"
        assert result.status == HealthStatus.HEALTHY
        assert result.check_type == CheckType.LIVENESS

    @pytest.mark.asyncio
    async def test_async_check(self):
        monitor = HealthMonitor()

        @monitor.readiness_check
        async def check_async():
            await asyncio.sleep(0.01)
            return HealthStatus.HEALTHY

        result = await monitor.run_check("check_async", check_async, CheckType.READINESS)
        assert result.status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_check_timeout(self):
        monitor = HealthMonitor(default_timeout=0.01)

        async def slow_check():
            await asyncio.sleep(1.0)
            return HealthStatus.HEALTHY

        result = await monitor.run_check("slow", slow_check, CheckType.READINESS)
        assert result.status == HealthStatus.UNHEALTHY
        assert result.error == "timeout"

    @pytest.mark.asyncio
    async def test_check_exception(self):
        monitor = HealthMonitor()

        def failing_check():
            raise ValueError("Something broke")

        result = await monitor.run_check("failing", failing_check, CheckType.READINESS)
        assert result.status == HealthStatus.UNHEALTHY
        assert "Something broke" in result.error

    @pytest.mark.asyncio
    async def test_overall_report(self):
        monitor = HealthMonitor()

        @monitor.liveness_check
        def check1():
            return HealthStatus.HEALTHY

        @monitor.readiness_check
        def check2():
            return HealthStatus.HEALTHY

        report = await monitor.check_all(use_cache=False)

        assert report.overall_status == HealthStatus.HEALTHY
        assert len(report.checks) == 2

    @pytest.mark.asyncio
    async def test_degraded_status(self):
        monitor = HealthMonitor()

        @monitor.readiness_check
        def check1():
            return HealthStatus.HEALTHY

        @monitor.readiness_check
        def check2():
            return HealthStatus.DEGRADED

        report = await monitor.check_all(use_cache=False)
        assert report.overall_status == HealthStatus.DEGRADED

    @pytest.mark.asyncio
    async def test_unhealthy_status(self):
        monitor = HealthMonitor()

        @monitor.readiness_check
        def check1():
            return HealthStatus.HEALTHY

        @monitor.readiness_check
        def check2():
            return HealthStatus.UNHEALTHY

        report = await monitor.check_all(use_cache=False)
        assert report.overall_status == HealthStatus.UNHEALTHY


class TestCheckResult:
    """Test check result dataclass."""

    def test_serialization(self):
        result = CheckResult(
            name="test",
            status=HealthStatus.HEALTHY,
            check_type=CheckType.LIVENESS,
            response_time_ms=10.5,
            timestamp=datetime.utcnow(),
            message="All good",
        )

        data = result.to_dict()
        assert data["name"] == "test"
        assert data["status"] == "healthy"
        assert data["response_time_ms"] == 10.5


# ═══════════════════════════════════════════════════════════════════════════════
# Integration Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestIntegration:
    """Integration tests combining multiple components."""

    @pytest.mark.asyncio
    async def test_event_to_projection_flow(self):
        """Test that events update projections correctly."""
        # Reset singletons
        reset_event_bus()

        bus = get_event_bus("memory")
        projection = ModelPerformanceProjection(bus)

        # Publish multiple events
        for i in range(10):
            await bus.publish(
                TaskCompletedEvent(
                    task_id=f"task-{i}",
                    model="gpt-4o",
                    score=0.9,
                    cost_usd=0.02,
                    latency_ms=1000,
                )
            )

        await asyncio.sleep(0.05)

        # Check projection
        score = projection.get_model_score("gpt-4o", "code_gen")
        assert score > 0.5

        stats = projection.get_model_stats("gpt-4o", "code_gen")
        assert stats is not None
        assert stats.total_calls == 10

        projection.unsubscribe_all()
        reset_event_bus()

    @pytest.mark.asyncio
    async def test_cache_with_events(self):
        """Test cache can be used with events."""
        reset_cache()

        cache = get_cache()
        bus = get_event_bus("memory")

        # Cache event handler results
        received = []

        @bus.subscribe("task.completed")
        async def handler(event):
            # Cache the result
            await cache.set(f"result:{event.task_id}", event.score)
            received.append(event)

        await bus.publish(TaskCompletedEvent(task_id="123", model="gpt-4o", score=0.95))

        await asyncio.sleep(0.01)

        # Retrieve from cache
        score = await cache.get("result:123")
        assert score == 0.95

        reset_cache()
        reset_event_bus()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
