"""
Performance Tests for Multi-LLM Orchestrator
============================================
Benchmarks and load tests for performance-critical components.

Run with:
    pytest tests/test_performance.py -v
    pytest tests/test_performance.py --benchmark-only
"""

import asyncio
import time
from typing import List

import pytest

from orchestrator.performance import (
    LRUCache,
    RedisCache,
    MetricsCollector,
    QueryOptimizer,
    ConnectionPool,
    cached,
)
from orchestrator.monitoring import (
    MetricsRegistry,
    KPIReporter,
    HealthChecker,
    monitor_endpoint,
    STANDARD_KPIS,
)

# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
async def lru_cache():
    """Create LRU cache fixture."""
    cache = LRUCache(max_size=1000, default_ttl=60)
    yield cache


@pytest.fixture
async def metrics_registry():
    """Create metrics registry fixture."""
    return MetricsRegistry()


@pytest.fixture
async def kpi_reporter():
    """Create KPI reporter fixture."""
    return KPIReporter()


# ═══════════════════════════════════════════════════════════════════════════════
# CACHE BENCHMARKS
# ═══════════════════════════════════════════════════════════════════════════════


class TestCachePerformance:
    """Benchmark cache operations."""

    @pytest.mark.asyncio
    async def test_lru_cache_hit_performance(self, lru_cache):
        """Test cache hit latency should be <1ms."""
        # Warm up cache
        await lru_cache.set("key", "value")

        # Measure hit latency
        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            await lru_cache.get("key")
            latencies.append((time.perf_counter() - start) * 1000)

        avg_latency = sum(latencies) / len(latencies)
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]

        print(f"\nLRU Cache Hit Latency: avg={avg_latency:.3f}ms, p95={p95:.3f}ms")

        assert avg_latency < 1.0, f"Cache hit too slow: {avg_latency:.3f}ms"
        assert p95 < 2.0, f"Cache hit P95 too slow: {p95:.3f}ms"

    @pytest.mark.asyncio
    async def test_lru_cache_throughput(self, lru_cache):
        """Test cache operations per second."""
        operations = 10000

        async def worker(cache, start_idx, count):
            for i in range(count):
                await cache.set(f"key_{start_idx}_{i}", f"value_{i}")
                await cache.get(f"key_{start_idx}_{i}")

        start = time.perf_counter()

        # Run concurrent workers
        workers = [worker(lru_cache, i, operations // 10) for i in range(10)]
        await asyncio.gather(*workers)

        elapsed = time.perf_counter() - start
        ops_per_sec = (operations * 2) / elapsed  # *2 for set+get

        print(f"\nLRU Cache Throughput: {ops_per_sec:,.0f} ops/sec")

        assert ops_per_sec > 10000, f"Cache throughput too low: {ops_per_sec:,.0f}"

    @pytest.mark.asyncio
    async def test_cache_memory_usage(self, lru_cache):
        """Test cache memory remains bounded."""
        max_size = 1000

        # Fill beyond capacity
        for i in range(max_size * 2):
            await lru_cache.set(f"key_{i}", {"data": "x" * 1000})

        stats = lru_cache.get_stats()

        print(f"\nCache Memory Stats: {stats}")

        assert stats["size"] <= max_size, "Cache exceeded max size"
        assert stats["evictions"] >= max_size, "Cache should have evicted entries"

    @pytest.mark.asyncio
    async def test_cache_hit_rate(self, lru_cache):
        """Test cache hit rate with realistic access pattern."""
        # Fill cache
        for i in range(100):
            await lru_cache.set(f"key_{i}", f"value_{i}")

        # 80/20 access pattern (Pareto)
        for _ in range(1000):
            # 80% of accesses hit 20% of keys
            if hash(time.time()) % 100 < 80:
                key_idx = hash(time.time()) % 20
            else:
                key_idx = 20 + (hash(time.time()) % 80)

            await lru_cache.get(f"key_{key_idx}")

        stats = lru_cache.get_stats()
        hit_rate = float(stats["hit_rate"].rstrip("%")) / 100

        print(f"\nCache Hit Rate: {stats['hit_rate']}")

        assert hit_rate > 0.5, f"Hit rate too low: {stats['hit_rate']}"


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS BENCHMARKS
# ═══════════════════════════════════════════════════════════════════════════════


class TestMetricsPerformance:
    """Benchmark metrics collection."""

    @pytest.mark.asyncio
    async def test_metrics_collection_latency(self, metrics_registry):
        """Test metrics recording latency."""
        latencies = []

        for _ in range(1000):
            start = time.perf_counter()
            await metrics_registry.record("test_metric", 42.0)
            latencies.append((time.perf_counter() - start) * 1000)

        avg_latency = sum(latencies) / len(latencies)
        p99 = sorted(latencies)[int(len(latencies) * 0.99)]

        print(f"\nMetrics Recording Latency: avg={avg_latency:.3f}ms, p99={p99:.3f}ms")

        assert avg_latency < 0.1, f"Metrics recording too slow: {avg_latency:.3f}ms"

    @pytest.mark.asyncio
    async def test_metrics_query_performance(self, metrics_registry):
        """Test metrics query performance."""
        # Populate metrics
        for i in range(1000):
            await metrics_registry.record("test_metric", i)

        # Query multiple times
        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            await metrics_registry.get_metric("test_metric")
            latencies.append((time.perf_counter() - start) * 1000)

        avg_latency = sum(latencies) / len(latencies)

        print(f"\nMetrics Query Latency: avg={avg_latency:.3f}ms")

        assert avg_latency < 10, f"Metrics query too slow: {avg_latency:.3f}ms"

    @pytest.mark.asyncio
    async def test_metrics_memory_efficiency(self, metrics_registry):
        """Test metrics don't grow unbounded."""
        max_history = 1000

        # Add many samples
        for i in range(10000):
            await metrics_registry.record("test_metric", i)

        metric_data = await metrics_registry.get_metric("test_metric")
        window = metric_data.get("window", {})

        print(f"\nMetrics Memory: {window.get('count', 0)} samples stored")

        assert window.get("count", 0) <= max_history, "Metrics exceeded max history"


# ═══════════════════════════════════════════════════════════════════════════════
# KPI BENCHMARKS
# ═══════════════════════════════════════════════════════════════════════════════


class TestKPIPerformance:
    """Benchmark KPI evaluation."""

    @pytest.mark.asyncio
    async def test_kpi_evaluation_latency(self, kpi_reporter):
        """Test KPI evaluation latency."""
        latencies = []

        for _ in range(100):
            start = time.perf_counter()
            await kpi_reporter.evaluate("response_time_p50", 150)
            latencies.append((time.perf_counter() - start) * 1000)

        avg_latency = sum(latencies) / len(latencies)
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]

        print(f"\nKPI Evaluation Latency: avg={avg_latency:.3f}ms, p95={p95:.3f}ms")

        assert avg_latency < 1, f"KPI evaluation too slow: {avg_latency:.3f}ms"

    @pytest.mark.asyncio
    async def test_health_score_calculation(self, kpi_reporter):
        """Test health score calculation performance."""
        # Populate metrics
        from orchestrator.monitoring import metrics

        for _ in range(100):
            await metrics.record("response_time_p50", 100)
            await metrics.record("error_rate", 0.001)

        latencies = []
        for _ in range(50):
            start = time.perf_counter()
            await kpi_reporter.get_health_score()
            latencies.append((time.perf_counter() - start) * 1000)

        avg_latency = sum(latencies) / len(latencies)

        print(f"\nHealth Score Latency: avg={avg_latency:.3f}ms")

        assert avg_latency < 50, f"Health score calculation too slow: {avg_latency:.3f}ms"


# ═══════════════════════════════════════════════════════════════════════════════
# DECORATOR BENCHMARKS
# ═══════════════════════════════════════════════════════════════════════════════


class TestDecoratorPerformance:
    """Benchmark caching decorators."""

    @pytest.mark.asyncio
    async def test_cached_decorator_overhead(self):
        """Test decorator overhead is minimal."""
        call_count = 0

        @cached(ttl=60)
        async def expensive_function(x: int) -> int:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.001)  # Simulate work
            return x * 2

        # First call - cache miss
        start = time.perf_counter()
        await expensive_function(5)
        miss_time = (time.perf_counter() - start) * 1000

        # Second call - cache hit
        start = time.perf_counter()
        await expensive_function(5)
        hit_time = (time.perf_counter() - start) * 1000

        print(f"\nCached Decorator: miss={miss_time:.3f}ms, hit={hit_time:.3f}ms")

        assert hit_time < miss_time / 10, "Cache hit not significantly faster"
        assert call_count == 1, "Function called more than once"


# ═══════════════════════════════════════════════════════════════════════════════
# CONNECTION POOL BENCHMARKS
# ═══════════════════════════════════════════════════════════════════════════════


class TestConnectionPoolPerformance:
    """Benchmark connection pool."""

    @pytest.mark.asyncio
    async def test_connection_pool_latency(self):
        """Test connection acquisition latency."""
        connection_count = 0

        async def create_connection():
            nonlocal connection_count
            connection_count += 1
            await asyncio.sleep(0.001)
            return {"id": connection_count}

        pool = ConnectionPool(
            create_connection,
            min_size=2,
            max_size=5,
        )

        latencies = []

        async with pool.acquire() as conn:
            pass

        for _ in range(50):
            start = time.perf_counter()
            async with pool.acquire() as conn:
                pass
            latencies.append((time.perf_counter() - start) * 1000)

        avg_latency = sum(latencies) / len(latencies)
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]

        print(f"\nConnection Pool Latency: avg={avg_latency:.3f}ms, p95={p95:.3f}ms")

        assert avg_latency < 5, f"Connection acquisition too slow: {avg_latency:.3f}ms"

        await pool.close()

    @pytest.mark.asyncio
    async def test_connection_pool_contention(self):
        """Test pool under concurrent load."""

        async def create_connection():
            await asyncio.sleep(0.001)
            return {"connection": True}

        pool = ConnectionPool(
            create_connection,
            min_size=2,
            max_size=5,
        )

        async def worker(worker_id: int):
            for _ in range(20):
                async with pool.acquire() as conn:
                    await asyncio.sleep(0.01)

        start = time.perf_counter()
        workers = [worker(i) for i in range(10)]
        await asyncio.gather(*workers)
        elapsed = time.perf_counter() - start

        stats = pool.get_stats()

        print(f"\nConnection Pool Contention: {elapsed:.2f}s, {stats}")

        assert stats["wait_timeouts"] == 0, "Connections timed out"

        await pool.close()


# ═══════════════════════════════════════════════════════════════════════════════
# END-TO-END BENCHMARKS
# ═══════════════════════════════════════════════════════════════════════════════


class TestEndToEndPerformance:
    """End-to-end performance tests."""

    @pytest.mark.asyncio
    async def test_dashboard_response_time(self):
        """Simulate dashboard API response times."""
        from orchestrator.performance import get_cache

        cache = get_cache()
        await cache.initialize()

        # Simulate API endpoint
        @cached(ttl=300)
        async def get_models():
            await asyncio.sleep(0.05)  # Simulate DB query
            return {"models": ["gpt-4", "claude"]}

        # Warm up
        await get_models()

        # Test cached response
        latencies = []
        for _ in range(100):
            start = time.perf_counter()
            await get_models()
            latencies.append((time.perf_counter() - start) * 1000)

        avg_latency = sum(latencies) / len(latencies)
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]
        p99 = sorted(latencies)[int(len(latencies) * 0.99)]

        print(
            f"\nDashboard API Response Time: avg={avg_latency:.3f}ms, p95={p95:.3f}ms, p99={p99:.3f}ms"
        )

        assert avg_latency < 10, f"API response too slow: {avg_latency:.3f}ms"
        assert p95 < 20, f"API P95 too slow: {p95:.3f}ms"

        await cache.close()

    @pytest.mark.asyncio
    async def test_full_system_throughput(self):
        """Test full system under load."""
        from orchestrator.performance import get_cache
        from orchestrator.monitoring import metrics

        cache = get_cache()
        await cache.initialize()

        async def operation():
            # Simulate cached operation
            key = f"key_{hash(time.time()) % 100}"
            cached_val = await cache.get(key)
            if not cached_val:
                await cache.set(key, {"data": "value"}, ttl=60)

            # Record metric
            await metrics.record("test_op", 42.0)

        # Run load test
        concurrent_requests = 50
        requests_per_worker = 20

        async def worker():
            for _ in range(requests_per_worker):
                await operation()

        start = time.perf_counter()
        await asyncio.gather(*[worker() for _ in range(concurrent_requests)])
        elapsed = time.perf_counter() - start

        total_requests = concurrent_requests * requests_per_worker
        rps = total_requests / elapsed

        print(
            f"\nFull System Throughput: {rps:,.0f} requests/sec ({total_requests} in {elapsed:.2f}s)"
        )

        assert rps > 500, f"Throughput too low: {rps:,.0f} RPS"

        await cache.close()


# ═══════════════════════════════════════════════════════════════════════════════
# PERFORMANCE TARGETS
# ═══════════════════════════════════════════════════════════════════════════════


class TestPerformanceTargets:
    """Verify performance meets targets."""

    TARGETS = {
        "ttfb_ms": 50,
        "response_time_p50_ms": 100,
        "response_time_p95_ms": 300,
        "cache_hit_latency_ms": 1,
        "cache_hit_rate": 0.85,
        "throughput_rps": 1000,
    }

    @pytest.mark.asyncio
    async def test_all_targets(self):
        """Verify all performance targets are met."""
        results = []

        # Test cache performance
        cache = LRUCache(max_size=1000)

        # Warm up
        for i in range(100):
            await cache.set(f"key_{i}", f"value_{i}")

        # Measure hit latency
        latencies = []
        for _ in range(1000):
            start = time.perf_counter()
            await cache.get("key_50")
            latencies.append((time.perf_counter() - start) * 1000)

        hit_latency = sum(latencies) / len(latencies)
        results.append(("Cache Hit Latency", hit_latency, self.TARGETS["cache_hit_latency_ms"]))

        # Measure hit rate
        hits = 0
        for i in range(1000):
            key = f"key_{i % 100}" if i < 800 else f"key_{i + 100}"
            if await cache.get(key):
                hits += 1

        hit_rate = hits / 1000
        results.append(("Cache Hit Rate %", hit_rate * 100, self.TARGETS["cache_hit_rate"] * 100))

        # Print results
        print("\n" + "=" * 60)
        print("PERFORMANCE TARGETS VALIDATION")
        print("=" * 60)

        all_passed = True
        for name, actual, target in results:
            status = "✓ PASS" if actual <= target else "✗ FAIL"
            if "Rate" in name:
                status = "✓ PASS" if actual >= target else "✗ FAIL"

            print(f"{status} {name}: {actual:.2f} (target: {target:.2f})")

            if "Rate" in name:
                if actual < target:
                    all_passed = False
            else:
                if actual > target:
                    all_passed = False

        print("=" * 60)

        assert all_passed, "Some performance targets not met"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
