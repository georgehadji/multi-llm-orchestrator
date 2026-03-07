"""
Performance Benchmarks for AI Orchestrator
==========================================

Run with: pytest tests/benchmarks/ --benchmark-only

Benchmarks cover:
- Task execution latency
- Cache hit/miss performance
- Memory tier operations
- Event bus throughput
- API client performance
"""

import asyncio
import time
from pathlib import Path
import pytest

from orchestrator import Orchestrator, Budget
from orchestrator.cache_optimizer import CacheOptimizer, CacheConfig
from orchestrator.memory_tier import MemoryTierManager
from orchestrator.unified_events import UnifiedEventBus, get_event_bus


# ═══════════════════════════════════════════════════════════════════════════════
# Task Execution Benchmarks
# ═══════════════════════════════════════════════════════════════════════════════

class TestTaskExecutionBenchmarks:
    """Benchmark task execution performance."""
    
    @pytest.mark.benchmark(group="task_execution")
    def test_task_decomposition_latency(self, benchmark):
        """Benchmark task decomposition speed."""
        async def run_decompose():
            orch = Orchestrator(budget=Budget(max_usd=1.0))
            # Mock decomposition (don't actually call LLM)
            return await orch._decompose("Build a simple API", max_tasks=5)
        
        result = benchmark(asyncio.run, run_decompose())
        assert result is not None
    
    @pytest.mark.benchmark(group="task_execution")
    def test_topological_sort_performance(self, benchmark):
        """Benchmark topological sort with 100 tasks."""
        orch = Orchestrator(budget=Budget(max_usd=1.0))
        
        # Create 100 tasks with dependencies
        tasks = {
            f"task_{i}": type('Task', (), {
                'id': f"task_{i}",
                'dependencies': [f"task_{j}" for j in range(max(0, i-3), i)]
            })()
            for i in range(100)
        }
        
        result = benchmark(orch._topological_sort, tasks)
        assert len(result) > 0
    
    @pytest.mark.benchmark(group="task_execution")
    def test_result_lock_contention(self, benchmark):
        """Benchmark lock contention for concurrent result writes."""
        async def concurrent_writes():
            orch = Orchestrator(budget=Budget(max_usd=1.0))
            orch.results = {}
            orch._results_lock = asyncio.Lock()
            
            async def write_result(i):
                async with orch._results_lock:
                    orch.results[f"task_{i}"] = f"result_{i}"
            
            await asyncio.gather(*[write_result(i) for i in range(50)])
        
        benchmark(asyncio.run, concurrent_writes())


# ═══════════════════════════════════════════════════════════════════════════════
# Cache Performance Benchmarks
# ═══════════════════════════════════════════════════════════════════════════════

class TestCacheBenchmarks:
    """Benchmark cache performance."""
    
    @pytest.fixture
    def cache_optimizer(self):
        return CacheOptimizer(CacheConfig(
            l1_max_size=1000,
            l1_ttl_seconds=3600,
            l2_ttl_hours=48,
            l3_quality_threshold=0.85,
        ))
    
    @pytest.mark.benchmark(group="cache")
    def test_l1_cache_write(self, benchmark, cache_optimizer):
        """Benchmark L1 cache write performance."""
        def write():
            cache_optimizer.l1_cache["test_key"] = {"value": "test", "timestamp": time.time()}
        
        benchmark(write)
    
    @pytest.mark.benchmark(group="cache")
    def test_l1_cache_read(self, benchmark, cache_optimizer):
        """Benchmark L1 cache read performance."""
        # Pre-populate
        cache_optimizer.l1_cache["test_key"] = {"value": "test", "timestamp": time.time()}
        
        def read():
            return cache_optimizer.l1_cache.get("test_key")
        
        result = benchmark(read)
        assert result is not None
    
    @pytest.mark.benchmark(group="cache")
    def test_l1_cache_lookup_miss(self, benchmark, cache_optimizer):
        """Benchmark L1 cache miss performance."""
        def lookup_miss():
            return cache_optimizer.l1_cache.get("nonexistent_key")
        
        result = benchmark(lookup_miss)
        assert result is None
    
    @pytest.mark.benchmark(group="cache")
    def test_cache_key_generation(self, benchmark, cache_optimizer):
        """Benchmark cache key generation."""
        prompt = "Build a FastAPI API with authentication"
        model = "gpt-4o-mini"
        temperature = 0.7
        
        def generate_key():
            return cache_optimizer.generate_key(prompt, model, temperature)
        
        result = benchmark(generate_key)
        assert isinstance(result, str)
        assert len(result) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Memory Tier Benchmarks
# ═══════════════════════════════════════════════════════════════════════════════

class TestMemoryTierBenchmarks:
    """Benchmark memory tier operations."""
    
    @pytest.fixture
    def memory_manager(self, tmp_path):
        return MemoryTierManager(storage_path=tmp_path / "memory")
    
    @pytest.mark.benchmark(group="memory")
    def test_memory_store_latency(self, benchmark, memory_manager):
        """Benchmark memory store operation."""
        async def store():
            await memory_manager.store(
                project_id="benchmark_proj",
                content="Test memory content for benchmarking",
                memory_type="task"
            )
        
        benchmark(asyncio.run, store())
    
    @pytest.mark.benchmark(group="memory")
    def test_memory_retrieve_latency(self, benchmark, memory_manager):
        """Benchmark memory retrieve operation."""
        # Pre-populate
        async def setup():
            for i in range(10):
                await memory_manager.store(
                    project_id="benchmark_proj",
                    content=f"Test memory {i}",
                    memory_type="task"
                )
        
        asyncio.run(setup())
        
        async def retrieve():
            return await memory_manager.retrieve(
                project_id="benchmark_proj",
                query="test",
                limit=5
            )
        
        result = benchmark(asyncio.run, retrieve())
        assert len(result) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Event Bus Benchmarks
# ═══════════════════════════════════════════════════════════════════════════════

class TestEventBusBenchmarks:
    """Benchmark event bus throughput."""
    
    @pytest.fixture
    def event_bus(self):
        return get_event_bus()
    
    @pytest.mark.benchmark(group="events")
    def test_event_publish_latency(self, benchmark, event_bus):
        """Benchmark single event publish."""
        from orchestrator.unified_events import TaskCompletedEvent
        
        event = TaskCompletedEvent(
            task_id="benchmark_task",
            status="completed"
        )
        
        async def publish():
            await event_bus.publish(event)
        
        benchmark(asyncio.run, publish())
    
    @pytest.mark.benchmark(group="events")
    def test_event_throughput_100(self, benchmark, event_bus):
        """Benchmark 100 events per second."""
        from orchestrator.unified_events import TaskCompletedEvent
        
        async def publish_many():
            events = [
                TaskCompletedEvent(
                    task_id=f"benchmark_task_{i}",
                    status="completed"
                )
                for i in range(100)
            ]
            await asyncio.gather(*[event_bus.publish(e) for e in events])
        
        benchmark(asyncio.run, publish_many())


# ═══════════════════════════════════════════════════════════════════════════════
# API Client Benchmarks (Mock)
# ═══════════════════════════════════════════════════════════════════════════════

class TestAPIClientBenchmarks:
    """Benchmark API client performance (mock)."""
    
    @pytest.mark.benchmark(group="api")
    def test_request_serialization(self, benchmark):
        """Benchmark request serialization."""
        from orchestrator.api_clients import UnifiedClient
        
        client = UnifiedClient()
        messages = [
            {"role": "user", "content": "Build an API" * 100}
        ]
        
        def serialize():
            return client._serialize_request("gpt-4o-mini", messages, temperature=0.7)
        
        result = benchmark(serialize)
        assert result is not None
    
    @pytest.mark.benchmark(group="api")
    def test_response_parsing(self, benchmark):
        """Benchmark response parsing."""
        from orchestrator.api_clients import APIResponse
        
        mock_response = {
            "choices": [{
                "message": {"content": "Test output" * 100}
            }],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 200,
                "total_tokens": 300
            }
        }
        
        def parse():
            return APIResponse.from_openai_response(mock_response, "gpt-4o-mini")
        
        result = benchmark(parse)
        assert result is not None


# ═══════════════════════════════════════════════════════════════════════════════
# State Manager Benchmarks
# ═══════════════════════════════════════════════════════════════════════════════

class TestStateManagerBenchmarks:
    """Benchmark state manager operations."""
    
    @pytest.mark.benchmark(group="state")
    def test_checkpoint_write(self, benchmark, tmp_path):
        """Benchmark checkpoint write performance."""
        from orchestrator.state import StateManager
        
        async def write_checkpoint():
            state_mgr = StateManager(cache_dir=str(tmp_path))
            await state_mgr.save_checkpoint(
                project_id="benchmark_proj",
                task_id="task_001",
                state={"tasks": {"task_001": {"status": "completed"}}}
            )
            await state_mgr.close()
        
        benchmark(asyncio.run, write_checkpoint())
    
    @pytest.mark.benchmark(group="state")
    def test_checkpoint_read(self, benchmark, tmp_path):
        """Benchmark checkpoint read performance."""
        from orchestrator.state import StateManager
        
        # Setup
        async def setup():
            state_mgr = StateManager(cache_dir=str(tmp_path))
            await state_mgr.save_checkpoint(
                project_id="benchmark_proj",
                task_id="task_001",
                state={"tasks": {"task_001": {"status": "completed"}}}
            )
            await state_mgr.close()
        
        asyncio.run(setup())
        
        # Benchmark
        async def read_checkpoint():
            state_mgr = StateManager(cache_dir=str(tmp_path))
            result = await state_mgr.load_project("benchmark_proj")
            await state_mgr.close()
            return result
        
        result = benchmark(asyncio.run, read_checkpoint())
        assert result is not None
