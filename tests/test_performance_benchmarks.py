"""
Performance Baseline Benchmarks
================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Establishes performance baselines for key operations.
Run periodically to detect regressions.

Benchmarks:
- Budget operations (charge, reserve, commit)
- File I/O (async write, read, append)
- Code validation (AST parse, security scan)
- State persistence (save, load, checkpoint)
- Concurrent operations (10, 50, 100 tasks)

USAGE:
    pytest tests/test_performance_benchmarks.py -v -m benchmark
    pytest tests/test_performance_benchmarks.py -v --benchmark-only
"""

from __future__ import annotations

import asyncio
import logging
import statistics
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Callable, Awaitable

import pytest

from orchestrator.models import Budget, Task, TaskType, TaskResult, ProjectState
from orchestrator.code_validator import validate_code
from orchestrator.async_file_io import (
    async_write_text,
    async_read_text,
    async_write_json,
)
from orchestrator.state import StateManager

logger = logging.getLogger("orchestrator.benchmarks")


# ─────────────────────────────────────────────
# Benchmark Configuration
# ─────────────────────────────────────────────

@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    iterations: int = 100
    warmup_iterations: int = 10
    concurrency_levels: List[int] = field(default_factory=lambda: [10, 50, 100])
    sample_sizes: List[int] = field(default_factory=lambda: [100, 1000, 10000])


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    name: str
    iterations: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    std_dev_ms: float
    min_ms: float
    max_ms: float
    ops_per_second: float
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "iterations": self.iterations,
            "mean_ms": round(self.mean_ms, 3),
            "median_ms": round(self.median_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "p99_ms": round(self.p99_ms, 3),
            "std_dev_ms": round(self.std_dev_ms, 3),
            "min_ms": round(self.min_ms, 3),
            "max_ms": round(self.max_ms, 3),
            "ops_per_second": round(self.ops_per_second, 1),
        }


# ─────────────────────────────────────────────
# Benchmark Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def benchmark_config() -> BenchmarkConfig:
    """Benchmark configuration."""
    return BenchmarkConfig()


@pytest.fixture
def temp_dir() -> Path:
    """Temporary directory for benchmarks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ─────────────────────────────────────────────
# Benchmark Helpers
# ─────────────────────────────────────────────

async def run_benchmark(
    name: str,
    func: Callable[[], Awaitable[Any]],
    iterations: int,
    warmup: int = 10,
) -> BenchmarkResult:
    """
    Run a benchmark and collect statistics.
    
    Args:
        name: Benchmark name
        func: Async function to benchmark
        iterations: Number of iterations
        warmup: Warmup iterations (not counted)
    
    Returns:
        BenchmarkResult with statistics
    """
    # Warmup
    for _ in range(warmup):
        await func()
    
    # Actual benchmark
    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        await func()
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to ms
    
    # Calculate statistics
    latencies.sort()
    mean_ms = statistics.mean(latencies)
    median_ms = statistics.median(latencies)
    p95_ms = latencies[int(len(latencies) * 0.95)]
    p99_ms = latencies[int(len(latencies) * 0.99)]
    std_dev_ms = statistics.stdev(latencies) if len(latencies) > 1 else 0
    min_ms = min(latencies)
    max_ms = max(latencies)
    ops_per_second = 1000 / mean_ms if mean_ms > 0 else 0
    
    return BenchmarkResult(
        name=name,
        iterations=iterations,
        mean_ms=mean_ms,
        median_ms=median_ms,
        p95_ms=p95_ms,
        p99_ms=p99_ms,
        std_dev_ms=std_dev_ms,
        min_ms=min_ms,
        max_ms=max_ms,
        ops_per_second=ops_per_second,
    )


# ─────────────────────────────────────────────
# Budget Benchmarks
# ─────────────────────────────────────────────

class TestBudgetBenchmarks:
    """Budget operation benchmarks."""

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_budget_charge_latency(
        self,
        benchmark_config: BenchmarkConfig,
    ):
        """Benchmark budget charge operation."""
        budget = Budget(max_usd=1000.0)
        
        async def charge():
            await budget.charge(0.01, "generation")
        
        result = await run_benchmark(
            "budget_charge",
            charge,
            benchmark_config.iterations,
        )
        
        logger.info(f"Budget charge: {result.mean_ms:.3f}ms mean, {result.ops_per_second:.1f} ops/sec")
        
        # Assert performance baseline
        assert result.mean_ms < 1.0, f"Budget charge too slow: {result.mean_ms}ms"

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_budget_reserve_latency(
        self,
        benchmark_config: BenchmarkConfig,
    ):
        """Benchmark budget reserve operation."""
        budget = Budget(max_usd=1000.0)
        
        async def reserve():
            await budget.reserve(0.01)
            await budget.release_reservation(0.01)
        
        result = await run_benchmark(
            "budget_reserve_release",
            reserve,
            benchmark_config.iterations,
        )
        
        logger.info(f"Budget reserve: {result.mean_ms:.3f}ms mean")
        
        # Assert performance baseline
        assert result.mean_ms < 1.0, f"Budget reserve too slow: {result.mean_ms}ms"


# ─────────────────────────────────────────────
# File I/O Benchmarks
# ─────────────────────────────────────────────

class TestFileIOBenchmarks:
    """File I/O operation benchmarks."""

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_async_write_latency(
        self,
        temp_dir: Path,
        benchmark_config: BenchmarkConfig,
    ):
        """Benchmark async file write."""
        file_path = temp_dir / "benchmark.txt"
        content = "Hello, World!" * 100
        
        async def write():
            await async_write_text(file_path, content)
        
        result = await run_benchmark(
            "async_write_1kb",
            write,
            benchmark_config.iterations,
        )
        
        logger.info(f"Async write: {result.mean_ms:.3f}ms mean")
        
        # Assert performance baseline
        assert result.mean_ms < 10.0, f"Async write too slow: {result.mean_ms}ms"

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_async_read_latency(
        self,
        temp_dir: Path,
        benchmark_config: BenchmarkConfig,
    ):
        """Benchmark async file read."""
        file_path = temp_dir / "benchmark.txt"
        content = "Hello, World!" * 100
        await async_write_text(file_path, content)
        
        async def read():
            await async_read_text(file_path)
        
        result = await run_benchmark(
            "async_read_1kb",
            read,
            benchmark_config.iterations,
        )
        
        logger.info(f"Async read: {result.mean_ms:.3f}ms mean")
        
        # Assert performance baseline
        assert result.mean_ms < 5.0, f"Async read too slow: {result.mean_ms}ms"

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_async_json_write_latency(
        self,
        temp_dir: Path,
        benchmark_config: BenchmarkConfig,
    ):
        """Benchmark async JSON file write."""
        file_path = temp_dir / "benchmark.json"
        data = {"key": "value", "number": 42, "list": [1, 2, 3]}
        
        async def write():
            await async_write_json(file_path, data)
        
        result = await run_benchmark(
            "async_write_json",
            write,
            benchmark_config.iterations,
        )
        
        logger.info(f"Async JSON write: {result.mean_ms:.3f}ms mean")
        
        # Assert performance baseline
        assert result.mean_ms < 10.0, f"Async JSON write too slow: {result.mean_ms}ms"


# ─────────────────────────────────────────────
# Code Validation Benchmarks
# ─────────────────────────────────────────────

class TestCodeValidationBenchmarks:
    """Code validation benchmarks."""

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_validate_simple_code(
        self,
        benchmark_config: BenchmarkConfig,
    ):
        """Benchmark validation of simple code."""
        code = "def hello(): return 'world'"
        
        def validate():
            return validate_code(code)
        
        async def validate_async():
            return validate()
        
        result = await run_benchmark(
            "validate_simple_code",
            validate_async,
            benchmark_config.iterations,
        )
        
        logger.info(f"Validate simple code: {result.mean_ms:.3f}ms mean")
        
        # Assert performance baseline
        assert result.mean_ms < 1.0, f"Simple validation too slow: {result.mean_ms}ms"

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_validate_complex_code(
        self,
        benchmark_config: BenchmarkConfig,
    ):
        """Benchmark validation of complex code."""
        code = """
class ComplexClass:
    def __init__(self):
        self.value = 0
    
    async def process(self, data: list) -> dict:
        result = {}
        for item in data:
            if item > 0:
                result[item] = item * 2
        return result

async def main():
    obj = ComplexClass()
    data = list(range(100))
    return await obj.process(data)
"""
        
        def validate():
            return validate_code(code)
        
        async def validate_async():
            return validate()
        
        result = await run_benchmark(
            "validate_complex_code",
            validate_async,
            benchmark_config.iterations,
        )
        
        logger.info(f"Validate complex code: {result.mean_ms:.3f}ms mean")
        
        # Assert performance baseline
        assert result.mean_ms < 2.0, f"Complex validation too slow: {result.mean_ms}ms"


# ─────────────────────────────────────────────
# State Persistence Benchmarks
# ─────────────────────────────────────────────

class TestStateBenchmarks:
    """State persistence benchmarks."""

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_state_save_latency(
        self,
        temp_dir: Path,
        benchmark_config: BenchmarkConfig,
    ):
        """Benchmark state save operation."""
        db_path = temp_dir / "state.db"
        state_mgr = StateManager(db_path)
        
        budget = Budget(max_usd=10.0)
        state = ProjectState(
            project_description="Benchmark project",
            success_criteria="Tests pass",
            budget=budget,
        )
        
        async def save():
            await state_mgr.save_project("benchmark", state)
        
        result = await run_benchmark(
            "state_save",
            save,
            benchmark_config.iterations // 10,  # Fewer iterations for DB ops
        )
        
        logger.info(f"State save: {result.mean_ms:.3f}ms mean")
        
        # Assert performance baseline
        assert result.mean_ms < 50.0, f"State save too slow: {result.mean_ms}ms"
        
        await state_mgr.close()

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_state_load_latency(
        self,
        temp_dir: Path,
        benchmark_config: BenchmarkConfig,
    ):
        """Benchmark state load operation."""
        db_path = temp_dir / "state.db"
        state_mgr = StateManager(db_path)
        
        # Pre-save state
        budget = Budget(max_usd=10.0)
        state = ProjectState(
            project_description="Benchmark project",
            success_criteria="Tests pass",
            budget=budget,
        )
        await state_mgr.save_project("benchmark", state)
        
        async def load():
            return await state_mgr.load_project("benchmark")
        
        result = await run_benchmark(
            "state_load",
            load,
            benchmark_config.iterations // 10,
        )
        
        logger.info(f"State load: {result.mean_ms:.3f}ms mean")
        
        # Assert performance baseline
        assert result.mean_ms < 20.0, f"State load too slow: {result.mean_ms}ms"
        
        await state_mgr.close()


# ─────────────────────────────────────────────
# Concurrent Operation Benchmarks
# ─────────────────────────────────────────────

class TestConcurrentBenchmarks:
    """Concurrent operation benchmarks."""

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_concurrent_budget_charges(
        self,
        benchmark_config: BenchmarkConfig,
    ):
        """Benchmark concurrent budget charges."""
        for concurrency in benchmark_config.concurrency_levels:
            budget = Budget(max_usd=1000.0)
            
            async def charge_all():
                tasks = [budget.charge(0.01, "generation") for _ in range(concurrency)]
                await asyncio.gather(*tasks)
            
            result = await run_benchmark(
                f"concurrent_budget_{concurrency}",
                charge_all,
                benchmark_config.iterations // 10,
            )
            
            logger.info(
                f"Concurrent budget charges ({concurrency}): "
                f"{result.mean_ms:.3f}ms mean, {result.p95_ms:.3f}ms p95"
            )
            
            # Assert performance baseline
            assert result.p95_ms < 100.0, f"Concurrent charges too slow at {concurrency}"

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_concurrent_file_writes(
        self,
        temp_dir: Path,
        benchmark_config: BenchmarkConfig,
    ):
        """Benchmark concurrent file writes."""
        for concurrency in benchmark_config.concurrency_levels[:2]:  # Skip 100 for file I/O
            async def write_all():
                tasks = [
                    async_write_text(
                        temp_dir / f"file_{i}.txt",
                        f"Content {i}" * 100
                    )
                    for i in range(concurrency)
                ]
                await asyncio.gather(*tasks)
            
            result = await run_benchmark(
                f"concurrent_write_{concurrency}",
                write_all,
                benchmark_config.iterations // 10,
            )
            
            logger.info(
                f"Concurrent file writes ({concurrency}): "
                f"{result.mean_ms:.3f}ms mean, {result.p95_ms:.3f}ms p95"
            )
            
            # Assert performance baseline
            assert result.p95_ms < 500.0, f"Concurrent writes too slow at {concurrency}"


# ─────────────────────────────────────────────
# Benchmark Report
# ─────────────────────────────────────────────

def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "benchmark: mark test as benchmark"
    )


# ─────────────────────────────────────────────
# Run Benchmarks
# ─────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "benchmark"])
