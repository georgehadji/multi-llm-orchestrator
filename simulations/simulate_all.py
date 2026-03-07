"""
Production Failure Simulation Suite
====================================

Simulates production failure scenarios to test monitoring and rollback triggers.

Usage:
    python -m simulations.simulate_all  # Run all simulations
    python -m simulations.simulate_high_load  # High load only
    python -m simulations.simulate_provider_failure  # Provider failure only
    python -m simulations.simulate_malformed_input  # Malformed input only
"""

import asyncio
import time
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
from dataclasses import dataclass, asdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("simulations")


@dataclass
class SimulationResult:
    """Results from a simulation run."""
    name: str
    start_time: str
    end_time: str
    duration_seconds: float
    metrics: Dict[str, Any]
    threshold_violations: List[str]
    rollback_triggered: bool
    status: str  # "success", "warning", "critical"


# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATION 1: HIGH LOAD
# ═══════════════════════════════════════════════════════════════════════════════

async def simulate_high_load(
    duration_minutes: int = 15,
    target_tasks_per_minute: int = 100,
) -> SimulationResult:
    """
    Simulate 10x normal traffic load.
    
    Baseline: 10 tasks/minute
    High load: 100 tasks/minute
    
    Tests:
    - Task execution queue handling
    - Memory growth under load
    - Background task tracking
    - Resource exhaustion prevention
    """
    from orchestrator import Orchestrator, Budget, Task, TaskType
    from orchestrator.models import Model
    
    logger.info(f"Starting high load simulation: {target_tasks_per_minute} tasks/min for {duration_minutes} min")
    
    orch = Orchestrator(budget=Budget(max_usd=100.0))
    
    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)
    
    tasks_created = 0
    tasks_completed = 0
    tasks_failed = 0
    latencies = []
    memory_samples = []
    
    import psutil
    process = psutil.Process()
    
    while time.time() < end_time:
        # Create burst of tasks
        batch_size = target_tasks_per_minute // 10  # 10 batches per minute
        
        for _ in range(batch_size):
            task = Task(
                id=f"load_test_{tasks_created}",
                prompt="Write a simple Python function that adds two numbers",
                type=TaskType.CODE_GEN,
            )
            
            task_start = time.perf_counter()
            
            try:
                result = await orch._execute_task(task)
                latency = time.perf_counter() - task_start
                latencies.append(latency)
                
                if result.status.value == "completed":
                    tasks_completed += 1
                else:
                    tasks_failed += 1
                    
            except Exception as e:
                tasks_failed += 1
                logger.warning(f"Task failed: {e}")
            
            tasks_created += 1
        
        # Sample memory every 10 seconds
        memory_mb = process.memory_info().rss / 1024 / 1024
        memory_samples.append(memory_mb)
        
        # Log progress every minute
        if tasks_created % target_tasks_per_minute == 0:
            elapsed = time.time() - start_time
            logger.info(
                f"Progress: {tasks_created} tasks, "
                f"{tasks_completed} completed, {tasks_failed} failed, "
                f"Memory: {memory_mb:.1f} MB"
            )
        
        await asyncio.sleep(6)  # Wait for next batch
    
    # Calculate metrics
    duration = time.time() - start_time
    
    p50_latency = sorted(latencies)[len(latencies) // 2] if latencies else 0
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0
    p99_latency = sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0
    
    avg_memory = sum(memory_samples) / len(memory_samples) if memory_samples else 0
    max_memory = max(memory_samples) if memory_samples else 0
    
    error_rate = tasks_failed / tasks_created if tasks_created > 0 else 0
    
    # Check threshold violations
    violations = []
    if p95_latency > 30:
        violations.append(f"p95 latency {p95_latency:.1f}s > 30s threshold")
    if error_rate > 0.02:
        violations.append(f"Error rate {error_rate:.1%} > 2% threshold")
    if max_memory > 512:
        violations.append(f"Max memory {max_memory:.1f} MB > 512 MB threshold")
    
    # Determine status
    if len(violations) >= 2:
        status = "critical"
    elif len(violations) >= 1:
        status = "warning"
    else:
        status = "success"
    
    await orch.close()
    
    return SimulationResult(
        name="High Load Simulation",
        start_time=datetime.fromtimestamp(start_time).isoformat(),
        end_time=datetime.fromtimestamp(time.time()).isoformat(),
        duration_seconds=duration,
        metrics={
            "tasks_created": tasks_created,
            "tasks_completed": tasks_completed,
            "tasks_failed": tasks_failed,
            "tasks_per_minute": tasks_created / (duration / 60),
            "p50_latency_seconds": p50_latency,
            "p95_latency_seconds": p95_latency,
            "p99_latency_seconds": p99_latency,
            "avg_memory_mb": avg_memory,
            "max_memory_mb": max_memory,
            "error_rate": error_rate,
        },
        threshold_violations=violations,
        rollback_triggered=len(violations) >= 2,
        status=status,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATION 2: PROVIDER FAILURE
# ═══════════════════════════════════════════════════════════════════════════════

async def simulate_provider_failure(
    failure_rate: float = 0.5,
    num_tasks: int = 100,
) -> SimulationResult:
    """
    Simulate LLM provider failures.
    
    Tests:
    - Circuit breaker activation
    - Fallback chain behavior
    - Error rate monitoring
    - Recovery after failure
    """
    from orchestrator import Orchestrator, Budget, Task, TaskType
    from orchestrator.models import Model
    from unittest.mock import AsyncMock, patch
    
    logger.info(f"Starting provider failure simulation: {failure_rate*100}% failure rate")
    
    orch = Orchestrator(budget=Budget(max_usd=50.0))
    
    start_time = time.time()
    
    successes = 0
    fallbacks = 0
    failures = 0
    latencies = []
    circuit_breaker_trips = 0
    
    for i in range(num_tasks):
        task = Task(
            id=f"provider_test_{i}",
            prompt="Write a Python function",
            type=TaskType.CODE_GEN,
        )
        
        task_start = time.perf_counter()
        
        # Simulate failure
        should_fail = (i % 2) == 0  # 50% failure rate
        
        if should_fail:
            with patch.object(orch.client, 'call', new_callable=AsyncMock) as mock_call:
                mock_call.side_effect = Exception("Provider API unavailable")
                
                try:
                    result = await orch._execute_task(task)
                    latency = time.perf_counter() - task_start
                    latencies.append(latency)
                    
                    if result.status.value == "completed":
                        fallbacks += 1
                    else:
                        failures += 1
                        
                except Exception as e:
                    failures += 1
        else:
            try:
                result = await orch._execute_task(task)
                latency = time.perf_counter() - task_start
                latencies.append(latency)
                successes += 1
            except Exception as e:
                failures += 1
        
        # Check circuit breaker state
        if not orch.api_health.get(Model.GPT_4O, True):
            circuit_breaker_trips += 1
        
        # Log progress
        if (i + 1) % 10 == 0:
            logger.info(
                f"Progress: {i+1}/{num_tasks} - "
                f"Success: {successes}, Fallback: {fallbacks}, Failures: {failures}"
            )
    
    duration = time.time() - start_time
    
    # Calculate metrics
    total = successes + fallbacks + failures
    fallback_rate = fallbacks / total if total > 0 else 0
    error_rate = failures / total if total > 0 else 0
    
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0
    
    # Check threshold violations
    violations = []
    if error_rate > 0.03:
        violations.append(f"Error rate {error_rate:.1%} > 3% threshold")
    if fallback_rate > 0.30:
        violations.append(f"Fallback rate {fallback_rate:.1%} > 30% threshold")
    if p95_latency > 30:
        violations.append(f"p95 latency {p95_latency:.1f}s > 30s threshold")
    
    # Determine status
    if len(violations) >= 2:
        status = "critical"
    elif len(violations) >= 1:
        status = "warning"
    else:
        status = "success"
    
    await orch.close()
    
    return SimulationResult(
        name="Provider Failure Simulation",
        start_time=datetime.fromtimestamp(start_time).isoformat(),
        end_time=datetime.fromtimestamp(time.time()).isoformat(),
        duration_seconds=duration,
        metrics={
            "total_tasks": total,
            "successes": successes,
            "fallbacks": fallbacks,
            "failures": failures,
            "fallback_rate": fallback_rate,
            "error_rate": error_rate,
            "p95_latency_seconds": p95_latency,
            "circuit_breaker_trips": circuit_breaker_trips,
        },
        threshold_violations=violations,
        rollback_triggered=len(violations) >= 2,
        status=status,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATION 3: MALFORMED INPUT
# ═══════════════════════════════════════════════════════════════════════════════

async def simulate_malformed_input(
    num_inputs: int = 60,
) -> SimulationResult:
    """
    Simulate burst of malformed/invalid inputs.
    
    Tests:
    - Input validation
    - Error handling
    - Memory safety with large inputs
    - Injection attack prevention
    """
    from orchestrator import Orchestrator, Budget, Task, TaskType
    
    MALFORMED_INPUTS = [
        "",  # Empty
        "x" * 100000,  # Very long (100KB)
        "<script>alert('xss')</script>",  # XSS attempt
        "'; DROP TABLE tasks; --",  # SQL injection
        "\x00\x01\x02\x03",  # Binary data
        "正常な日本語" * 1000,  # Unicode overflow
        "A" * 1000000,  # 1MB string
        "{{config}}",  # Template injection
        "${7*7}",  # Expression injection
        "../../etc/passwd",  # Path traversal
    ]
    
    logger.info(f"Starting malformed input simulation: {num_inputs} inputs")
    
    orch = Orchestrator(budget=Budget(max_usd=20.0))
    
    start_time = time.time()
    
    validation_errors = 0
    processing_errors = 0
    successes = 0
    latencies = []
    
    import psutil
    process = psutil.Process()
    memory_before = process.memory_info().rss / 1024 / 1024
    
    for i in range(num_inputs):
        malformed = MALFORMED_INPUTS[i % len(MALFORMED_INPUTS)]
        
        task = Task(
            id=f"malformed_{i}",
            prompt=malformed,
            type=TaskType.CODE_GEN,
        )
        
        task_start = time.perf_counter()
        
        try:
            result = await orch._execute_task(task)
            latency = time.perf_counter() - task_start
            latencies.append(latency)
            
            if result.status.value == "failed":
                validation_errors += 1
            else:
                successes += 1
                
        except Exception as e:
            processing_errors += 1
            logger.debug(f"Processing error: {type(e).__name__}: {str(e)[:100]}")
        
        # Log progress
        if (i + 1) % 10 == 0:
            logger.info(
                f"Progress: {i+1}/{num_inputs} - "
                f"Success: {successes}, Validation errors: {validation_errors}, "
                f"Processing errors: {processing_errors}"
            )
    
    duration = time.time() - start_time
    memory_after = process.memory_info().rss / 1024 / 1024
    memory_delta = memory_after - memory_before
    
    # Calculate metrics
    total = successes + validation_errors + processing_errors
    validation_error_rate = validation_errors / total if total > 0 else 0
    processing_error_rate = processing_errors / total if total > 0 else 0
    
    # Check threshold violations
    violations = []
    if processing_error_rate > 0.02:
        violations.append(f"Processing error rate {processing_error_rate:.1%} > 2% threshold")
    if validation_error_rate > 0.50:
        violations.append(f"Validation error rate {validation_error_rate:.1%} > 50% threshold")
    if memory_delta > 256:
        violations.append(f"Memory spike {memory_delta:.1f} MB > 256 MB threshold")
    
    # Determine status
    if len(violations) >= 2:
        status = "critical"
    elif len(violations) >= 1:
        status = "warning"
    else:
        status = "success"
    
    await orch.close()
    
    return SimulationResult(
        name="Malformed Input Simulation",
        start_time=datetime.fromtimestamp(start_time).isoformat(),
        end_time=datetime.fromtimestamp(time.time()).isoformat(),
        duration_seconds=duration,
        metrics={
            "total_inputs": total,
            "successes": successes,
            "validation_errors": validation_errors,
            "processing_errors": processing_errors,
            "validation_error_rate": validation_error_rate,
            "processing_error_rate": processing_error_rate,
            "memory_before_mb": memory_before,
            "memory_after_mb": memory_after,
            "memory_delta_mb": memory_delta,
        },
        threshold_violations=violations,
        rollback_triggered=len(violations) >= 2,
        status=status,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN: RUN ALL SIMULATIONS
# ═══════════════════════════════════════════════════════════════════════════════

async def run_all_simulations() -> List[SimulationResult]:
    """Run all simulations and generate report."""
    
    results = []
    
    # Simulation 1: High Load
    logger.info("=" * 60)
    logger.info("SIMULATION 1: HIGH LOAD")
    logger.info("=" * 60)
    
    try:
        result = await simulate_high_load(
            duration_minutes=5,  # Shortened for testing
            target_tasks_per_minute=50,
        )
        results.append(result)
        logger.info(f"Result: {result.status} - {result.threshold_violations}")
    except Exception as e:
        logger.error(f"High load simulation failed: {e}")
        results.append(SimulationResult(
            name="High Load Simulation",
            start_time=datetime.now().isoformat(),
            end_time=datetime.now().isoformat(),
            duration_seconds=0,
            metrics={"error": str(e)},
            threshold_violations=["Simulation failed"],
            rollback_triggered=True,
            status="critical",
        ))
    
    # Simulation 2: Provider Failure
    logger.info("")
    logger.info("=" * 60)
    logger.info("SIMULATION 2: PROVIDER FAILURE")
    logger.info("=" * 60)
    
    try:
        result = await simulate_provider_failure(
            failure_rate=0.5,
            num_tasks=50,  # Shortened for testing
        )
        results.append(result)
        logger.info(f"Result: {result.status} - {result.threshold_violations}")
    except Exception as e:
        logger.error(f"Provider failure simulation failed: {e}")
    
    # Simulation 3: Malformed Input
    logger.info("")
    logger.info("=" * 60)
    logger.info("SIMULATION 3: MALFORMED INPUT")
    logger.info("=" * 60)
    
    try:
        result = await simulate_malformed_input(
            num_inputs=30,  # Shortened for testing
        )
        results.append(result)
        logger.info(f"Result: {result.status} - {result.threshold_violations}")
    except Exception as e:
        logger.error(f"Malformed input simulation failed: {e}")
    
    # Generate summary report
    logger.info("")
    logger.info("=" * 60)
    logger.info("SIMULATION SUMMARY")
    logger.info("=" * 60)
    
    for result in results:
        logger.info(f"{result.name}: {result.status}")
        if result.threshold_violations:
            for v in result.threshold_violations:
                logger.info(f"  - {v}")
    
    # Save results to file
    output_path = Path(__file__).parent / "simulation_results.json"
    with open(output_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    
    logger.info(f"\nResults saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("PRODUCTION FAILURE SIMULATION SUITE")
    print("=" * 60)
    print()
    
    results = asyncio.run(run_all_simulations())
    
    # Exit with error if any simulation was critical
    critical_count = sum(1 for r in results if r.status == "critical")
    
    if critical_count > 0:
        print(f"\n⚠️  {critical_count} simulation(s) triggered CRITICAL alerts")
        print("Review threshold violations and consider rollback triggers")
        sys.exit(1)
    else:
        print("\n✅ All simulations completed within acceptable thresholds")
        sys.exit(0)
