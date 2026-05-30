"""
Resilient Streaming Pipeline with Backpressure
===============================================

Implements the minimax regret improvement for Black Swan Scenario 3:
Streaming Memory Exhaustion

Features:
- Bounded queue sizes
- Backpressure strategies
- Memory monitoring
- Circuit breaker pattern
- Event sampling under pressure

Usage:
    from orchestrator.streaming_resilient import ResilientStreamingPipeline

    pipeline = ResilientStreamingPipeline(
        max_parallel=3,
        max_queue_size=1000,
        max_memory_mb=1024,
    )
"""

from __future__ import annotations

import asyncio
import gc
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

from .log_config import get_logger
from .streaming import (
    PipelineEvent,
    PipelineEventType,
    StreamingContext,
    StreamingPipeline,
    StreamingStage,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = get_logger(__name__)


class BackpressureStrategy(Enum):
    """Strategies for handling queue overflow."""

    DROP_OLDEST = auto()  # Drop oldest events
    DROP_NEWEST = auto()  # Drop newest events
    BLOCK = auto()  # Block producer (risk: deadlock)
    SAMPLE = auto()  # Keep every Nth event
    PAUSE = auto()  # Pause pipeline temporarily


class MemoryPressure(Enum):
    """Memory pressure levels."""

    NORMAL = auto()
    HIGH = auto()
    CRITICAL = auto()


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = auto()  # Normal operation
    OPEN = auto()  # Failing, rejecting requests
    HALF_OPEN = auto()  # Testing if recovered


@dataclass
class MemoryPressureConfig:
    """Configuration for memory pressure handling."""

    max_queue_size: int = 1000
    max_memory_mb: int = 1024
    warning_threshold_percent: float = 70.0
    critical_threshold_percent: float = 90.0
    backpressure_strategy: BackpressureStrategy = BackpressureStrategy.SAMPLE
    sampling_rate: int = 10  # Keep every Nth event under pressure
    pause_duration_seconds: float = 1.0
    gc_threshold_mb: int = 100  # Force GC when available below this


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_max_calls: int = 3


class MemoryMonitor:
    """Monitor system memory usage."""

    def __init__(
        self,
        warning_threshold: float = 70.0,
        critical_threshold: float = 90.0,
    ):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self._psutil_available = self._check_psutil()

    def _check_psutil(self) -> bool:
        """Check if psutil is available."""
        try:
            import psutil

            return True
        except ImportError:
            logger.warning("psutil not available - using fallback memory monitoring")
            return False

    def get_pressure_level(self) -> MemoryPressure:
        """Get current memory pressure level."""
        usage_percent = self.get_usage_percent()

        if usage_percent > self.critical_threshold:
            return MemoryPressure.CRITICAL
        elif usage_percent > self.warning_threshold:
            return MemoryPressure.HIGH
        return MemoryPressure.NORMAL

    def get_usage_percent(self) -> float:
        """Get memory usage as percentage."""
        if self._psutil_available:
            import psutil

            return psutil.virtual_memory().percent
        else:
            # Fallback: estimate from process memory
            try:
                import resource

                usage = resource.getrusage(resource.RUSAGE_SELF)
                # Rough estimate - not accurate but better than nothing
                return min(usage.ru_maxrss / (1024 * 10), 100.0)
            except Exception:
                return 50.0  # Unknown, assume moderate

    def get_available_mb(self) -> int:
        """Get available memory in MB."""
        if self._psutil_available:
            import psutil

            return psutil.virtual_memory().available // (1024 * 1024)
        return 1024  # Conservative estimate

    def get_used_mb(self) -> int:
        """Get used memory in MB."""
        if self._psutil_available:
            import psutil

            return psutil.virtual_memory().used // (1024 * 1024)
        return 0


class CircuitBreaker:
    """
    Circuit breaker for preventing cascade failures.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, reject requests quickly
    - HALF_OPEN: Testing if service recovered
    """

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.failures = 0
        self.last_failure_time: float | None = None
        self.state = CircuitState.CLOSED
        self.half_open_calls = 0

    def is_open(self) -> bool:
        """Check if circuit breaker is open."""
        if self.state == CircuitState.OPEN:
            # Check if recovery time has passed
            if time.time() - self.last_failure_time > self.config.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
                logger.info("Circuit breaker entering half-open state")
                return False
            return True

        if self.state == CircuitState.HALF_OPEN:
            self.half_open_calls += 1
            if self.half_open_calls > self.config.half_open_max_calls:
                # Too many test calls, stay half-open
                return False

        return False

    def record_success(self) -> None:
        """Record a successful call."""
        if self.state == CircuitState.HALF_OPEN:
            self.failures = 0
            self.state = CircuitState.CLOSED
            logger.info("Circuit breaker closed - service recovered")

    def record_failure(self) -> None:
        """Record a failed call."""
        self.failures += 1
        self.last_failure_time = time.time()

        if self.failures >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            logger.critical("Circuit breaker opened - too many failures")


class ResilientStreamingPipeline(StreamingPipeline):
    """
    Streaming pipeline with backpressure and memory protection.

    Prevents memory exhaustion through:
    - Bounded queues
    - Backpressure strategies
    - Memory monitoring
    - Circuit breaker
    """

    def __init__(
        self,
        max_parallel: int = 3,
        memory_config: MemoryPressureConfig | None = None,
        circuit_config: CircuitBreakerConfig | None = None,
    ):
        super().__init__(max_parallel)
        self.memory_config = memory_config or MemoryPressureConfig()
        self.circuit = CircuitBreaker(circuit_config or CircuitBreakerConfig())
        self.memory_monitor = MemoryMonitor(
            self.memory_config.warning_threshold_percent,
            self.memory_config.critical_threshold_percent,
        )
        self._event_counter = 0
        self._paused = False

    async def execute_streaming(
        self,
        project_description: str,
        success_criteria: str,
        budget: float = 5.0,
        project_id: str | None = None,
    ) -> AsyncIterator[PipelineEvent]:
        """
        Execute with memory protection and backpressure.

        Features:
        - Circuit breaker for failure isolation
        - Memory pressure detection
        - Bounded queue with backpressure
        - Event sampling under pressure
        """
        # Check circuit breaker
        if self.circuit.is_open():
            raise StreamingUnavailableError("Streaming circuit breaker is open - too many failures")

        # Check memory before starting
        pressure = self.memory_monitor.get_pressure_level()
        if pressure == MemoryPressure.CRITICAL:
            logger.error("Memory pressure critical, rejecting new project")
            raise ResourceExhaustedError("System under memory pressure")

        # Create bounded queue
        event_queue: asyncio.Queue[PipelineEvent] = asyncio.Queue(
            maxsize=self.memory_config.max_queue_size
        )

        # Create context
        context = StreamingContext(
            project_id=project_id or f"proj_{int(time.time())}",
            project_description=project_description,
            success_criteria=success_criteria,
            budget=budget,
            metadata={},
        )

        # Start pipeline
        pipeline_task = asyncio.create_task(
            self._run_pipeline_with_backpressure(context, event_queue)
        )

        try:
            while True:
                # Periodic memory check
                pressure = self.memory_monitor.get_pressure_level()

                if pressure == MemoryPressure.CRITICAL:
                    await self._handle_critical_memory()

                try:
                    # Wait for event with timeout
                    event = await asyncio.wait_for(event_queue.get(), timeout=1.0)

                    # Apply sampling under pressure
                    if pressure == MemoryPressure.HIGH:
                        self._event_counter += 1
                        if self._event_counter % self.memory_config.sampling_rate != 0:
                            event_queue.task_done()
                            continue

                    yield event
                    event_queue.task_done()

                except asyncio.TimeoutError:
                    # Check if pipeline completed
                    if pipeline_task.done():
                        # Drain remaining events
                        while not event_queue.empty():
                            try:
                                event = event_queue.get_nowait()
                                yield event
                                event_queue.task_done()
                            except asyncio.QueueEmpty:
                                break
                        break

                    # Handle backpressure if queue full
                    if event_queue.full():
                        await self._apply_backpressure(event_queue)

        except Exception:
            self.circuit.record_failure()
            raise
        else:
            self.circuit.record_success()
        finally:
            if not pipeline_task.done():
                pipeline_task.cancel()
                try:
                    await pipeline_task
                except asyncio.CancelledError:
                    pass

    async def _run_pipeline_with_backpressure(
        self,
        context: StreamingContext,
        event_queue: asyncio.Queue,
    ) -> None:
        """Run pipeline stages with memory monitoring."""

        # Adjust concurrency based on memory
        safe_parallel = self._calculate_safe_concurrency()
        semaphore = asyncio.Semaphore(safe_parallel)

        for stage in self.stages:
            # Check memory before each stage
            available_mb = self.memory_monitor.get_available_mb()
            if available_mb < self.memory_config.gc_threshold_mb:
                logger.warning(f"Low memory ({available_mb}MB), forcing GC")
                gc.collect()

            try:
                await self._run_stage_safe(stage, context, event_queue, semaphore)
            except MemoryError:
                logger.critical("Memory exhausted during stage execution")
                await event_queue.put(
                    PipelineEvent(
                        type=PipelineEventType.ERROR,
                        project_id=context.project_id,
                        timestamp=datetime.utcnow(),
                        data={"error": "Memory exhausted", "stage": stage.name},
                    )
                )
                raise ResourceExhaustedError("Memory limit exceeded")

    async def _run_stage_safe(
        self,
        stage: StreamingStage,
        context: StreamingContext,
        event_queue: asyncio.Queue,
        semaphore: asyncio.Semaphore,
    ) -> None:
        """Run stage with concurrency limit."""
        async with semaphore:
            await super()._run_stage(stage, context, event_queue)

    async def _apply_backpressure(self, queue: asyncio.Queue) -> None:
        """Apply backpressure strategy when queue is full."""

        strategy = self.memory_config.backpressure_strategy

        if strategy == BackpressureStrategy.DROP_OLDEST:
            try:
                dropped = queue.get_nowait()
                queue.task_done()
                logger.debug(f"Dropped oldest event: {dropped.type}")
            except asyncio.QueueEmpty:
                pass

        elif strategy == BackpressureStrategy.SAMPLE:
            # Aggressive: drop half the queue
            items_to_drop = queue.qsize() // 2
            for _ in range(items_to_drop):
                try:
                    queue.get_nowait()
                    queue.task_done()
                except asyncio.QueueEmpty:
                    break
            logger.warning(f"Dropped {items_to_drop} events due to memory pressure")

        elif strategy == BackpressureStrategy.PAUSE:
            if not self._paused:
                self._paused = True
                logger.info(f"Pausing pipeline for {self.memory_config.pause_duration_seconds}s")
                await asyncio.sleep(self.memory_config.pause_duration_seconds)
                self._paused = False

        elif strategy == BackpressureStrategy.BLOCK:
            # Just wait a bit
            await asyncio.sleep(0.1)

    async def _handle_critical_memory(self) -> None:
        """Handle critical memory pressure."""
        logger.critical("Critical memory pressure detected")
        gc.collect()
        await asyncio.sleep(0.5)

    def _calculate_safe_concurrency(self) -> int:
        """Calculate safe concurrency based on available memory."""
        available_mb = self.memory_monitor.get_available_mb()

        # Rough estimate: each task needs ~50MB
        safe_tasks = max(1, int(available_mb / 50))
        return min(safe_tasks, self.max_parallel)

    def get_health(self) -> dict[str, Any]:
        """Get pipeline health status."""
        return {
            "circuit_state": self.circuit.state.name,
            "circuit_failures": self.circuit.failures,
            "memory_pressure": self.memory_monitor.get_pressure_level().name,
            "memory_used_mb": self.memory_monitor.get_used_mb(),
            "memory_available_mb": self.memory_monitor.get_available_mb(),
        }


class ResourceExhaustedError(Exception):
    """Raised when system resources are exhausted."""

    pass


class StreamingUnavailableError(Exception):
    """Raised when streaming is unavailable due to failures."""

    pass
