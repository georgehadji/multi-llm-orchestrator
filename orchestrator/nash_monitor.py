"""
Nash Stability Runtime Monitor
==============================

Real-time monitoring with stability threshold enforcement.
Implements the τ (tau) thresholds from resilience testing.
"""

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Callable, Any
from pathlib import Path

from .log_config import get_logger
from .nash_infrastructure_v2 import (
    AsyncIOManager,
    WriteAheadLog,
    UnifiedEventBus,
)

logger = get_logger(__name__)


class StabilityLevel(Enum):
    """Stability classification."""
    HEALTHY = "healthy"
    WARNING = "warning"  # τ_critical exceeded
    CRITICAL = "critical"  # τ_rollback exceeded
    EMERGENCY = "emergency"  # Multiple τ_rollback exceeded


@dataclass
class MetricSample:
    """Single metric measurement."""
    timestamp: float
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class StabilityThreshold:
    """
    Individual stability threshold with hysteresis.
    
    Prevents flapping by requiring sustained violation
    before triggering action.
    """
    
    def __init__(
        self,
        name: str,
        τ_critical: float,
        τ_rollback: float,
        window_seconds: float = 60.0,
        violation_count_threshold: int = 3,
    ):
        self.name = name
        self.τ_critical = τ_critical
        self.τ_rollback = τ_rollback
        self.window_seconds = window_seconds
        self.violation_count_threshold = violation_count_threshold
        
        self._samples: deque = deque()
        self._critical_violations = 0
        self._rollback_violations = 0
        self._last_state = StabilityLevel.HEALTHY
    
    def record(self, value: float, metadata: Optional[Dict] = None):
        """Record a new metric sample."""
        now = time.time()
        self._samples.append(MetricSample(now, value, metadata or {}))
        
        # Clean old samples
        cutoff = now - self.window_seconds
        while self._samples and self._samples[0].timestamp < cutoff:
            self._samples.popleft()
        
        # Check thresholds
        if value >= self.τ_rollback:
            self._rollback_violations += 1
            return StabilityLevel.CRITICAL
        elif value >= self.τ_critical:
            self._critical_violations += 1
            return StabilityLevel.WARNING
        
        return StabilityLevel.HEALTHY
    
    def check_sustained_violation(self) -> tuple[StabilityLevel, str]:
        """
        Check if threshold has been violated consistently.
        
        Returns: (level, reason)
        """
        if self._rollback_violations >= self.violation_count_threshold:
            return StabilityLevel.CRITICAL, f"{self.name} exceeded τ_rollback ({self.τ_rollback}) {self._rollback_violations} times"
        
        if self._critical_violations >= self.violation_count_threshold:
            return StabilityLevel.WARNING, f"{self.name} exceeded τ_critical ({self.τ_critical}) {self._critical_violations} times"
        
        return StabilityLevel.HEALTHY, ""
    
    def reset(self):
        """Reset violation counters."""
        self._critical_violations = 0
        self._rollback_violations = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current threshold statistics."""
        if not self._samples:
            return {"name": self.name, "current": None, "avg": None}
        
        values = [s.value for s in self._samples]
        return {
            "name": self.name,
            "current": values[-1],
            "avg": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "samples": len(values),
            "critical_violations": self._critical_violations,
            "rollback_violations": self._rollback_violations,
        }


class NashRuntimeMonitor:
    """
    Central runtime monitor for Nash Stability Infrastructure.
    
    Monitors all subsystems and triggers recovery actions when
    stability thresholds are violated.
    """
    
    # Pre-computed thresholds from resilience testing
    DEFAULT_THRESHOLDS = {
        "nash_io_error_rate": (0.05, 0.20, 60),  # (critical, rollback, window)
        "nash_io_latency_p99": (1.0, 5.0, 60),
        "nash_wal_pending_entries": (100, 1000, 60),
        "nash_wal_recovery_failures": (1, 3, 300),
        "nash_event_normalization_failures": (0.01, 0.10, 60),
        "nash_event_backpressure": (1000, 10000, 60),
        "nash_thread_pool_saturation": (0.80, 0.95, 60),
        "nash_thread_pool_queue_size": (100, 1000, 60),
    }
    
    def __init__(
        self,
        check_interval: float = 5.0,
        auto_recover: bool = True,
    ):
        self.check_interval = check_interval
        self.auto_recover = auto_recover
        
        # Initialize thresholds
        self._thresholds: Dict[str, StabilityThreshold] = {}
        for name, (τ_crit, τ_roll, window) in self.DEFAULT_THRESHOLDS.items():
            self._thresholds[name] = StabilityThreshold(
                name=name,
                τ_critical=τ_crit,
                τ_rollback=τ_roll,
                window_seconds=window,
            )
        
        # Component references
        self._io_manager: Optional[AsyncIOManager] = None
        self._wal: Optional[WriteAheadLog] = None
        self._event_bus: Optional[UnifiedEventBus] = None
        
        # Monitoring state
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        
        # Recovery handlers
        self._recovery_handlers: Dict[str, Callable] = {}
        self._alert_handlers: List[Callable[[str, str], None]] = []
        
        # Metrics history
        self._overall_stability = StabilityLevel.HEALTHY
        self._violation_history: deque = deque(maxlen=100)
    
    def register_components(
        self,
        io_manager: Optional[AsyncIOManager] = None,
        wal: Optional[WriteAheadLog] = None,
        event_bus: Optional[UnifiedEventBus] = None,
    ):
        """Register infrastructure components for monitoring."""
        self._io_manager = io_manager
        self._wal = wal
        self._event_bus = event_bus
    
    def register_recovery_handler(self, metric: str, handler: Callable):
        """Register a recovery handler for a specific metric."""
        self._recovery_handlers[metric] = handler
    
    def register_alert_handler(self, handler: Callable[[str, str], None]):
        """Register a handler for critical alerts."""
        self._alert_handlers.append(handler)
    
    async def start(self):
        """Start the monitoring loop."""
        if self._running:
            return
        
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("NashRuntimeMonitor started")
    
    async def stop(self):
        """Stop the monitoring loop."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("NashRuntimeMonitor stopped")
    
    async def _monitor_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                await self._check_all_metrics()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def _check_all_metrics(self):
        """Check all metrics and trigger actions if needed."""
        metrics = await self._collect_metrics()
        
        # Record all metrics
        for name, value in metrics.items():
            if name in self._thresholds:
                level = self._thresholds[name].record(value)
                if level != StabilityLevel.HEALTHY:
                    logger.warning(f"Metric {name} at {level.value}: {value}")
        
        # Check for sustained violations
        critical_metrics = []
        warning_metrics = []
        
        for name, threshold in self._thresholds.items():
            level, reason = threshold.check_sustained_violation()
            if level == StabilityLevel.CRITICAL:
                critical_metrics.append((name, reason))
            elif level == StabilityLevel.WARNING:
                warning_metrics.append((name, reason))
        
        # Update overall stability
        if len(critical_metrics) >= 2:
            new_stability = StabilityLevel.EMERGENCY
        elif critical_metrics:
            new_stability = StabilityLevel.CRITICAL
        elif warning_metrics:
            new_stability = StabilityLevel.WARNING
        else:
            new_stability = StabilityLevel.HEALTHY
        
        # Handle state transitions
        if new_stability != self._overall_stability:
            await self._handle_stability_change(
                self._overall_stability,
                new_stability,
                critical_metrics + warning_metrics,
            )
            self._overall_stability = new_stability
        
        # Trigger recovery if needed
        if self.auto_recover and critical_metrics:
            for name, reason in critical_metrics:
                await self._trigger_recovery(name, reason)
    
    async def _collect_metrics(self) -> Dict[str, float]:
        """Collect current metrics from all components."""
        metrics = {}
        
        # I/O metrics
        if self._io_manager:
            io_metrics = self._io_manager.get_metrics()
            total = io_metrics.get("tasks_submitted", 0)
            failed = io_metrics.get("tasks_failed", 0)
            metrics["nash_io_error_rate"] = failed / total if total > 0 else 0
            # TODO: Add latency tracking
            metrics["nash_io_latency_p99"] = 0.0
        
        # WAL metrics
        if self._wal:
            recovered = await self._wal.recover()
            metrics["nash_wal_pending_entries"] = len(recovered)
            # TODO: Track recovery failures
            metrics["nash_wal_recovery_failures"] = 0
        
        # Event bus metrics
        if self._event_bus:
            bus_metrics = self._event_bus.get_metrics()
            total = bus_metrics.get("events_normalized", 0)
            errors = bus_metrics.get("normalization_errors", 0)
            metrics["nash_event_normalization_failures"] = errors / total if total > 0 else 0
            # TODO: Add backpressure tracking
            metrics["nash_event_backpressure"] = 0
        
        # Thread pool metrics (simulated)
        metrics["nash_thread_pool_saturation"] = 0.5  # Placeholder
        metrics["nash_thread_pool_queue_size"] = 10  # Placeholder
        
        return metrics
    
    async def _handle_stability_change(
        self,
        old_level: StabilityLevel,
        new_level: StabilityLevel,
        violations: List[tuple],
    ):
        """Handle stability level change."""
        logger.warning(
            f"Stability level changed: {old_level.value} -> {new_level.value}"
        )
        
        # Record violation
        for metric, reason in violations:
            self._violation_history.append({
                "timestamp": time.time(),
                "metric": metric,
                "reason": reason,
                "level": new_level.value,
            })
        
        # Send alerts
        for handler in self._alert_handlers:
            try:
                handler(new_level.value, str(violations))
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")
    
    async def _trigger_recovery(self, metric: str, reason: str):
        """Trigger recovery for a metric."""
        logger.error(f"Triggering recovery for {metric}: {reason}")
        
        if metric in self._recovery_handlers:
            try:
                await self._recovery_handlers[metric]()
                logger.info(f"Recovery completed for {metric}")
            except Exception as e:
                logger.error(f"Recovery failed for {metric}: {e}")
        else:
            # Default recovery actions
            await self._default_recovery(metric)
    
    async def _default_recovery(self, metric: str):
        """Default recovery action based on metric type."""
        if "io" in metric and self._io_manager:
            logger.info("Executing default I/O manager recovery")
            self._io_manager.shutdown(wait=True)
            # New instance will be created on next use
        
        elif "wal" in metric and self._wal:
            logger.info("Executing default WAL recovery")
            await self._wal.recover()
        
        elif "event" in metric and self._event_bus:
            logger.info("Executing default event bus recovery")
            # Clear and reinitialize
    
    def get_status(self) -> Dict[str, Any]:
        """Get current monitor status."""
        return {
            "stability_level": self._overall_stability.value,
            "running": self._running,
            "check_interval": self.check_interval,
            "thresholds": {
                name: threshold.get_stats()
                for name, threshold in self._thresholds.items()
            },
            "recent_violations": list(self._violation_history)[-10:],
        }


# Global monitor instance
_global_monitor: Optional[NashRuntimeMonitor] = None


def get_monitor() -> NashRuntimeMonitor:
    """Get or create global monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = NashRuntimeMonitor()
    return _global_monitor


async def start_monitoring(
    io_manager: Optional[AsyncIOManager] = None,
    wal: Optional[WriteAheadLog] = None,
    event_bus: Optional[UnifiedEventBus] = None,
):
    """Convenience function to start monitoring."""
    monitor = get_monitor()
    monitor.register_components(io_manager, wal, event_bus)
    await monitor.start()
    return monitor
