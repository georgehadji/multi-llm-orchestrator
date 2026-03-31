"""
Health Checks & Readiness Probes
================================

Kubernetes-style health monitoring for the orchestrator.

Features:
- Liveness probes (is the process running?)
- Readiness probes (is it ready to accept traffic?)
- Startup probes (is initialization complete?)
- Deep health checks (are dependencies healthy?)

Usage:
    from orchestrator.health import HealthMonitor, health_check

    monitor = HealthMonitor()

    @monitor.liveness_check
    def check_basic():
        return HealthStatus.HEALTHY

    @monitor.readiness_check
    async def check_db():
        if await db.ping():
            return HealthStatus.HEALTHY
        return HealthStatus.UNHEALTHY

    # In Kubernetes probe endpoint
    @app.get("/health")
    async def health():
        return await monitor.check_all()
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger("orchestrator.health")


# ═══════════════════════════════════════════════════════════════════════════════
# Health Status Enum
# ═══════════════════════════════════════════════════════════════════════════════


class HealthStatus(Enum):
    """Health status levels."""

    HEALTHY = "healthy"  # All good
    DEGRADED = "degraded"  # Working but with issues
    UNHEALTHY = "unhealthy"  # Not working
    UNKNOWN = "unknown"  # Status unknown


class CheckType(Enum):
    """Types of health checks."""

    LIVENESS = "liveness"  # Is process alive?
    READINESS = "readiness"  # Is ready to serve?
    STARTUP = "startup"  # Has initialization completed?
    DEEP = "deep"  # Full dependency check


# ═══════════════════════════════════════════════════════════════════════════════
# Health Check Result
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class CheckResult:
    """Result of a single health check."""

    name: str
    status: HealthStatus
    check_type: CheckType
    response_time_ms: float
    timestamp: datetime
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "type": self.check_type.value,
            "response_time_ms": round(self.response_time_ms, 2),
            "timestamp": self.timestamp.isoformat(),
            "message": self.message,
            "metadata": self.metadata,
            "error": self.error,
        }


@dataclass
class HealthReport:
    """Complete health report."""

    overall_status: HealthStatus
    checks: list[CheckResult]
    timestamp: datetime
    version: str = "1.0.0"
    uptime_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.overall_status.value,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
            "uptime_seconds": round(self.uptime_seconds, 2),
            "checks": [c.to_dict() for c in self.checks],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Health Check Function Types
# ═══════════════════════════════════════════════════════════════════════════════

HealthCheckFunc = Callable[[], HealthStatus | Awaitable[HealthStatus]]


# ═══════════════════════════════════════════════════════════════════════════════
# Health Monitor
# ═══════════════════════════════════════════════════════════════════════════════


class HealthMonitor:
    """
    Central health monitoring system.

    Supports multiple check types with different criticality levels.
    """

    def __init__(
        self,
        default_timeout: float = 5.0,
        check_interval: float = 30.0,
    ):
        self.default_timeout = default_timeout
        self.check_interval = check_interval
        self._checks: dict[CheckType, dict[str, HealthCheckFunc]] = defaultdict(dict)
        self._cache: dict[str, CheckResult] = {}
        self._cache_timestamp: datetime | None = None
        self._start_time = time.time()
        self._running = False
        self._check_task: asyncio.Task | None = None

    # Registration methods

    def liveness_check(self, func: HealthCheckFunc) -> HealthCheckFunc:
        """Decorator to register a liveness check."""
        self._checks[CheckType.LIVENESS][func.__name__] = func
        return func

    def readiness_check(self, func: HealthCheckFunc) -> HealthCheckFunc:
        """Decorator to register a readiness check."""
        self._checks[CheckType.READINESS][func.__name__] = func
        return func

    def startup_check(self, func: HealthCheckFunc) -> HealthCheckFunc:
        """Decorator to register a startup check."""
        self._checks[CheckType.STARTUP][func.__name__] = func
        return func

    def deep_check(self, func: HealthCheckFunc) -> HealthCheckFunc:
        """Decorator to register a deep health check."""
        self._checks[CheckType.DEEP][func.__name__] = func
        return func

    def add_check(
        self,
        name: str,
        func: HealthCheckFunc,
        check_type: CheckType = CheckType.READINESS,
    ) -> None:
        """Programmatically add a health check."""
        self._checks[check_type][name] = func

    # Check execution

    async def run_check(
        self,
        name: str,
        func: HealthCheckFunc,
        check_type: CheckType,
        timeout: float | None = None,
    ) -> CheckResult:
        """Run a single health check with timeout."""
        timeout = timeout or self.default_timeout
        start_time = time.time()

        try:
            # Check if function is async
            if inspect.iscoroutinefunction(func):
                status = await asyncio.wait_for(func(), timeout=timeout)
            else:
                # Run sync function in thread pool
                loop = asyncio.get_event_loop()
                status = await asyncio.wait_for(loop.run_in_executor(None, func), timeout=timeout)

            elapsed = (time.time() - start_time) * 1000

            return CheckResult(
                name=name,
                status=status,
                check_type=check_type,
                response_time_ms=elapsed,
                timestamp=datetime.utcnow(),
                message="Check passed" if status == HealthStatus.HEALTHY else "Check failed",
            )

        except asyncio.TimeoutError:
            elapsed = (time.time() - start_time) * 1000
            return CheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                check_type=check_type,
                response_time_ms=elapsed,
                timestamp=datetime.utcnow(),
                message=f"Check timed out after {timeout}s",
                error="timeout",
            )
        except Exception as e:
            elapsed = (time.time() - start_time) * 1000
            return CheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                check_type=check_type,
                response_time_ms=elapsed,
                timestamp=datetime.utcnow(),
                message=f"Check failed: {str(e)}",
                error=str(e),
            )

    async def run_checks(self, check_type: CheckType | None = None) -> list[CheckResult]:
        """Run all checks of a specific type, or all checks."""
        results = []

        types_to_run = [check_type] if check_type else list(CheckType)

        for ct in types_to_run:
            for name, func in self._checks[ct].items():
                result = await self.run_check(name, func, ct)
                results.append(result)

        return results

    async def check_all(self, use_cache: bool = True) -> HealthReport:
        """
        Run all health checks and return comprehensive report.

        Args:
            use_cache: If True and cache is fresh, return cached result
        """
        # Check if we can use cached result
        if use_cache and self._cache_timestamp:
            age = (datetime.utcnow() - self._cache_timestamp).total_seconds()
            if age < self.check_interval:
                checks = list(self._cache.values())
                return self._build_report(checks)

        # Run all checks
        checks = await self.run_checks()

        # Update cache
        for check in checks:
            self._cache[check.name] = check
        self._cache_timestamp = datetime.utcnow()

        return self._build_report(checks)

    def _build_report(self, checks: list[CheckResult]) -> HealthReport:
        """Build health report from check results."""
        # Determine overall status
        statuses = [c.status for c in checks]

        if HealthStatus.UNHEALTHY in statuses:
            overall = HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            overall = HealthStatus.DEGRADED
        elif all(s == HealthStatus.HEALTHY for s in statuses):
            overall = HealthStatus.HEALTHY
        else:
            overall = HealthStatus.UNKNOWN

        uptime = time.time() - self._start_time

        return HealthReport(
            overall_status=overall,
            checks=checks,
            timestamp=datetime.utcnow(),
            uptime_seconds=uptime,
        )

    # Convenience check methods

    async def is_alive(self) -> bool:
        """Quick liveness check."""
        if not self._checks[CheckType.LIVENESS]:
            return True  # No liveness checks = assume alive

        results = await self.run_checks(CheckType.LIVENESS)
        return all(r.status == HealthStatus.HEALTHY for r in results)

    async def is_ready(self) -> bool:
        """Quick readiness check."""
        if not self._checks[CheckType.READINESS]:
            return True

        results = await self.run_checks(CheckType.READINESS)
        return all(r.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED) for r in results)

    async def is_started(self) -> bool:
        """Check if startup is complete."""
        if not self._checks[CheckType.STARTUP]:
            return True

        results = await self.run_checks(CheckType.STARTUP)
        return all(r.status == HealthStatus.HEALTHY for r in results)

    # Background monitoring

    async def start_monitoring(self) -> None:
        """Start background health check loop."""
        self._running = True
        self._check_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Health monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop background health check loop."""
        self._running = False
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass
        logger.info("Health monitoring stopped")

    async def _monitoring_loop(self) -> None:
        """Background loop for periodic health checks."""
        while self._running:
            try:
                await self.check_all(use_cache=False)
            except Exception as e:
                logger.error(f"Health check loop error: {e}")

            await asyncio.sleep(self.check_interval)


# ═══════════════════════════════════════════════════════════════════════════════
# Built-in Health Checks
# ═══════════════════════════════════════════════════════════════════════════════


def create_default_health_monitor() -> HealthMonitor:
    """Create a health monitor with default checks."""
    monitor = HealthMonitor()

    # Liveness check - always returns healthy if process is running
    @monitor.liveness_check
    def basic_liveness() -> HealthStatus:
        """Basic liveness check - process is running."""
        return HealthStatus.HEALTHY

    # Readiness check - can we accept work?
    @monitor.readiness_check
    async def basic_readiness() -> HealthStatus:
        """Check if system is ready to accept work."""
        # This would check if core components are initialized
        return HealthStatus.HEALTHY

    return monitor


# ═══════════════════════════════════════════════════════════════════════════════
# Kubernetes Probe Helpers
# ═══════════════════════════════════════════════════════════════════════════════


class KubernetesProbes:
    """Helper class for Kubernetes probe endpoints."""

    def __init__(self, monitor: HealthMonitor | None = None):
        self.monitor = monitor or create_default_health_monitor()

    async def liveness_probe(self) -> dict[str, Any]:
        """Kubernetes liveness probe endpoint."""
        is_alive = await self.monitor.is_alive()
        return {
            "status": "alive" if is_alive else "dead",
            "timestamp": datetime.utcnow().isoformat(),
        }

    async def readiness_probe(self) -> dict[str, Any]:
        """Kubernetes readiness probe endpoint."""
        is_ready = await self.monitor.is_ready()
        return {
            "status": "ready" if is_ready else "not_ready",
            "timestamp": datetime.utcnow().isoformat(),
        }

    async def startup_probe(self) -> dict[str, Any]:
        """Kubernetes startup probe endpoint."""
        is_started = await self.monitor.is_started()
        return {
            "status": "started" if is_started else "starting",
            "timestamp": datetime.utcnow().isoformat(),
        }

    async def health_check(self) -> tuple[dict[str, Any], int]:
        """Full health check for /health endpoint."""
        report = await self.monitor.check_all()

        # HTTP status code based on health
        status_code = 200
        if report.overall_status == HealthStatus.UNHEALTHY:
            status_code = 503
        elif report.overall_status == HealthStatus.DEGRADED:
            status_code = 200  # Or 429 if you want to indicate load shedding

        return report.to_dict(), status_code


# ═══════════════════════════════════════════════════════════════════════════════
# Example Usage
# ═══════════════════════════════════════════════════════════════════════════════


async def example():
    """Example of health monitor usage."""
    monitor = HealthMonitor()

    # Define checks
    @monitor.liveness_check
    def check_process():
        return HealthStatus.HEALTHY

    @monitor.readiness_check
    async def check_db():
        # Simulate DB check
        await asyncio.sleep(0.1)
        return HealthStatus.HEALTHY

    @monitor.readiness_check
    async def check_cache():
        # Simulate cache check
        return HealthStatus.DEGRADED  # Cache is slow but working

    # Run checks
    report = await monitor.check_all()

    print(f"Overall status: {report.overall_status.value}")
    for check in report.checks:
        print(f"  {check.name}: {check.status.value} ({check.response_time_ms:.2f}ms)")


if __name__ == "__main__":
    asyncio.run(example())
