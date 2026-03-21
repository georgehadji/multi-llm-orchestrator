"""
Production Monitoring & KPIs Module
===================================
Comprehensive monitoring for Multi-LLM Orchestrator deployment.

Tracks critical KPIs:
- Performance (TTFB, latency, throughput)
- Reliability (error rates, uptime)
- Cost efficiency (token usage, budget burn)
- Resource utilization (memory, CPU, connections)

Usage:
    from orchestrator.monitoring import KPIReporter, monitor_endpoint
    
    @monitor_endpoint("/api/models")
    async def get_models():
        return await fetch_models()
"""
from __future__ import annotations

import asyncio
import functools
import json
import platform
try:
    import resource
except ModuleNotFoundError:
    resource = None
import time
from collections import deque
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar
from typing_extensions import ParamSpec

from .log_config import get_logger

logger = get_logger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


# ═══════════════════════════════════════════════════════════════════════════════
# KPI DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

class KPITier(Enum):
    """KPI criticality tiers."""
    CRITICAL = "critical"  # Page on failure
    HIGH = "high"          # Alert immediately
    MEDIUM = "medium"      # Daily report
    LOW = "low"            # Weekly report


@dataclass
class KPIThreshold:
    """Threshold configuration for a KPI."""
    warning: float
    critical: float
    unit: str
    
    def check(self, value: float) -> Tuple[bool, str]:
        """Check value against thresholds. Returns (is_alert, severity)."""
        if value >= self.critical:
            return True, "critical"
        elif value >= self.warning:
            return True, "warning"
        return False, "ok"


@dataclass
class KPIDefinition:
    """Definition of a Key Performance Indicator."""
    name: str
    description: str
    tier: KPITier
    threshold: KPIThreshold
    target_value: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "tier": self.tier.value,
            "threshold": asdict(self.threshold),
            "target_value": self.target_value,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# STANDARD KPIs FOR MULTI-LLM ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

STANDARD_KPIS = {
    # Performance KPIs
    "ttfb": KPIDefinition(
        name="Time to First Byte",
        description="Time from request to first byte of response",
        tier=KPITier.CRITICAL,
        threshold=KPIThreshold(warning=100, critical=500, unit="ms"),
        target_value=50,
    ),
    "response_time_p50": KPIDefinition(
        name="Median Response Time",
        description="50th percentile response time",
        tier=KPITier.CRITICAL,
        threshold=KPIThreshold(warning=200, critical=1000, unit="ms"),
        target_value=100,
    ),
    "response_time_p95": KPIDefinition(
        name="P95 Response Time",
        description="95th percentile response time",
        tier=KPITier.HIGH,
        threshold=KPIThreshold(warning=500, critical=2000, unit="ms"),
        target_value=300,
    ),
    "throughput": KPIDefinition(
        name="Request Throughput",
        description="Requests per second",
        tier=KPITier.HIGH,
        threshold=KPIThreshold(warning=10, critical=100, unit="req/s"),
        target_value=50,
    ),
    
    # Reliability KPIs
    "error_rate": KPIDefinition(
        name="Error Rate",
        description="Percentage of failed requests",
        tier=KPITier.CRITICAL,
        threshold=KPIThreshold(warning=0.01, critical=0.05, unit="percent"),
        target_value=0.001,
    ),
    "uptime": KPIDefinition(
        name="Uptime",
        description="Service availability percentage",
        tier=KPITier.CRITICAL,
        threshold=KPIThreshold(warning=99.5, critical=99.0, unit="percent"),
        target_value=99.99,
    ),
    "cache_hit_rate": KPIDefinition(
        name="Cache Hit Rate",
        description="Percentage of cache hits",
        tier=KPITier.MEDIUM,
        threshold=KPIThreshold(warning=0.5, critical=0.3, unit="percent"),
        target_value=0.85,
    ),
    
    # Cost KPIs
    "daily_cost": KPIDefinition(
        name="Daily API Cost",
        description="Total API costs per day",
        tier=KPITier.HIGH,
        threshold=KPIThreshold(warning=50, critical=100, unit="USD"),
        target_value=25,
    ),
    "token_efficiency": KPIDefinition(
        name="Token Efficiency",
        description="Output tokens per input tokens",
        tier=KPITier.MEDIUM,
        threshold=KPIThreshold(warning=0.3, critical=0.2, unit="ratio"),
        target_value=0.5,
    ),
    
    # Resource KPIs
    "memory_usage": KPIDefinition(
        name="Memory Usage",
        description="Current memory utilization",
        tier=KPITier.HIGH,
        threshold=KPIThreshold(warning=70, critical=90, unit="percent"),
        target_value=50,
    ),
    "connection_pool_utilization": KPIDefinition(
        name="Connection Pool Utilization",
        description="Percentage of connections in use",
        tier=KPITier.MEDIUM,
        threshold=KPIThreshold(warning=80, critical=95, unit="percent"),
        target_value=50,
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS COLLECTION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MetricSample:
    """Single metric sample."""
    value: float
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)


class SlidingWindow:
    """Sliding window for metric samples."""
    
    def __init__(self, duration_seconds: int = 300):
        self.duration = duration_seconds
        self._samples: deque = deque()
        self._lock = asyncio.Lock()
    
    async def add(self, value: float, labels: Optional[Dict[str, str]] = None):
        """Add sample to window."""
        async with self._lock:
            now = time.time()
            self._samples.append(MetricSample(value, now, labels or {}))
            self._prune(now)
    
    def _prune(self, now: float):
        """Remove old samples."""
        cutoff = now - self.duration
        while self._samples and self._samples[0].timestamp < cutoff:
            self._samples.popleft()
    
    async def get_samples(self) -> List[MetricSample]:
        """Get all samples in window."""
        async with self._lock:
            self._prune(time.time())
            return list(self._samples)
    
    async def get_stats(self) -> Dict[str, float]:
        """Get statistics for window."""
        samples = await self.get_samples()
        
        if not samples:
            return {"count": 0}
        
        values = [s.value for s in samples]
        values_sorted = sorted(values)
        n = len(values)
        
        return {
            "count": n,
            "min": values_sorted[0],
            "max": values_sorted[-1],
            "avg": sum(values) / n,
            "p50": values_sorted[int(n * 0.50)],
            "p95": values_sorted[min(int(n * 0.95), n - 1)],
            "p99": values_sorted[min(int(n * 0.99), n - 1)],
        }


class MetricsRegistry:
    """Central registry for all metrics."""
    
    def __init__(self):
        self._windows: Dict[str, SlidingWindow] = {}
        self._counters: Dict[str, int] = {}
        self._gauges: Dict[str, float] = {}
        self._lock = asyncio.Lock()
    
    async def record(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a metric value."""
        async with self._lock:
            if name not in self._windows:
                self._windows[name] = SlidingWindow()
        
        await self._windows[name].add(value, labels)
    
    async def increment(self, name: str, value: int = 1):
        """Increment a counter."""
        async with self._lock:
            self._counters[name] = self._counters.get(name, 0) + value
    
    async def gauge(self, name: str, value: float):
        """Set a gauge value."""
        async with self._lock:
            self._gauges[name] = value
    
    async def get_metric(self, name: str) -> Dict[str, Any]:
        """Get metric statistics."""
        async with self._lock:
            window = self._windows.get(name)
            counter = self._counters.get(name)
            gauge = self._gauges.get(name)
        
        result = {}
        
        if window:
            result["window"] = await window.get_stats()
        
        if counter is not None:
            result["counter"] = counter
        
        if gauge is not None:
            result["gauge"] = gauge
        
        return result
    
    async def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get all metrics."""
        all_metrics = {}
        
        async with self._lock:
            names = list(self._windows.keys()) + list(self._counters.keys()) + list(self._gauges.keys())
        
        for name in set(names):
            all_metrics[name] = await self.get_metric(name)
        
        return all_metrics


# Global registry
metrics = MetricsRegistry()


# ═══════════════════════════════════════════════════════════════════════════════
# KPI REPORTER
# ═══════════════════════════════════════════════════════════════════════════════

class KPIReporter:
    """
    Reports KPI status and generates alerts.
    
    Features:
    - Real-time KPI monitoring
    - Alert generation on threshold breaches
    - Health score calculation
    - Trend analysis
    """
    
    def __init__(self, kpis: Optional[Dict[str, KPIDefinition]] = None):
        self.kpis = kpis or STANDARD_KPIS
        self._alert_history: deque = deque(maxlen=100)
        self._health_score = 100.0
    
    async def evaluate(self, kpi_name: str, value: float) -> Dict[str, Any]:
        """Evaluate a KPI value against thresholds."""
        definition = self.kpis.get(kpi_name)
        if not definition:
            return {"error": f"Unknown KPI: {kpi_name}"}
        
        is_alert, severity = definition.threshold.check(value)
        
        result = {
            "kpi": definition.name,
            "value": value,
            "unit": definition.threshold.unit,
            "threshold_warning": definition.threshold.warning,
            "threshold_critical": definition.threshold.critical,
            "target": definition.target_value,
            "status": severity,
            "is_alert": is_alert,
            "tier": definition.tier.value,
        }
        
        if is_alert:
            alert = {
                "timestamp": datetime.now().isoformat(),
                "kpi": kpi_name,
                "severity": severity,
                "value": value,
                "threshold": definition.threshold.critical if severity == "critical" else definition.threshold.warning,
            }
            self._alert_history.append(alert)
            
            # Log alert
            log_method = logger.error if severity == "critical" else logger.warning
            log_method(
                f"KPI ALERT: {definition.name} = {value}{definition.threshold.unit} "
                f"(threshold: {definition.threshold.critical}{definition.threshold.unit})"
            )
        
        return result
    
    async def get_health_score(self) -> Dict[str, Any]:
        """Calculate overall health score."""
        all_metrics = await metrics.get_all_metrics()
        
        scores = []
        details = []
        
        for kpi_name, definition in self.kpis.items():
            metric_data = all_metrics.get(kpi_name, {})
            window = metric_data.get("window", {})
            
            if window.get("count", 0) > 0:
                current_value = window.get("avg", 0)
                evaluation = await self.evaluate(kpi_name, current_value)
                
                # Calculate individual score
                if evaluation["is_alert"]:
                    if evaluation["status"] == "critical":
                        score = 0
                    else:
                        score = 50
                else:
                    score = 100
                
                scores.append(score)
                details.append({
                    "kpi": kpi_name,
                    "score": score,
                    "current": current_value,
                    "target": definition.target_value,
                })
        
        overall_score = sum(scores) / len(scores) if scores else 100
        self._health_score = overall_score
        
        return {
            "overall": round(overall_score, 1),
            "status": "healthy" if overall_score >= 90 else "degraded" if overall_score >= 70 else "critical",
            "details": details,
            "timestamp": datetime.now().isoformat(),
        }
    
    def get_alert_summary(self, since_minutes: int = 60) -> Dict[str, Any]:
        """Get summary of recent alerts."""
        cutoff = time.time() - (since_minutes * 60)
        
        recent_alerts = [
            a for a in self._alert_history
            if datetime.fromisoformat(a["timestamp"]).timestamp() > cutoff
        ]
        
        by_severity = {"critical": 0, "warning": 0}
        by_kpi: Dict[str, int] = {}
        
        for alert in recent_alerts:
            by_severity[alert["severity"]] += 1
            by_kpi[alert["kpi"]] = by_kpi.get(alert["kpi"], 0) + 1
        
        return {
            "total": len(recent_alerts),
            "by_severity": by_severity,
            "by_kpi": by_kpi,
            "recent": recent_alerts[-10:],  # Last 10 alerts
        }
    
    async def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive KPI report."""
        all_metrics = await metrics.get_all_metrics()
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "health": await self.get_health_score(),
            "alerts": self.get_alert_summary(),
            "kpis": {},
            "system": await self._get_system_metrics(),
        }
        
        for kpi_name, definition in self.kpis.items():
            metric_data = all_metrics.get(kpi_name, {})
            window = metric_data.get("window", {})
            
            if window.get("count", 0) > 0:
                current_value = window.get("avg", 0)
                evaluation = await self.evaluate(kpi_name, current_value)
                
                report["kpis"][kpi_name] = {
                    "definition": definition.to_dict(),
                    "current": current_value,
                    "evaluation": evaluation,
                    "stats": window,
                }
        
        return report
    
    async def _get_system_metrics(self) -> Dict[str, Any]:
        """Get system-level metrics."""
        try:
            import psutil
            
            return {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent,
                "network_io": psutil.net_io_counters()._asdict(),
            }
        except ImportError:
            # Fallback without psutil
            return {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
            }


# ═══════════════════════════════════════════════════════════════════════════════
# MONITORING DECORATORS
# ═══════════════════════════════════════════════════════════════════════════════

def monitor_endpoint(name: Optional[str] = None):
    """
    Decorator to monitor endpoint performance.
    
    Automatically tracks:
    - Response time
    - Request count
    - Error count
    
    Usage:
        @monitor_endpoint("/api/models")
        async def get_models():
            return await fetch_models()
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        endpoint_name = name or func.__name__
        
        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                
                # Record success
                response_time = (time.time() - start_time) * 1000
                await metrics.record("response_time_p50", response_time, {"endpoint": endpoint_name})
                await metrics.record("response_time_p95", response_time, {"endpoint": endpoint_name})
                await metrics.increment(f"requests_total:{endpoint_name}")
                
                return result
                
            except Exception as e:
                # Record error
                await metrics.increment(f"errors_total:{endpoint_name}")
                await metrics.increment("error_rate")
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Fire and forget metrics for sync functions
            asyncio.create_task(metrics.increment(f"requests_total:{endpoint_name}"))
            return func(*args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


def monitor_async_task(name: Optional[str] = None):
    """
    Decorator to monitor async task execution.
    
    Usage:
        @monitor_async_task("model_routing")
        async def route_model(task):
            # ... routing logic
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        task_name = name or func.__name__
        
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                
                duration = (time.time() - start_time) * 1000
                await metrics.record(f"task_duration:{task_name}", duration)
                await metrics.increment(f"tasks_completed:{task_name}")
                
                return result
                
            except Exception as e:
                await metrics.increment(f"tasks_failed:{task_name}")
                raise
        
        return wrapper
    return decorator


# ═══════════════════════════════════════════════════════════════════════════════
# HEALTH CHECK
# ═══════════════════════════════════════════════════════════════════════════════

class HealthChecker:
    """Comprehensive health check for the orchestrator."""
    
    def __init__(self):
        self._checks: Dict[str, Callable[[], Any]] = {}
    
    def register(self, name: str, check_func: Callable[[], Any]):
        """Register a health check."""
        self._checks[name] = check_func
    
    async def check(self) -> Dict[str, Any]:
        """Run all health checks."""
        results = {}
        overall_healthy = True
        
        for name, check_func in self._checks.items():
            try:
                if asyncio.iscoroutinefunction(check_func):
                    result = await check_func()
                else:
                    result = check_func()
                
                results[name] = {
                    "status": "healthy" if result else "unhealthy",
                    "healthy": bool(result),
                }
                
                if not result:
                    overall_healthy = False
                    
            except Exception as e:
                results[name] = {
                    "status": "error",
                    "healthy": False,
                    "error": str(e),
                }
                overall_healthy = False
        
        return {
            "overall": "healthy" if overall_healthy else "unhealthy",
            "healthy": overall_healthy,
            "checks": results,
            "timestamp": datetime.now().isoformat(),
        }


# Global health checker
health_checker = HealthChecker()


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════════════════════

async def example():
    """Example monitoring usage."""
    
    # Create reporter
    reporter = KPIReporter()
    
    # Simulate metrics
    await metrics.record("response_time_p50", 45)
    await metrics.record("response_time_p95", 120)
    await metrics.record("error_rate", 0.005)
    await metrics.gauge("memory_usage", 65.5)
    
    # Evaluate KPIs
    result = await reporter.evaluate("response_time_p50", 45)
    print(f"Response time status: {result['status']}")
    
    # Get health score
    health = await reporter.get_health_score()
    print(f"Health score: {health['overall']}")
    
    # Generate report
    report = await reporter.generate_report()
    print(json.dumps(report, indent=2))
    
    # Health checks
    health_checker.register("cache", lambda: True)
    health_checker.register("database", lambda: True)
    
    health_status = await health_checker.check()
    print(f"Health check: {health_status['overall']}")


if __name__ == "__main__":
    asyncio.run(example())
