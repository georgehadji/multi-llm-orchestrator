"""
Meta-Optimization Monitoring & Metrics
=======================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Production monitoring for meta-optimization:
- Prometheus metrics export
- Alert rules and evaluation
- Health checks for all components

USAGE:
    from orchestrator.meta_monitoring import MetricsExporter, HealthChecker
    
    exporter = MetricsExporter()
    metrics = exporter.get_metrics()  # Prometheus format
    
    checker = HealthChecker(meta_v2)
    health = await checker.check_all()
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List, Any, Callable
from pathlib import Path

logger = logging.getLogger("orchestrator.meta_monitoring")


# ─────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────

class HealthStatus(str, Enum):
    """Health status for components."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────

@dataclass
class Metric:
    """A single metric with labels."""
    name: str
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    help_text: str = ""
    
    def to_prometheus(self) -> str:
        """Convert to Prometheus exposition format."""
        labels_str = ",".join(
            f'{k}="{v}"' for k, v in self.labels.items()
        )
        
        if labels_str:
            return f"{self.name}{{{labels_str}}} {self.value}"
        else:
            return f"{self.name} {self.value}"
    
    @classmethod
    def gauge(cls, name: str, value: float, **labels) -> "Metric":
        """Create a gauge metric."""
        return cls(name=name, value=value, labels=labels)
    
    @classmethod
    def counter(cls, name: str, value: float, **labels) -> "Metric":
        """Create a counter metric."""
        return cls(name=name, value=value, labels=labels)


@dataclass
class Alert:
    """An alert triggered by rule evaluation."""
    name: str
    severity: AlertSeverity
    message: str
    triggered_at: float = field(default_factory=time.time)
    resolved_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "severity": self.severity.value,
            "message": self.message,
            "triggered_at": self.triggered_at,
            "resolved_at": self.resolved_at,
            "metadata": self.metadata,
        }


@dataclass
class AlertRule:
    """Rule for triggering alerts."""
    name: str
    metric_name: str
    condition: str  # "gt", "lt", "eq", "gte", "lte"
    threshold: float
    severity: AlertSeverity
    cooldown_seconds: int = 300  # Prevent alert fatigue
    last_triggered: Optional[float] = None
    
    def evaluate(self, value: float) -> bool:
        """Evaluate if alert should trigger."""
        # Check cooldown
        if self.last_triggered:
            if time.time() - self.last_triggered < self.cooldown_seconds:
                return False
        
        # Evaluate condition
        if self.condition == "gt":
            return value > self.threshold
        elif self.condition == "lt":
            return value < self.threshold
        elif self.condition == "eq":
            return value == self.threshold
        elif self.condition == "gte":
            return value >= self.threshold
        elif self.condition == "lte":
            return value <= self.threshold
        
        return False
    
    def trigger(self):
        """Mark alert as triggered."""
        self.last_triggered = time.time()


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    component: str
    status: HealthStatus
    message: str
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "component": self.component,
            "status": self.status.value,
            "message": self.message,
            "latency_ms": self.latency_ms,
            "metadata": self.metadata,
        }


# ─────────────────────────────────────────────
# Metrics Exporter
# ─────────────────────────────────────────────

class MetricsExporter:
    """
    Export metrics in Prometheus exposition format.
    
    Metrics exposed:
    - meta_optimization_proposals_total{status}
    - meta_optimization_experiments_active
    - meta_optimization_rollouts_in_progress
    - meta_optimization_hitl_pending_count
    - meta_optimization_transfer_patterns_active
    - meta_optimization_archive_size
    - meta_optimization_optimization_latency_seconds
    """
    
    def __init__(self, meta_v2: Optional[Any] = None):
        self.meta_v2 = meta_v2
        self._custom_metrics: List[Metric] = []
    
    def register_metric(self, metric: Metric):
        """Register a custom metric."""
        self._custom_metrics.append(metric)
    
    def get_metrics(self) -> str:
        """
        Get all metrics in Prometheus exposition format.
        
        Returns:
            Metrics in Prometheus format
        """
        lines = []
        
        # Header
        lines.append("# HELP meta_optimization_info Meta-optimization system info")
        lines.append("# TYPE meta_optimization_info gauge")
        lines.append(f'meta_optimization_info{{version="1.0.0"}} 1')
        lines.append("")
        
        # Collect metrics from meta_v2
        if self.meta_v2:
            lines.extend(self._collect_meta_v2_metrics())
        
        # Custom metrics
        if self._custom_metrics:
            lines.append("# Custom metrics")
            for metric in self._custom_metrics:
                lines.append(metric.to_prometheus())
        
        return "\n".join(lines)
    
    def _collect_meta_v2_metrics(self) -> List[str]:
        """Collect metrics from meta_v2."""
        lines = []
        
        try:
            status = self.meta_v2.get_status()
            
            # Archive metrics
            if "archive_stats" in status:
                archive = status["archive_stats"]
                lines.append("# HELP meta_optimization_archive_projects Total projects in archive")
                lines.append("# TYPE meta_optimization_archive_projects gauge")
                lines.append(f'meta_optimization_archive_projects {archive.get("total_projects", 0)}')
                lines.append("")
                
                lines.append("# HELP meta_optimization_archive_executions Total executions recorded")
                lines.append("# TYPE meta_optimization_archive_executions gauge")
                lines.append(f'meta_optimization_archive_executions {archive.get("total_executions", 0)}')
                lines.append("")
            
            # A/B Testing metrics
            if "ab_testing" in status:
                ab = status["ab_testing"]
                lines.append("# HELP meta_optimization_experiments_active Active A/B experiments")
                lines.append("# TYPE meta_optimization_experiments_active gauge")
                lines.append(f'meta_optimization_experiments_active {ab.get("active_experiments", 0)}')
                lines.append("")
                
                lines.append("# HELP meta_optimization_experiments_total Total experiments created")
                lines.append("# TYPE meta_optimization_experiments_total gauge")
                lines.append(f'meta_optimization_experiments_total {ab.get("total_experiments", 0)}')
                lines.append("")
            
            # HITL metrics
            if "hitl" in status:
                hitl = status["hitl"]
                lines.append("# HELP meta_optimization_hitl_pending Pending HITL approvals")
                lines.append("# TYPE meta_optimization_hitl_pending gauge")
                lines.append(f'meta_optimization_hitl_pending {hitl.get("pending_count", 0)}')
                lines.append("")
                
                lines.append("# HELP meta_optimization_hitl_auto_approved Auto-approved requests")
                lines.append("# TYPE meta_optimization_hitl_auto_approved gauge")
                lines.append(f'meta_optimization_hitl_auto_approved {hitl.get("auto_approved_count", 0)}')
                lines.append("")
            
            # Rollout metrics
            if "rollout" in status:
                rollout = status["rollout"]
                lines.append("# HELP meta_optimization_rollouts_active Active gradual rollouts")
                lines.append("# TYPE meta_optimization_rollouts_active gauge")
                lines.append(f'meta_optimization_rollouts_active {rollout.get("active_rollouts", 0)}')
                lines.append("")
                
                lines.append("# HELP meta_optimization_rollouts_completed Completed rollouts")
                lines.append("# TYPE meta_optimization_rollouts_completed gauge")
                lines.append(f'meta_optimization_rollouts_completed {rollout.get("completed_rollouts", 0)}')
                lines.append("")
            
            # Transfer learning metrics
            if "transfer" in status:
                transfer = status["transfer"]
                lines.append("# HELP meta_optimization_transfer_patterns Active transfer patterns")
                lines.append("# TYPE meta_optimization_transfer_patterns gauge")
                lines.append(f'meta_optimization_transfer_patterns {transfer.get("active_patterns", 0)}')
                lines.append("")
            
            # Optimization metrics
            lines.append("# HELP meta_optimization_cycles_total Total optimization cycles run")
            lines.append("# TYPE meta_optimization_cycles_total gauge")
            lines.append(f'meta_optimization_cycles_total {status.get("optimization_count", 0)}')
            lines.append("")
            
        except Exception as e:
            logger.warning(f"Failed to collect meta_v2 metrics: {e}")
            lines.append(f'# ERROR collecting metrics: {e}')
        
        return lines
    
    def get_metrics_dict(self) -> Dict[str, Any]:
        """Get metrics as a dictionary."""
        return {
            "meta_v2": self.meta_v2.get_status() if self.meta_v2 else {},
            "custom": [
                {"name": m.name, "value": m.value, "labels": m.labels}
                for m in self._custom_metrics
            ],
        }


# ─────────────────────────────────────────────
# Alert Rules Engine
# ─────────────────────────────────────────────

class AlertRulesEngine:
    """
    Define and evaluate alert rules.
    
    Prevents alert fatigue through cooldown periods
    and smart threshold management.
    """
    
    def __init__(self):
        self._rules: Dict[str, AlertRule] = {}
        self._active_alerts: Dict[str, Alert] = {}
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default alert rules for meta-optimization."""
        self.add_rule(AlertRule(
            name="high_hitl_pending",
            metric_name="meta_optimization_hitl_pending",
            condition="gt",
            threshold=10,
            severity=AlertSeverity.WARNING,
            cooldown_seconds=600,
        ))
        
        self.add_rule(AlertRule(
            name="critical_hitl_pending",
            metric_name="meta_optimization_hitl_pending",
            condition="gt",
            threshold=50,
            severity=AlertSeverity.CRITICAL,
            cooldown_seconds=300,
        ))
        
        self.add_rule(AlertRule(
            name="high_active_experiments",
            metric_name="meta_optimization_experiments_active",
            condition="gt",
            threshold=20,
            severity=AlertSeverity.WARNING,
            cooldown_seconds=600,
        ))
        
        self.add_rule(AlertRule(
            name="rollout_failures",
            metric_name="meta_optimization_rollouts_rolled_back",
            condition="gt",
            threshold=5,
            severity=AlertSeverity.CRITICAL,
            cooldown_seconds=300,
        ))
    
    def add_rule(self, rule: AlertRule):
        """Add an alert rule."""
        self._rules[rule.name] = rule
    
    def remove_rule(self, name: str):
        """Remove an alert rule."""
        self._rules.pop(name, None)
    
    def evaluate(
        self,
        metrics: Dict[str, float],
    ) -> List[Alert]:
        """
        Evaluate all rules against current metrics.
        
        Args:
            metrics: Current metric values
        
        Returns:
            List of triggered alerts
        """
        triggered = []
        
        for rule in self._rules.values():
            value = metrics.get(rule.metric_name)
            if value is None:
                continue
            
            if rule.evaluate(value):
                alert = Alert(
                    name=rule.name,
                    severity=rule.severity,
                    message=f"{rule.name}: {rule.metric_name}={value} (threshold: {rule.condition} {rule.threshold})",
                    metadata={
                        "metric": rule.metric_name,
                        "value": value,
                        "threshold": rule.threshold,
                    },
                )
                
                rule.trigger()
                self._active_alerts[rule.name] = alert
                triggered.append(alert)
                
                logger.warning(f"Alert triggered: {alert.name} - {alert.message}")
        
        return triggered
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return list(self._active_alerts.values())
    
    def resolve_alert(self, name: str):
        """Resolve an alert."""
        if name in self._active_alerts:
            self._active_alerts[name].resolved_at = time.time()
            del self._active_alerts[name]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get alert engine statistics."""
        return {
            "rules_count": len(self._rules),
            "active_alerts": len(self._active_alerts),
            "rules": {
                name: {
                    "threshold": rule.threshold,
                    "severity": rule.severity.value,
                    "last_triggered": rule.last_triggered,
                }
                for name, rule in self._rules.items()
            },
        }


# ─────────────────────────────────────────────
# Health Checker
# ─────────────────────────────────────────────

class HealthChecker:
    """
    Health checks for meta-optimization components.
    
    Provides quick status checks for monitoring dashboards
    and orchestration systems.
    """
    
    def __init__(self, meta_v2: Optional[Any] = None):
        self.meta_v2 = meta_v2
    
    async def check_all(self) -> Dict[str, HealthCheckResult]:
        """
        Run all health checks.
        
        Returns:
            Dictionary of component -> health result
        """
        results = {}
        
        results["meta_v2"] = await self._check_meta_v2()
        results["archive"] = await self._check_archive()
        results["ab_testing"] = await self._check_ab_testing()
        results["hitl"] = await self._check_hitl()
        results["rollout"] = await self._check_rollout()
        results["transfer"] = await self._check_transfer()
        
        return results
    
    async def _check_meta_v2(self) -> HealthCheckResult:
        """Check meta_v2 health."""
        start = time.time()
        
        try:
            if not self.meta_v2:
                return HealthCheckResult(
                    component="meta_v2",
                    status=HealthStatus.UNHEALTHY,
                    message="Meta-optimization not initialized",
                )
            
            status = self.meta_v2.get_status()
            
            if not status.get("enabled", False):
                return HealthCheckResult(
                    component="meta_v2",
                    status=HealthStatus.DEGRADED,
                    message="Meta-optimization disabled",
                )
            
            latency = (time.time() - start) * 1000
            
            return HealthCheckResult(
                component="meta_v2",
                status=HealthStatus.HEALTHY,
                message="Meta-optimization healthy",
                latency_ms=latency,
                metadata=status,
            )
            
        except Exception as e:
            return HealthCheckResult(
                component="meta_v2",
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                latency_ms=(time.time() - start) * 1000,
            )
    
    async def _check_archive(self) -> HealthCheckResult:
        """Check archive health."""
        start = time.time()
        
        try:
            if not self.meta_v2:
                return self._unhealthy("archive", "Meta-optimization not initialized", start)
            
            status = self.meta_v2.get_status()
            archive_stats = status.get("archive_stats", {})
            
            total_projects = archive_stats.get("total_projects", 0)
            
            latency = (time.time() - start) * 1000
            
            return HealthCheckResult(
                component="archive",
                status=HealthStatus.HEALTHY,
                message=f"Archive contains {total_projects} projects",
                latency_ms=latency,
                metadata=archive_stats,
            )
            
        except Exception as e:
            return self._unhealthy("archive", str(e), start)
    
    async def _check_ab_testing(self) -> HealthCheckResult:
        """Check A/B testing health."""
        start = time.time()
        
        try:
            if not self.meta_v2:
                return self._unhealthy("ab_testing", "Meta-optimization not initialized", start)
            
            status = self.meta_v2.get_status()
            ab_status = status.get("ab_testing", {})
            
            active = ab_status.get("active_experiments", 0)
            
            latency = (time.time() - start) * 1000
            
            return HealthCheckResult(
                component="ab_testing",
                status=HealthStatus.HEALTHY,
                message=f"{active} active experiments",
                latency_ms=latency,
                metadata=ab_status,
            )
            
        except Exception as e:
            return self._unhealthy("ab_testing", str(e), start)
    
    async def _check_hitl(self) -> HealthCheckResult:
        """Check HITL health."""
        start = time.time()
        
        try:
            if not self.meta_v2:
                return self._unhealthy("hitl", "Meta-optimization not initialized", start)
            
            status = self.meta_v2.get_status()
            hitl_status = status.get("hitl", {})
            
            pending = hitl_status.get("pending_count", 0)
            
            latency = (time.time() - start) * 1000
            
            # Degraded if too many pending
            if pending > 50:
                status_level = HealthStatus.DEGRADED
                message = f"{pending} pending approvals (high backlog)"
            else:
                status_level = HealthStatus.HEALTHY
                message = f"{pending} pending approvals"
            
            return HealthCheckResult(
                component="hitl",
                status=status_level,
                message=message,
                latency_ms=latency,
                metadata=hitl_status,
            )
            
        except Exception as e:
            return self._unhealthy("hitl", str(e), start)
    
    async def _check_rollout(self) -> HealthCheckResult:
        """Check rollout health."""
        start = time.time()
        
        try:
            if not self.meta_v2:
                return self._unhealthy("rollout", "Meta-optimization not initialized", start)
            
            status = self.meta_v2.get_status()
            rollout_status = status.get("rollout", {})
            
            active = rollout_status.get("active_rollouts", 0)
            completed = rollout_status.get("completed_rollouts", 0)
            
            latency = (time.time() - start) * 1000
            
            return HealthCheckResult(
                component="rollout",
                status=HealthStatus.HEALTHY,
                message=f"{active} active, {completed} completed",
                latency_ms=latency,
                metadata=rollout_status,
            )
            
        except Exception as e:
            return self._unhealthy("rollout", str(e), start)
    
    async def _check_transfer(self) -> HealthCheckResult:
        """Check transfer learning health."""
        start = time.time()
        
        try:
            from .transfer_learning import get_transfer_engine
            
            transfer_engine = get_transfer_engine()
            
            if not transfer_engine:
                return HealthCheckResult(
                    component="transfer",
                    status=HealthStatus.UNKNOWN,
                    message="Transfer learning not initialized",
                    latency_ms=(time.time() - start) * 1000,
                )
            
            stats = transfer_engine.get_stats()
            
            latency = (time.time() - start) * 1000
            
            return HealthCheckResult(
                component="transfer",
                status=HealthStatus.HEALTHY,
                message=f"{stats.get('active_patterns', 0)} active patterns",
                latency_ms=latency,
                metadata=stats,
            )
            
        except Exception as e:
            return self._unhealthy("transfer", str(e), start)
    
    def _unhealthy(self, component: str, message: str, start: float) -> HealthCheckResult:
        """Create unhealthy result."""
        return HealthCheckResult(
            component=component,
            status=HealthStatus.UNHEALTHY,
            message=message,
            latency_ms=(time.time() - start) * 1000,
        )
    
    def get_overall_status(self, results: Dict[str, HealthCheckResult]) -> HealthStatus:
        """
        Get overall health status from individual results.
        
        Returns:
            Overall status (worst of all components)
        """
        if not results:
            return HealthStatus.UNKNOWN
        
        statuses = [r.status for r in results.values()]
        
        if HealthStatus.UNHEALTHY in statuses:
            return HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        elif HealthStatus.HEALTHY in statuses:
            return HealthStatus.HEALTHY
        
        return HealthStatus.UNKNOWN


# ─────────────────────────────────────────────
# Convenience Functions
# ─────────────────────────────────────────────

_default_exporter: Optional[MetricsExporter] = None
_default_alerts: Optional[AlertRulesEngine] = None


def get_metrics_exporter(meta_v2: Optional[Any] = None) -> MetricsExporter:
    """Get or create default metrics exporter."""
    global _default_exporter
    if _default_exporter is None:
        _default_exporter = MetricsExporter(meta_v2)
    return _default_exporter


def get_alert_rules_engine() -> AlertRulesEngine:
    """Get or create default alert rules engine."""
    global _default_alerts
    if _default_alerts is None:
        _default_alerts = AlertRulesEngine()
    return _default_alerts


def reset_monitoring() -> None:
    """Reset monitoring singletons (for testing)."""
    global _default_exporter, _default_alerts
    _default_exporter = None
    _default_alerts = None
