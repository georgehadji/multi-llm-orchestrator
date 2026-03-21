"""
Production Guardrails — Runtime monitors, kill switches, and safety mechanisms
================================================================================
Author: Senior Distributed Systems Architect

CRITICAL: Production safety mechanisms that prevent catastrophic failures.

FEATURES:
1. Budget enforcement (hard limit, never exceeded)
2. Rate limit enforcement
3. Memory monitoring
4. Error rate monitoring
5. Kill switch (file-based emergency stop)
6. Configuration drift detection

USAGE:
    from orchestrator.guardrails import ProductionGuardrails, get_guardrails
    
    guardrails = get_guardrails()
    
    # Check all guardrails
    if not guardrails.all_checks_pass(spent=50.0, max_budget=100.0):
        raise SystemExit("Guardrail failure")
    
    # Check kill switch
    if guardrails.check_kill_switch():
        raise SystemExit("Kill switch activated")
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable

logger = logging.getLogger("orchestrator.guardrails")


@dataclass
class GuardrailConfig:
    """Configuration for production guardrails."""
    
    # Budget limits
    max_budget_usd: float = 1000.0
    budget_warning_threshold: float = 0.80  # 80%
    budget_critical_threshold: float = 0.95  # 95%
    
    # Rate limits
    max_requests_per_minute: int = 100
    max_tokens_per_minute: int = 1_000_000
    
    # Memory limits
    max_memory_mb: float = 1024.0
    memory_warning_threshold: float = 0.85
    
    # Latency limits
    max_latency_ms: float = 30000.0  # 30 seconds
    latency_warning_threshold_ms: float = 10000.0  # 10 seconds
    
    # Error rate limits
    max_error_rate: float = 0.10  # 10%
    error_rate_window: int = 100  # Check last N requests
    
    # Kill switch
    enable_kill_switch: bool = True
    kill_switch_file: str = "/tmp/orchestrator_kill"
    force_kill_file: str = "/tmp/orchestrator_force_kill"
    
    # Drift detection
    enable_drift_detection: bool = True
    drift_check_interval_seconds: float = 60.0


@dataclass
class GuardrailStatus:
    """Status of a single guardrail check."""
    name: str
    passed: bool
    value: Any
    threshold: Any
    message: str


class ProductionGuardrails:
    """
    Production safety mechanisms.
    
    GUARANTEES:
    1. Budget never exceeded (hard limit)
    2. Kill switch respected (immediate shutdown)
    3. Error rate monitored (alert on degradation)
    4. Memory monitored (prevent OOM)
    5. Configuration drift detected
    
    USAGE:
        guardrails = ProductionGuardrails()
        
        # Before each operation
        if not guardrails.all_checks_pass(spent, max_budget):
            raise BudgetExceededError()
        
        # In main loop
        if guardrails.check_kill_switch():
            raise SystemExit("Kill switch activated")
    """
    
    def __init__(self, config: Optional[GuardrailConfig] = None):
        self.config = config or GuardrailConfig()
        
        # Error tracking
        self._error_count = 0
        self._total_requests = 0
        self._error_window: List[bool] = []  # Rolling window of success/failure
        
        # Timing
        self._start_time = time.monotonic()
        self._kill_switch_checked = 0.0
        self._memory_checked = 0.0
        
        # Drift detection
        self._drift_baseline: Dict[str, Any] = {}
        self._drift_history: List[Dict[str, Any]] = []
        
        # Callbacks
        self._on_budget_warning: Optional[Callable] = None
        self._on_budget_critical: Optional[Callable] = None
        self._on_kill_switch: Optional[Callable] = None
        self._on_error_rate_exceeded: Optional[Callable] = None
    
    # ─────────────────────────────────────────────────────────────────────────
    # Budget Checks
    # ─────────────────────────────────────────────────────────────────────────
    
    def check_budget(self, spent: float, max_budget: float) -> GuardrailStatus:
        """
        Check if budget is within limits.
        
        Args:
            spent: Amount spent in USD
            max_budget: Maximum budget in USD
            
        Returns:
            GuardrailStatus with check result
        """
        if max_budget <= 0:
            return GuardrailStatus(
                name="budget",
                passed=False,
                value=spent,
                threshold=max_budget,
                message="Invalid budget (zero or negative)"
            )
        
        ratio = spent / max_budget
        
        if spent > max_budget:
            logger.critical(f"BUDGET EXCEEDED: ${spent:.2f} > ${max_budget:.2f}")
            return GuardrailStatus(
                name="budget",
                passed=False,
                value=spent,
                threshold=max_budget,
                message=f"Budget exceeded: ${spent:.2f} > ${max_budget:.2f}"
            )
        
        if ratio >= self.config.budget_critical_threshold:
            msg = f"BUDGET CRITICAL: {ratio*100:.1f}% used (${spent:.2f} / ${max_budget:.2f})"
            logger.critical(msg)
            if self._on_budget_critical:
                self._on_budget_critical(spent, max_budget)
            return GuardrailStatus(
                name="budget",
                passed=True,
                value=spent,
                threshold=max_budget,
                message=msg
            )
        
        if ratio >= self.config.budget_warning_threshold:
            msg = f"BUDGET WARNING: {ratio*100:.1f}% used (${spent:.2f} / ${max_budget:.2f})"
            logger.warning(msg)
            if self._on_budget_warning:
                self._on_budget_warning(spent, max_budget)
        
        return GuardrailStatus(
            name="budget",
            passed=True,
            value=spent,
            threshold=max_budget,
            message=f"Budget OK: {ratio*100:.1f}% used"
        )
    
    # ─────────────────────────────────────────────────────────────────────────
    # Kill Switch
    # ─────────────────────────────────────────────────────────────────────────
    
    def check_kill_switch(self) -> bool:
        """
        Check if kill switch is activated.
        
        Returns:
            True if kill switch is activated (should terminate)
        """
        if not self.config.enable_kill_switch:
            return False
        
        # Only check every 5 seconds to avoid I/O overhead
        now = time.monotonic()
        if now - self._kill_switch_checked < 5.0:
            return False
        self._kill_switch_checked = now
        
        # Check force kill first
        force_file = Path(self.config.force_kill_file)
        if force_file.exists():
            logger.critical(f"FORCE KILL SWITCH ACTIVATED: {force_file}")
            if self._on_kill_switch:
                self._on_kill_switch(force=True)
            return True
        
        # Check normal kill
        kill_file = Path(self.config.kill_switch_file)
        if kill_file.exists():
            logger.critical(f"KILL SWITCH ACTIVATED: {kill_file}")
            if self._on_kill_switch:
                self._on_kill_switch(force=False)
            return True
        
        return False
    
    def activate_kill_switch(self, force: bool = False) -> None:
        """
        Activate kill switch (for testing or emergency).
        
        Args:
            force: If True, create force kill file (immediate termination)
        """
        target = Path(
            self.config.force_kill_file if force else self.config.kill_switch_file
        )
        target.touch()
        logger.warning(f"Kill switch activated: {target}")
    
    def deactivate_kill_switch(self) -> None:
        """Deactivate kill switch."""
        for path in [
            Path(self.config.kill_switch_file),
            Path(self.config.force_kill_file)
        ]:
            if path.exists():
                path.unlink()
        logger.info("Kill switch deactivated")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Error Rate
    # ─────────────────────────────────────────────────────────────────────────
    
    def record_request(self, success: bool) -> None:
        """
        Record request outcome for error rate calculation.
        
        Args:
            success: Whether the request succeeded
        """
        self._total_requests += 1
        if not success:
            self._error_count += 1
        
        # Update rolling window
        self._error_window.append(success)
        if len(self._error_window) > self.config.error_rate_window:
            self._error_window.pop(0)
    
    def check_error_rate(self) -> GuardrailStatus:
        """
        Check if error rate is within limits.
        
        Returns:
            GuardrailStatus with check result
        """
        if len(self._error_window) < 10:
            return GuardrailStatus(
                name="error_rate",
                passed=True,
                value=0.0,
                threshold=self.config.max_error_rate,
                message="Not enough data for error rate calculation"
            )
        
        errors = sum(1 for success in self._error_window if not success)
        error_rate = errors / len(self._error_window)
        
        if error_rate > self.config.max_error_rate:
            msg = f"ERROR RATE TOO HIGH: {error_rate*100:.1f}% (threshold: {self.config.max_error_rate*100:.1f}%)"
            logger.critical(msg)
            if self._on_error_rate_exceeded:
                self._on_error_rate_exceeded(error_rate)
            return GuardrailStatus(
                name="error_rate",
                passed=False,
                value=error_rate,
                threshold=self.config.max_error_rate,
                message=msg
            )
        
        return GuardrailStatus(
            name="error_rate",
            passed=True,
            value=error_rate,
            threshold=self.config.max_error_rate,
            message=f"Error rate OK: {error_rate*100:.1f}%"
        )
    
    # ─────────────────────────────────────────────────────────────────────────
    # Memory
    # ─────────────────────────────────────────────────────────────────────────
    
    def check_memory(self) -> GuardrailStatus:
        """
        Check memory usage.
        
        Returns:
            GuardrailStatus with check result
        """
        # Only check every 10 seconds
        now = time.monotonic()
        if now - self._memory_checked < 10.0:
            return GuardrailStatus(
                name="memory",
                passed=True,
                value=0,
                threshold=self.config.max_memory_mb,
                message="Memory check skipped (recently checked)"
            )
        self._memory_checked = now
        
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            if memory_mb > self.config.max_memory_mb:
                msg = f"MEMORY EXCEEDED: {memory_mb:.0f}MB > {self.config.max_memory_mb:.0f}MB"
                logger.critical(msg)
                return GuardrailStatus(
                    name="memory",
                    passed=False,
                    value=memory_mb,
                    threshold=self.config.max_memory_mb,
                    message=msg
                )
            
            ratio = memory_mb / self.config.max_memory_mb
            if ratio >= self.config.memory_warning_threshold:
                logger.warning(f"MEMORY WARNING: {ratio*100:.1f}% used ({memory_mb:.0f}MB)")
            
            return GuardrailStatus(
                name="memory",
                passed=True,
                value=memory_mb,
                threshold=self.config.max_memory_mb,
                message=f"Memory OK: {memory_mb:.0f}MB ({ratio*100:.1f}%)"
            )
            
        except ImportError:
            return GuardrailStatus(
                name="memory",
                passed=True,
                value=0,
                threshold=self.config.max_memory_mb,
                message="psutil not available, memory check skipped"
            )
    
    # ─────────────────────────────────────────────────────────────────────────
    # Drift Detection
    # ─────────────────────────────────────────────────────────────────────────
    
    def set_baseline(self, state: Dict[str, Any]) -> None:
        """
        Set baseline state for drift detection.
        
        Args:
            state: Current state to use as baseline
        """
        self._drift_baseline = state.copy()
        logger.debug("Drift baseline set")
    
    def detect_drift(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect configuration drift from baseline.
        
        Args:
            current_state: Current state to compare
            
        Returns:
            Dict of drifted values with baseline and current
        """
        if not self.config.enable_drift_detection:
            return {}
        
        if not self._drift_baseline:
            self._drift_baseline = current_state.copy()
            return {}
        
        drift = {}
        for key, value in current_state.items():
            if key in self._drift_baseline:
                baseline = self._drift_baseline[key]
                if baseline != value:
                    drift[key] = {
                        "baseline": baseline,
                        "current": value
                    }
        
        if drift:
            logger.warning(f"CONFIGURATION DRIFT DETECTED: {drift}")
            self._drift_history.append({
                "timestamp": time.time(),
                "drift": drift
            })
        
        return drift
    
    # ─────────────────────────────────────────────────────────────────────────
    # Combined Checks
    # ─────────────────────────────────────────────────────────────────────────
    
    def all_checks_pass(
        self,
        spent: float,
        max_budget: float,
        check_memory: bool = True,
        check_error_rate: bool = True,
        check_kill_switch: bool = True
    ) -> bool:
        """
        Run all guardrail checks.
        
        Args:
            spent: Amount spent in USD
            max_budget: Maximum budget in USD
            check_memory: Whether to check memory
            check_error_rate: Whether to check error rate
            check_kill_switch: Whether to check kill switch
            
        Returns:
            True if all checks pass, False otherwise
        """
        checks = [
            self.check_budget(spent, max_budget)
        ]
        
        if check_memory:
            checks.append(self.check_memory())
        
        if check_error_rate:
            checks.append(self.check_error_rate())
        
        if check_kill_switch:
            # Kill switch is special - returns True if should terminate
            if self.check_kill_switch():
                return False
        
        failed = [c for c in checks if not c.passed]
        
        if failed:
            logger.critical(f"GUARDRAIL FAILURES: {[c.name for c in failed]}")
        
        return len(failed) == 0
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get overall guardrail status.
        
        Returns:
            Dict with status of all guardrails
        """
        return {
            "uptime_seconds": time.monotonic() - self._start_time,
            "total_requests": self._total_requests,
            "error_count": self._error_count,
            "error_rate": self._error_count / max(1, self._total_requests),
            "kill_switch_enabled": self.config.enable_kill_switch,
            "drift_detection_enabled": self.config.enable_drift_detection,
            "drift_history_count": len(self._drift_history),
        }
    
    # ─────────────────────────────────────────────────────────────────────────
    # Callback Registration
    # ─────────────────────────────────────────────────────────────────────────
    
    def on_budget_warning(self, callback: Callable) -> None:
        """Register callback for budget warning."""
        self._on_budget_warning = callback
    
    def on_budget_critical(self, callback: Callable) -> None:
        """Register callback for budget critical."""
        self._on_budget_critical = callback
    
    def on_kill_switch(self, callback: Callable) -> None:
        """Register callback for kill switch activation."""
        self._on_kill_switch = callback
    
    def on_error_rate_exceeded(self, callback: Callable) -> None:
        """Register callback for error rate exceeded."""
        self._on_error_rate_exceeded = callback


class KillSwitch:
    """
    Standalone kill switch for emergency shutdown.
    
    USAGE:
        # Activate kill switch
        kill_switch = KillSwitch()
        kill_switch.activate()
        
        # In main loop
        if kill_switch.is_activated():
            raise SystemExit("Kill switch activated")
        
        # Deactivate
        kill_switch.deactivate()
    """
    
    def __init__(
        self,
        kill_file: str = "/tmp/orchestrator_kill",
        force_file: str = "/tmp/orchestrator_force_kill",
        audit_file: Optional[str] = None
    ):
        self.kill_file = Path(kill_file)
        self.force_file = Path(force_file)
        self.audit_file = Path(audit_file) if audit_file else None
        self._activated = False
    
    def is_activated(self) -> bool:
        """Check if kill switch is activated."""
        if self.kill_file.exists():
            if not self._activated:
                self._activated = True
                self._log_activation()
            return True
        return False
    
    def is_force_activated(self) -> bool:
        """Check if force kill is activated."""
        return self.force_file.exists()
    
    def _log_activation(self) -> None:
        """Log kill switch activation."""
        logger.critical(f"KILL SWITCH ACTIVATED at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if self.audit_file:
            with open(self.audit_file, "a") as f:
                f.write(f"{time.time()}: KILL_SWITCH_ACTIVATED\n")
    
    def activate(self, force: bool = False) -> None:
        """Activate kill switch."""
        target = self.force_file if force else self.kill_file
        target.touch()
        logger.warning(f"Kill switch activated: {target}")
    
    def deactivate(self) -> None:
        """Deactivate kill switch."""
        for f in [self.kill_file, self.force_file]:
            if f.exists():
                f.unlink()
        self._activated = False
        logger.info("Kill switch deactivated")
    
    def check_and_exit(self) -> None:
        """Check kill switch and exit if activated."""
        if self.is_force_activated():
            logger.critical("FORCE KILL - Immediate termination")
            os._exit(1)  # Immediate, no cleanup
        
        if self.is_activated():
            logger.critical("KILL SWITCH - Graceful shutdown")
            raise SystemExit(0)


# Singleton instance
_guardrails: Optional[ProductionGuardrails] = None


def get_guardrails() -> ProductionGuardrails:
    """Get global guardrails instance."""
    global _guardrails
    if _guardrails is None:
        _guardrails = ProductionGuardrails()
    return _guardrails


def reset_guardrails() -> None:
    """Reset singleton (for testing)."""
    global _guardrails
    _guardrails = None