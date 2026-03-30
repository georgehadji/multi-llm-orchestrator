"""
System Diagnostics Module
=========================
Automated diagnostic tools for troubleshooting.

Usage:
    from orchestrator.diagnostics import SystemDiagnostic

    diag = SystemDiagnostic()
    report = await diag.run_full_check()
    print(report.health_status)
"""
from __future__ import annotations

import asyncio
import importlib
import os
import socket
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .log_config import get_logger
from .models import Model

logger = get_logger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"


class Severity(Enum):
    """Issue severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Issue:
    """Diagnostic issue."""
    component: str
    description: str
    severity: Severity
    suggested_fix: str
    error_code: str | None = None


@dataclass
class DiagnosticReport:
    """Complete diagnostic report."""
    health_status: HealthStatus
    issues: list[Issue] = field(default_factory=list)
    checks_passed: int = 0
    checks_failed: int = 0
    timestamp: str = field(default_factory=lambda: __import__('datetime').datetime.now().isoformat())

    @property
    def is_healthy(self) -> bool:
        return self.health_status == HealthStatus.HEALTHY


class SystemDiagnostic:
    """
    Comprehensive system diagnostic tool.

    Checks:
    - Environment configuration
    - API key validity
    - Network connectivity
    - Dependencies
    - Disk space
    - Performance metrics
    """

    def __init__(self):
        self.issues: list[Issue] = []
        self.checks_passed = 0
        self.checks_failed = 0

    async def run_full_check(self) -> DiagnosticReport:
        """Run all diagnostic checks."""
        logger.info("Running full system diagnostic...")

        await self._check_environment()
        await self._check_api_keys()
        await self._check_network()
        await self._check_dependencies()
        await self._check_disk_space()
        await self._check_performance()

        # Determine overall health
        critical_count = sum(1 for i in self.issues if i.severity == Severity.CRITICAL)
        error_count = sum(1 for i in self.issues if i.severity == Severity.ERROR)

        if critical_count > 0:
            health = HealthStatus.CRITICAL
        elif error_count > 0:
            health = HealthStatus.DEGRADED
        else:
            health = HealthStatus.HEALTHY

        report = DiagnosticReport(
            health_status=health,
            issues=self.issues,
            checks_passed=self.checks_passed,
            checks_failed=self.checks_failed,
        )

        logger.info(f"Diagnostic complete: {health.value}")
        return report

    async def _check_environment(self):
        """Check environment variables."""
        logger.debug("Checking environment variables...")

        required_vars = [
            "OPENAI_API_KEY",
            "GOOGLE_API_KEY",
            "ANTHROPIC_API_KEY",
            "DEEPSEEK_API_KEY",
            "MINIMAX_API_KEY",
        ]

        found_keys = []
        for var in required_vars:
            if os.getenv(var):
                found_keys.append(var)

        if not found_keys:
            self.issues.append(Issue(
                component="environment",
                description="No API keys found. At least one provider key is required.",
                severity=Severity.CRITICAL,
                suggested_fix="Set at least one API key: export OPENAI_API_KEY=sk-...",
                error_code="ENV001",
            ))
            self.checks_failed += 1
        elif len(found_keys) < 2:
            self.issues.append(Issue(
                component="environment",
                description=f"Only one API key found ({found_keys[0]}). Multiple providers recommended for fallback.",
                severity=Severity.WARNING,
                suggested_fix="Add at least one more API key for redundancy.",
                error_code="ENV002",
            ))
            self.checks_passed += 1
        else:
            self.checks_passed += 1

        # Check optional vars
        if not os.getenv("ORCHESTRATOR_CACHE_DIR"):
            self.issues.append(Issue(
                component="environment",
                description="ORCHESTRATOR_CACHE_DIR not set. Using default.",
                severity=Severity.INFO,
                suggested_fix="Optional: Set ORCHESTRATOR_CACHE_DIR=~/.orchestrator_cache",
            ))

    async def _check_api_keys(self):
        """Test API key validity."""
        logger.debug("Checking API key validity...")

        from .api_clients import UnifiedClient

        test_models = [
            (Model.GPT_4O_MINI, "OpenAI"),
            (Model.DEEPSEEK_CHAT, "DeepSeek"),
            (Model.GEMINI_FLASH, "Google"),
        ]

        working_providers = []

        for model, provider_name in test_models:
            try:
                client = UnifiedClient(model)
                # Simple test call
                await client.generate("Hello", max_tokens=5)
                working_providers.append(provider_name)
            except Exception as e:
                if "authentication" in str(e).lower() or "api key" in str(e).lower():
                    self.issues.append(Issue(
                        component=f"api:{provider_name}",
                        description=f"{provider_name} API key invalid or expired.",
                        severity=Severity.ERROR,
                        suggested_fix=f"Check {provider_name.upper()}_API_KEY environment variable.",
                        error_code="API001",
                    ))
                elif "rate limit" in str(e).lower():
                    self.issues.append(Issue(
                        component=f"api:{provider_name}",
                        description=f"{provider_name} rate limit hit.",
                        severity=Severity.WARNING,
                        suggested_fix="Wait before retrying or upgrade plan.",
                    ))

        if working_providers:
            self.checks_passed += 1
        else:
            self.checks_failed += 1

    async def _check_network(self):
        """Check network connectivity."""
        logger.debug("Checking network connectivity...")

        hosts_to_check = [
            ("api.openai.com", 443),
            ("api.deepseek.com", 443),
            ("generativelanguage.googleapis.com", 443),
        ]

        for host, port in hosts_to_check:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                result = sock.connect_ex((host, port))
                sock.close()

                if result != 0:
                    self.issues.append(Issue(
                        component="network",
                        description=f"Cannot connect to {host}:{port}",
                        severity=Severity.ERROR,
                        suggested_fix="Check internet connection and firewall settings.",
                        error_code="NET001",
                    ))
                    self.checks_failed += 1
                else:
                    self.checks_passed += 1
            except Exception as e:
                self.issues.append(Issue(
                    component="network",
                    description=f"Network check failed for {host}: {e}",
                    severity=Severity.WARNING,
                    suggested_fix="Check DNS and network connectivity.",
                ))

    async def _check_dependencies(self):
        """Check Python dependencies."""
        logger.debug("Checking dependencies...")

        required_packages = [
            ("pydantic", "2.0.0"),
            ("openai", "1.0.0"),
            ("aiosqlite", "0.19.0"),
        ]

        optional_packages = [
            ("fastapi", "Dashboard"),
            ("uvicorn", "Dashboard"),
            ("pytest", "Testing"),
            ("ruff", "Linting"),
            ("redis", "Caching"),
        ]

        for package, min_version in required_packages:
            try:
                importlib.import_module(package)
                self.checks_passed += 1
            except ImportError:
                self.issues.append(Issue(
                    component="dependencies",
                    description=f"Required package '{package}' not installed.",
                    severity=Severity.CRITICAL,
                    suggested_fix=f"pip install {package}>={min_version}",
                    error_code="DEP001",
                ))
                self.checks_failed += 1

        for package, feature in optional_packages:
            try:
                importlib.import_module(package)
            except ImportError:
                self.issues.append(Issue(
                    component="dependencies",
                    description=f"Optional package '{package}' not installed.",
                    severity=Severity.INFO,
                    suggested_fix=f"pip install {package} (for {feature})",
                ))

    async def _check_disk_space(self):
        """Check available disk space."""
        logger.debug("Checking disk space...")

        try:
            import shutil

            # Check current directory
            total, used, free = shutil.disk_usage(".")
            free_gb = free / (1024**3)

            if free_gb < 1:  # Less than 1GB
                self.issues.append(Issue(
                    component="disk",
                    description=f"Low disk space: {free_gb:.1f}GB remaining",
                    severity=Severity.WARNING,
                    suggested_fix="Free up disk space or change output directory.",
                    error_code="DISK001",
                ))
            else:
                self.checks_passed += 1
        except Exception as e:
            logger.warning(f"Could not check disk space: {e}")

    async def _check_performance(self):
        """Check performance metrics."""
        logger.debug("Checking performance...")

        # Check cache
        try:
            from .performance import get_cache
            cache = get_cache()
            stats = cache.get_stats()

            hit_rate = stats.get("hit_rate", "0%")
            if hit_rate != "0%":
                hit_rate_val = float(hit_rate.rstrip("%")) / 100
                if hit_rate_val < 0.5:
                    self.issues.append(Issue(
                        component="performance",
                        description=f"Low cache hit rate: {hit_rate}",
                        severity=Severity.INFO,
                        suggested_fix="Consider increasing cache TTL or review cache usage.",
                    ))
        except Exception:
            pass  # Cache not initialized

        self.checks_passed += 1


class ProjectDiagnostic:
    """Diagnose issues with a specific project."""

    def __init__(self, project_id: str):
        self.project_id = project_id

    async def diagnose(self) -> dict[str, Any]:
        """Diagnose specific project issues."""
        from .state import StateManager

        state_mgr = StateManager()
        state = state_mgr.load_state(self.project_id)

        if not state:
            return {"error": f"Project {self.project_id} not found"}

        issues = []

        # Check for failed tasks
        failed_tasks = [
            task_id for task_id, result in state.results.items()
            if result.error or result.score < 0.5
        ]

        if failed_tasks:
            issues.append({
                "type": "failed_tasks",
                "count": len(failed_tasks),
                "tasks": failed_tasks,
                "suggestion": "Review task logs and consider retrying with different models",
            })

        # Check budget
        if state.budget.spent_usd > state.budget.max_usd * 0.9:
            issues.append({
                "type": "budget_warning",
                "spent": state.budget.spent_usd,
                "max": state.budget.max_usd,
                "suggestion": "Project near budget limit. Consider increasing budget or optimizing.",
            })

        # Check completion
        completion = state.completion_percentage
        if completion < 100 and state.status.value == "in_progress":
            issues.append({
                "type": "incomplete",
                "completion": completion,
                "suggestion": f"Project incomplete ({completion:.1f}%). Resume with --resume {self.project_id}",
            })

        return {
            "project_id": self.project_id,
            "status": state.status.value,
            "completion": completion,
            "budget_spent": f"${state.budget.spent_usd:.2f}",
            "issues": issues,
            "is_healthy": len(issues) == 0,
        }


def print_diagnostic_report(report: DiagnosticReport):
    """Pretty print diagnostic report."""
    print("\n" + "="*70)
    print(" 🔍 SYSTEM DIAGNOSTIC REPORT")
    print("="*70)

    # Status
    status_color = {
        HealthStatus.HEALTHY: "🟢",
        HealthStatus.DEGRADED: "🟡",
        HealthStatus.CRITICAL: "🔴",
    }
    print(f"\n{status_color[report.health_status]} Overall Status: {report.health_status.value.upper()}")
    print(f"   Checks: {report.checks_passed} passed, {report.checks_failed} failed")

    # Issues
    if report.issues:
        print(f"\n📋 Issues Found ({len(report.issues)}):")

        for issue in report.issues:
            severity_icon = {
                Severity.CRITICAL: "🔴",
                Severity.ERROR: "❌",
                Severity.WARNING: "⚠️ ",
                Severity.INFO: "ℹ️ ",
            }

            print(f"\n{severity_icon[issue.severity]} [{issue.severity.value.upper()}] {issue.component}")
            print(f"   Problem: {issue.description}")
            print(f"   Fix: {issue.suggested_fix}")
            if issue.error_code:
                print(f"   Code: {issue.error_code}")
    else:
        print("\n✅ No issues found!")

    print("\n" + "="*70)


# CLI interface
if __name__ == "__main__":
    import asyncio

    async def main():
        diag = SystemDiagnostic()
        report = await diag.run_full_check()
        print_diagnostic_report(report)

        sys.exit(0 if report.is_healthy else 1)

    asyncio.run(main())
