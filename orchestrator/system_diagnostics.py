"""
SystemDiagnostics - Real-time build/install/test/error status.
================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis
Part of Cat 7, Phase D6 (Dyad-inspired).
"""

from __future__ import annotations
from dataclasses import dataclass, field
import subprocess, time, logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class DiagnosticCheck:
    name: str
    status: str = "pending"
    message: str = ""
    duration_ms: float = 0.0
    details: str = ""


@dataclass
class DiagnosticsReport:
    checks: list = field(default_factory=list)
    overall_status: str = "pending"

    @property
    def is_healthy(self):
        return all(c.status == "ok" for c in self.checks)

    def to_dict(self):
        return {
            "overall": self.overall_status,
            "is_healthy": self.is_healthy,
            "checks": [
                {
                    "name": c.name,
                    "status": c.status,
                    "message": c.message,
                    "duration_ms": c.duration_ms,
                    "details": c.details,
                }
                for c in self.checks
            ],
        }


class SystemDiagnostics:
    """Runs real-time system health checks."""

    def __init__(self, project_dir="."):
        self._dir = Path(project_dir)

    async def run_all(self):
        checks = []
        checks.append(self._check_python())
        checks.append(self._check_dependencies())
        checks.append(self._check_git())
        checks.append(self._check_tests())
        overall = "ok" if all(c.status == "ok" for c in checks) else "warning"
        return DiagnosticsReport(checks=checks, overall_status=overall)

    def _check_python(self):
        import sys

        t0 = time.monotonic()
        return DiagnosticCheck(
            name="python",
            status="ok",
            message=f"Python {sys.version.split()[0]}",
            duration_ms=(time.monotonic() - t0) * 1000,
        )

    def _check_dependencies(self):
        t0 = time.monotonic()
        try:
            subprocess.run(
                ["pip", "list", "--format=json"], capture_output=True, text=True, timeout=15
            )
            return DiagnosticCheck(
                name="dependencies",
                status="ok",
                message="pip packages installed",
                duration_ms=(time.monotonic() - t0) * 1000,
            )
        except Exception as e:
            return DiagnosticCheck(name="dependencies", status="failed", message=str(e))

    def _check_git(self):
        t0 = time.monotonic()
        try:
            r = subprocess.run(
                ["git", "status", "--short"],
                cwd=str(self._dir),
                capture_output=True,
                text=True,
                timeout=10,
            )
            changes = len([l for l in r.stdout.strip().split(chr(10)) if l])
            return DiagnosticCheck(
                name="git",
                status="ok",
                message=f"{changes} changes" if changes else "Clean",
                duration_ms=(time.monotonic() - t0) * 1000,
                details=r.stdout[:500],
            )
        except Exception:
            return DiagnosticCheck(name="git", status="warning", message="Not a git repo")

    def _check_tests(self):
        t0 = time.monotonic()
        try:
            r = subprocess.run(
                ["pytest", "--co", "-q"],
                cwd=str(self._dir),
                capture_output=True,
                text=True,
                timeout=20,
            )
            count = 0
            for l in r.stdout.split(chr(10)):
                if "selected" in l:
                    try:
                        count = int(l.split()[0])
                    except:
                        pass
            return DiagnosticCheck(
                name="tests",
                status="ok" if r.returncode == 0 else "warning",
                message=f"{count} tests",
                duration_ms=(time.monotonic() - t0) * 1000,
                details=r.stdout[:500],
            )
        except Exception as e:
            return DiagnosticCheck(name="tests", status="warning", message=str(e))
