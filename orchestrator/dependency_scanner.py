"""
Dependency Security Scanner — Adapter + Strategy Pattern
=========================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Dependency vulnerability scanning using Adapter Pattern for different
package managers and Strategy Pattern for scanning algorithms.

Paradigm: OOP with Functional utilities
Patterns: Adapter, Strategy, Repository

Usage:
    from orchestrator.safety.dependency_scanner import DependencyScannerContext, NpmAdapter

    scanner = DependencyScannerContext(NpmAdapter())
    vulnerabilities = scanner.scan_project("/path/to/project")
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
import subprocess
import json

# ═══════════════════════════════════════════════════════════════════
# IMMUTABLE DATA CLASSES
# ═══════════════════════════════════════════════════════════════════


class VulnerabilitySeverity(str, Enum):
    """Vulnerability severity levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass(frozen=True)
class Vulnerability:
    """
    Immutable vulnerability record.

    Attributes:
        package: Package name
        severity: Vulnerability severity
        cve_id: CVE identifier (if available)
        description: Vulnerability description
        fix_version: Version that fixes the vulnerability
        current_version: Currently installed version
        cwes: List of CWE identifiers
    """

    package: str
    severity: VulnerabilitySeverity
    cve_id: Optional[str] = None
    description: str = ""
    fix_version: Optional[str] = None
    current_version: Optional[str] = None
    cwes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "package": self.package,
            "severity": self.severity.value,
            "cve_id": self.cve_id,
            "description": self.description,
            "fix_version": self.fix_version,
            "current_version": self.current_version,
            "cwes": self.cwes,
        }


@dataclass(frozen=True)
class ScanResult:
    """
    Immutable scan result.

    Attributes:
        project_path: Scanned project path
        vulnerabilities: List of vulnerabilities found
        total_dependencies: Total number of dependencies
        vulnerable_dependencies: Number of vulnerable dependencies
        scan_time: Scan time in seconds
    """

    project_path: str
    vulnerabilities: List[Vulnerability]
    total_dependencies: int
    vulnerable_dependencies: int
    scan_time: float

    @property
    def is_safe(self) -> bool:
        """Check if project is safe (no critical/high vulnerabilities)."""
        critical_or_high = {VulnerabilitySeverity.CRITICAL, VulnerabilitySeverity.HIGH}
        return not any(v.severity in critical_or_high for v in self.vulnerabilities)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "project_path": self.project_path,
            "vulnerabilities": [v.to_dict() for v in self.vulnerabilities],
            "total_dependencies": self.total_dependencies,
            "vulnerable_dependencies": self.vulnerable_dependencies,
            "scan_time": self.scan_time,
            "is_safe": self.is_safe,
        }


# ═══════════════════════════════════════════════════════════════════
# STRATEGY PATTERN — SCANNER INTERFACE
# ═══════════════════════════════════════════════════════════════════


class DependencyScanner(ABC):
    """
    Strategy Pattern for different package managers.

    Defines common interface for all dependency scanners.
    """

    @abstractmethod
    def scan(self, project_path: str) -> ScanResult:
        """
        Scan project for vulnerabilities.

        Args:
            project_path: Path to project root

        Returns:
            ScanResult with vulnerabilities
        """
        pass

    @abstractmethod
    def parse_results(self, output: str) -> List[Vulnerability]:
        """
        Parse scanner output.

        Args:
            output: Raw scanner output

        Returns:
            List of Vulnerability objects
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if scanner is available (installed).

        Returns:
            True if scanner is available
        """
        pass


# ═══════════════════════════════════════════════════════════════════
# ADAPTER PATTERN — NPM SCANNER
# ═══════════════════════════════════════════════════════════════════


class NpmAdapter(DependencyScanner):
    """
    Adapter for npm audit (Node.js projects).

    Adapts npm audit JSON output to common Vulnerability interface.
    """

    def __init__(self):
        self._command = ["npm", "audit", "--json"]

    def is_available(self) -> bool:
        """Check if npm is installed."""
        try:
            subprocess.run(["npm", "--version"], capture_output=True, check=True, timeout=10)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def scan(self, project_path: str) -> ScanResult:
        """
        Run npm audit and parse results.

        Args:
            project_path: Path to Node.js project

        Returns:
            ScanResult with vulnerabilities
        """
        import time

        start_time = time.time()

        try:
            result = subprocess.run(
                self._command, cwd=project_path, capture_output=True, text=True, timeout=120
            )

            # npm audit returns non-zero exit code if vulnerabilities found
            # but still provides JSON output
            output = result.stdout

            if not output.strip():
                return ScanResult(
                    project_path=project_path,
                    vulnerabilities=[],
                    total_dependencies=0,
                    vulnerable_dependencies=0,
                    scan_time=time.time() - start_time,
                )

            # Parse JSON output
            data = json.loads(output)
            vulnerabilities = self.parse_results(json.dumps(data))

            # Get dependency count from package-lock.json
            total_deps = data.get("metadata", {}).get("dependencies", 0)

            return ScanResult(
                project_path=project_path,
                vulnerabilities=vulnerabilities,
                total_dependencies=total_deps,
                vulnerable_dependencies=len(vulnerabilities),
                scan_time=time.time() - start_time,
            )

        except subprocess.TimeoutExpired:
            return ScanResult(
                project_path=project_path,
                vulnerabilities=[],
                total_dependencies=0,
                vulnerable_dependencies=0,
                scan_time=time.time() - start_time,
            )
        except json.JSONDecodeError:
            # No vulnerabilities or invalid JSON
            return ScanResult(
                project_path=project_path,
                vulnerabilities=[],
                total_dependencies=0,
                vulnerable_dependencies=0,
                scan_time=time.time() - start_time,
            )

    def parse_results(self, output: str) -> List[Vulnerability]:
        """
        Parse npm audit JSON output.

        Args:
            output: JSON output from npm audit

        Returns:
            List of Vulnerability objects
        """
        vulnerabilities = []

        try:
            data = json.loads(output)
            vulnerabilities_data = data.get("vulnerabilities", {})

            for package, vuln_info in vulnerabilities_data.items():
                # npm audit v2+ structure
                if isinstance(vuln_info, dict):
                    for via in vuln_info.get("via", []):
                        if isinstance(via, dict):  # Skip string references
                            severity_str = via.get("severity", "info").lower()
                            severity = (
                                VulnerabilitySeverity(severity_str)
                                if severity_str in [s.value for s in VulnerabilitySeverity]
                                else VulnerabilitySeverity.INFO
                            )

                            vulnerabilities.append(
                                Vulnerability(
                                    package=package,
                                    severity=severity,
                                    cve_id=via.get("cwe"),  # npm uses CWE instead of CVE
                                    description=via.get("title", ""),
                                    fix_version=via.get("fixVersion"),
                                    current_version=vuln_info.get("version"),
                                    cwes=[via.get("cwe")] if via.get("cwe") else [],
                                )
                            )
        except json.JSONDecodeError:
            pass

        return vulnerabilities


# ═══════════════════════════════════════════════════════════════════
# ADAPTER PATTERN — PIP SCANNER
# ═══════════════════════════════════════════════════════════════════


class PipAdapter(DependencyScanner):
    """
    Adapter for pip safety check (Python projects).

    Adapts pip-audit or safety output to common Vulnerability interface.
    """

    def __init__(self):
        self._command = ["safety", "check", "--json"]

    def is_available(self) -> bool:
        """Check if safety is installed."""
        try:
            subprocess.run(["safety", "--version"], capture_output=True, check=True, timeout=10)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def scan(self, project_path: str) -> ScanResult:
        """
        Run safety check and parse results.

        Args:
            project_path: Path to Python project

        Returns:
            ScanResult with vulnerabilities
        """
        import time

        start_time = time.time()

        try:
            result = subprocess.run(
                self._command, cwd=project_path, capture_output=True, text=True, timeout=120
            )

            output = result.stdout

            if not output.strip():
                return ScanResult(
                    project_path=project_path,
                    vulnerabilities=[],
                    total_dependencies=0,
                    vulnerable_dependencies=0,
                    scan_time=time.time() - start_time,
                )

            vulnerabilities = self.parse_results(output)

            # Count dependencies from requirements.txt
            total_deps = self._count_dependencies(project_path)

            return ScanResult(
                project_path=project_path,
                vulnerabilities=vulnerabilities,
                total_dependencies=total_deps,
                vulnerable_dependencies=len(vulnerabilities),
                scan_time=time.time() - start_time,
            )

        except (subprocess.TimeoutExpired, FileNotFoundError):
            return ScanResult(
                project_path=project_path,
                vulnerabilities=[],
                total_dependencies=0,
                vulnerable_dependencies=0,
                scan_time=time.time() - start_time,
            )

    def _count_dependencies(self, project_path: str) -> int:
        """Count dependencies from requirements.txt."""
        import os

        req_file = os.path.join(project_path, "requirements.txt")

        if not os.path.exists(req_file):
            return 0

        with open(req_file, "r") as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
            return len(lines)

    def parse_results(self, output: str) -> List[Vulnerability]:
        """Parse safety JSON output."""
        vulnerabilities = []

        try:
            data = json.loads(output)

            for vuln in data:
                # Safety output format
                package = vuln.get("package_name", "")
                severity_str = vuln.get("severity", "medium").lower()
                severity = (
                    VulnerabilitySeverity(severity_str)
                    if severity_str in [s.value for s in VulnerabilitySeverity]
                    else VulnerabilitySeverity.MEDIUM
                )

                vulnerabilities.append(
                    Vulnerability(
                        package=package,
                        severity=severity,
                        cve_id=vuln.get("cve_id"),
                        description=vuln.get("advisory", ""),
                        fix_version=(
                            vuln.get("remediation", "").split("->")[-1].strip()
                            if "->" in vuln.get("remediation", "")
                            else None
                        ),
                        current_version=vuln.get("analyzed_version"),
                        cwes=[],
                    )
                )
        except json.JSONDecodeError:
            pass

        return vulnerabilities


# ═══════════════════════════════════════════════════════════════════
# CONTEXT CLASS — USES STRATEGY
# ═══════════════════════════════════════════════════════════════════


class DependencyScannerContext:
    """
    Context class that uses Strategy pattern.

    Allows switching scanners at runtime.

    Usage:
        context = DependencyScannerContext(NpmAdapter())
        result = context.scan_project("/path/to/node/project")

        # Switch strategy
        context.scanner = PipAdapter()
        result = context.scan_project("/path/to/python/project")
    """

    def __init__(self, scanner: DependencyScanner):
        """
        Initialize with scanner strategy.

        Args:
            scanner: Scanner strategy to use
        """
        self._scanner = scanner

    @property
    def scanner(self) -> DependencyScanner:
        """Get current scanner."""
        return self._scanner

    @scanner.setter
    def scanner(self, scanner: DependencyScanner) -> None:
        """Set new scanner strategy."""
        self._scanner = scanner

    def scan_project(self, project_path: str) -> ScanResult:
        """
        Scan project using current scanner.

        Args:
            project_path: Path to project

        Returns:
            ScanResult with vulnerabilities
        """
        return self._scanner.scan(project_path)

    def get_vulnerabilities_by_severity(
        self, project_path: str
    ) -> Dict[VulnerabilitySeverity, List[Vulnerability]]:
        """
        Get vulnerabilities grouped by severity.

        Args:
            project_path: Path to project

        Returns:
            Dictionary mapping severity to vulnerabilities
        """
        result = self.scan_project(project_path)

        grouped = {}
        for vuln in result.vulnerabilities:
            if vuln.severity not in grouped:
                grouped[vuln.severity] = []
            grouped[vuln.severity].append(vuln)

        return grouped

    def has_critical_vulnerabilities(self, project_path: str) -> bool:
        """
        Check if project has critical vulnerabilities.

        Args:
            project_path: Path to project

        Returns:
            True if critical vulnerabilities found
        """
        result = self.scan_project(project_path)
        return any(v.severity == VulnerabilitySeverity.CRITICAL for v in result.vulnerabilities)


# ═══════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════


def scan_npm_project(project_path: str) -> ScanResult:
    """
    Scan Node.js project for vulnerabilities.

    Args:
        project_path: Path to project

    Returns:
        ScanResult with vulnerabilities
    """
    context = DependencyScannerContext(NpmAdapter())
    return context.scan_project(project_path)


def scan_python_project(project_path: str) -> ScanResult:
    """
    Scan Python project for vulnerabilities.

    Args:
        project_path: Path to project

    Returns:
        ScanResult with vulnerabilities
    """
    context = DependencyScannerContext(PipAdapter())
    return context.scan_project(project_path)


def auto_scan_project(project_path: str) -> ScanResult:
    """
    Automatically detect project type and scan.

    Args:
        project_path: Path to project

    Returns:
        ScanResult with vulnerabilities
    """
    import os

    # Detect project type
    if os.path.exists(os.path.join(project_path, "package.json")):
        return scan_npm_project(project_path)
    elif os.path.exists(os.path.join(project_path, "requirements.txt")):
        return scan_python_project(project_path)
    else:
        return ScanResult(
            project_path=project_path,
            vulnerabilities=[],
            total_dependencies=0,
            vulnerable_dependencies=0,
            scan_time=0.0,
        )
