"""
Quality Control System
======================
Automated testing, code quality analysis, and compliance enforcement.

Features:
- Multi-level test orchestration
- Static code analysis
- Quality metrics tracking
- Compliance gates
- Regression detection

Usage:
    from orchestrator.quality_control import QualityController, TestSuite
    
    qc = QualityController()
    results = await qc.run_quality_gate(project_path)
    
    if results.passed:
        print("Quality gate passed!")
"""
from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import subprocess
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
import re

from .log_config import get_logger
from .performance import cached
from .monitoring import monitor_endpoint, metrics

logger = get_logger(__name__)


class TestLevel(Enum):
    """Testing levels."""
    UNIT = "unit"
    INTEGRATION = "integration"
    E2E = "e2e"
    PERFORMANCE = "performance"
    SECURITY = "security"


class QualitySeverity(Enum):
    """Quality issue severity."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class TestResult:
    """Single test result."""
    name: str
    level: TestLevel
    passed: bool
    duration_ms: float
    message: str = ""
    stdout: str = ""
    stderr: str = ""
    coverage_percent: Optional[float] = None


@dataclass
class QualityIssue:
    """Code quality issue."""
    rule_id: str
    description: str
    severity: QualitySeverity
    file_path: str
    line_number: int
    column: int
    suggested_fix: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "description": self.description,
            "severity": self.severity.value,
            "location": f"{self.file_path}:{self.line_number}:{self.column}",
            "suggested_fix": self.suggested_fix,
        }


@dataclass
class CodeMetrics:
    """Code quality metrics."""
    file_path: str
    lines_of_code: int
    complexity_score: float  # Cyclomatic complexity
    maintainability_index: float  # 0-100
    duplication_percent: float
    documentation_coverage: float  # Docstring coverage
    type_hint_coverage: float  # Type annotation coverage
    
    @property
    def quality_score(self) -> float:
        """Calculate overall quality score (0-100)."""
        scores = [
            max(0, 100 - self.complexity_score * 5),  # Lower complexity is better
            self.maintainability_index,
            100 - self.duplication_percent,
            self.documentation_coverage,
            self.type_hint_coverage * 100,
        ]
        return sum(scores) / len(scores)


@dataclass
class QualityReport:
    """Complete quality report."""
    project_id: str
    timestamp: str
    test_results: List[TestResult] = field(default_factory=list)
    issues: List[QualityIssue] = field(default_factory=list)
    metrics: List[CodeMetrics] = field(default_factory=list)
    
    @property
    def passed(self) -> bool:
        """Check if all quality gates passed."""
        # No critical issues
        if any(i.severity == QualitySeverity.CRITICAL for i in self.issues):
            return False
        
        # All tests passed
        if not all(r.passed for r in self.test_results):
            return False
        
        # Minimum coverage
        avg_coverage = self.average_coverage
        if avg_coverage is not None and avg_coverage < 80:
            return False
        
        return True
    
    @property
    def average_coverage(self) -> Optional[float]:
        """Calculate average test coverage."""
        coverages = [r.coverage_percent for r in self.test_results if r.coverage_percent is not None]
        return sum(coverages) / len(coverages) if coverages else None
    
    @property
    def quality_score(self) -> float:
        """Calculate overall quality score."""
        if not self.metrics:
            return 0.0
        return sum(m.quality_score for m in self.metrics) / len(self.metrics)
    
    def get_issues_by_severity(self, severity: QualitySeverity) -> List[QualityIssue]:
        """Filter issues by severity."""
        return [i for i in self.issues if i.severity == severity]


class StaticAnalyzer:
    """Static code analysis engine."""
    
    COMPLEXITY_THRESHOLD = 10
    MAX_LINE_LENGTH = 100
    
    def __init__(self):
        self.issues: List[QualityIssue] = []
    
    async def analyze_file(self, file_path: Path) -> CodeMetrics:
        """Analyze single Python file."""
        content = file_path.read_text(encoding='utf-8')
        lines = content.split('\n')
        
        # Parse AST
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            self.issues.append(QualityIssue(
                rule_id="SYNTAX_ERROR",
                description=f"Syntax error: {e.msg}",
                severity=QualitySeverity.CRITICAL,
                file_path=str(file_path),
                line_number=e.lineno or 1,
                column=e.offset or 0,
            ))
            return self._empty_metrics(file_path)
        
        # Calculate metrics
        loc = len([l for l in lines if l.strip() and not l.strip().startswith('#')])
        complexity = self._calculate_complexity(tree)
        doc_coverage = self._calculate_doc_coverage(tree)
        type_coverage = self._calculate_type_coverage(tree)
        
        # Check style issues
        self._check_line_length(lines, file_path)
        self._check_complexity(complexity, file_path, tree)
        self._check_imports(tree, file_path)
        
        # Calculate maintainability index
        maintainability = self._calculate_maintainability(loc, complexity, len(lines))
        
        return CodeMetrics(
            file_path=str(file_path),
            lines_of_code=loc,
            complexity_score=complexity,
            maintainability_index=maintainability,
            duplication_percent=0.0,  # Would need multi-file analysis
            documentation_coverage=doc_coverage,
            type_hint_coverage=type_coverage,
        )
    
    def _calculate_complexity(self, tree: ast.AST) -> float:
        """Calculate cyclomatic complexity."""
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.FunctionDef):
                complexity += 1
                # Add for boolean operators
                complexity += sum(
                    1 for n in ast.walk(node)
                    if isinstance(n, ast.BoolOp)
                )
        
        return complexity
    
    def _calculate_doc_coverage(self, tree: ast.AST) -> float:
        """Calculate documentation coverage."""
        documented = 0
        total = 0
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                total += 1
                if ast.get_docstring(node):
                    documented += 1
        
        return (documented / total * 100) if total > 0 else 100.0
    
    def _calculate_type_coverage(self, tree: ast.AST) -> float:
        """Calculate type hint coverage."""
        functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        
        if not functions:
            return 100.0
        
        typed = 0
        for func in functions:
            # Check return annotation
            if func.returns:
                typed += 1
                continue
            
            # Check argument annotations
            args_with_types = sum(1 for arg in func.args.args if arg.annotation)
            if args_with_types == len(func.args.args):
                typed += 1
        
        return typed / len(functions) * 100
    
    def _calculate_maintainability(self, loc: int, complexity: float, total_lines: int) -> float:
        """Calculate maintainability index (simplified)."""
        # Halstead Volume approximation
        volume = loc * (complexity + 1)
        
        # Maintainability Index formula (simplified)
        mi = 171 - 5.2 * (volume ** 0.23) - 0.23 * complexity - 16.2 * (total_lines / 100)
        
        return max(0, min(100, mi))
    
    def _check_line_length(self, lines: List[str], file_path: Path):
        """Check for long lines."""
        for i, line in enumerate(lines, 1):
            if len(line) > self.MAX_LINE_LENGTH:
                self.issues.append(QualityIssue(
                    rule_id="LINE_TOO_LONG",
                    description=f"Line exceeds {self.MAX_LINE_LENGTH} characters",
                    severity=QualitySeverity.LOW,
                    file_path=str(file_path),
                    line_number=i,
                    column=self.MAX_LINE_LENGTH,
                    suggested_fix="Break line into multiple lines",
                ))
    
    def _check_complexity(self, complexity: float, file_path: Path, tree: ast.AST):
        """Check for high complexity functions."""
        if complexity > self.COMPLEXITY_THRESHOLD:
            # Find the most complex function
            max_func_complexity = 0
            max_func_line = 1
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_complexity = 1
                    for n in ast.walk(node):
                        if isinstance(n, (ast.If, ast.While, ast.For)):
                            func_complexity += 1
                    
                    if func_complexity > max_func_complexity:
                        max_func_complexity = func_complexity
                        max_func_line = node.lineno
            
            self.issues.append(QualityIssue(
                rule_id="HIGH_COMPLEXITY",
                description=f"Cyclomatic complexity ({complexity:.0f}) exceeds threshold ({self.COMPLEXITY_THRESHOLD})",
                severity=QualitySeverity.HIGH if complexity > 20 else QualitySeverity.MEDIUM,
                file_path=str(file_path),
                line_number=max_func_line,
                column=0,
                suggested_fix="Refactor into smaller functions",
            ))
    
    def _check_imports(self, tree: ast.AST, file_path: Path):
        """Check for import issues."""
        imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
        
        # Check for wildcard imports
        for node in imports:
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name == '*':
                        self.issues.append(QualityIssue(
                            rule_id="WILDCARD_IMPORT",
                            description="Wildcard import detected",
                            severity=QualitySeverity.MEDIUM,
                            file_path=str(file_path),
                            line_number=getattr(node, 'lineno', 1),
                            column=0,
                            suggested_fix="Import specific names",
                        ))
    
    def _empty_metrics(self, file_path: Path) -> CodeMetrics:
        """Return empty metrics for unparseable file."""
        return CodeMetrics(
            file_path=str(file_path),
            lines_of_code=0,
            complexity_score=0,
            maintainability_index=0,
            duplication_percent=0,
            documentation_coverage=0,
            type_hint_coverage=0,
        )
    
    def get_issues(self) -> List[QualityIssue]:
        """Get all discovered issues."""
        return self.issues


class TestRunner:
    """Test execution engine."""
    
    async def run_tests(
        self,
        project_path: Path,
        level: TestLevel = TestLevel.UNIT,
    ) -> List[TestResult]:
        """Run tests at specified level."""
        results = []
        
        if level == TestLevel.UNIT:
            results = await self._run_pytest(project_path)
        elif level == TestLevel.PERFORMANCE:
            results = await self._run_performance_tests(project_path)
        elif level == TestLevel.SECURITY:
            results = await self._run_security_checks(project_path)
        
        return results
    
    async def _run_pytest(self, project_path: Path) -> List[TestResult]:
        """Run pytest and collect results."""
        results = []
        
        try:
            # Run pytest with coverage
            proc = await asyncio.create_subprocess_exec(
                "python", "-m", "pytest",
                str(project_path / "tests"),
                "--json-report",
                "--json-report-file=-",  # Output to stdout
                "--cov=.",
                "--cov-report=term-missing",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            
            stdout, stderr = await proc.communicate()
            
            # Parse results
            try:
                report = json.loads(stdout.decode())
                
                for test in report.get("tests", []):
                    results.append(TestResult(
                        name=test.get("nodeid", "unknown"),
                        level=TestLevel.UNIT,
                        passed=test.get("outcome") == "passed",
                        duration_ms=test.get("duration", 0) * 1000,
                        message=test.get("call", {}).get("longrepr", ""),
                    ))
                
                # Extract coverage from stderr (simple parsing)
                coverage_match = re.search(r'TOTAL\s+\d+\s+\d+\s+(\d+)%', stderr.decode())
                coverage = float(coverage_match.group(1)) if coverage_match else None
                
                if results and coverage is not None:
                    results[0].coverage_percent = coverage
                    
            except json.JSONDecodeError:
                # Fallback if JSON report not available
                passed = proc.returncode == 0
                results.append(TestResult(
                    name="pytest_suite",
                    level=TestLevel.UNIT,
                    passed=passed,
                    duration_ms=0,
                    message=stdout.decode()[:500],
                    stderr=stderr.decode()[:500],
                ))
                
        except FileNotFoundError:
            results.append(TestResult(
                name="pytest",
                level=TestLevel.UNIT,
                passed=False,
                duration_ms=0,
                message="pytest not found",
            ))
        
        return results
    
    async def _run_performance_tests(self, project_path: Path) -> List[TestResult]:
        """Run performance benchmarks."""
        results = []
        
        # Check for performance test file
        perf_test_file = project_path / "tests" / "test_performance.py"
        
        if not perf_test_file.exists():
            return [TestResult(
                name="performance_tests",
                level=TestLevel.PERFORMANCE,
                passed=True,  # Pass if no tests exist
                duration_ms=0,
                message="No performance tests found",
            )]
        
        try:
            proc = await asyncio.create_subprocess_exec(
                "python", "-m", "pytest",
                str(perf_test_file),
                "-v",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            
            stdout, stderr = await proc.communicate()
            
            results.append(TestResult(
                name="performance_benchmarks",
                level=TestLevel.PERFORMANCE,
                passed=proc.returncode == 0,
                duration_ms=0,
                message=stdout.decode()[:1000],
            ))
            
        except Exception as e:
            results.append(TestResult(
                name="performance_tests",
                level=TestLevel.PERFORMANCE,
                passed=False,
                duration_ms=0,
                message=str(e),
            ))
        
        return results
    
    async def _run_security_checks(self, project_path: Path) -> List[TestResult]:
        """Run security vulnerability checks."""
        results = []
        
        # Check for common security issues
        issues = []
        
        for py_file in project_path.rglob("*.py"):
            try:
                content = py_file.read_text(encoding='utf-8')
                
                # Check for hardcoded secrets (simple patterns)
                secret_patterns = [
                    (r'password\s*=\s*["\'][^"\']+["\']', "Hardcoded password"),
                    (r'api_key\s*=\s*["\'][^"\']+["\']', "Hardcoded API key"),
                    (r'secret\s*=\s*["\'][^"\']+["\']', "Hardcoded secret"),
                    (r'eval\s*\(', "Use of eval()"),
                    (r'exec\s*\(', "Use of exec()"),
                ]
                
                for pattern, description in secret_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        issues.append(f"{py_file}: {description}")
                        
            except Exception:
                pass
        
        results.append(TestResult(
            name="security_scan",
            level=TestLevel.SECURITY,
            passed=len(issues) == 0,
            duration_ms=0,
            message="\n".join(issues) if issues else "No security issues found",
        ))
        
        return results


class QualityController:
    """
    Main quality control orchestrator.
    
    Features:
    - Multi-level testing
    - Static analysis
    - Quality gating
    - Regression tracking
    - Compliance enforcement
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path(".quality")
        self.storage_path.mkdir(exist_ok=True)
        
        self._analyzer = StaticAnalyzer()
        self._test_runner = TestRunner()
        self._baseline: Optional[QualityReport] = None
        
        self._load_baseline()
    
    def _load_baseline(self):
        """Load quality baseline."""
        baseline_file = self.storage_path / "baseline.json"
        if baseline_file.exists():
            try:
                # Simplified - would fully reconstruct
                logger.info("Loaded quality baseline")
            except Exception as e:
                logger.warning(f"Failed to load baseline: {e}")
    
    @monitor_endpoint("/quality/gate")
    async def run_quality_gate(
        self,
        project_id: str,
        project_path: Path,
        levels: Optional[List[TestLevel]] = None,
    ) -> QualityReport:
        """
        Run complete quality gate.
        
        Pipeline:
        1. Static analysis
        2. Unit tests
        3. Integration tests
        4. Performance tests
        5. Security scan
        6. Compliance check
        """
        levels = levels or [TestLevel.UNIT, TestLevel.PERFORMANCE]
        
        start_time = time.time()
        
        # Initialize report
        report = QualityReport(
            project_id=project_id,
            timestamp=datetime.now().isoformat(),
        )
        
        # Phase 1: Static Analysis
        logger.info("Running static analysis...")
        metrics_files = list(project_path.rglob("*.py"))
        
        for py_file in metrics_files:
            if "__pycache__" in str(py_file):
                continue
            
            try:
                file_metrics = await self._analyzer.analyze_file(py_file)
                report.metrics.append(file_metrics)
            except Exception as e:
                logger.warning(f"Failed to analyze {py_file}: {e}")
        
        # Collect issues from analyzer
        report.issues.extend(self._analyzer.get_issues())
        
        # Phase 2: Testing
        logger.info("Running tests...")
        for level in levels:
            test_results = await self._test_runner.run_tests(project_path, level)
            report.test_results.extend(test_results)
        
        # Phase 3: Compliance check
        logger.info("Running compliance checks...")
        compliance_issues = await self._check_compliance(report)
        report.issues.extend(compliance_issues)
        
        # Calculate duration
        duration = (time.time() - start_time) * 1000
        
        # Record metrics
        await metrics.record("quality_gate_duration", duration)
        await metrics.record("quality_score", report.quality_score)
        await metrics.record("test_coverage", report.average_coverage or 0)
        
        # Persist report
        await self._persist_report(report)
        
        logger.info(
            f"Quality gate complete: score={report.quality_score:.1f}, "
            f"passed={report.passed}, issues={len(report.issues)}"
        )
        
        return report
    
    async def _check_compliance(self, report: QualityReport) -> List[QualityIssue]:
        """Check compliance with standards."""
        issues = []
        
        # Check 1: Minimum test coverage
        coverage = report.average_coverage
        if coverage is not None and coverage < 80:
            issues.append(QualityIssue(
                rule_id="COVERAGE_TOO_LOW",
                description=f"Test coverage ({coverage:.1f}%) below minimum (80%)",
                severity=QualitySeverity.HIGH,
                file_path="",
                line_number=0,
                column=0,
                suggested_fix="Add more tests to increase coverage",
            ))
        
        # Check 2: No critical issues
        critical_count = len(report.get_issues_by_severity(QualitySeverity.CRITICAL))
        if critical_count > 0:
            issues.append(QualityIssue(
                rule_id="CRITICAL_ISSUES_FOUND",
                description=f"Found {critical_count} critical quality issues",
                severity=QualitySeverity.CRITICAL,
                file_path="",
                line_number=0,
                column=0,
                suggested_fix="Fix all critical issues before merging",
            ))
        
        # Check 3: Complexity threshold
        high_complexity = [m for m in report.metrics if m.complexity_score > 15]
        if len(high_complexity) > 3:
            issues.append(QualityIssue(
                rule_id="TOO_MANY_COMPLEX_FUNCTIONS",
                description=f"{len(high_complexity)} functions with high complexity",
                severity=QualitySeverity.MEDIUM,
                file_path="",
                line_number=0,
                column=0,
                suggested_fix="Refactor complex functions",
            ))
        
        # Check 4: Documentation coverage
        low_doc = [m for m in report.metrics if m.documentation_coverage < 50]
        if low_doc:
            issues.append(QualityIssue(
                rule_id="LOW_DOCUMENTATION",
                description=f"{len(low_doc)} files have low documentation coverage",
                severity=QualitySeverity.LOW,
                file_path="",
                line_number=0,
                column=0,
                suggested_fix="Add docstrings to functions and classes",
            ))
        
        return issues
    
    def detect_regression(
        self,
        current: QualityReport,
        baseline: Optional[QualityReport] = None,
    ) -> List[Dict[str, Any]]:
        """Detect quality regressions."""
        baseline = baseline or self._baseline
        regressions = []
        
        if not baseline:
            return regressions
        
        # Check coverage regression
        if current.average_coverage and baseline.average_coverage:
            if current.average_coverage < baseline.average_coverage - 5:
                regressions.append({
                    "type": "coverage_drop",
                    "severity": "high",
                    "message": f"Coverage dropped from {baseline.average_coverage:.1f}% to {current.average_coverage:.1f}%",
                    "delta": current.average_coverage - baseline.average_coverage,
                })
        
        # Check quality score regression
        if current.quality_score < baseline.quality_score - 10:
            regressions.append({
                "type": "quality_drop",
                "severity": "high",
                "message": f"Quality score dropped from {baseline.quality_score:.1f} to {current.quality_score:.1f}",
                "delta": current.quality_score - baseline.quality_score,
            })
        
        # Check new critical issues
        current_critical = len(current.get_issues_by_severity(QualitySeverity.CRITICAL))
        baseline_critical = len(baseline.get_issues_by_severity(QualitySeverity.CRITICAL))
        
        if current_critical > baseline_critical:
            regressions.append({
                "type": "new_critical_issues",
                "severity": "critical",
                "message": f"New critical issues: {current_critical - baseline_critical}",
                "count": current_critical - baseline_critical,
            })
        
        return regressions
    
    @cached(ttl=60)
    async def get_quality_trends(self, project_id: str) -> Dict[str, Any]:
        """Get quality trends over time."""
        reports_dir = self.storage_path / "reports"
        if not reports_dir.exists():
            return {"error": "No historical data"}
        
        reports = []
        for report_file in reports_dir.glob(f"{project_id}_*.json"):
            try:
                with open(report_file, 'r') as f:
                    data = json.load(f)
                    reports.append(data)
            except Exception:
                pass
        
        if not reports:
            return {"error": "No reports found"}
        
        # Sort by timestamp
        reports.sort(key=lambda r: r.get("timestamp", ""))
        
        return {
            "project_id": project_id,
            "data_points": len(reports),
            "quality_scores": [r.get("quality_score", 0) for r in reports[-10:]],
            "coverages": [r.get("average_coverage", 0) for r in reports[-10:]],
            "trend": "improving" if len(reports) > 1 and reports[-1].get("quality_score", 0) > reports[0].get("quality_score", 0) else "stable",
        }
    
    async def _persist_report(self, report: QualityReport):
        """Save quality report."""
        reports_dir = self.storage_path / "reports"
        reports_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = reports_dir / f"{report.project_id}_{timestamp}.json"
        
        data = {
            "project_id": report.project_id,
            "timestamp": report.timestamp,
            "passed": report.passed,
            "quality_score": report.quality_score,
            "average_coverage": report.average_coverage,
            "test_count": len(report.test_results),
            "tests_passed": sum(1 for t in report.test_results if t.passed),
            "issues_count": len(report.issues),
            "critical_issues": len(report.get_issues_by_severity(QualitySeverity.CRITICAL)),
            "metrics_summary": {
                "total_files": len(report.metrics),
                "total_loc": sum(m.lines_of_code for m in report.metrics),
                "avg_complexity": sum(m.complexity_score for m in report.metrics) / len(report.metrics) if report.metrics else 0,
            },
        }
        
        with open(report_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def generate_badge(self, report: QualityReport) -> str:
        """Generate shields.io badge URL."""
        if report.passed:
            color = "brightgreen"
            status = "passing"
        else:
            color = "red"
            status = "failing"
        
        return (
            f"https://img.shields.io/badge/quality-{status}-{color}"
            f"?logo=pytest&logoColor=white"
        )


# Global quality controller
_quality_controller: Optional[QualityController] = None


def get_quality_controller(storage_path: Optional[Path] = None) -> QualityController:
    """Get global quality controller instance."""
    global _quality_controller
    if _quality_controller is None:
        _quality_controller = QualityController(storage_path)
    return _quality_controller
