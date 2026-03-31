"""
Project Analyzer - Post-Completion Codebase Analysis
====================================================
Automatically analyzes completed projects and suggests improvements.

Features:
- Code quality analysis (complexity, coverage, patterns)
- Architecture assessment
- Security review
- Performance optimization suggestions
- Feature gap analysis
- Learning extraction for Knowledge Base

Usage:
    from orchestrator.project_analyzer import ProjectAnalyzer

    analyzer = ProjectAnalyzer()
    report = await analyzer.analyze_project(project_path, project_id)

    for suggestion in report.suggestions:
        print(f"[{suggestion.priority}] {suggestion.title}")
        print(f"  {suggestion.description}")
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

from .knowledge_base import KnowledgeType, get_knowledge_base
from .log_config import get_logger
from .quality_control import QualityController, TestLevel

if TYPE_CHECKING:
    from pathlib import Path

logger = get_logger(__name__)


# Nexus Search integration for security checks
def _get_nexus_search():
    """Lazy import of Nexus Search for security vulnerability checks."""
    try:
        from orchestrator.nexus_search import SearchSource
        from orchestrator.nexus_search import search as nexus_search

        return nexus_search, SearchSource
    except ImportError:
        return None, None


class SuggestionPriority(Enum):
    """Priority levels for suggestions."""

    CRITICAL = "critical"  # Must fix - security, crashes
    HIGH = "high"  # Should fix - performance, maintainability
    MEDIUM = "medium"  # Nice to have - optimizations
    LOW = "low"  # Consider - style, documentation


class SuggestionCategory(Enum):
    """Categories of suggestions."""

    CODE_QUALITY = "code_quality"
    ARCHITECTURE = "architecture"
    SECURITY = "security"
    PERFORMANCE = "performance"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    FEATURES = "features"
    BEST_PRACTICES = "best_practices"


@dataclass
class CodeIssue:
    """Identified code issue."""

    file_path: str
    line_number: int
    issue_type: str
    description: str
    severity: str
    suggested_fix: str | None = None


@dataclass
class ImprovementSuggestion:
    """Suggestion for improvement."""

    id: str
    title: str
    description: str
    category: SuggestionCategory
    priority: SuggestionPriority
    affected_files: list[str]
    estimated_effort: str  # "1h", "2d", "1w"
    expected_impact: str
    code_example: str | None = None
    rationale: str = ""


@dataclass
class ArchitectureInsight:
    """Architecture analysis insight."""

    pattern_detected: str
    quality_score: float  # 0-100
    strengths: list[str]
    weaknesses: list[str]
    recommendations: list[str]


@dataclass
class ProjectAnalysisReport:
    """Complete project analysis report."""

    project_id: str
    analyzed_at: str

    # Overall metrics
    total_files: int
    total_lines: int
    languages: dict[str, int]  # language -> file count

    # Quality metrics
    quality_score: float  # 0-100
    test_coverage: float | None
    complexity_score: float
    documentation_coverage: float

    # Analysis results
    issues: list[CodeIssue]
    suggestions: list[ImprovementSuggestion]
    architecture_insights: list[ArchitectureInsight]

    # Learning extraction
    patterns_found: list[str]
    lessons_learned: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def get_critical_suggestions(self) -> list[ImprovementSuggestion]:
        """Get critical priority suggestions."""
        return [s for s in self.suggestions if s.priority == SuggestionPriority.CRITICAL]

    def get_suggestions_by_category(
        self, category: SuggestionCategory
    ) -> list[ImprovementSuggestion]:
        """Get suggestions by category."""
        return [s for s in self.suggestions if s.category == category]


class CodeMetricsAnalyzer:
    """Analyze code metrics and quality."""

    def __init__(self):
        self.issues: list[CodeIssue] = []

    async def analyze_file(self, file_path: Path) -> dict[str, Any]:
        """Analyze a single file."""
        if file_path.suffix == ".py":
            return await self._analyze_python(file_path)
        elif file_path.suffix in [".js", ".ts"]:
            return await self._analyze_javascript(file_path)
        else:
            return {"language": "unknown", "lines": 0}

    async def _analyze_python(self, file_path: Path) -> dict[str, Any]:
        """Analyze Python file."""
        try:
            content = file_path.read_text(encoding="utf-8")
            lines = content.split("\n")

            # Parse AST
            try:
                tree = ast.parse(content)
            except SyntaxError as e:
                self.issues.append(
                    CodeIssue(
                        file_path=str(file_path),
                        line_number=e.lineno or 1,
                        issue_type="syntax_error",
                        description=f"Syntax error: {e.msg}",
                        severity="critical",
                    )
                )
                return {"error": "syntax_error", "lines": len(lines)}

            # Calculate metrics
            metrics = {
                "total_lines": len(lines),
                "code_lines": len(
                    [l for l in lines if l.strip() and not l.strip().startswith("#")]
                ),
                "functions": len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]),
                "classes": len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]),
                "complexity": self._calculate_complexity(tree),
                "imports": len(
                    [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
                ),
            }

            # Check for issues
            self._check_python_issues(file_path, tree, content)

            return metrics

        except Exception as e:
            logger.warning(f"Failed to analyze {file_path}: {e}")
            return {"error": str(e)}

    def _calculate_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler, ast.FunctionDef)):
                complexity += 1
        return complexity

    def _check_python_issues(self, file_path: Path, tree: ast.AST, content: str):
        """Check for common Python issues."""
        # Check for bare excepts
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                self.issues.append(
                    CodeIssue(
                        file_path=str(file_path),
                        line_number=node.lineno,
                        issue_type="bare_except",
                        description="Bare except clause - catches KeyboardInterrupt, SystemExit",
                        severity="high",
                        suggested_fix="Use 'except Exception:' instead",
                    )
                )

        # Check for TODO/FIXME comments
        for i, line in enumerate(content.split("\n"), 1):
            if "TODO" in line or "FIXME" in line:
                self.issues.append(
                    CodeIssue(
                        file_path=str(file_path),
                        line_number=i,
                        issue_type="todo_comment",
                        description=f"Incomplete task: {line.strip()}",
                        severity="low",
                    )
                )

        # Check line length
        for i, line in enumerate(content.split("\n"), 1):
            if len(line) > 100:
                self.issues.append(
                    CodeIssue(
                        file_path=str(file_path),
                        line_number=i,
                        issue_type="long_line",
                        description=f"Line too long ({len(line)} > 100 chars)",
                        severity="low",
                    )
                )

    async def _analyze_javascript(self, file_path: Path) -> dict[str, Any]:
        """Analyze JavaScript/TypeScript file."""
        try:
            content = file_path.read_text(encoding="utf-8")
            lines = content.split("\n")

            # Basic metrics
            metrics = {
                "total_lines": len(lines),
                "code_lines": len(
                    [l for l in lines if l.strip() and not l.strip().startswith("//")]
                ),
                "functions": content.count("function ") + content.count("=>"),
            }

            return metrics

        except Exception as e:
            return {"error": str(e)}


class ArchitectureAnalyzer:
    """Analyze project architecture and patterns."""

    def analyze_structure(self, project_path: Path) -> ArchitectureInsight:
        """Analyze project structure and patterns."""

        # Detect patterns
        patterns = []
        strengths = []
        weaknesses = []
        recommendations = []

        # Check for common patterns
        files = list(project_path.rglob("*"))

        # MVC/MVT pattern
        has_models = any("model" in f.name.lower() for f in files if f.is_file())
        has_views = any("view" in f.name.lower() for f in files if f.is_file())
        any("controller" in f.name.lower() for f in files if f.is_file())

        if has_models and has_views:
            patterns.append("MVC/MVT")
            strengths.append("Separation of concerns with models and views")

        # Layered architecture
        has_services = any("service" in f.name.lower() for f in files if f.is_file())
        has_repositories = any(
            "repository" in f.name.lower() or "repo" in f.name.lower() for f in files if f.is_file()
        )

        if has_services and has_repositories:
            patterns.append("Layered Architecture")
            strengths.append("Clear separation between business logic and data access")

        # Check for tests
        has_tests = any("test" in f.name.lower() for f in files if f.is_file())
        if not has_tests:
            weaknesses.append("No test files detected")
            recommendations.append("Add unit tests for core functionality")
        else:
            strengths.append("Testing infrastructure present")

        # Check for documentation
        has_docs = any(f.suffix in [".md", ".rst"] for f in files if f.is_file())
        if not has_docs:
            weaknesses.append("No documentation files")
            recommendations.append("Add README and API documentation")

        # Check for config management
        has_config = (
            (project_path / "config.py").exists()
            or (project_path / "settings.py").exists()
            or (project_path / ".env.example").exists()
        )

        if not has_config:
            weaknesses.append("No clear configuration management")
            recommendations.append("Add configuration files and environment templates")

        # Calculate quality score
        score = 70  # Base score
        score += len(strengths) * 5
        score -= len(weaknesses) * 10
        score = max(0, min(100, score))

        return ArchitectureInsight(
            pattern_detected=", ".join(patterns) if patterns else "No clear pattern",
            quality_score=score,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations,
        )


class ImprovementSuggester:
    """Generate improvement suggestions based on analysis."""

    def generate_suggestions(
        self,
        metrics: dict[str, Any],
        issues: list[CodeIssue],
        architecture: ArchitectureInsight,
        project_path: Path,
    ) -> list[ImprovementSuggestion]:
        """Generate actionable suggestions."""

        suggestions = []

        # Critical issues
        critical_issues = [i for i in issues if i.severity == "critical"]
        for issue in critical_issues[:3]:  # Top 3
            suggestions.append(
                ImprovementSuggestion(
                    id=f"critical_{hash(issue.description) % 10000}",
                    title=f"Fix {issue.issue_type}",
                    description=issue.description,
                    category=SuggestionCategory.CODE_QUALITY,
                    priority=SuggestionPriority.CRITICAL,
                    affected_files=[issue.file_path],
                    estimated_effort="30m",
                    expected_impact="Prevents crashes/errors",
                    suggested_fix=issue.suggested_fix,
                    rationale="Critical issue that affects stability",
                )
            )

        # Architecture suggestions
        for rec in architecture.recommendations[:3]:
            suggestions.append(
                ImprovementSuggestion(
                    id=f"arch_{hash(rec) % 10000}",
                    title=rec,
                    description=f"Based on architecture analysis: {architecture.pattern_detected}",
                    category=SuggestionCategory.ARCHITECTURE,
                    priority=SuggestionPriority.HIGH,
                    affected_files=[str(project_path)],
                    estimated_effort="2h",
                    expected_impact="Improves maintainability",
                    rationale="Architecture best practice",
                )
            )

        # Test coverage
        if "test" not in str(project_path).lower() or not any(
            "test" in i.issue_type for i in issues
        ):
            suggestions.append(
                ImprovementSuggestion(
                    id="test_coverage",
                    title="Add comprehensive test suite",
                    description="Project lacks adequate test coverage. Add unit and integration tests.",
                    category=SuggestionCategory.TESTING,
                    priority=SuggestionPriority.HIGH,
                    affected_files=[str(project_path)],
                    estimated_effort="4h",
                    expected_impact="Reduces bugs, enables refactoring",
                    code_example="""
# Example test structure
def test_feature():
    # Arrange
    input_data = {...}

    # Act
    result = process(input_data)

    # Assert
    assert result == expected
""",
                    rationale="Testing ensures code quality and prevents regressions",
                )
            )

        # Documentation
        suggestions.append(
            ImprovementSuggestion(
                id="documentation",
                title="Improve API documentation",
                description="Add docstrings, README sections, and usage examples",
                category=SuggestionCategory.DOCUMENTATION,
                priority=SuggestionPriority.MEDIUM,
                affected_files=[str(project_path)],
                estimated_effort="2h",
                expected_impact="Easier onboarding and usage",
                code_example='''
"""
Function description.

Args:
    param1: Description of param1
    param2: Description of param2

Returns:
    Description of return value

Example:
    >>> result = my_function(1, 2)
    >>> print(result)
    3
"""
''',
                rationale="Good documentation reduces support burden",
            )
        )

        # Error handling
        suggestions.append(
            ImprovementSuggestion(
                id="error_handling",
                title="Add comprehensive error handling",
                description="Add try-except blocks, input validation, and error logging",
                category=SuggestionCategory.BEST_PRACTICES,
                priority=SuggestionPriority.HIGH,
                affected_files=[str(project_path)],
                estimated_effort="3h",
                expected_impact="Better user experience, easier debugging",
                code_example="""
try:
    result = risky_operation()
except SpecificException as e:
    logger.error(f"Operation failed: {e}")
    # Fallback or re-raise
    raise CustomException("User-friendly message") from e
""",
                rationale="Proper error handling prevents crashes and aids debugging",
            )
        )

        # Performance monitoring
        suggestions.append(
            ImprovementSuggestion(
                id="performance_monitoring",
                title="Add performance monitoring",
                description="Track execution time, memory usage, and bottlenecks",
                category=SuggestionCategory.PERFORMANCE,
                priority=SuggestionPriority.MEDIUM,
                affected_files=[str(project_path)],
                estimated_effort="2h",
                expected_impact="Identify and fix performance issues",
                code_example="""
import time
from functools import wraps

def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info(f"{func.__name__} took {elapsed:.2f}s")
        return result
    return wrapper
""",
                rationale="Monitoring helps identify performance regressions",
            )
        )

        return suggestions


class ProjectAnalyzer:
    """
    Main project analyzer that coordinates all analysis.

    Usage:
        analyzer = ProjectAnalyzer()
        report = await analyzer.analyze_project(
            project_path=Path("./results/my_project"),
            project_id="proj_123"
        )

        # Print suggestions
        for suggestion in report.suggestions:
            print(f"[{suggestion.priority.value}] {suggestion.title}")
    """

    def __init__(self, nexus_enabled: bool = True):
        self.metrics_analyzer = CodeMetricsAnalyzer()
        self.architecture_analyzer = ArchitectureAnalyzer()
        self.suggester = ImprovementSuggester()
        self.quality_controller = QualityController()
        self.nexus_enabled = nexus_enabled

    async def _check_security_vulnerabilities(
        self,
        project_path: Path,
    ) -> list[ImprovementSuggestion]:
        """
        Check for security vulnerabilities using Nexus Search.

        Args:
            project_path: Path to the project directory

        Returns:
            List of security-related suggestions
        """
        if not self.nexus_enabled:
            return []

        nexus_search, SearchSource = _get_nexus_search()
        if nexus_search is None:
            logger.debug("Nexus Search not available for security checks")
            return []

        suggestions = []

        try:
            # Look for requirements.txt or package.json
            deps_file = project_path / "requirements.txt"
            if deps_file.exists():
                # Read dependencies
                deps = []
                with open(deps_file) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#"):
                            # Extract package name
                            pkg = line.split("==")[0].split(">=")[0].split("<=")[0]
                            if pkg:
                                deps.append(pkg)

                # Check top 5 dependencies for vulnerabilities
                for dep in deps[:5]:
                    try:
                        results = await nexus_search(
                            query=f"{dep} CVE vulnerability security 2026",
                            sources=[SearchSource.WEB, SearchSource.NEWS],
                            num_results=3,
                        )

                        # Check if any vulnerabilities found
                        for result in results.top[:2]:
                            if any(
                                kw in result.title.lower() or kw in result.content.lower()
                                for kw in ["cve", "vulnerability", "security", "exploit"]
                            ):
                                suggestions.append(
                                    ImprovementSuggestion(
                                        id=f"security_{dep}",
                                        title=f"Check {dep} for vulnerabilities",
                                        description=f"Potential security issues found in {dep}. Review the dependency for known CVEs.",
                                        category=SuggestionCategory.SECURITY,
                                        priority=SuggestionPriority.HIGH,
                                        affected_files=[str(deps_file)],
                                        estimated_effort="2h",
                                        expected_impact="Improved security posture",
                                        code_example=f"# Consider updating {dep} to latest secure version\n# pip install {dep} --upgrade",
                                        rationale=f"Search results indicate potential vulnerabilities in {dep}",
                                    )
                                )
                                break  # One suggestion per dependency
                    except Exception as e:
                        logger.debug(f"Security check for {dep} failed: {e}")

        except Exception as e:
            logger.warning(f"Security vulnerability check failed: {e}")

        return suggestions

    async def analyze_project(
        self, project_path: Path, project_id: str, run_quality_gate: bool = True
    ) -> ProjectAnalysisReport:
        """
        Analyze a completed project comprehensively.

        Args:
            project_path: Path to the project directory
            project_id: Unique project identifier
            run_quality_gate: Whether to run quality control checks

        Returns:
            Complete analysis report with suggestions
        """
        logger.info(f"Analyzing project {project_id} at {project_path}")

        if not project_path.exists():
            raise ValueError(f"Project path does not exist: {project_path}")

        # Collect all files
        all_files = list(project_path.rglob("*"))
        code_files = [
            f
            for f in all_files
            if f.is_file()
            and f.suffix in [".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs"]
        ]

        # Analyze individual files
        file_metrics = []
        languages = {}

        for file_path in code_files[:50]:  # Limit to 50 files for performance
            metrics = await self.metrics_analyzer.analyze_file(file_path)
            if "error" not in metrics:
                file_metrics.append({"path": str(file_path), **metrics})
                lang = file_path.suffix.lstrip(".")
                languages[lang] = languages.get(lang, 0) + 1

        # Calculate totals
        total_lines = sum(m.get("total_lines", 0) for m in file_metrics)
        avg_complexity = (
            sum(m.get("complexity", 0) for m in file_metrics) / len(file_metrics)
            if file_metrics
            else 0
        )

        # Architecture analysis
        architecture = self.architecture_analyzer.analyze_structure(project_path)

        # Quality gate (optional)
        test_coverage = None
        if run_quality_gate and (project_path / "tests").exists():
            try:
                quality_report = await self.quality_controller.run_quality_gate(
                    project_id=project_id, project_path=project_path, levels=[TestLevel.UNIT]
                )
                test_coverage = quality_report.average_coverage
            except Exception as e:
                logger.warning(f"Quality gate failed: {e}")

        # Security vulnerability check (Nexus Search)
        security_suggestions = await self._check_security_vulnerabilities(project_path)

        # Generate suggestions
        suggestions = self.suggester.generate_suggestions(
            metrics={"total_lines": total_lines, "avg_complexity": avg_complexity},
            issues=self.metrics_analyzer.issues,
            architecture=architecture,
            project_path=project_path,
        )

        # Add security suggestions
        suggestions.extend(security_suggestions)

        # Calculate overall quality score
        quality_score = self._calculate_quality_score(file_metrics, architecture, test_coverage)

        # Create report
        report = ProjectAnalysisReport(
            project_id=project_id,
            analyzed_at=datetime.now().isoformat(),
            total_files=len(code_files),
            total_lines=total_lines,
            languages=languages,
            quality_score=quality_score,
            test_coverage=test_coverage,
            complexity_score=avg_complexity,
            documentation_coverage=0.0,  # TODO: Calculate
            issues=self.metrics_analyzer.issues,
            suggestions=suggestions,
            architecture_insights=[architecture],
            patterns_found=(
                architecture.pattern_detected.split(", ")
                if architecture.pattern_detected != "No clear pattern"
                else []
            ),
            lessons_learned=[],
        )

        # Store in Knowledge Base
        await self._store_in_knowledge_base(report, project_path)

        logger.info(
            f"Analysis complete. Score: {quality_score:.1f}/100, "
            f"{len(suggestions)} suggestions generated"
        )

        return report

    def _calculate_quality_score(
        self,
        file_metrics: list[dict],
        architecture: ArchitectureInsight,
        test_coverage: float | None,
    ) -> float:
        """Calculate overall quality score."""
        score = 50  # Base

        # Architecture score
        score += architecture.quality_score * 0.3

        # Complexity (lower is better)
        if file_metrics:
            avg_complexity = sum(m.get("complexity", 0) for m in file_metrics) / len(file_metrics)
            score += max(0, 20 - avg_complexity)  # Bonus for low complexity

        # Test coverage
        if test_coverage:
            score += test_coverage * 0.2  # Up to 20 points for coverage

        return min(100, max(0, score))

    async def _store_in_knowledge_base(self, report: ProjectAnalysisReport, project_path: Path):
        """Store analysis findings in Knowledge Base."""
        try:
            kb = get_knowledge_base()

            # Store patterns found
            for pattern in report.patterns_found:
                await kb.add_artifact(
                    type=KnowledgeType.PATTERN,
                    title=f"Pattern: {pattern}",
                    content=f"Detected in project {report.project_id}",
                    tags=["pattern", "architecture"],
                    source_project=report.project_id,
                )

            # Store lessons learned
            for suggestion in report.suggestions[:3]:  # Top 3
                await kb.add_artifact(
                    type=KnowledgeType.LESSON,
                    title=suggestion.title,
                    content=suggestion.rationale,
                    tags=["lesson", suggestion.category.value],
                    source_project=report.project_id,
                )

            logger.info("Stored findings in Knowledge Base")

        except Exception as e:
            logger.warning(f"Could not store in Knowledge Base: {e}")

    def generate_summary(self, report: ProjectAnalysisReport) -> str:
        """Generate human-readable summary."""
        lines = [
            f"📊 Project Analysis: {report.project_id}",
            f"{'='*60}",
            "",
            f"Overall Quality Score: {report.quality_score:.1f}/100",
            (
                f"Test Coverage: {report.test_coverage:.1f}%"
                if report.test_coverage
                else "Test Coverage: N/A"
            ),
            f"Total Files: {report.total_files}",
            f"Total Lines: {report.total_lines:,}",
            "",
            "🎯 Top Suggestions:",
        ]

        critical = report.get_critical_suggestions()
        if critical:
            lines.append(f"\n🔴 Critical ({len(critical)}):")
            for s in critical[:2]:
                lines.append(f"  - {s.title}")

        high_priority = [s for s in report.suggestions if s.priority == SuggestionPriority.HIGH][:3]
        if high_priority:
            lines.append(f"\n🟠 High Priority ({len(high_priority)}):")
            for s in high_priority:
                lines.append(f"  - {s.title} (~{s.estimated_effort})")

        lines.append(f"\n💡 Architecture: {report.architecture_insights[0].pattern_detected}")
        lines.append(f"   Quality: {report.architecture_insights[0].quality_score:.0f}/100")

        return "\n".join(lines)


# Convenience function
async def analyze_project(
    project_path: Path, project_id: str, print_summary: bool = True
) -> ProjectAnalysisReport:
    """
    Quick analysis function.

    Usage:
        report = await analyze_project(
            Path("./results/my_project"),
            "my_project_123"
        )
    """
    analyzer = ProjectAnalyzer()
    report = await analyzer.analyze_project(project_path, project_id)

    if print_summary:
        print(analyzer.generate_summary(report))

    return report
