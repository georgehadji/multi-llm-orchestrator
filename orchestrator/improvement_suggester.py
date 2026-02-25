"""Generate improvement suggestions from codebase profile"""

from dataclasses import dataclass
from typing import List

from orchestrator.codebase_profile import CodebaseProfile


@dataclass
class Improvement:
    """A suggested improvement to the codebase"""
    title: str
    description: str
    impact: str  # Why this matters
    effort_hours: int
    priority: str  # "HIGH", "MEDIUM", "LOW"
    category: str  # "testing", "refactoring", "documentation", "features"

    def __str__(self) -> str:
        return f"[{self.priority}] {self.title} ({self.effort_hours}h)"


class ImprovementSuggester:
    """Generate improvement suggestions based on codebase analysis"""

    def suggest(self, profile: CodebaseProfile) -> List[Improvement]:
        """
        Generate prioritized improvement suggestions.

        Args:
            profile: CodebaseProfile from semantic analysis

        Returns:
            List of Improvement suggestions, prioritized by impact
        """
        improvements = []

        # Test coverage issues
        if profile.test_coverage in ["low", "moderate"]:
            improvements.append(Improvement(
                title="Add comprehensive test suite",
                description="Increase test coverage to >80%",
                impact="Confidence in changes, safety for refactoring",
                effort_hours=6 if profile.test_coverage == "low" else 3,
                priority="HIGH",
                category="testing",
            ))

        # Documentation issues
        if profile.documentation in ["minimal", "unknown"]:
            improvements.append(Improvement(
                title="Create API documentation",
                description="Add OpenAPI/Swagger docs or README API section",
                impact="Improves discoverability and onboarding",
                effort_hours=2,
                priority="MEDIUM",
                category="documentation",
            ))

        # Anti-pattern fixes
        if "no error handling" in profile.anti_patterns:
            improvements.append(Improvement(
                title="Add custom exception hierarchy",
                description="Create structured error handling with custom exceptions",
                impact="Better error diagnostics and user feedback",
                effort_hours=3,
                priority="MEDIUM",
                category="refactoring",
            ))

        if "no type hints" in profile.anti_patterns:
            improvements.append(Improvement(
                title="Add type hints",
                description="Add Python type annotations across codebase",
                impact="Better IDE support, catch errors early",
                effort_hours=4,
                priority="MEDIUM",
                category="refactoring",
            ))

        if "no logging" in profile.anti_patterns:
            improvements.append(Improvement(
                title="Add structured logging",
                description="Implement logging with levels and formatters",
                impact="Production debugging and monitoring",
                effort_hours=2,
                priority="MEDIUM",
                category="features",
            ))

        # Missing infrastructure
        if profile.primary_language == "python" and profile.project_type in ["fastapi", "django"]:
            improvements.append(Improvement(
                title="Add database migrations (Alembic)",
                description="Replace manual SQL with Alembic versioning",
                impact="Safe schema evolution, rollback capability",
                effort_hours=4,
                priority="HIGH",
                category="refactoring",
            ))

        # Code quality improvements
        improvements.append(Improvement(
            title="Add pre-commit hooks",
            description="Setup ruff, mypy, black in CI/CD",
            impact="Automated code quality checks",
            effort_hours=1,
            priority="LOW",
            category="testing",
        ))

        # Sort by priority (HIGH > MEDIUM > LOW), then by effort
        priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        improvements.sort(
            key=lambda x: (priority_order[x.priority], x.effort_hours)
        )

        return improvements
