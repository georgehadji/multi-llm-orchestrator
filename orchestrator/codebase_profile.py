"""Codebase semantic profile (understanding)"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CodebaseProfile:
    """Semantic understanding of a codebase"""

    purpose: str  # "What does this project do?"
    primary_patterns: list[str] = field(default_factory=list)  # ["layered", "REST API"]
    anti_patterns: list[str] = field(default_factory=list)  # ["no type hints", "god functions"]
    test_coverage: str = "unknown"  # low, moderate, good, excellent
    documentation: str = "unknown"  # minimal, good, excellent
    primary_language: str = "unknown"
    project_type: str = "generic"

    def __str__(self) -> str:
        return f"""
CodebaseProfile
───────────────
Purpose: {self.purpose}

Patterns: {', '.join(self.primary_patterns) if self.primary_patterns else '(none detected)'}
Anti-patterns: {', '.join(self.anti_patterns) if self.anti_patterns else '(none detected)'}

Language: {self.primary_language}
Type: {self.project_type}
Test Coverage: {self.test_coverage}
Documentation: {self.documentation}
"""
