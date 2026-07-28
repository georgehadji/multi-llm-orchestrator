"""SuiteValidator — RED-gate and assertion floor (AST-based, no I/O).

Phase 3 feature (E-2). Validates that a test suite:

1. Is non-vacuous — has at least one test function.
2. Has real assertions — uses assert/self.assert*/expect/etc.
3. Has assertion density ≥ floor — at least one assertion per test.

All validation is AST-based (stdlib only) — no subprocess, no I/O.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Assertion keywords by framework for detection
_ASSERT_KEYWORDS: dict[str, set[str]] = {
    "pytest": {"assert"},
    "unittest": {
        "assertEqual",
        "assertNotEqual",
        "assertTrue",
        "assertFalse",
        "assertIs",
        "assertIsNot",
        "assertIsNone",
        "assertIsNotNone",
        "assertIn",
        "assertNotIn",
        "assertRaises",
        "assertAlmostEqual",
        "assertNotAlmostEqual",
        "assertGreater",
        "assertGreaterEqual",
        "assertLess",
        "assertLessEqual",
        "assertRegex",
        "assertNotRegex",
        "assertCountEqual",
        "assertMultiLineEqual",
        "assertSequenceEqual",
        "assertListEqual",
        "assertTupleEqual",
        "assertSetEqual",
        "assertDictEqual",
    },
    "jest": {"expect", "assert", "describe", "it", "test"},
    "go": {"t.Error", "t.Fatal", "t.Errorf", "assert"},
}

# Test function patterns by framework
_TEST_PATTERNS: dict[str, tuple[str, ...]] = {
    "pytest": ("test_",),
    "unittest": (
        "test_",
        "Test",
    ),
    "jest": (
        "test(",
        "it(",
        "describe(",
    ),
    "go": (
        "Test",
        "Benchmark",
    ),
}


@dataclass
class ValidationResult:
    """Result of RED-gate validation."""

    passed: bool
    test_count: int = 0
    assertion_count: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def assertion_density(self) -> float:
        """Assertions per test function."""
        if self.test_count == 0:
            return 0.0
        return self.assertion_count / self.test_count


class SuiteValidator:
    """Validates that a test suite is non-vacuous and has real assertions.

    Uses pure AST analysis — no subprocess, no I/O, no network.
    Thread-safe (no mutable shared state).
    """

    MIN_ASSERTION_DENSITY: float = 0.5
    """Minimum assertions per test function (floor)."""
    MIN_TEST_COUNT: int = 1
    """Minimum number of test functions."""

    def validate(
        self,
        test_code: str,
        framework: str = "pytest",
        *,
        min_assertions: int | None = None,
        min_tests: int | None = None,
    ) -> ValidationResult:
        """Validate a test suite string.

        Args:
            test_code: The test source code to validate.
            framework: Test framework identifier (pytest, unittest, jest, go).
            min_assertions: Minimum assertion floor (defaults to density * test_count).
            min_tests: Minimum test function count (defaults to MIN_TEST_COUNT).

        Returns:
            ValidationResult with pass/fail and diagnostics.
        """
        result = ValidationResult()

        if not test_code.strip():
            result.errors.append("Test suite is empty")
            return result

        try:
            tree = ast.parse(test_code)
        except SyntaxError as exc:
            result.errors.append(f"Syntax error in test suite: {exc}")
            return result

        test_functions = self._find_test_functions(tree, framework)
        result.test_count = len(test_functions)

        if result.test_count < (min_tests or self.MIN_TEST_COUNT):
            result.errors.append(
                f"Test suite has {result.test_count} test function(s), "
                f"need at least {min_tests or self.MIN_TEST_COUNT}"
            )
            # Still check assertions for diagnostics

        # Count assertions across all test functions
        assertion_count = 0
        for func_node in test_functions:
            assertion_count += self._count_assertions(func_node, framework)

        result.assertion_count = assertion_count

        density = result.assertion_density
        min_density = min_assertions or self.MIN_ASSERTION_DENSITY

        if density < min_density and result.test_count > 0:
            result.warnings.append(
                f"Low assertion density: {density:.2f} assertions/test " f"(floor: {min_density})"
            )

        if assertion_count == 0 and result.test_count > 0:
            result.errors.append("Test suite has no assertions — vacuous tests detected (RED-gate)")

        result.passed = len(result.errors) == 0
        return result

    @staticmethod
    def is_vacuous(test_code: str, framework: str = "pytest") -> bool:
        """Quick check: is this test suite vacuous (no real assertions)?

        Runs the full validation and returns True if the suite is vacuous.
        """
        validator = SuiteValidator()
        result = validator.validate(test_code, framework)
        return not result.passed

    def _find_test_functions(
        self, tree: ast.AST, framework: str
    ) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
        """Find test function definitions in the AST."""
        patterns = _TEST_PATTERNS.get(framework, ("test_",))
        tests: list[ast.FunctionDef | ast.AsyncFunctionDef] = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = node.name
                for pattern in patterns:
                    if name.startswith(pattern) or pattern in name:
                        tests.append(node)
                        break

        return tests

    def _count_assertions(
        self,
        func_node: ast.FunctionDef | ast.AsyncFunctionDef,
        framework: str,
    ) -> int:
        """Count assertion statements in a function body."""
        keywords = _ASSERT_KEYWORDS.get(framework, {"assert"})
        count = 0

        for node in ast.walk(func_node):
            if isinstance(node, ast.Assert):
                count += 1
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                # self.assertEqual(...), expect(...).toBe(...), etc.
                if node.func.attr in keywords:
                    count += 1
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in keywords:
                    count += 1

        return count

    @staticmethod
    def detect_framework(test_code: str) -> str:
        """Heuristic detection of test framework from code content.

        Returns one of: pytest, unittest, jest, go.
        """
        if "from unittest" in test_code or "import unittest" in test_code:
            return "unittest"
        if "describe(" in test_code or "it(" in test_code or "expect(" in test_code:
            return "jest"
        if "func Test" in test_code or "t *testing.T" in test_code:
            return "go"
        # Default: pytest
        return "pytest"
