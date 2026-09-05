"""
Tests for SuiteValidator (E-2 RED-gate and assertion floor).
=============================================================
"""

from __future__ import annotations

import pytest

from orchestrator.application.testing.suite_validator import SuiteValidator

pytestmark = pytest.mark.unit


class TestSuiteValidator:
    """Verify RED-gate detects vacuous suites and assertion floors."""

    def test_non_vacuous_suite_passes(self) -> None:
        """A suite with real tests and assertions passes."""
        code = """
def test_add():
    result = add(1, 2)
    assert result == 3

def test_subtract():
    result = subtract(5, 3)
    assert result == 2
"""
        result = SuiteValidator().validate(code, framework="pytest")
        assert result.passed, f"Expected pass, got errors: {result.errors}"
        assert result.test_count == 2
        assert result.assertion_count >= 2

    def test_vacuous_suite_fails(self) -> None:
        """A suite with test functions but no assertions fails."""
        code = """
def test_add():
    result = add(1, 2)

def test_subtract():
    result = subtract(5, 3)
"""
        result = SuiteValidator().validate(code, framework="pytest")
        assert not result.passed
        assert any("no assertions" in e.lower() for e in result.errors)

    def test_empty_suite_fails(self) -> None:
        """An empty string fails validation."""
        result = SuiteValidator().validate("", framework="pytest")
        assert not result.passed
        assert any("empty" in e.lower() for e in result.errors)

    def test_no_test_functions_fails(self) -> None:
        """A file with no test functions fails."""
        code = """
def helper():
    return 42
"""
        result = SuiteValidator().validate(code, framework="pytest")
        assert not result.passed
        assert any("test function" in e.lower() for e in result.errors)

    def test_syntax_error_fails(self) -> None:
        """Invalid Python syntax fails validation."""
        code = "def test_bad(::"
        result = SuiteValidator().validate(code, framework="pytest")
        assert not result.passed
        assert any("syntax error" in e.lower() for e in result.errors)

    def test_unittest_assertions_detected(self) -> None:
        """Unittest-style self.assertEqual is detected."""
        code = """
from unittest import TestCase

class TestMath(TestCase):
    def test_add(self):
        self.assertEqual(add(1, 2), 3)
"""
        result = SuiteValidator().validate(code, framework="unittest")
        assert result.passed, f"Expected pass, got errors: {result.errors}"
        assert result.assertion_count >= 1

    def test_assertion_density_warning(self) -> None:
        """Low assertion density produces a warning but still passes."""
        code = """
def test_a():
    assert True

def test_b():
    pass  # no assertion here
"""
        result = SuiteValidator().validate(code, framework="pytest")
        # Still passes because there's at least one assertion
        assert result.passed
        assert len(result.warnings) > 0

    def test_is_vacuous_static_method(self) -> None:
        """Static is_vacuous() returns True for empty suites."""
        assert SuiteValidator.is_vacuous("", framework="pytest")
        assert not SuiteValidator.is_vacuous("def test_x(): assert 1", framework="pytest")

    def test_detect_framework_heuristic(self) -> None:
        """Framework detection works correctly."""
        assert SuiteValidator.detect_framework("from unittest import") == "unittest"
        assert SuiteValidator.detect_framework('describe("group",') == "jest"
        assert SuiteValidator.detect_framework("def test_x(): pass") == "pytest"
