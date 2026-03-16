"""Tests for REPLVerifier."""
from __future__ import annotations
import pytest


class TestVerificationLevel:
    def test_four_levels_exist(self):
        from orchestrator.verification import VerificationLevel
        assert VerificationLevel.NONE is not None
        assert VerificationLevel.SYNTAX is not None
        assert VerificationLevel.EXECUTION is not None
        assert VerificationLevel.FULL is not None


class TestVerificationResult:
    def test_passed_result(self):
        from orchestrator.verification import VerificationResult, VerificationLevel
        r = VerificationResult(passed=True, level=VerificationLevel.SYNTAX)
        assert r.passed is True
        assert r.error_message is None

    def test_failed_result(self):
        from orchestrator.verification import VerificationResult, VerificationLevel
        r = VerificationResult(passed=False, level=VerificationLevel.SYNTAX, error_message="bad syntax")
        assert r.passed is False
        assert r.error_message == "bad syntax"


class TestREPLVerifierNone:
    def test_none_always_passes(self):
        from orchestrator.verification import REPLVerifier, VerificationLevel
        v = REPLVerifier(level=VerificationLevel.NONE)
        result = v.verify("this is not valid python at all!!!")
        assert result.passed is True


class TestREPLVerifierSyntax:
    def test_valid_syntax_passes(self):
        from orchestrator.verification import REPLVerifier, VerificationLevel
        v = REPLVerifier(level=VerificationLevel.SYNTAX)
        result = v.verify("x = 1 + 2\nprint(x)")
        assert result.passed is True

    def test_invalid_syntax_fails(self):
        from orchestrator.verification import REPLVerifier, VerificationLevel
        v = REPLVerifier(level=VerificationLevel.SYNTAX)
        result = v.verify("def foo(\n  # missing closing paren")
        assert result.passed is False
        assert result.error_message is not None

    def test_empty_code_passes_syntax(self):
        from orchestrator.verification import REPLVerifier, VerificationLevel
        v = REPLVerifier(level=VerificationLevel.SYNTAX)
        result = v.verify("")
        assert result.passed is True


class TestREPLVerifierExecution:
    def test_valid_code_executes(self):
        from orchestrator.verification import REPLVerifier, VerificationLevel
        v = REPLVerifier(level=VerificationLevel.EXECUTION)
        result = v.verify("x = 1 + 2")
        assert result.passed is True

    def test_runtime_error_fails(self):
        from orchestrator.verification import REPLVerifier, VerificationLevel
        v = REPLVerifier(level=VerificationLevel.EXECUTION)
        result = v.verify("raise ValueError('intentional')")
        assert result.passed is False


class TestREPLVerifierFull:
    def test_code_with_output_passes(self):
        from orchestrator.verification import REPLVerifier, VerificationLevel
        v = REPLVerifier(level=VerificationLevel.FULL)
        result = v.verify("print('hello world')")
        assert result.passed is True

    def test_code_without_output_fails(self):
        from orchestrator.verification import REPLVerifier, VerificationLevel
        v = REPLVerifier(level=VerificationLevel.FULL)
        result = v.verify("x = 1")  # no stdout
        assert result.passed is False


class TestSelfHealingLoop:
    def test_passing_code_returns_immediately(self):
        from orchestrator.verification import REPLVerifier, VerificationLevel, self_healing_loop
        v = REPLVerifier(level=VerificationLevel.SYNTAX)
        code, result = self_healing_loop("x = 1", v, max_attempts=3)
        assert result.passed is True
        assert code == "x = 1"

    def test_failing_code_returns_after_max_attempts(self):
        from orchestrator.verification import REPLVerifier, VerificationLevel, self_healing_loop
        v = REPLVerifier(level=VerificationLevel.SYNTAX)
        bad_code = "def foo(\n  # broken"
        code, result = self_healing_loop(bad_code, v, max_attempts=2)
        assert result.passed is False
