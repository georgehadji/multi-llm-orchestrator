"""
Tests for the repair policy (E-3) — plateau, hash lock, escalation.
====================================================================
Pure-logic tests: no subprocess, no filesystem, no clock.
"""

from __future__ import annotations

import pytest

from orchestrator.application.testing.repair_policy import (
    FailureSignature,
    RepairPolicy,
    TestHashLockError,
    normalize_failure_signature,
)


@pytest.mark.unit
class TestSignatureNormalization:
    """Failure signatures must be comparable and stable."""

    def test_extracts_node_exception_frame(self) -> None:
        error = (
            "test_main.py::test_bad: assert 3 == 99\n"
            "Traceback (most recent call last):\n"
            '  File "test_main.py", line 3, in test_bad\n'
            "AssertionError: assert 3 == 99"
        )
        sig = normalize_failure_signature(error)
        assert sig is not None
        assert sig.node_id == "test_main.py::test_bad"
        assert sig.exception_type == "AssertionError"
        assert sig.first_frame == "test_main.py:3"

    def test_empty_error_returns_none(self) -> None:
        assert normalize_failure_signature("") is None
        assert normalize_failure_signature("   ") is None

    def test_identical_messages_same_signature(self) -> None:
        a = normalize_failure_signature("test_main.py::test_x: TypeError: boom")
        b = normalize_failure_signature("test_main.py::test_x: TypeError: boom")
        assert a == b

    def test_different_node_differs(self) -> None:
        a = normalize_failure_signature("test_main.py::test_a: TypeError: boom")
        b = normalize_failure_signature("test_main.py::test_b: TypeError: boom")
        assert a != b


@pytest.mark.unit
class TestRepairPolicyPlateau:
    """Two consecutive identical signatures => plateau."""

    def _errors(self, node: str = "test_main.py::test_bad") -> list[str]:
        return [f"{node}: AssertionError: assert 3 == 99"]

    def test_no_plateau_after_one_failure(self) -> None:
        policy = RepairPolicy(max_iterations=5)
        policy.record_failure(self._errors())
        assert policy.is_plateau() is False

    def test_plateau_after_two_identical(self) -> None:
        policy = RepairPolicy(max_iterations=5)
        policy.record_failure(self._errors())
        assert policy.is_plateau() is False
        policy.record_failure(self._errors())
        assert policy.is_plateau() is True

    def test_no_plateau_on_different_failures(self) -> None:
        policy = RepairPolicy(max_iterations=5)
        policy.record_failure(self._errors())
        policy.record_failure(self._errors(node="test_main.py::test_other"))
        assert policy.is_plateau() is False

    def test_repeated_identical_failure_stops_at_iteration_two(self) -> None:
        """E-3 acceptance: stops at iteration 2 of the tier, not iteration 5."""
        policy = RepairPolicy(max_iterations=5)
        simulated_iterations = 0
        for _ in range(5):
            policy.record_failure(self._errors())
            simulated_iterations += 1
            if policy.is_plateau():
                break
        assert simulated_iterations == 2


@pytest.mark.unit
class TestHashLock:
    """The oracle is immutable during repair."""

    def test_lock_arms_and_verifies(self) -> None:
        policy = RepairPolicy()
        code = "def test_x():\n    assert 1\n"
        policy.lock_tests(code)
        policy.assert_tests_locked(code)  # unchanged: no raise

    def test_mutation_raises(self) -> None:
        policy = RepairPolicy()
        policy.lock_tests("def test_x():\n    assert 1\n")
        with pytest.raises(TestHashLockError):
            policy.assert_tests_locked("def test_x():\n    assert True\n  # weakened")


@pytest.mark.unit
class TestEscalation:
    """One model-tier escalation maximum."""

    def test_escalate_uses_chain(self, monkeypatch) -> None:
        class FakeA:
            value = "fake/a"

        class FakeB:
            value = "fake/b"

        policy = RepairPolicy(max_escalations=1)
        monkeypatch.setattr(
            "orchestrator.application.testing.repair_policy._get_fallback_chain",
            lambda: {FakeA: FakeB},
        )
        escalated = policy.escalate(FakeA)
        assert escalated is FakeB
        assert policy.escalations == 1
        # Second escalation: budget exhausted -> unchanged.
        assert policy.escalate(FakeB) is FakeB
        assert policy.escalations == 1

    def test_no_chain_entry_returns_same_model(self) -> None:
        policy = RepairPolicy()
        assert policy.escalate("unknown-model") == "unknown-model"
        assert policy.escalations == 0

    def test_diagnostic_mentions_history(self) -> None:
        policy = RepairPolicy()
        policy.record_failure(["test_main.py::test_x: TypeError: boom"])
        diag = policy.diagnostic()
        assert "repair iterations: 1" in diag
        assert "TypeError" in diag
