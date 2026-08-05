"""
Tests for the RED-gate decision logic (E-2).
=============================================
Pure-function tests: every degenerate suite form is rejected, a genuine
suite survives with zero discards, and mode (off/warn/enforce) drives the
decision matrix.
"""

from __future__ import annotations

import pytest

from orchestrator.application.testing.red_gate import (
    decide_red_gate,
    discard_vacuous_tests,
)


@pytest.mark.unit
class TestRedGateDecisions:
    """Decision matrix for vacuity/assertion outcomes."""

    def _decide(self, *, assertion_ok=True, total=4, vacuous=(), mode="enforce", retry=False):
        return decide_red_gate(
            assertion_ok=assertion_ok,
            assertion_errors=("bad" if not assertion_ok else ""),
            total=total,
            vacuous_node_ids=tuple(vacuous),
            mode=mode,
            retry_used=retry,
        )

    def test_genuine_suite_proceeds_no_discards(self) -> None:
        result = self._decide(total=4, vacuous=())
        assert result.decision == "proceed"
        assert result.discard_node_ids == ()

    def test_all_vacuous_regenerates(self) -> None:
        result = self._decide(total=4, vacuous=("t::a", "t::b", "t::c", "t::d"))
        assert result.decision == "regenerate"

    def test_mostly_vacuous_regenerates_once_then_fails(self) -> None:
        first = self._decide(total=4, vacuous=("t::a", "t::b", "t::c"))
        assert first.decision == "regenerate"
        second = self._decide(total=4, vacuous=("t::a", "t::b", "t::c"), retry=True)
        assert second.decision == "fail"

    def test_minority_vacuous_discarded_not_regenerated(self) -> None:
        result = self._decide(total=4, vacuous=("t::a",))
        assert result.decision == "proceed"
        assert result.discard_node_ids == ("t::a",)

    def test_warn_mode_never_discards_or_fails(self) -> None:
        result = self._decide(total=4, vacuous=("t::a", "t::b", "t::c"), mode="warn")
        assert result.decision == "proceed"
        assert result.discard_node_ids == ()
        assert result.diagnosis  # measured and logged

    def test_off_mode_skips(self) -> None:
        result = self._decide(total=4, vacuous=("t::a",), mode="off")
        assert result.decision == "proceed"
        assert result.discard_node_ids == ()

    def test_assertion_floor_failure_enforce_regenerates(self) -> None:
        result = self._decide(assertion_ok=False, total=0, mode="enforce")
        assert result.decision == "regenerate"
        assert result.assertion_ok is False

    def test_assertion_floor_failure_retry_fails(self) -> None:
        result = self._decide(assertion_ok=False, total=0, mode="enforce", retry=True)
        assert result.decision == "fail"

    def test_assertion_floor_failure_warn_proceeds(self) -> None:
        result = self._decide(assertion_ok=False, total=0, mode="warn")
        assert result.decision == "proceed"

    def test_zero_executed_is_not_a_vacuity_verdict(self) -> None:
        result = self._decide(total=0, mode="enforce")
        assert result.decision == "proceed"

    def test_vacuous_ratio_property(self) -> None:
        result = self._decide(total=4, vacuous=("a", "b"))
        assert result.vacuous_ratio == 0.5


@pytest.mark.unit
class TestDiscardVacuous:
    """AST-based removal of vacuous tests."""

    def test_removes_named_tests_only(self) -> None:
        code = (
            "from main import add\n"
            "def test_real():\n    assert add(1, 2) == 3\n"
            "def test_vacuous():\n    assert True\n"
        )
        cleaned = discard_vacuous_tests(code, {"test_main.py::test_vacuous"})
        assert "def test_vacuous" not in cleaned
        assert "def test_real" in cleaned
        assert "assert add(1, 2) == 3" in cleaned

    def test_noop_when_no_match(self) -> None:
        code = "def test_a():\n    assert 1\n"
        assert discard_vacuous_tests(code, {"other.py::test_b"}) == code

    def test_empty_set_returns_unchanged(self) -> None:
        code = "def test_a():\n    assert 1\n"
        assert discard_vacuous_tests(code, set()) == code
