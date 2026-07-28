"""
Characterization tests for test-execution behavior (Phase 0, B-1).
==================================================================
Author: Implementation Plan (Autonomous Testing Engine)

Locks present behavior before refactoring so consolidation regressions
are detectable. Tests use only domain types — no subprocess, no I/O.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator.domain.testing_models import (
    CheckScope,
    IsolationLevel,
    SuiteReport,
    TestOutcome,
    TestSelection,
    TestStatus,
    Workspace,
)


@pytest.mark.unit
class TestSuiteReportIsVacuous:
    """Characterize SuiteReport.is_vacuous_result across real-world shapes."""

    def test_empty_outcomes_is_vacuous(self) -> None:
        """Zero outcomes → vacuous, even if passed=True (D-7 false-positive guard)."""
        report = SuiteReport(passed=True, exit_code=0, outcomes=())
        assert report.is_vacuous_result is True
        assert report.executed == 0

    def test_collection_error_no_outcomes_is_vacuous(self) -> None:
        """Collection errors with no outcomes → vacuous."""
        report = SuiteReport(
            passed=False,
            exit_code=2,
            collection_errors=("ImportError: No module named missing_dep",),
        )
        assert report.is_vacuous_result is True
        assert report.executed == 0
        assert len(report.collection_errors) == 1

    def test_all_pass_with_outcomes_is_not_vacuous(self) -> None:
        """Real passing outcomes → not vacuous."""
        report = SuiteReport(
            passed=True,
            exit_code=0,
            outcomes=(
                TestOutcome(node_id="test_a", status=TestStatus.PASSED, duration_ms=10.0),
                TestOutcome(node_id="test_b", status=TestStatus.PASSED, duration_ms=20.0),
            ),
        )
        assert report.is_vacuous_result is False
        assert report.executed == 2

    def test_mixed_outcomes_is_not_vacuous(self) -> None:
        """Mixed pass/fail/error/skip → not vacuous."""
        report = SuiteReport(
            passed=False,
            exit_code=1,
            outcomes=(
                TestOutcome(node_id="test_a", status=TestStatus.PASSED, duration_ms=10.0),
                TestOutcome(
                    node_id="test_b",
                    status=TestStatus.FAILED,
                    duration_ms=5.0,
                    message="AssertionError",
                ),
                TestOutcome(
                    node_id="test_c",
                    status=TestStatus.ERROR,
                    duration_ms=0.0,
                    message="ZeroDivisionError",
                ),
                TestOutcome(node_id="test_d", status=TestStatus.SKIPPED, duration_ms=0.0),
            ),
        )
        assert report.is_vacuous_result is False
        assert report.executed == 4


@pytest.mark.unit
class TestSuiteReportExecuted:
    """Characterize SuiteReport.executed property."""

    def test_empty_outcomes_executed_zero(self) -> None:
        report = SuiteReport(passed=True, exit_code=0)
        assert report.executed == 0

    def test_single_outcome_executed_one(self) -> None:
        report = SuiteReport(
            passed=True,
            exit_code=0,
            outcomes=(TestOutcome(node_id="x", status=TestStatus.PASSED, duration_ms=1.0),),
        )
        assert report.executed == 1

    def test_error_outcome_still_counts_as_executed(self) -> None:
        """An ERROR outcome is still an executed test — it ran but failed."""
        report = SuiteReport(
            passed=False,
            exit_code=1,
            outcomes=(
                TestOutcome(
                    node_id="test_with_bad_fixture",
                    status=TestStatus.ERROR,
                    duration_ms=2.0,
                    message="fixture 'db' not found",
                ),
            ),
        )
        assert report.executed == 1
        assert report.outcomes[0].status == TestStatus.ERROR
        assert "fixture" in report.outcomes[0].message


@pytest.mark.unit
class TestTestOutcome:
    """Characterize TestOutcome construction and field access."""

    def test_minimal_construction(self) -> None:
        outcome = TestOutcome(node_id="test_x", status=TestStatus.PASSED, duration_ms=15.0)
        assert outcome.node_id == "test_x"
        assert outcome.status == TestStatus.PASSED
        assert outcome.duration_ms == 15.0
        assert outcome.message == ""

    def test_with_message(self) -> None:
        outcome = TestOutcome(
            node_id="test_fail",
            status=TestStatus.FAILED,
            duration_ms=3.0,
            message="Expected 42, got 0",
        )
        assert "42" in outcome.message

    def test_frozen_prevents_mutation(self) -> None:
        outcome = TestOutcome(node_id="t", status=TestStatus.PASSED, duration_ms=1.0)
        with pytest.raises(AttributeError):
            outcome.status = TestStatus.FAILED  # type: ignore[misc]

    def test_skipped_outcome(self) -> None:
        outcome = TestOutcome(
            node_id="test_skip",
            status=TestStatus.SKIPPED,
            duration_ms=0.0,
            message="requires_db marker",
        )
        assert outcome.status == TestStatus.SKIPPED
        assert outcome.duration_ms == 0.0


@pytest.mark.unit
class TestWorkspace:
    """Characterize Workspace construction."""

    def test_minimal_workspace(self) -> None:
        ws = Workspace(root=Path("/tmp/test"), framework="pytest")
        assert ws.root == Path("/tmp/test")
        assert ws.framework == "pytest"
        assert len(ws.source_files) == 0
        assert len(ws.test_files) == 0
        assert ws.manifest is None
        assert ws.env == {}

    def test_with_source_and_test_files(self) -> None:
        ws = Workspace(
            root=Path("/tmp/test"),
            framework="pytest",
            source_files=(Path("src/main.py"), Path("src/utils.py")),
            test_files=(Path("tests/test_main.py"),),
        )
        assert len(ws.source_files) == 2
        assert len(ws.test_files) == 1

    def test_with_manifest(self) -> None:
        ws = Workspace(root=Path("/tmp/test"), framework="pytest", manifest=Path("pyproject.toml"))
        assert ws.manifest == Path("pyproject.toml")

    def test_with_env(self) -> None:
        ws = Workspace(
            root=Path("/tmp/test"),
            framework="pytest",
            env={"PYTHONPATH": "/tmp/test/src", "CI": "true"},
        )
        assert ws.env["CI"] == "true"

    def test_frozen(self) -> None:
        ws = Workspace(root=Path("/tmp/test"), framework="pytest")
        with pytest.raises(AttributeError):
            ws.framework = "jest"  # type: ignore[misc]


@pytest.mark.unit
class TestTestSelection:
    """Characterize TestSelection construction and defaults."""

    def test_default_full_suite(self) -> None:
        sel = TestSelection()
        assert len(sel.node_ids) == 0
        assert sel.reason == "full"

    def test_with_node_ids_and_reason(self) -> None:
        sel = TestSelection(node_ids=("test_a", "test_b"), reason="changed_files")
        assert len(sel.node_ids) == 2
        assert sel.reason == "changed_files"

    def test_frozen(self) -> None:
        sel = TestSelection(node_ids=("test_a",), reason="changed_files")
        with pytest.raises(AttributeError):
            sel.reason = "full"  # type: ignore[misc]


@pytest.mark.unit
class TestEnums:
    """Characterize enum values for TestStatus, IsolationLevel, CheckScope."""

    def test_test_status_values(self) -> None:
        assert TestStatus.PASSED.value == "passed"
        assert TestStatus.FAILED.value == "failed"
        assert TestStatus.ERROR.value == "error"
        assert TestStatus.SKIPPED.value == "skipped"
        assert TestStatus.XFAILED.value == "xfailed"

    def test_isolation_level_values(self) -> None:
        assert IsolationLevel.NONE.value == "none"
        assert IsolationLevel.SUBPROCESS.value == "subprocess"
        assert IsolationLevel.DOCKER.value == "docker"

    def test_check_scope_values(self) -> None:
        assert CheckScope.ARTIFACT.value == "artifact"
        assert CheckScope.WORKSPACE.value == "workspace"

    def test_test_status_is_str_enum(self) -> None:
        """String enums are comparable to strings."""
        assert TestStatus.PASSED == "passed"

    def test_isolation_level_has_subprocess_default(self) -> None:
        """SUBPROCESS is the default isolation level on SuiteReport."""
        report = SuiteReport(passed=True, exit_code=0)
        assert report.isolation == IsolationLevel.SUBPROCESS


@pytest.mark.unit
class TestSuiteReportEdgeCases:
    """Characterize SuiteReport with various combinations of outcomes and metadata."""

    def test_line_coverage_attached(self) -> None:
        report = SuiteReport(
            passed=True,
            exit_code=0,
            outcomes=(TestOutcome(node_id="test_a", status=TestStatus.PASSED, duration_ms=10.0),),
            line_coverage=0.85,
        )
        assert report.line_coverage == 0.85

    def test_mutation_score_attached(self) -> None:
        report = SuiteReport(
            passed=True,
            exit_code=0,
            line_coverage=0.85,
            mutation_score=0.72,
        )
        assert report.mutation_score == 0.72

    def test_flaky_node_ids(self) -> None:
        report = SuiteReport(
            passed=True,
            exit_code=0,
            outcomes=(
                TestOutcome(node_id="flaky_test", status=TestStatus.PASSED, duration_ms=5.0),
            ),
            flaky_node_ids=("flaky_test",),
        )
        assert "flaky_test" in report.flaky_node_ids

    def test_duration_tracking(self) -> None:
        report = SuiteReport(passed=True, exit_code=0, duration_ms=1234.5)
        assert report.duration_ms == 1234.5

    def test_truncated_output(self) -> None:
        report = SuiteReport(passed=True, exit_code=0, truncated_output="test output here")
        assert "test output" in report.truncated_output

    def test_frozen_prevents_mutation(self) -> None:
        """SuiteReport is a frozen dataclass."""
        report = SuiteReport(passed=True, exit_code=0)
        with pytest.raises(AttributeError):
            report.passed = False  # type: ignore[misc]

    def test_vacuous_even_when_passed_is_true(self) -> None:
        """passed=True with zero outcomes is still vacuous (D-7 false-positive guard)."""
        report = SuiteReport(passed=True, exit_code=0)
        # The report says 'passed' but nothing was executed — must be marked vacuous
        assert report.is_vacuous_result is True
        assert report.executed == 0
