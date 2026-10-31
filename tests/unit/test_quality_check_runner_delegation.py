"""F-6 consolidation of ``orchestrator/quality_control.py``.

``QualityCheckRunner`` owns the quality-gating checks (performance, security,
level dispatch). Running the unit-test suite is NOT its job: ``_run_pytest``
delegates to the authoritative runner behind ``TestExecutorPort``
(``orchestrator.infrastructure.test_runners``). The old ``TestRunner`` name
survives only as a deprecated subclass shim.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from orchestrator.domain.testing_models import (
    SuiteReport,
    TestOutcome,
    TestSelection,
    TestStatus,
    Workspace,
)
from orchestrator.quality_control import (
    QualityCheckRunner,
    QualityController,
    QualityReport,
    TestLevel,
)


class FakeExecutor:
    """Stands in for the infrastructure runner; records calls, never runs pytest."""

    def __init__(self, report: SuiteReport) -> None:
        self.report = report
        self.calls: list[tuple[Workspace, TestSelection | None, float]] = []

    async def run(self, workspace, selection=None, *, timeout_s=120.0) -> SuiteReport:
        self.calls.append((workspace, selection, timeout_s))
        return self.report

    def supports(self, framework: str) -> bool:
        return framework == "pytest"


def _report(*outcomes: TestOutcome, **kw) -> SuiteReport:
    kw.setdefault("passed", all(o.status is TestStatus.PASSED for o in outcomes))
    kw.setdefault("exit_code", 0 if kw["passed"] else 1)
    return SuiteReport(outcomes=tuple(outcomes), **kw)


PASS = TestOutcome("tests/test_a.py::test_a", TestStatus.PASSED, 12.0)
FAIL = TestOutcome("tests/test_a.py::test_b", TestStatus.FAILED, 3.0, "assert 1 == 2")


@pytest.mark.asyncio
async def test_run_pytest_delegates_to_executor(tmp_path: Path) -> None:
    fake = FakeExecutor(_report(PASS))
    await QualityCheckRunner(test_executor=fake)._run_pytest(tmp_path)

    assert len(fake.calls) == 1
    workspace, selection, _ = fake.calls[0]
    assert workspace.root == tmp_path
    assert workspace.framework == "pytest"
    assert selection is not None and selection.node_ids == ("tests",)


@pytest.mark.asyncio
async def test_run_pytest_maps_outcomes_and_coverage(tmp_path: Path) -> None:
    fake = FakeExecutor(_report(PASS, FAIL, line_coverage=87.0))
    results = await QualityCheckRunner(test_executor=fake)._run_pytest(tmp_path)

    assert [r.name for r in results] == [PASS.node_id, FAIL.node_id]
    assert all(r.level is TestLevel.UNIT for r in results)
    assert [r.passed for r in results] == [True, False]
    assert results[0].duration_ms == 12.0
    assert results[1].message == "assert 1 == 2"
    assert results[0].coverage_percent == 87.0
    assert results[1].coverage_percent is None


@pytest.mark.asyncio
async def test_run_pytest_without_coverage_leaves_none(tmp_path: Path) -> None:
    fake = FakeExecutor(_report(PASS))
    results = await QualityCheckRunner(test_executor=fake)._run_pytest(tmp_path)
    assert results[0].coverage_percent is None


@pytest.mark.asyncio
async def test_run_pytest_vacuous_or_broken_suite_is_never_green(tmp_path: Path) -> None:
    broken = SuiteReport(
        passed=False,
        exit_code=2,
        collection_errors=("tests/test_x.py: ImportError",),
        truncated_output="boom",
    )
    results = await QualityCheckRunner(test_executor=FakeExecutor(broken))._run_pytest(tmp_path)

    assert len(results) == 1
    assert results[0].name == "pytest_suite"
    assert results[0].level is TestLevel.UNIT
    assert results[0].passed is False
    assert "ImportError" in results[0].message


@pytest.mark.asyncio
async def test_run_pytest_runner_unavailable_reports_failure(tmp_path: Path) -> None:
    from orchestrator.quality.run_tests import TestRunnerUnavailableError

    class Missing(QualityCheckRunner):
        def _resolve_executor(self):
            raise TestRunnerUnavailableError("nope")

    results = await Missing()._run_pytest(tmp_path)
    assert len(results) == 1
    assert results[0].passed is False
    assert "nope" in results[0].message


@pytest.mark.asyncio
async def test_run_tests_dispatches_unit_level_to_executor(tmp_path: Path) -> None:
    fake = FakeExecutor(_report(PASS))
    results = await QualityCheckRunner(test_executor=fake).run_tests(tmp_path, TestLevel.UNIT)
    assert len(fake.calls) == 1
    assert [r.name for r in results] == [PASS.node_id]


def test_legacy_testrunner_is_deprecated_subclass_shim() -> None:
    from orchestrator.quality_control import TestRunner

    assert issubclass(TestRunner, QualityCheckRunner)
    with pytest.warns(DeprecationWarning, match="QualityCheckRunner"):
        TestRunner()


@pytest.mark.asyncio
async def test_legacy_testrunner_still_works(tmp_path: Path) -> None:
    from orchestrator.quality_control import TestRunner

    (tmp_path / "clean.py").write_text("x = 1\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        results = await TestRunner()._run_security_checks(tmp_path)
    assert results[0].passed is True


def test_legacy_testrunner_is_same_object_via_both_paths() -> None:
    from orchestrator.quality.quality_control import TestRunner as ViaShim
    from orchestrator.quality_control import TestRunner as ViaRoot

    assert ViaShim is ViaRoot


@pytest.mark.asyncio
async def test_quality_controller_default_constructible_and_gate_shape(tmp_path: Path) -> None:
    qc = QualityController(storage_path=tmp_path / ".quality")
    assert isinstance(qc._test_runner, QualityCheckRunner)

    qc = QualityController(
        storage_path=tmp_path / ".quality2",
        test_executor=FakeExecutor(_report(PASS, line_coverage=90.0)),
    )
    project = tmp_path / "proj"
    project.mkdir()
    (project / "m.py").write_text("def f():\n    return 1\n")

    report = await qc.run_quality_gate("p1", project, [TestLevel.UNIT])

    assert isinstance(report, QualityReport)
    assert report.project_id == "p1"
    assert [r.name for r in report.test_results] == [PASS.node_id]
    assert report.average_coverage == 90.0


def test_resolve_test_executor_returns_pytest_runner() -> None:
    from orchestrator.quality.run_tests import (
        TestRunnerUnavailableError,
        resolve_test_executor,
    )

    assert resolve_test_executor("pytest").supports("pytest")
    with pytest.raises(TestRunnerUnavailableError):
        resolve_test_executor("no-such-framework")
