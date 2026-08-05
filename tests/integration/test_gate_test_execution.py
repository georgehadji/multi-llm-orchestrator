"""
Tests for E-6: test execution wired into VerificationGate.
===========================================================
Covers the WORKSPACE dispatch, the test_execution check factory, and the
ORCH_TEST_GATE tri-state (off / shadow / enforce).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator.application.verification_gate import VerificationGate
from orchestrator.domain.testing_models import CheckScope, Workspace
from orchestrator.domain.verification import CheckOutcome, VerificationPolicy
from orchestrator.infrastructure.verification_checks import (
    _make_test_execution_check,
    default_checks,
)


def _failing_workspace(tmp_path: Path) -> Workspace:
    (tmp_path / "main.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (tmp_path / "test_main.py").write_text(
        "from main import add\ndef test_bad():\n    assert add(1, 2) == 99\n",
        encoding="utf-8",
    )
    return Workspace(
        root=tmp_path,
        framework="pytest",
        source_files=(tmp_path / "main.py",),
        test_files=(tmp_path / "test_main.py",),
        env={"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
    )


def _passing_workspace(tmp_path: Path) -> Workspace:
    (tmp_path / "main.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (tmp_path / "test_main.py").write_text(
        "from main import add\ndef test_add():\n    assert add(1, 2) == 3\n",
        encoding="utf-8",
    )
    return Workspace(
        root=tmp_path,
        framework="pytest",
        source_files=(tmp_path / "main.py",),
        test_files=(tmp_path / "test_main.py",),
        env={"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
    )


@pytest.mark.unit
class TestTestExecutionCheck:
    """The check factory produces a WORKSPACE-scoped check."""

    def test_scope_is_workspace(self) -> None:
        check = _make_test_execution_check()
        assert check.scope is CheckScope.WORKSPACE
        assert check.name == "test_execution"

    def test_check_is_in_default_checks(self) -> None:
        names = {c.name for c in default_checks()}
        assert "test_execution" in names


@pytest.mark.integration
class TestGateTestExecution:
    """Gate dispatch: workspace drives the test_execution check."""

    @pytest.mark.asyncio
    async def test_workspace_absent_emits_not_run(self) -> None:
        gate = VerificationGate(checks=[_make_test_execution_check()])
        result = await gate.run("def add(a, b): return a + b")
        receipt = next(r for r in result.receipts if r.check_name == "test_execution")
        assert receipt.outcome is CheckOutcome.NOT_RUN
        # NOT_RUN must not floor the score
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_passing_workspace_passes(self, tmp_path: Path) -> None:
        gate = VerificationGate(checks=[_make_test_execution_check()])
        result = await gate.run("", workspace=_passing_workspace(tmp_path))
        receipt = next(r for r in result.receipts if r.check_name == "test_execution")
        assert receipt.outcome is CheckOutcome.PASSED
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_failing_workspace_floors_in_enforce(self, tmp_path: Path) -> None:
        gate = VerificationGate(checks=[_make_test_execution_check()])
        result = await gate.run("", workspace=_failing_workspace(tmp_path))
        receipt = next(r for r in result.receipts if r.check_name == "test_execution")
        assert receipt.outcome is CheckOutcome.FAILED
        assert result.score <= VerificationGate.FAIL_SCORE_FLOOR

    @pytest.mark.asyncio
    async def test_failing_workspace_shadow_does_not_floor(self, tmp_path: Path) -> None:
        gate = VerificationGate(checks=[_make_test_execution_check()])
        result = await gate.run(
            "", workspace=_failing_workspace(tmp_path), non_blocking={"test_execution"}
        )
        receipt = next(r for r in result.receipts if r.check_name == "test_execution")
        assert receipt.outcome is CheckOutcome.FAILED  # outcome still recorded
        assert result.score == 1.0  # but shadow mode does not floor

    @pytest.mark.asyncio
    async def test_vacuous_workspace_fails(self, tmp_path: Path) -> None:
        (tmp_path / "main.py").write_text("def add(a, b):\n    return a + b\n")
        (tmp_path / "test_main.py").write_text("def helper():\n    return 1\n")
        ws = Workspace(
            root=tmp_path,
            framework="pytest",
            env={"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
        )
        gate = VerificationGate(checks=[_make_test_execution_check()])
        result = await gate.run("", workspace=ws)
        receipt = next(r for r in result.receipts if r.check_name == "test_execution")
        assert receipt.outcome is CheckOutcome.FAILED
        assert "vacuous" in (receipt.reason or "")
