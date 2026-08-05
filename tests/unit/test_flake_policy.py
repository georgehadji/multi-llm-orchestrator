"""
Tests for E-5: flake rerun policy and quarantine tracking.
===========================================================
The rerun policy lives in TestRunnerBase (single execution path); the
quarantine accounting in FlakeTracker. Unit tests use fake sandboxes so
no real subprocess is needed.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from orchestrator.application.testing.flake_tracker import FlakeTracker
from orchestrator.domain.testing_models import IsolationLevel, Workspace
from orchestrator.infrastructure.test_runners.pytest_runner import PytestRunner


class _FakeSandbox:
    """Sandbox that writes a pytest-json-report file (like the real flow)."""

    level = IsolationLevel.SUBPROCESS

    def __init__(self, script: list[tuple[int, dict]]) -> None:
        # script: list of (exit_code, report_dict)
        self.script = list(script)
        self.calls = 0

    async def exec(self, argv, *, cwd, env, timeout_s):
        import json

        self.calls += 1
        if len(self.script) < self.calls:
            return (-1, "", "no more scripted responses")
        exit_code, report_dict = self.script[self.calls - 1]
        (Path(cwd) / ".orch-pytest-report.json").write_text(
            json.dumps(report_dict), encoding="utf-8"
        )
        return (exit_code, "", "")


def _flaky_report(outcome: str) -> dict:
    return {
        "exitcode": 1 if outcome == "failed" else 0,
        "tests": [{"nodeid": "test_main.py::test_flaky", "outcome": outcome, "duration": 0.01}],
    }


def _fail_then_pass_script() -> list[tuple[int, dict]]:
    """Run 1: one test fails. Run 2: same test passes (flake)."""
    return [(1, _flaky_report("failed")), (0, _flaky_report("passed"))]


@pytest.mark.unit
class TestFlakeRerun:
    """E-5: pass-on-rerun => flaky, excluded from the failure verdict."""

    def test_flaky_test_marked_and_suite_passes(self, monkeypatch) -> None:
        monkeypatch.setenv("ORCH_FLAKE_RERUN", "1")
        sandbox = _FakeSandbox(_fail_then_pass_script())
        runner = PytestRunner(sandbox=sandbox)

        report = asyncio.run(
            runner.run(Workspace(root=Path("."), framework="pytest"), timeout_s=10)
        )
        assert sandbox.calls == 2, "failing suite must be re-run exactly once"
        assert report.passed is True
        assert report.flaky_node_ids == ("test_main.py::test_flaky",)

    def test_genuine_failure_not_masked(self, monkeypatch) -> None:
        monkeypatch.setenv("ORCH_FLAKE_RERUN", "1")
        fail_twice = [(1, _flaky_report("failed")), (1, _flaky_report("failed"))]
        sandbox = _FakeSandbox(fail_twice)
        runner = PytestRunner(sandbox=sandbox)
        report = asyncio.run(
            runner.run(Workspace(root=Path("."), framework="pytest"), timeout_s=10)
        )
        assert report.passed is False
        assert report.flaky_node_ids == ()

    def test_rerun_disabled_flag(self, monkeypatch) -> None:
        monkeypatch.setenv("ORCH_FLAKE_RERUN", "0")
        sandbox = _FakeSandbox(_fail_then_pass_script())
        runner = PytestRunner(sandbox=sandbox)
        report = asyncio.run(
            runner.run(Workspace(root=Path("."), framework="pytest"), timeout_s=10)
        )
        assert sandbox.calls == 1, "ORCH_FLAKE_RERUN=0 must disable the rerun"
        assert report.passed is False

    def test_vacuous_suite_not_rerun(self, monkeypatch) -> None:
        monkeypatch.setenv("ORCH_FLAKE_RERUN", "1")
        sandbox = _FakeSandbox([(5, {"exitcode": 5, "tests": []})])
        runner = PytestRunner(sandbox=sandbox)
        report = asyncio.run(
            runner.run(Workspace(root=Path("."), framework="pytest"), timeout_s=10)
        )
        assert sandbox.calls == 1, "vacuous suites must not be re-run"
        assert report.passed is False


@pytest.mark.unit
class TestFlakeTracker:
    """Three observations quarantine a node id."""

    def test_quarantine_after_three(self) -> None:
        tracker = FlakeTracker(quarantine_threshold=3)
        assert tracker.record(("t::x",)) == ()
        assert tracker.record(("t::x",)) == ()
        assert tracker.record(("t::x",)) == ("t::x",)
        assert tracker.quarantined == ("t::x",)
        assert tracker.observation_count("t::x") == 3

    def test_no_quarantine_below_threshold(self) -> None:
        tracker = FlakeTracker(quarantine_threshold=3)
        tracker.record(("t::x",))
        tracker.record(("t::y",))
        assert tracker.quarantined == ()
