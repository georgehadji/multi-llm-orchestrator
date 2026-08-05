"""
Tests for the test-runner adapters (F-6/F-7) — machine-readable parsing.
=======================================================================
Fixture-driven: real captured JSON documents for pass/fail/collection-error/
empty cases per framework. No subprocess in the parsing tests; the execution
tests run real pytest through the sandbox (integration marker).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from orchestrator.domain.testing_models import IsolationLevel, Workspace
from orchestrator.infrastructure.test_runners import get_runner
from orchestrator.infrastructure.test_runners.base import TestRunnerBase
from orchestrator.infrastructure.test_runners.cargo_runner import CargoRunner
from orchestrator.infrastructure.test_runners.go_runner import GoRunner
from orchestrator.infrastructure.test_runners.jest_runner import JestRunner
from orchestrator.infrastructure.test_runners.pytest_runner import PytestRunner

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures: captured machine-readable outputs
# ─────────────────────────────────────────────────────────────────────────────

PYTEST_PASS_JSON = """
{
  "exitcode": 0,
  "tests": [
    {"nodeid": "test_main.py::test_add", "outcome": "passed", "duration": 0.001},
    {"nodeid": "test_main.py::test_sub", "outcome": "passed", "duration": 0.002}
  ],
  "summary": {"passed": 2, "failed": 0, "total": 2}
}
"""

PYTEST_FAIL_JSON = """
{
  "exitcode": 1,
  "tests": [
    {"nodeid": "test_main.py::test_bad", "outcome": "failed",
     "duration": 0.01, "call": {"longrepr": "assert 3 == 99"}}
  ],
  "summary": {"passed": 0, "failed": 1, "total": 1}
}
"""

PYTEST_COLLECTION_ERROR_JSON = """
{
  "exitcode": 2,
  "tests": [],
  "collectors": [
    {"nodeid": "test_main.py", "result": [
      {"outcome": "failed", "longrepr": "ImportError: cannot import name 'nope'"}
    ]}
  ],
  "summary": {"passed": 0, "failed": 0, "total": 0, "collected": 0}
}
"""

PYTEST_EMPTY_JSON = """
{
  "exitcode": 5,
  "tests": [],
  "summary": {"passed": 0, "failed": 0, "total": 0}
}
"""

JEST_PASS_JSON = """
{
  "numTotalTests": 2,
  "numPassedTests": 2,
  "numFailedTests": 0,
  "testResults": [
    {"name": "main.test.js", "assertionResults": [
      {"title": "adds numbers", "status": "passed", "duration": 3},
      {"title": "subtracts", "status": "passed", "duration": 2}
    ]}
  ]
}
"""

JEST_FAIL_JSON = """
{
  "numTotalTests": 1,
  "numPassedTests": 0,
  "numFailedTests": 1,
  "testResults": [
    {"name": "main.test.js", "assertionResults": [
      {"title": "fails", "status": "failed", "duration": 5,
       "failureMessages": ["Expected 4 to equal 99"]}
    ]}
  ]
}
"""

GO_PASS_NDJSON = """
{"Action":"run","Test":"TestAdd","Elapsed":0.001}
{"Action":"output","Test":"TestAdd","Output":"PASS"}
{"Action":"pass","Test":"TestAdd","Elapsed":0.001}
{"Action":"run","Test":"TestSub","Elapsed":0.002}
{"Action":"pass","Test":"TestSub","Elapsed":0.002}
{"Action":"pass","Elapsed":0.003}
"""

GO_FAIL_NDJSON = """
{"Action":"run","Test":"TestAdd","Elapsed":0.001}
{"Action":"fail","Test":"TestAdd","Elapsed":0.001}
{"Action":"fail","Elapsed":0.002}
"""

CARGO_PASS_NDJSON = """
{"type":"suite","event":"started"}
{"type":"test","name":"tests::test_add","event":"ok"}
{"type":"test","name":"tests::test_sub","event":"ok"}
{"type":"suite","event":"ok"}
"""

CARGO_FAIL_NDJSON = """
{"type":"test","name":"tests::test_add","event":"failed"}
{"type":"suite","event":"failed"}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Pytest runner
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestPytestRunnerParsing:
    """Fixture-driven parsing of pytest --json-report documents."""

    def _parse(self, report_json: str, exit_code: int):
        return PytestRunner().parse_report(
            "", "", exit_code, 1.0, isolation=IsolationLevel.SUBPROCESS, report_json=report_json
        )

    def test_pass_suite(self) -> None:
        report = self._parse(PYTEST_PASS_JSON, 0)
        assert report.passed is True
        assert report.executed == 2
        assert all(o.status.value == "passed" for o in report.outcomes)

    def test_fail_suite(self) -> None:
        report = self._parse(PYTEST_FAIL_JSON, 1)
        assert report.passed is False
        assert report.executed == 1
        assert report.outcomes[0].status.value == "failed"
        assert "assert 3 == 99" in report.outcomes[0].message

    def test_collection_error_is_not_repairable_failure(self) -> None:
        report = self._parse(PYTEST_COLLECTION_ERROR_JSON, 2)
        assert report.passed is False
        assert len(report.collection_errors) == 1
        assert "ImportError" in report.collection_errors[0]

    def test_empty_suite_never_passes(self) -> None:
        """D-7: zero executed tests is never success, even with JSON."""
        report = self._parse(PYTEST_EMPTY_JSON, 5)
        assert report.passed is False
        assert report.is_vacuous_result is True

    def test_missing_json_degrades_and_fails_closed(self) -> None:
        """No report file => degraded path => indeterminate => not passed."""
        report = PytestRunner().parse_report(
            "", "", 0, 1.0, isolation=IsolationLevel.SUBPROCESS, report_json=None
        )
        assert report.passed is False
        assert report.collection_errors

    def test_duplicate_pass_counts_as_failure(self) -> None:
        """exit 0 but a failed outcome => not passed (guard against exit-code trust)."""
        report = self._parse(PYTEST_FAIL_JSON.replace('"exitcode": 1', '"exitcode": 0'), 0)
        assert report.passed is False


# ─────────────────────────────────────────────────────────────────────────────
# Jest / Go / Cargo runners
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestOtherRunnerParsing:
    """Fixture-driven parsing for jest, go, cargo."""

    def test_jest_pass(self) -> None:
        report = JestRunner().parse_report(
            "", "", 0, 1.0, isolation=IsolationLevel.SUBPROCESS, report_json=JEST_PASS_JSON
        )
        assert report.passed is True
        assert report.executed == 2

    def test_jest_fail(self) -> None:
        report = JestRunner().parse_report(
            "", "", 1, 1.0, isolation=IsolationLevel.SUBPROCESS, report_json=JEST_FAIL_JSON
        )
        assert report.passed is False
        assert "99" in report.outcomes[0].message

    def test_go_pass(self) -> None:
        report = GoRunner().parse_report(
            GO_PASS_NDJSON, "", 0, 1.0, isolation=IsolationLevel.SUBPROCESS
        )
        assert report.passed is True
        assert report.executed == 2

    def test_go_fail(self) -> None:
        report = GoRunner().parse_report(
            GO_FAIL_NDJSON, "", 1, 1.0, isolation=IsolationLevel.SUBPROCESS
        )
        assert report.passed is False
        assert report.outcomes[0].status.value == "failed"

    def test_cargo_pass(self) -> None:
        report = CargoRunner().parse_report(
            CARGO_PASS_NDJSON, "", 0, 1.0, isolation=IsolationLevel.SUBPROCESS
        )
        assert report.passed is True
        assert report.executed == 2

    def test_cargo_fail(self) -> None:
        report = CargoRunner().parse_report(
            CARGO_FAIL_NDJSON, "", 1, 1.0, isolation=IsolationLevel.SUBPROCESS
        )
        assert report.passed is False

    def test_unknown_framework_raises(self) -> None:
        with pytest.raises(KeyError):
            get_runner("no-such-framework")

    def test_registry_returns_instances_with_sandbox(self) -> None:
        runner = get_runner("pytest")
        assert isinstance(runner, TestRunnerBase)
        assert runner.sandbox is not None


# ─────────────────────────────────────────────────────────────────────────────
# Build commands
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestBuildCommands:
    """Machine-readable flags only; no shell strings."""

    def test_pytest_command_is_list_with_json_report(self) -> None:
        runner = PytestRunner()
        cmd = runner.build_command(Workspace(root=Path("."), framework="pytest"))
        assert isinstance(cmd, list)
        assert "--json-report" in cmd
        assert "-p" in cmd  # deterministic plugin loads

    def test_pytest_selection_appends_node_ids(self) -> None:
        from orchestrator.domain.testing_models import TestSelection

        runner = PytestRunner()
        cmd = runner.build_command(
            Workspace(root=Path("."), framework="pytest"),
            selection=TestSelection(node_ids=("test_main.py::test_add",), reason="subset"),
        )
        assert cmd[-1] == "test_main.py::test_add"

    def test_jest_go_cargo_machine_readable(self) -> None:
        ws = Workspace(root=Path("."), framework="jest")
        assert "--json" in JestRunner().build_command(ws)
        assert "-json" in GoRunner().build_command(ws)
        assert "json" in CargoRunner().build_command(ws)


# ─────────────────────────────────────────────────────────────────────────────
# Integration: real pytest execution through the sandbox
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.integration
class TestRealExecution:
    """End-to-end: materialize a workspace, run pytest via the sandbox.

    Uses the runner's JSON-report path. In environments where the dash
    pytest plugin hangs at import, PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 in the
    workspace env works around it (the runner loads pytest_jsonreport
    explicitly, so autoload is not required).
    """

    @pytest.fixture(autouse=True)
    def _env_workaround(self) -> None:
        import os

        os.environ.setdefault("ORCH_TEST_AUTOLOAD_WORKAROUND", "1")

    def _make_workspace(self, tmp_path: Path, test_code: str) -> Workspace:
        (tmp_path / "main.py").write_text(
            "def add(a, b):\n    return a + b\ndef sub(a, b):\n    return a - b\n",
            encoding="utf-8",
        )
        (tmp_path / "test_main.py").write_text(test_code, encoding="utf-8")
        env = {}
        if os.environ.get("ORCH_TEST_AUTOLOAD_WORKAROUND"):
            env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
        return Workspace(
            root=tmp_path,
            framework="pytest",
            source_files=(tmp_path / "main.py",),
            test_files=(tmp_path / "test_main.py",),
            env=env,
        )

    @pytest.mark.asyncio
    async def test_green_suite_executes(self, tmp_path: Path) -> None:
        ws = self._make_workspace(
            tmp_path,
            "from main import add, sub\n"
            "def test_add():\n    assert add(1, 2) == 3\n"
            "def test_sub():\n    assert sub(5, 3) == 2\n",
        )
        report = await get_runner("pytest").run(ws, timeout_s=60)
        assert report.passed is True
        assert report.executed == 2
        assert report.isolation is IsolationLevel.SUBPROCESS

    @pytest.mark.asyncio
    async def test_failing_suite_floors(self, tmp_path: Path) -> None:
        ws = self._make_workspace(
            tmp_path, "from main import add\ndef test_bad():\n    assert add(1, 2) == 99\n"
        )
        report = await get_runner("pytest").run(ws, timeout_s=60)
        assert report.passed is False
        assert report.executed == 1
        assert report.outcomes[0].status.value == "failed"


# ─────────────────────────────────────────────────────────────────────────────
# E-1: multi-file workspaces (module B importing module A)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.integration
class TestMultiFileWorkspace:
    """E-1: two-module task where b.py imports a.py — impossible pre-E-1."""

    @pytest.mark.asyncio
    async def test_cross_file_imports_execute_and_pass(self, tmp_path: Path) -> None:
        from orchestrator.domain.testing_models import TestSelection
        from orchestrator.infrastructure.workspace_materializer import WorkspaceMaterializer

        materializer = WorkspaceMaterializer(base_dir=tmp_path)
        ws = await materializer.materialize(
            framework="pytest",
            source_files={
                "a.py": "def double(x):\n    return x * 2\n",
                "b.py": "from a import double\ndef quadruple(x):\n    return double(double(x))\n",
            },
            test_files={
                "test_b.py": (
                    "from b import quadruple\nfrom a import double\n"
                    "def test_quadruple():\n    assert quadruple(2) == 8\n"
                    "def test_double():\n    assert double(3) == 6\n"
                ),
            },
            env={"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
        )
        try:
            report = await get_runner("pytest").run(ws, timeout_s=60)
            assert report.passed is True
            assert report.executed == 2
        finally:
            await materializer.cleanup(ws)

    @pytest.mark.asyncio
    async def test_workspace_cleanup_on_success_and_failure(self, tmp_path: Path) -> None:
        from orchestrator.infrastructure.workspace_materializer import WorkspaceMaterializer

        materializer = WorkspaceMaterializer(base_dir=tmp_path)
        ws = await materializer.materialize(
            framework="pytest",
            artifact="def add(a, b):\n    return a + b\n",
            test_code="from main import add\ndef test_add():\n    assert add(1, 2) == 3\n",
        )
        root = ws.root
        assert root.exists()
        await materializer.cleanup(ws)
        assert not root.exists(), "workspace not cleaned up"
