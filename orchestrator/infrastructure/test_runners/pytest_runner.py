"""PytestRunner — pytest with --json-report (machine-readable, F-7).

Parses the ``pytest-json-report`` JSON document (file, not stdout — stdout
mixes with captured test output). ``passed`` requires exit code 0, at least
one executed test (never vacuous), and no collection errors. Exit code 5
(no tests collected) and exit code 2 (usage/interrupt) are treated as
collection failures, never as green.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from ...domain.testing_models import (
    IsolationLevel,
    SuiteReport,
    TestOutcome,
    TestSelection,
    TestStatus,
    Workspace,
)
from .base import TestRunnerBase

logger = logging.getLogger(__name__)


class PytestRunner(TestRunnerBase):
    """Runs pytest with --json-report and parses structured output."""

    report_format = "pytest-report"

    @property
    def framework(self) -> str:
        return "pytest"

    def build_command(
        self,
        workspace: Workspace,
        selection: TestSelection | None = None,
        *,
        report_path: Path | None = None,
    ) -> list[str]:
        cmd = [
            "python",
            "-m",
            "pytest",
            "-q",
            "--no-header",
            "-p",
            "pytest_jsonreport.plugin",  # F-7: machine-readable output (explicit load)
        ]
        # E-5: keep run order deterministic (no random plugin side effects).
        cmd.extend(["-p", "no:randomly"])
        # Workspaces may declare extra plugins (comma-separated), e.g.
        # ORCH_PYTEST_PLUGINS=asyncio,mock for generated async suites.
        extra_plugins = (workspace.env or {}).get("ORCH_PYTEST_PLUGINS", "")
        for plugin in (p.strip() for p in extra_plugins.split(",") if p.strip()):
            cmd.extend(["-p", plugin])
        if report_path is not None:
            cmd.extend(["--json-report", f"--json-report-file={report_path.name}"])
        else:
            # Degraded-compatible mode: plugin writes .report.json in cwd.
            cmd.append("--json-report")
        if selection and selection.node_ids:
            cmd.extend(selection.node_ids)
        return cmd

    def parse_report(
        self,
        stdout: str,
        stderr: str,
        exit_code: int,
        duration_ms: float,
        isolation: IsolationLevel = IsolationLevel.SUBPROCESS,
        report_json: str | None = None,
    ) -> SuiteReport:
        if not report_json or not report_json.strip():
            return self._degraded_report(
                stdout, exit_code, stderr=stderr, isolation=isolation, error_regex=r"(\d+) error"
            )

        try:
            data = json.loads(report_json)
        except json.JSONDecodeError as exc:
            logger.warning("pytest JSON report unparseable: %s", exc)
            return self._degraded_report(
                stdout, exit_code, stderr=stderr, isolation=isolation, error_regex=r"(\d+) error"
            )

        outcome_to_status = {
            "passed": TestStatus.PASSED,
            "failed": TestStatus.FAILED,
            "error": TestStatus.ERROR,
            "skipped": TestStatus.SKIPPED,
            "xfailed": TestStatus.XFAILED,
        }

        outcomes: list[TestOutcome] = []
        collection_errors: list[str] = []

        for test in data.get("tests", []):
            nodeid = test.get("nodeid", "unknown")
            outcome = test.get("outcome", "passed")
            status = outcome_to_status.get(outcome, TestStatus.FAILED)
            duration = float(test.get("duration", 0) or 0) * 1000
            message = ""
            call = test.get("call", {}) or {}
            if isinstance(call, dict) and call.get("longrepr"):
                message = str(call["longrepr"])[:2000]
            outcomes.append(
                TestOutcome(
                    node_id=nodeid,
                    status=status,
                    duration_ms=duration,
                    message=message,
                )
            )

        for collector in data.get("collectors", []):
            result = collector.get("result", [])
            if isinstance(result, list):
                for entry in result:
                    if isinstance(entry, dict) and entry.get("outcome") in (
                        "failed",
                        "error",
                    ):
                        collection_errors.append(
                            f"{collector.get('nodeid', '?')}: {entry.get('longrepr', '')}"
                        )

        if data.get("collection", {}).get("longrepr"):
            collection_errors.append(str(data["collection"]["longrepr"]))

        executed = len(outcomes)
        passed = (
            exit_code == 0
            and executed > 0
            and not collection_errors
            and not any(o.status is TestStatus.FAILED for o in outcomes)
        )

        return SuiteReport(
            passed=passed,
            exit_code=exit_code,
            outcomes=tuple(outcomes),
            collection_errors=tuple(collection_errors),
            isolation=isolation,
            duration_ms=duration_ms,
            truncated_output=(stdout + stderr)[:4000],
        )
