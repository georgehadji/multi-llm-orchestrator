"""JestRunner — jest --json (machine-readable, F-7)."""

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


class JestRunner(TestRunnerBase):
    """Runs Jest with --json and parses structured output."""

    report_format = "jest"

    @property
    def framework(self) -> str:
        return "jest"

    def build_command(
        self,
        workspace: Workspace,
        selection: TestSelection | None = None,
        *,
        report_path: Path | None = None,
    ) -> list[str]:
        cmd = ["npx", "jest", "--json", "--silent", "--no-cache"]
        if report_path is not None:
            cmd.extend(["--outputFile", report_path.name])  # relative: cwd == workspace.root
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
                stdout,
                exit_code,
                stderr=stderr,
                isolation=isolation,
                passed_regex=r"Tests:\s+(\d+) passed",
                failed_regex=r"Tests:\s+\d+ failed, (\d+) passed",
            )

        try:
            data = json.loads(report_json)
        except json.JSONDecodeError as exc:
            logger.warning("jest JSON report unparseable: %s", exc)
            return self._degraded_report(
                stdout,
                exit_code,
                stderr=stderr,
                isolation=isolation,
                passed_regex=r"Tests:\s+(\d+) passed",
                failed_regex=r"Tests:\s+\d+ failed, (\d+) passed",
            )

        outcomes: list[TestOutcome] = []
        total = int(data.get("numTotalTests", 0))
        for suite in data.get("testResults", []):
            suite_name = suite.get("name", "unknown")
            for assertion in suite.get("assertionResults", []):
                title = assertion.get("title", "unknown")
                status = assertion.get("status", "failed")
                duration_ms_test = float(assertion.get("duration", 0) or 0)
                message = assertion.get("failureMessages", [])
                outcomes.append(
                    TestOutcome(
                        node_id=f"{suite_name}::{title}",
                        status={
                            "passed": TestStatus.PASSED,
                            "failed": TestStatus.FAILED,
                            "pending": TestStatus.SKIPPED,
                            "todo": TestStatus.SKIPPED,
                            "disabled": TestStatus.SKIPPED,
                        }.get(status, TestStatus.FAILED),
                        duration_ms=duration_ms_test,
                        message="\n".join(message)[:2000] if message else "",
                    )
                )

        executed = len(outcomes)
        passed = (
            exit_code == 0
            and executed > 0
            and not any(o.status is TestStatus.FAILED for o in outcomes)
        )
        return SuiteReport(
            passed=passed,
            exit_code=exit_code,
            outcomes=tuple(outcomes),
            isolation=isolation,
            duration_ms=duration_ms,
            truncated_output=(stdout + stderr)[:4000],
        )
