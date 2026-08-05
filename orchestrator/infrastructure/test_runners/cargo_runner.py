"""CargoRunner — cargo test --format json (NDJSON, F-7)."""

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


class CargoRunner(TestRunnerBase):
    """Runs ``cargo test --format json`` and parses the NDJSON stream.

    ``--format json`` is the libtest JSON reporter. If the toolchain does
    not support it, output falls back to the degraded regex path (labelled
    on the report).
    """

    report_format = "cargo-ndjson"

    @property
    def framework(self) -> str:
        return "cargo"

    def build_command(
        self,
        workspace: Workspace,
        selection: TestSelection | None = None,
        *,
        report_path: Path | None = None,
    ) -> list[str]:
        cmd = ["cargo", "test", "--format", "json"]
        if selection and selection.node_ids:
            cmd.append("--")
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
        text = report_json if report_json else stdout
        if not text.strip():
            return self._degraded_report(
                stdout,
                exit_code,
                stderr=stderr,
                isolation=isolation,
                passed_regex=r"test result: ok\. (\d+) passed",
                failed_regex=r"test result: FAILED\. (\d+) failed",
                error_regex=None,
            )

        outcomes: list[TestOutcome] = []
        statuses: dict[str, TestStatus] = {}
        saw_failure = False

        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") != "test":
                continue
            name = event.get("name", "")
            if not name:
                continue
            ev = event.get("event", "")
            if ev == "ok":
                statuses[name] = TestStatus.PASSED
            elif ev == "failed":
                statuses[name] = TestStatus.FAILED
                saw_failure = True
            elif ev == "ignored" or ev == "filtered_out":
                statuses[name] = TestStatus.SKIPPED

        outcomes = [
            TestOutcome(
                node_id=f"cargo::{name}",
                status=status,
                duration_ms=0.0,
            )
            for name, status in statuses.items()
        ]

        executed = len(outcomes)
        passed = (
            exit_code == 0
            and executed > 0
            and not saw_failure
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
