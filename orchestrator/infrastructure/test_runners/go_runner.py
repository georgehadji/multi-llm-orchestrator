"""GoRunner — go test -json (NDJSON stream on stdout, F-7)."""

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


class GoRunner(TestRunnerBase):
    """Runs ``go test -json`` and parses the NDJSON event stream."""

    report_format = "go-ndjson"

    @property
    def framework(self) -> str:
        return "go"

    def build_command(
        self,
        workspace: Workspace,
        selection: TestSelection | None = None,
        *,
        report_path: Path | None = None,
    ) -> list[str]:
        cmd = ["go", "test", "-json", "./..."]
        if selection and selection.node_ids:
            cmd = ["go", "test", "-json", "-run", ",".join(selection.node_ids)]
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
                passed_regex=r"--- PASS: (\w+)",
                failed_regex=r"--- FAIL: (\w+)",
                error_regex=None,
            )

        # Aggregate NDJSON events per test: last Action wins.
        statuses: dict[str, TestStatus] = {}
        durations: dict[str, float] = {}
        messages: dict[str, str] = {}
        collection_errors: list[str] = []
        saw_fail = False

        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue  # non-JSON lines (e.g. "ok pkg" summary) ignored
            action = event.get("Action", "")
            test = event.get("Test", "")
            if action == "output" and test and event.get("Output"):
                messages.setdefault(test, "")
                messages[test] = (messages[test] + event["Output"])[:2000]
                continue
            if not test:
                if action == "fail":
                    saw_fail = True
                continue
            if action in ("run", "start"):
                statuses.setdefault(test, TestStatus.FAILED)
                durations.setdefault(test, 0.0)
                continue
            if action == "pass":
                statuses[test] = TestStatus.PASSED
                durations[test] = float(event.get("Elapsed", 0) or 0) * 1000
            elif action == "fail":
                statuses[test] = TestStatus.FAILED
                durations[test] = float(event.get("Elapsed", 0) or 0) * 1000
            elif action == "skip":
                statuses[test] = TestStatus.SKIPPED
                durations[test] = float(event.get("Elapsed", 0) or 0) * 1000

        outcomes = tuple(
            TestOutcome(
                node_id=f"go::{name}",
                status=status,
                duration_ms=durations.get(name, 0.0),
                message=messages.get(name, ""),
            )
            for name, status in statuses.items()
        )

        executed = len(outcomes)
        passed = (
            exit_code == 0
            and executed > 0
            and not saw_fail
            and not any(o.status is TestStatus.FAILED for o in outcomes)
        )
        return SuiteReport(
            passed=passed,
            exit_code=exit_code,
            outcomes=outcomes,
            collection_errors=tuple(collection_errors),
            isolation=isolation,
            duration_ms=duration_ms,
            truncated_output=(stdout + stderr)[:4000],
        )
