"""TestRunnerBase — shared execution and machine-readable parsing (F-6/F-7).

All framework runners share the same execution path:

1. :meth:`build_command` — machine-readable flags only (``--json-report``,
   ``--json``, ``-json``, ``--format json``). Regex parsing of human output
   survives only as an explicitly-labelled *degraded* path when the JSON
   plugin is absent, and that path returns ``passed=False`` when counts
   cannot be determined (closes the D-7 ``total=0 => success`` hole).
2. Execute via a :class:`SandboxPort` (never directly on the host).
3. :meth:`parse_report` — structured JSON → :class:`SuiteReport`.
"""

from __future__ import annotations

import logging
import os
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path

from ...domain.testing_models import (
    IsolationLevel,
    SuiteReport,
    TestOutcome,
    TestSelection,
    TestStatus,
    Workspace,
)
from ..sandboxes import SubprocessSandbox

logger = logging.getLogger(__name__)


class TestRunnerBase(ABC):
    """Abstract base for all framework-specific test runners.

    Subclasses implement :meth:`framework`, :meth:`build_command` and
    :meth:`parse_report`. Execution and timeout handling live here.
    """

    #: Identifier of the JSON report file format: "pytest-report", "jest",
    #: "go-ndjson", "cargo-ndjson". Used to gate degraded-path detection.
    report_format: str = "json"

    def __init__(self, sandbox=None) -> None:
        """Initialize the runner.

        Args:
            sandbox: SandboxPort to execute inside. Defaults to a
                SubprocessSandbox (never executes unisolated).
        """
        self.sandbox = sandbox or SubprocessSandbox()

    @property
    @abstractmethod
    def framework(self) -> str:
        """Framework identifier, e.g. 'pytest', 'jest'."""
        ...

    def supports(self, framework: str) -> bool:
        return framework == self.framework

    async def run(
        self,
        workspace: Workspace,
        selection: TestSelection | None = None,
        *,
        timeout_s: float = 120.0,
    ) -> SuiteReport:
        """Execute tests and return a structured report.

        Args:
            workspace: Materialized project tree.
            selection: Optional test subset (empty = full suite).
            timeout_s: Maximum wall-clock time.

        Returns:
            SuiteReport with outcomes parsed from machine-readable output.
        """
        started = time.monotonic()
        report_path: Path | None = None

        # pytest/jest write JSON to a report file; go/cargo stream NDJSON
        # to stdout. Use a RELATIVE path (workspace root == cwd): on
        # Windows, pytest-json-report <1.6 does not write absolute paths.
        if self._uses_report_file():
            report_path = workspace.root / f".orch-{self.framework}-report.json"

        try:
            argv = self.build_command(workspace, selection, report_path=report_path)
        except Exception as exc:
            logger.error("build_command failed for %s: %s", self.framework, exc)
            return SuiteReport(
                passed=False,
                exit_code=2,
                collection_errors=(f"runner failed to build command: {exc}",),
                isolation=self.sandbox.level,
                duration_ms=(time.monotonic() - started) * 1000,
            )

        env = dict(workspace.env or {})

        # E-5 determinism: run pytest with plugin autoload disabled so third-
        # party plugin side effects (e.g. a hung dash/flask import) cannot
        # affect generated-suite validation. The runner loads exactly the
        # plugins it needs explicitly (see build_command), and workspaces can
        # add more via ORCH_PYTEST_PLUGINS (comma-separated, e.g. asyncio).
        if self.framework == "pytest":
            env.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")

        try:
            exit_code, stdout, stderr = await self.sandbox.exec(
                argv,
                cwd=workspace.root,
                env=env,
                timeout_s=timeout_s,
            )
        except Exception as exc:
            logger.error("sandbox execution failed for %s: %s", self.framework, exc)
            return SuiteReport(
                passed=False,
                exit_code=-1,
                collection_errors=(f"sandbox execution failed: {exc}",),
                isolation=self.sandbox.level,
                duration_ms=(time.monotonic() - started) * 1000,
            )

        report_json: str | None = None
        if report_path is not None and report_path.exists():
            try:
                report_json = report_path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:  # pragma: no cover - rare
                logger.warning("Could not read report file %s: %s", report_path, exc)

        report = self.parse_report(
            stdout,
            stderr,
            exit_code,
            duration_ms=(time.monotonic() - started) * 1000,
            isolation=self.sandbox.level,
            report_json=report_json,
        )

        # E-5: flake rerun — a failing suite is re-run exactly once. Tests
        # that pass on rerun are flaky: excluded from the failure verdict
        # (and therefore from the repair loop) but surfaced on the report.
        rerun_enabled = os.environ.get("ORCH_FLAKE_RERUN", "1") == "1"
        if (
            rerun_enabled
            and not report.passed
            and not report.is_vacuous_result
            and not report.collection_errors
        ):
            report = await self._rerun_and_merge(report, workspace, selection, timeout_s)

        # Remove the transient report file (workspace is a temp tree, but
        # keep it clean for ORCH_KEEP_FAILED_WORKSPACES debugging).
        if report_path is not None:
            try:
                report_path.unlink(missing_ok=True)
            except OSError:  # pragma: no cover - best-effort
                pass
        return report

    async def _rerun_and_merge(
        self,
        first: SuiteReport,
        workspace: Workspace,
        selection: TestSelection | None,
        timeout_s: float,
    ) -> SuiteReport:
        """Re-run a failing suite once; merge pass-on-rerun tests as flaky.

        Returns *first* unchanged when nothing flaky was observed (a genuine
        failure must not be masked), otherwise the second run's report with
        ``flaky_node_ids`` populated.
        """
        import os as _os

        argv = self.build_command(workspace, selection)
        env = dict(workspace.env or {})
        if self.framework == "pytest":
            env.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
        try:
            exit_code2, stdout2, stderr2 = await self.sandbox.exec(
                argv, cwd=workspace.root, env=env, timeout_s=timeout_s
            )
        except Exception as exc:  # pragma: no cover - rerun failure keeps first verdict
            logger.warning("flake rerun failed: %s", exc)
            return first

        report_path = workspace.root / f".orch-{self.framework}-report.json"
        report_json: str | None = None
        if report_path.exists():
            try:
                report_json = report_path.read_text(encoding="utf-8", errors="replace")
            except OSError:  # pragma: no cover
                pass
            try:
                report_path.unlink(missing_ok=True)
            except OSError:  # pragma: no cover
                pass

        second = self.parse_report(
            stdout2,
            stderr2,
            exit_code2,
            duration_ms=0.0,
            isolation=self.sandbox.level,
            report_json=report_json,
        )
        if second.is_vacuous_result or second.collection_errors:
            return first

        failed_nodes = {
            o.node_id for o in first.outcomes if o.status in (TestStatus.FAILED, TestStatus.ERROR)
        }
        flaky = {
            nid
            for nid in failed_nodes
            if any(o.node_id == nid and o.status is TestStatus.PASSED for o in second.outcomes)
        }
        if not flaky:
            return first

        logger.warning("flake rerun: %d test(s) passed on rerun: %s", len(flaky), sorted(flaky))
        return SuiteReport(
            passed=second.passed,
            exit_code=second.exit_code,
            outcomes=second.outcomes,
            collection_errors=second.collection_errors,
            line_coverage=second.line_coverage,
            mutation_score=second.mutation_score,
            flaky_node_ids=tuple(sorted(flaky)),
            isolation=second.isolation,
            duration_ms=second.duration_ms,
            truncated_output=second.truncated_output,
        )

    def _uses_report_file(self) -> bool:
        """Whether this runner writes JSON to a report file (vs stdout NDJSON)."""
        return self.report_format in ("pytest-report", "jest")

    @abstractmethod
    def build_command(
        self,
        workspace: Workspace,
        selection: TestSelection | None = None,
        *,
        report_path: Path | None = None,
    ) -> list[str]:
        """Build the CLI command for this framework (list form, no shell)."""
        ...

    def parse_report(
        self,
        stdout: str,
        stderr: str,
        exit_code: int,
        duration_ms: float,
        isolation: IsolationLevel = IsolationLevel.SUBPROCESS,
        report_json: str | None = None,
    ) -> SuiteReport:
        """Parse framework output into a SuiteReport."""
        raise NotImplementedError

    # ── degraded-path helpers (F-7: regex only as an explicit fallback) ────

    def _degraded_report(
        self,
        stdout: str,
        exit_code: int,
        *,
        stderr: str = "",
        isolation: IsolationLevel,
        passed_regex: str = r"(\d+) passed",
        failed_regex: str = r"(\d+) failed",
        error_regex: str | None = r"(\d+) error",
    ) -> SuiteReport:
        """Build a report from human-readable output (degraded path).

        If no counts can be determined, ``passed`` is False — a zero-test
        suite is never success (D-7).
        """
        passed = self._first_int(stdout, passed_regex) or 0
        failed = self._first_int(stdout, failed_regex) or 0
        errors = self._first_int(stdout, error_regex) or 0 if error_regex else 0

        executed = passed + failed + errors
        if executed == 0:
            # Indeterminate — cannot claim success without counts.
            return SuiteReport(
                passed=False,
                exit_code=exit_code,
                collection_errors=(
                    "degraded parsing: no test counts found in output; "
                    "suite result indeterminate",
                ),
                isolation=isolation,
                truncated_output=(stdout + stderr)[:4000],
            )

        outcomes = tuple(
            TestOutcome(
                node_id=f"degraded-{i}",
                status=TestStatus.PASSED if i < passed else TestStatus.FAILED,
                duration_ms=0.0,
            )
            for i in range(executed)
        )
        return SuiteReport(
            passed=exit_code == 0 and failed == 0 and errors == 0,
            exit_code=exit_code,
            outcomes=outcomes,
            isolation=isolation,
            truncated_output=(stdout + stderr)[:4000],
        )

    @staticmethod
    def _first_int(text: str, pattern: str | None) -> int | None:
        if not pattern:
            return None
        match = re.search(pattern, text)
        return int(match.group(1)) if match else None
