"""Pytest runner using --json-report for machine-readable output."""

from __future__ import annotations

from .base import TestRunnerBase


class PytestRunner(TestRunnerBase):
    """Runs pytest with --json-report and parses structured output."""

    @property
    def framework(self) -> str:
        return "pytest"

    def build_command(
        self,
        workspace: AnyWorkspace,
        selection: AnyTestSelection | None = None,
    ) -> list[str]:
        cmd = ["python", "-m", "pytest", "--json-report", "-q"]
        return cmd


AnyWorkspace = object
AnyTestSelection = object
