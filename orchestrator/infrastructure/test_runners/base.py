"""TestRunnerBase — shared machine-readable-output parsing for all runners."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from ...domain.testing_models import (
    IsolationLevel,
    SuiteReport,
    TestOutcome,
    TestSelection,
    TestStatus,
    Workspace,
)

logger = logging.getLogger(__name__)


class TestRunnerBase(ABC):
    """Abstract base for all framework-specific test runners.

    Subclasses implement :meth:`build_command` and :meth:`parse_report`.
    """

    @property
    @abstractmethod
    def framework(self) -> str:
        """Framework identifier, e.g. 'pytest', 'jest'."""
        ...

    async def run(
        self,
        workspace: Workspace,
        selection: TestSelection | None = None,
        *,
        timeout_s: float = 120.0,
    ) -> SuiteReport:
        """Execute tests and return a structured report."""
        raise NotImplementedError

    def supports(self, framework: str) -> bool:
        return framework == self.framework

    def build_command(
        self,
        workspace: Workspace,
        selection: TestSelection | None = None,
    ) -> list[str]:
        """Build the CLI command for this framework."""
        raise NotImplementedError

    def parse_report(
        self,
        stdout: str,
        stderr: str,
        exit_code: int,
        duration_ms: float,
        isolation: IsolationLevel = IsolationLevel.SUBPROCESS,
    ) -> SuiteReport:
        """Parse framework output into a SuiteReport."""
        raise NotImplementedError
