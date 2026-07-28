"""TestingService — orchestrates test execution via port injection."""

from __future__ import annotations

import logging
from pathlib import Path

from ...domain.testing_models import (
    IsolationLevel,
    SuiteReport,
    TestSelection,
    Workspace,
)
from ...domain.ports import TestExecutorPort, SandboxPort

logger = logging.getLogger(__name__)


class TestingService:
    """Coordinates test execution from workspace materialization through reporting."""

    def __init__(
        self,
        executor: TestExecutorPort,
    ) -> None:
        self._executor = executor

    async def run_suite(
        self,
        workspace: Workspace,
        selection: TestSelection | None = None,
        *,
        timeout_s: float = 120.0,
    ) -> SuiteReport:
        """Run a test suite and return structured results."""
        logger.info(
            "Running %s suite in %s",
            selection.reason if selection else "full",
            workspace.root,
        )
        return await self._executor.run(workspace, selection, timeout_s=timeout_s)
