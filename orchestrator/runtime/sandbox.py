"""
SandboxExecutor — Isolated code execution and testing
=======================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Capability 7 of the Agentic System Implementation Plan.
Executes generated code in isolated environments, runs tests,
and produces structured execution results.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import sys

logger = logging.getLogger("orchestrator.runtime.sandbox")


@dataclass
class ExecutionResult:
    """Result of code execution."""

    success: bool
    output: str = ""
    error: str = ""
    exit_code: int = 0


class SandboxExecutor:
    """Execute code in isolated temporary environments."""

    async def execute(
        self, code: str, language: str = "python", timeout: int = 30
    ) -> ExecutionResult:
        """Execute code and return results.

        Args:
            code: Source code to execute.
            language: Language of the code.
            timeout: Maximum execution time in seconds.

        Returns:
            ExecutionResult with output, error, and exit_code.
        """
        with tempfile.TemporaryDirectory() as tmp:
            if language == "python":
                filepath = Path(tmp) / "script.py"
                filepath.write_text(code, encoding="utf-8")
                cmd = [sys.executable, str(filepath)]
            else:
                return ExecutionResult(success=False, error=f"Unsupported language: {language}")

            try:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout,
                )
                return ExecutionResult(
                    success=process.returncode == 0,
                    output=stdout.decode().strip(),
                    error=stderr.decode().strip(),
                    exit_code=process.returncode or 0,
                )
            except asyncio.TimeoutError:
                return ExecutionResult(success=False, error=f"Execution timed out ({timeout}s)")
            except Exception as exc:
                return ExecutionResult(success=False, error=str(exc))


class TestRunner:
    """Run test suites — DEPRECATED shim (F-6).

    Delegates to the authoritative runner in ``orchestrator.infrastructure.test_runners``.
    Emits a ``DeprecationWarning``; scheduled for removal after one minor release.
    """

    def __init__(self) -> None:
        """Initialize the shim (no external resources acquired)."""
        import warnings

        warnings.warn(
            "runtime.sandbox.TestRunner is deprecated; use "
            "orchestrator.infrastructure.test_runners.get_runner('pytest') "
            "via the TestingService instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    async def run_tests(self, test_path: Path, timeout: int = 60) -> dict[str, Any]:
        """Run tests at the given path (delegates to the authoritative runner).

        Returns:
            dict with success/passed/failed/total counts (backward compatible).
        """
        from ..infrastructure.test_runners import get_runner
        from ..domain.testing_models import Workspace

        root = test_path if test_path.is_dir() else test_path.parent
        runner = get_runner("pytest")
        report = await runner.run(
            Workspace(root=root, framework="pytest", test_files=(test_path,)),
            timeout_s=timeout,
        )
        passed = sum(1 for o in report.outcomes if o.status.value == "passed")
        failed = sum(1 for o in report.outcomes if o.status.value == "failed")
        return {
            "success": report.passed,
            "passed": passed,
            "failed": failed,
            "total": passed + failed,
        }
