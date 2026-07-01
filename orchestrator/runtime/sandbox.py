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
    """Run test suites and parse results."""

    async def run_tests(self, test_path: Path, timeout: int = 60) -> dict[str, Any]:
        """Run tests at the given path.

        Supports pytest for Python projects.
        Returns dict with passed/failed/total counts.
        """
        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "pytest",
                str(test_path),
                "-q",
                "--no-header",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            return {"success": False, "error": "Test execution timed out"}

        output = stdout.decode()
        import re

        match = re.search(r"(\d+) passed", output)
        passed = int(match.group(1)) if match else 0
        match = re.search(r"(\d+) failed", output)
        failed = int(match.group(1)) if match else 0

        return {
            "success": process.returncode == 0,
            "passed": passed,
            "failed": failed,
            "total": passed + failed,
        }
