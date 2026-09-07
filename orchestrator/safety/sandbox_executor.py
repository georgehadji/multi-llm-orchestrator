"""
SandboxExecutor - Isolated task execution with diff/review/merge.
===================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis
Part of Category 10, Phase 4 (Replit-inspired).
"""

from __future__ import annotations
import asyncio
import difflib
import logging
import shlex
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SandboxResult:
    success: bool
    output: str = ""
    diff: str = ""  # Unified diff of changes
    files_changed: list = field(default_factory=list)
    error: str = ""
    duration_ms: float = 0.0
    review_approved: bool = False


class SandboxExecutor:
    """Executes tasks in an isolated temporary directory.

    Usage:
        sandbox = SandboxExecutor(project_dir="outputs/myapp")
        result = await sandbox.run(
            task_id="task_001",
            code="# ...",
            output_path="src/auth.py",
        )
        if result.success:
            sandbox.apply(result)
    """

    def __init__(self, project_dir="."):
        self.project_dir = Path(project_dir)
        self._sandbox_dir = self.project_dir / ".sandbox"
        self._pending: SandboxResult | None = None

    async def run(self, task_id, code, output_path, test_command=""):
        """Execute code in a sandboxed temp directory and capture diff.

        Args:
            task_id: Task identifier
            code: Code to write/execute
            output_path: Relative path within project to write the output
            test_command: Optional command to run for validation

        Returns:
            SandboxResult with diff and execution output
        """
        t0 = time.monotonic()
        sandbox = self._sandbox_dir / task_id
        sandbox.mkdir(parents=True, exist_ok=True)

        # Write code to sandbox
        target = sandbox / output_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(code, encoding="utf-8")

        # Execute test command if provided
        output = ""
        test_ok = True
        if test_command:
            # argv, not a shell. test_command comes from project config
            # (app_detector reads it out of a project's own settings), so it
            # travels with a cloned or generated project — `pytest; curl evil`
            # would otherwise be two commands here, in the module named
            # "sandbox" (SEC-003).
            argv = (
                shlex.split(test_command) if isinstance(test_command, str) else list(test_command)
            )
            if not argv:
                return SandboxResult(success=False, error="empty test command")
            try:
                proc = await asyncio.create_subprocess_exec(
                    *argv,
                    cwd=str(sandbox),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
                output = stdout.decode() + stderr.decode()
                test_ok = proc.returncode == 0
            except asyncio.TimeoutError:
                output = "Test command timed out"
                test_ok = False
            except (FileNotFoundError, OSError) as exc:
                # A shell reported a missing binary as exit 127; exec raises.
                output = f"Could not run test command: {exc}"
                test_ok = False

        # Generate diff against project dir
        diff = self._generate_diff(target, self.project_dir / output_path)

        result = SandboxResult(
            success=test_ok,
            output=output,
            diff=diff,
            files_changed=[output_path],
            duration_ms=(time.monotonic() - t0) * 1000,
        )
        self._pending = result
        return result

    def _generate_diff(self, new_file, existing_file):
        """Generate unified diff between sandbox output and existing file."""
        new_content = new_file.read_text(encoding="utf-8") if new_file.exists() else ""
        old_content = existing_file.read_text(encoding="utf-8") if existing_file.exists() else ""

        diff = difflib.unified_diff(
            old_content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=str(existing_file),
            tofile=str(new_file),
        )
        return "".join(diff)

    def apply(self, result=None):
        """Apply sandbox changes to the project directory."""
        r = result or self._pending
        if not r or not r.review_approved:
            return False

        for filepath in r.files_changed:
            src = self._sandbox_dir / "task" / filepath
            dst = self.project_dir / filepath
            if src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

        self._pending = None
        return True

    def reject(self):
        """Discard sandbox changes."""
        self._pending = None
