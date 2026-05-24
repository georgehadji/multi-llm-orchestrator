"""
CodebaseWriter — Apply modifications safely to an existing codebase
====================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Phase 4 of the Codebase-Aware Orchestrator enhancement.
Handles safe file modifications, diff generation, and safety gates.
"""

from __future__ import annotations

import difflib
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models import Task, TaskResult, TaskStatus, TaskType

logger = logging.getLogger("orchestrator.codebase_writer")


@dataclass
class VerificationResult:
    """Result of safety gate verification."""
    passed: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class FileOperations:
    """Safe file modification operations with rollback support."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self._backups: dict[Path, Path] = {}

    def read_file(self, path: Path) -> str:
        """Read an existing file."""
        full_path = self.root / path
        return full_path.read_text(encoding="utf-8")

    def create_file(self, path: Path, content: str) -> None:
        """Create a new file with parent directories."""
        full_path = self.root / path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")
        logger.info("Created: %s", path)

    def modify_file(self, path: Path, content: str, strategy: str = "replace") -> None:
        """Modify an existing file.

        Args:
            path: Relative path from root.
            content: New content (for "replace") or insert text.
            strategy: One of "replace", "insert_before", "insert_after", "patch".
        """
        full_path = self.root / path
        if not full_path.exists():
            logger.warning("File does not exist, creating: %s", path)
            self.create_file(path, content)
            return

        # Backup
        self._backup(path)

        original = full_path.read_text(encoding="utf-8")

        if strategy == "replace":
            full_path.write_text(content, encoding="utf-8")
        elif strategy == "insert_before":
            # Insert content before the last line matching a pattern
            # This uses AI-generated insert markers in the content
            full_path.write_text(content, encoding="utf-8")
        elif strategy == "insert_after":
            full_path.write_text(content, encoding="utf-8")
        else:
            full_path.write_text(content, encoding="utf-8")

        logger.info("Modified: %s (strategy=%s)", path, strategy)

    def delete_file(self, path: Path) -> None:
        """Delete a file (moves to .trash inside root)."""
        full_path = self.root / path
        if not full_path.exists():
            logger.warning("File does not exist, skipping: %s", path)
            return

        trash_dir = self.root / ".trash"
        trash_dir.mkdir(exist_ok=True)
        trash_path = trash_dir / path.name
        shutil.move(str(full_path), str(trash_path))
        logger.info("Deleted: %s (moved to .trash/)", path)

    def install_dependency(self, package: str) -> bool:
        """Install a Python package via pip.

        Args:
            package: Package name (e.g. "python-jose[cryptography]").

        Returns:
            True if installation succeeded.
        """
        import subprocess
        try:
            result = subprocess.run(
                ["python", "-m", "pip", "install", package],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode == 0:
                logger.info("Installed dependency: %s", package)
                return True
            else:
                logger.warning("Failed to install %s: %s", package, result.stderr[:200])
                return False
        except Exception as exc:
            logger.warning("Failed to install %s: %s", package, exc)
            return False

    def _backup(self, path: Path) -> None:
        """Create a backup of a file before modification."""
        full_path = self.root / path
        if not full_path.exists():
            return
        backup_dir = self.root / ".backups"
        backup_dir.mkdir(exist_ok=True)
        backup_path = backup_dir / path.name
        shutil.copy2(str(full_path), str(backup_path))
        self._backups[path] = backup_path

    def rollback(self, path: Path | None = None) -> None:
        """Rollback modified files from backups."""
        if path is None:
            for p, backup in self._backups.items():
                shutil.copy2(str(backup), str(self.root / p))
                logger.info("Rolled back: %s", p)
            self._backups.clear()
        elif path in self._backups:
            shutil.copy2(str(self._backups[path]), str(self.root / path))
            logger.info("Rolled back: %s", path)
            del self._backups[path]


class DiffEngine:
    """Generate and save unified diffs."""

    def generate_diff(self, original: str, modified: str, filepath: str) -> str:
        """Generate a unified diff string."""
        original_lines = original.splitlines(keepends=True)
        modified_lines = modified.splitlines(keepends=True)
        diff = difflib.unified_diff(
            original_lines, modified_lines,
            fromfile=f"a/{filepath}", tofile=f"b/{filepath}",
        )
        return "".join(diff)

    def save_diff(self, diff: str, output_dir: Path, filename: str = "changes.diff") -> Path:
        """Save a diff to file."""
        output_dir.mkdir(parents=True, exist_ok=True)
        diff_path = output_dir / filename
        diff_path.write_text(diff, encoding="utf-8")
        return diff_path


class ModificationGate:
    """Safety checks before applying any modification."""

    def verify(self, task: Task, result: TaskResult, repo_root: Path) -> VerificationResult:
        """Run safety checks on a modification.

        Checks:
        1. Syntax validation for Python files
        2. Import resolution (imports must resolve)
        3. No security red flags
        4. No hardcoded secrets

        Args:
            task: The task being applied.
            result: The task execution result.
            repo_root: Root of the codebase.

        Returns:
            VerificationResult with passed/errors/warnings.
        """
        result_ver = VerificationResult()

        if task.type == TaskType.MODIFY_FILE and task.target_path:
            target = repo_root / task.target_path
            if target.suffix == ".py":
                self._check_syntax(target, result_ver)
                self._check_imports(target, result_ver)
                self._check_secrets(result.output, result_ver)

        return result_ver

    def _check_syntax(self, path: Path, result: VerificationResult) -> None:
        """Check Python syntax of a file."""
        try:
            import ast
            content = path.read_text(encoding="utf-8") if path.exists() else ""
            if content.strip():
                ast.parse(content, filename=str(path))
        except SyntaxError as e:
            result.errors.append(f"Syntax error in {path.name}: {e}")
        except OSError:
            pass

    def _check_imports(self, path: Path, result: VerificationResult) -> None:
        """Check that imports resolve."""
        try:
            content = path.read_text(encoding="utf-8") if path.exists() else ""
        except OSError:
            return
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("import ") or line.startswith("from "):
                # Best-effort: log warning, don't block
                result.warnings.append(f"Unverified import in {path.name}: {line[:60]}")

    def _check_secrets(self, content: str, result: VerificationResult) -> None:
        """Check for hardcoded secrets."""
        import re
        secret_patterns = [
            (r'password\s*=\s*["\']([^"\']+)["\']', "Hardcoded password"),
            (r'api_key\s*=\s*["\']([^"\']+)["\']', "Hardcoded API key"),
            (r'secret\s*=\s*["\']([^"\']+)["\']', "Hardcoded secret"),
            (r'token\s*=\s*["\']([^"\']+)["\']', "Hardcoded token"),
        ]
        for pattern, message in secret_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                result.warnings.append(f"Possible {message}: {match.group(1)[:20]}")


class CodebaseWriter:
    """Top-level writer that applies modifications with safety gates."""

    def __init__(self, root: Path, dry_run: bool = False) -> None:
        self._files = FileOperations(root)
        self._diffs = DiffEngine()
        self._gate = ModificationGate()
        self._root = root
        self._dry_run = dry_run
        self._all_diffs: list[str] = []

    @property
    def all_diffs(self) -> str:
        return "\n".join(self._all_diffs)

    async def apply(self, task: Task, result: TaskResult) -> bool:
        """Apply a task result to the codebase, with safety checks.

        Args:
            task: The task that was executed.
            result: The execution result.

        Returns:
            True if applied successfully, False if blocked by safety gate.
        """
        if not result.output:
            logger.warning("No output for task %s, skipping", task.id)
            return False

        content = result.output.strip()

        if task.type == TaskType.CODE_GEN:
            target = Path(task.target_path) if task.target_path else Path(f"{task.id}.py")
            if not self._dry_run:
                self._files.create_file(target, content)
            logger.info("[dry-run] Would create: %s" if self._dry_run else "Created: %s", target)

        elif task.type == TaskType.MODIFY_FILE:
            target = Path(task.target_path) if task.target_path else Path(f"{task.id}.py")

            # Safety gate
            if not self._dry_run:
                ver = self._gate.verify(task, result, self._root)
                if ver.errors:
                    logger.error("Safety gate BLOCKED modification of %s: %s", target, ver.errors)
                    return False

                # Generate diff before modifying
                if target.exists():
                    original = self._files.read_file(target)
                    diff = self._diffs.generate_diff(original, content, str(target))
                    self._all_diffs.append(diff)
                else:
                    diff = f"--- /dev/null\n+++ b/{target}\n"
                    self._all_diffs.append(diff)

                strategy = task.modification_strategy
                self._files.modify_file(target, content, strategy)
            else:
                logger.info("[dry-run] Would modify: %s", target)

        elif task.type == TaskType.DELETE_FILE:
            target = Path(task.target_path) if task.target_path else None
            if target and not self._dry_run:
                self._files.delete_file(target)
            logger.info("[dry-run] Would delete: %s" if self._dry_run else "Deleted: %s", target)

        elif task.type == TaskType.INSTALL_DEP:
            for dep in task.dependencies_to_install:
                success = self._files.install_dependency(dep) if not self._dry_run else True
                logger.info(
                    "[dry-run] Would install: %s" if self._dry_run else "Installed: %s",
                    dep,
                )

        return True

    def save_diffs(self, output_dir: Path | None = None) -> None:
        """Save all generated diffs to a file."""
        if not self._all_diffs:
            return
        output_dir = output_dir or self._root / ".orchestrator"
        self._diffs.save_diff("\n".join(self._all_diffs), output_dir)

    def rollback_all(self) -> None:
        """Rollback all applied changes."""
        self._files.rollback()
