"""
GitIntegration — Git best practices for agentic development
=============================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Ensures every stage of development follows Git best practices:
- Feature branches per milestone
- Atomic commits with conventional commit messages
- Pull request creation per completed feature
- Push to GitHub after every successful stage
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("orchestrator.git_integration")


@dataclass
class CommitResult:
    success: bool
    hash: str = ""
    message: str = ""
    error: str = ""

    @property
    def short_hash(self) -> str:
        return self.hash[:8] if self.hash else ""


@dataclass
class BranchInfo:
    name: str
    base_branch: str = "main"
    created_at: str = ""
    url: str = ""


class GitIntegration:
    """Git operations for the agentic pipeline.

    Each milestone creates a feature branch. After each completed task,
    changes are committed with conventional commit messages. After each
    milestone, a PR is created and pushed.
    """

    def __init__(self, repo_path: Path, remote: str = "origin") -> None:
        self.repo = repo_path
        self.remote = remote
        self._current_branch: str = "main"

    async def init(self) -> bool:
        """Initialize git repo if not already initialized."""
        if not (self.repo / ".git").exists():
            return await self._run("git init")[0]
        return True

    async def create_branch(self, name: str, base: str = "main") -> bool:
        """Create a feature branch from base."""
        await self._run(f"git checkout {base}")
        ok, _ = await self._run(f"git checkout -b {name}")
        if ok:
            self._current_branch = name
            logger.info("Created branch: %s (from %s)", name, base)
        return ok

    async def commit(self, message: str, files: list[str] | None = None) -> CommitResult:
        """Stage and commit changes with conventional commit message."""
        # Stage specified files or all
        if files:
            for f in files:
                await self._run(f"git add {f}")
        else:
            await self._run("git add -A")

        # Check if there's anything to commit
        ok, status = await self._run("git status --porcelain")
        if not status.strip():
            return CommitResult(success=True, hash="", message="Nothing to commit")

        # Commit with conventional message format
        ok, output = await self._run(f'git commit -m "{message}"')
        if ok:
            h, _ = await self._run("git rev-parse HEAD")
            return CommitResult(
                success=True,
                hash=h.strip(),
                message=message,
            )
        return CommitResult(success=False, error=output)

    async def commit_task(self, task_id: str, task_type: str, description: str) -> CommitResult:
        """Commit with conventional commit message for a task.

        Format: type(scope): description
        Examples:
            feat(auth): add JWT authentication middleware
            fix(api): correct pagination offset
            test(models): add unit tests for User model
        """
        type_map = {
            "code_generation": "feat",
            "code_review": "review",
            "modify_file": "feat",
            "delete_file": "refactor",
            "install_dependency": "chore",
            "evaluation": "test",
            "reasoning": "docs",
        }
        prefix = type_map.get(task_type, "chore")
        message = f"{prefix}({task_id}): {description[:80]}"
        return await self.commit(message)

    async def push(self, branch: str = "") -> bool:
        """Push current branch to remote."""
        target = branch or self._current_branch
        ok, output = await self._run(f"git push {self.remote} {target}")
        if ok:
            logger.info("Pushed %s to %s", target, self.remote)
        else:
            logger.warning("Push failed for %s: %s", target, output[:200])
        return ok

    async def create_pr(self, title: str, body: str = "") -> str:
        """Create a GitHub PR using gh CLI."""
        body_text = body or "Automated PR from AI Orchestrator"
        ok, output = await self._run(
            f'gh pr create --title "{title[:80]}" --body "{body_text[:500]}"'
        )
        if ok:
            logger.info("Created PR: %s", output.strip()[:80])
            return output.strip()
        logger.warning("PR creation failed: %s", output[:200])
        return ""

    async def stage_commit_push(self, message: str, branch: str = "") -> CommitResult:
        """Convenience: stage all -> commit -> push in one call."""
        result = await self.commit(message)
        if result.success:
            await self.push(branch)
        return result

    async def _run(self, cmd: str) -> tuple[bool, str]:
        """Run a git command in the repo directory."""
        try:
            proc = await asyncio.create_subprocess_shell(
                cmd, cwd=str(self.repo),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            success = proc.returncode == 0
            output = stdout.decode().strip() or stderr.decode().strip()
            return success, output
        except asyncio.TimeoutError:
            return False, "Command timed out"
        except Exception as exc:
            return False, str(exc)

    async def status(self) -> str:
        """Get repo status."""
        _, output = await self._run("git status")
        return output

    async def log(self, n: int = 10) -> str:
        """Get recent commit log."""
        _, output = await self._run(f"git log --oneline -{n}")
        return output

    async def milestone_flow(
        self, milestone_name: str, task_results: list[tuple[str, str, str]]
    ) -> list[CommitResult]:
        """Execute a complete milestone flow: branch -> commits -> push -> PR.

        Args:
            milestone_name: Name for the branch and milestone (e.g. "auth-system")
            task_results: List of (task_id, task_type, description) tuples

        Returns:
            List of CommitResult for each task committed.
        """
        results: list[CommitResult] = []

        # 1. Create feature branch
        branch_ok = await self.create_branch(f"feature/{milestone_name}")
        if not branch_ok:
            logger.error("Failed to create branch for milestone: %s", milestone_name)
            return results

        # 2. Commit each task
        for task_id, task_type, description in task_results:
            result = await self.commit_task(task_id, task_type, description)
            results.append(result)

        # 3. Push branch
        await self.push()

        # 4. Create PR
        await self.create_pr(f"Milestone: {milestone_name.replace('-', ' ').title()}")

        return results
