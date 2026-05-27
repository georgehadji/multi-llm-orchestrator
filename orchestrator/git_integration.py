"""
GitIntegration — Auto-commit messages per task execution.
===========================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Part of Category 2, Phase N5 (Newly-inspired): Auto-generates descriptive
git commit messages after each task completes. Two modes: fast (template-based)
and full (LLM-generated summary of the task's changes).
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass


@dataclass
class CommitMessage:
    """Generated commit message for a task."""

    summary: str        # 50-char first line
    body: str = ""      # Detailed description
    task_id: str = ""
    co_authored_by: str = "AI Orchestrator <orchestrator@local>"

    @property
    def full_message(self) -> str:
        msg = self.summary
        if self.body:
            msg += f"\n\n{self.body}"
        if self.co_authored_by:
            msg += f"\n\nCo-Authored-By: {self.co_authored_by}"
        return msg


class GitIntegration:
    """Auto-commits with descriptive messages after task execution."""

    def __init__(self, repo_dir: str = "."):
        self.repo_dir = repo_dir

    def generate_message_fast(self, task_id: str, task_prompt: str, score: float = 0.0) -> CommitMessage:
        """Generate a template-based commit message (fast, no LLM).

        Args:
            task_id: Task identifier
            task_prompt: Original task description
            score: Quality score

        Returns:
            CommitMessage with summary and body
        """
        # Extract first meaningful line from prompt
        prompt_short = task_prompt[:80].strip()
        if len(prompt_short) == 80:
            prompt_short = prompt_short[:77] + "..."

        summary = f"feat({task_id}): {prompt_short}"[:72]

        body = f"Task: {task_id}\nScore: {score:.2f}\nPrompt: {task_prompt[:500]}"

        return CommitMessage(
            summary=summary,
            body=body,
            task_id=task_id,
        )

    async def generate_message_llm(
        self, task_id: str, task_prompt: str, diff: str, client=None
    ) -> CommitMessage:
        """Generate an LLM-based commit message (requires API call).

        Args:
            task_id: Task identifier
            task_prompt: Original task description
            diff: Git diff of changes
            client: Optional LLM client for message generation

        Returns:
            CommitMessage with LLM-generated summary
        """
        if not client:
            # Fall back to fast mode
            msg = self.generate_message_fast(task_id, task_prompt)
            msg.body += f"\nDiff length: {len(diff)} chars"
            return msg

        prompt = f"""Write a conventional commit message for this code change.

Task: {task_prompt[:200]}

Diff (first 3000 chars):
{diff[:3000]}

Format:
First line: type(scope): short description (max 72 chars)
Then blank line, then body explaining what was changed and why.

Return only the exact commit message, no JSON."""

        try:
            response = await client.call(
                model=None,
                prompt=prompt,
                system="You are a precise git commit message writer. Follow conventional commits format.",
                max_tokens=200,
                temperature=0.3,
                timeout=20,
            )
            text = response.text.strip()
            lines = text.split("\n")
            summary = lines[0][:72] if lines else f"feat({task_id}): code update"
            body = "\n".join(lines[1:]) if len(lines) > 1 else ""
            return CommitMessage(
                summary=summary,
                body=body,
                task_id=task_id,
            )
        except Exception:
            return self.generate_message_fast(task_id, task_prompt)

    def commit(self, message: CommitMessage, files: list[str] | None = None) -> bool:
        """Execute git commit with the generated message.

        Args:
            message: Commit message to use
            files: Specific files to commit (None = all)

        Returns:
            True if commit succeeded
        """
        try:
            if files:
                subprocess.run(
                    ["git", "add"] + files, cwd=self.repo_dir, capture_output=True, check=True
                )
            else:
                subprocess.run(
                    ["git", "add", "-A"], cwd=self.repo_dir, capture_output=True, check=True
                )

            subprocess.run(
                ["git", "commit", "-m", message.full_message],
                cwd=self.repo_dir, capture_output=True, check=True,
            )
            return True
        except subprocess.CalledProcessError as e:
            return False

    def get_diff(self, files: list[str] | None = None) -> str:
        """Get the current git diff for message generation.

        Args:
            files: Specific files (None = all)

        Returns:
            Git diff output
        """
        try:
            cmd = ["git", "diff"]
            if files:
                cmd += ["--"] + files
            result = subprocess.run(cmd, cwd=self.repo_dir, capture_output=True, text=True)
            return result.stdout
        except Exception:
            return ""
