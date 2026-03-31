"""
Git Integration Hooks for Orchestrator Lifecycle
================================================

Hooks that integrate Git service calls into the orchestrator pipeline:
- Check run creation/updates
- PR comments from code review
- Auto-commit / PR creation on Quality Gate pass

Author: Georgios-Chrysovalantis Chatzivantsidis
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from orchestrator.git_service import (
    CheckRunConclusion,
    CheckRunOutput,
    CheckRunStatus,
    GitHubService,
    GitIntegrationConfig,
    GitService,
    PRComment,
)
from orchestrator.models import ProjectState, ProjectStatus

if TYPE_CHECKING:
    from orchestrator.policy import PolicySet

logger = logging.getLogger("orchestrator.git_hooks")


@dataclass
class GitHookContext:
    """Context passed to Git hooks containing run information."""

    run_id: str
    commit_sha: str
    branch: str
    pr_number: int | None = None
    repository: str = ""
    triggered_by: str = "manual"  # manual, ci, webhook


class GitIntegrationHooks:
    """Hooks for integrating Git service into orchestrator lifecycle."""

    def __init__(self, git_service: GitService | None = None):
        self.git = git_service
        self._posted_comment_hashes: set[str] = set()

    @classmethod
    def from_config(cls, config: GitIntegrationConfig | None = None) -> GitIntegrationHooks:
        """Create hooks instance from configuration."""
        if config is None:
            config = GitIntegrationConfig.from_env()

        if not config.enabled or not config.validate():
            return cls(None)

        if config.provider == "github":
            return cls(GitHubService(config))
        else:
            logger.warning(f"Git provider {config.provider} not yet implemented")
            return cls(None)

    async def on_run_start(
        self,
        context: GitHookContext,
        project_description: str,
    ) -> None:
        """Called when orchestrator run starts. Creates 'queued' check run."""
        if not self.git:
            return

        try:
            output = CheckRunOutput(
                title="AI Orchestrator: Queued",
                summary=f"Starting orchestration for: {project_description[:80]}...",
            )

            await self.git.create_or_update_check_run(
                run_id=context.run_id,
                commit_sha=context.commit_sha,
                name="AI Orchestrator",
                status=CheckRunStatus.QUEUED,
                output=output,
            )
            logger.info(f"Created check run for {context.run_id}")
        except Exception as e:
            logger.error(f"Failed to create check run: {e}")

    async def on_run_in_progress(
        self,
        context: GitHookContext,
        total_tasks: int,
    ) -> None:
        """Called when orchestrator starts executing tasks."""
        if not self.git:
            return

        try:
            output = CheckRunOutput(
                title="AI Orchestrator: In Progress",
                summary=f"Executing {total_tasks} tasks...",
            )

            await self.git.create_or_update_check_run(
                run_id=context.run_id,
                commit_sha=context.commit_sha,
                name="AI Orchestrator",
                status=CheckRunStatus.IN_PROGRESS,
                output=output,
            )
        except Exception as e:
            logger.error(f"Failed to update check run: {e}")

    async def on_run_complete(
        self,
        context: GitHookContext,
        state: ProjectState,
        dashboard_url: str,
    ) -> None:
        """Called when orchestrator run completes. Updates check run with final status."""
        if not self.git:
            return

        try:
            # Determine conclusion based on quality gate
            quality_passed = state.status in (ProjectStatus.SUCCESS, ProjectStatus.PARTIAL_SUCCESS)
            conclusion = (
                CheckRunConclusion.SUCCESS if quality_passed else CheckRunConclusion.FAILURE
            )

            # Build summary
            budget = state.budget
            tasks_completed = len([r for r in state.results.values() if r.score >= 0.85])
            total_tasks = len(state.results)

            # Get main models used
            models_used = set()
            for result in state.results.values():
                models_used.add(result.model_used.value)
            main_models = ", ".join(list(models_used)[:3])

            summary = f"""**Status:** {state.status.value}
**Tasks:** {tasks_completed}/{total_tasks} passed (score ≥ 0.85)
**Budget:** ${budget.spent_usd:.4f} / ${budget.max_usd}
**Models:** {main_models}
**Time:** {budget.elapsed_seconds:.0f}s

[View Dashboard]({dashboard_url}/runs/{context.run_id})
"""

            title = "AI Orchestrator: PASSED ✅" if quality_passed else "AI Orchestrator: FAILED ❌"

            output = CheckRunOutput(
                title=title,
                summary=summary,
                text=self._format_detailed_output(state),
            )

            await self.git.create_or_update_check_run(
                run_id=context.run_id,
                commit_sha=context.commit_sha,
                name="AI Orchestrator",
                status=CheckRunStatus.COMPLETED,
                conclusion=conclusion,
                output=output,
            )
            logger.info(f"Updated check run for {context.run_id}: {conclusion.value}")
        except Exception as e:
            logger.error(f"Failed to complete check run: {e}")

    def _format_detailed_output(self, state: ProjectState) -> str:
        """Format detailed markdown output for check run."""
        lines = ["## Task Results", ""]
        lines.append("| Task | Type | Score | Model | Status |")
        lines.append("|------|------|-------|-------|--------|")

        for task_id in state.execution_order or state.results.keys():
            result = state.results.get(task_id)
            if not result:
                continue
            status_icon = "✅" if result.status.value == "completed" else "❌"
            lines.append(
                f"| {task_id} | {result.model_used.value} | "
                f"{result.score:.2f} | {result.model_used.value[:20]} | {status_icon} |"
            )

        return "\n".join(lines)

    async def post_code_review_comments(
        self,
        context: GitHookContext,
        review_results: list[dict],  # From code review tasks
        policy_set: PolicySet | None = None,
    ) -> None:
        """Post PR comments from code review results."""
        if not self.git or not context.pr_number:
            return

        # Check if enabled in policy
        if policy_set:
            enable_comments = True
            for policy in policy_set.global_policies:
                if hasattr(policy, "enable_pr_comments"):
                    enable_comments = policy.enable_pr_comments
                    break
            if not enable_comments:
                logger.debug("PR comments disabled by policy")
                return

        try:
            # Get existing comments for deduplication
            existing = await self.git.get_existing_pr_comments(context.pr_number)
            existing_hashes = {c.compute_hash() for c in existing}

            for review in review_results:
                # Convert review to PR comment
                comment = PRComment(
                    path=review.get("file", "unknown"),
                    line=review.get("line"),
                    body=self._format_review_comment(review),
                    commit_id=context.commit_sha,
                )

                # Skip duplicates
                comment_hash = comment.compute_hash()
                if comment_hash in existing_hashes or comment_hash in self._posted_comment_hashes:
                    logger.debug(f"Skipping duplicate comment on {comment.path}")
                    continue

                await self.git.post_pr_comment(context.pr_number, comment)
                self._posted_comment_hashes.add(comment_hash)

            logger.info(f"Posted {len(review_results)} PR comments")
        except Exception as e:
            logger.error(f"Failed to post PR comments: {e}")

    def _format_review_comment(self, review: dict) -> str:
        """Format code review result as PR comment body."""
        issues = review.get("issues", [])
        suggestions = review.get("suggestions", [])

        lines = ["### 🤖 AI Code Review", ""]

        if issues:
            lines.append("**Issues:**")
            for issue in issues:
                lines.append(f"- ⚠️ {issue}")
            lines.append("")

        if suggestions:
            lines.append("**Suggestions:**")
            for suggestion in suggestions:
                lines.append(f"- 💡 {suggestion}")
            lines.append("")

        lines.append("---")
        lines.append("*Reviewed by Multi-LLM Orchestrator*")

        return "\n".join(lines)

    async def auto_commit_or_create_pr(
        self,
        context: GitHookContext,
        state: ProjectState,
        files: dict[str, str],  # path -> content
        policy_set: PolicySet | None = None,
    ) -> dict | None:
        """Auto-commit changes or create PR based on policy."""
        if not self.git:
            return None

        # Check quality gate
        quality_passed = state.status in (ProjectStatus.SUCCESS, ProjectStatus.PARTIAL_SUCCESS)
        if not quality_passed:
            logger.info("Quality gate failed, skipping auto-commit")
            return None

        # Check policy flags
        allow_auto_commit = False
        require_human_approval = True

        if policy_set:
            for policy in policy_set.global_policies:
                if hasattr(policy, "allow_auto_commit"):
                    allow_auto_commit = policy.allow_auto_commit
                if hasattr(policy, "require_human_approval"):
                    require_human_approval = policy.require_human_approval

        if not allow_auto_commit:
            logger.info("Auto-commit disabled by policy")
            return None

        try:
            commit_message = (
                f"AI Orchestrator: apply codegen + fixes for {state.project_description[:50]}\n\n"
                f"- Quality Score: {self._get_average_score(state):.2f}\n"
                f"- Tasks: {len(state.results)}\n"
                f"- Budget: ${state.budget.spent_usd:.4f}"
            )

            if context.pr_number and not require_human_approval:
                # Commit to existing PR branch
                await self.git.commit_changes(
                    branch=context.branch,
                    message=commit_message,
                    files=files,
                )
                logger.info(f"Committed changes to {context.branch}")
                return {"action": "commit", "branch": context.branch}

            else:
                # Create new branch and PR
                new_branch = f"ai-orchestrator/{context.run_id[:8]}"
                await self.git.create_branch(new_branch, context.commit_sha)

                await self.git.commit_changes(
                    branch=new_branch,
                    message=commit_message,
                    files=files,
                )

                # Build PR body
                pr_body = self._format_pr_body(state, require_human_approval)

                pr_number = await self.git.create_pull_request(
                    title=f"AI Orchestrator suggestions: {state.project_description[:50]}",
                    body=pr_body,
                    head_branch=new_branch,
                    base_branch=context.branch,
                )

                logger.info(f"Created PR #{pr_number} from {new_branch}")
                return {
                    "action": "pr_created",
                    "pr_number": pr_number,
                    "branch": new_branch,
                }

        except Exception as e:
            logger.error(f"Failed to auto-commit/create PR: {e}")
            return None

    def _get_average_score(self, state: ProjectState) -> float:
        """Calculate average task score."""
        if not state.results:
            return 0.0
        return sum(r.score for r in state.results.values()) / len(state.results)

    def _format_pr_body(self, state: ProjectState, require_human_approval: bool) -> str:
        """Format PR description body."""
        lines = [
            "## 🤖 AI-Generated Code Changes",
            "",
            f"**Project:** {state.project_description}",
            f"**Quality Score:** {self._get_average_score(state):.2f}/1.0",
            f"**Status:** {state.status.value}",
            "",
            "### Summary",
            "",
        ]

        # List main changes
        for task_id in state.execution_order or []:
            task = state.tasks.get(task_id)
            if task and task.type.value == "code_generation":
                lines.append(f"- {task.prompt[:80]}...")

        lines.extend(
            [
                "",
                "### Checklist",
                "",
            ]
        )

        if require_human_approval:
            lines.extend(
                [
                    "- [ ] Human review completed",
                    "- [ ] Tests pass",
                    "- [ ] Code style checked",
                ]
            )

        lines.extend(
            [
                "",
                "---",
                "*Generated by Multi-LLM Orchestrator*",
            ]
        )

        return "\n".join(lines)


def create_git_hooks_from_env() -> GitIntegrationHooks:
    """Factory function to create Git hooks from environment variables."""
    return GitIntegrationHooks.from_config()
