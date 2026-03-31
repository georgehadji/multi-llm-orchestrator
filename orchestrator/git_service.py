"""
Git Integration for Multi-LLM Orchestrator
==========================================

Provides CI-style integration with GitHub/GitLab:
- Check Run / Status API per orchestrator run
- Structured PR comments from code review results
- Optional auto-commit / PR creation on Quality Gate pass

Author: Georgios-Chrysovalantis Chatzivantsidis
"""

from __future__ import annotations

import hashlib
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum

logger = logging.getLogger("orchestrator.git")


class CheckRunStatus(StrEnum):
    """Check run status states."""

    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


class CheckRunConclusion(StrEnum):
    """Check run conclusion states."""

    SUCCESS = "success"
    FAILURE = "failure"
    NEUTRAL = "neutral"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    ACTION_REQUIRED = "action_required"


@dataclass
class CheckRunOutput:
    """Structured output for check runs."""

    title: str
    summary: str
    text: str | None = None

    def to_dict(self) -> dict:
        result = {
            "title": self.title,
            "summary": self.summary,
        }
        if self.text:
            result["text"] = self.text
        return result


@dataclass
class PRComment:
    """Structured PR comment."""

    path: str  # File path
    line: int | None = None  # Line number (optional)
    start_line: int | None = None  # For multi-line comments
    body: str = ""
    commit_id: str | None = None

    def compute_hash(self) -> str:
        """Compute unique hash for deduplication."""
        content = f"{self.path}:{self.start_line or self.line}:{self.body}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class GitIntegrationConfig:
    """Configuration for Git integration."""

    enabled: bool = False
    provider: str = "github"  # github, gitlab
    token: str = ""
    repository: str = ""  # owner/repo format
    base_url: str = ""  # For GitLab self-hosted
    enable_pr_comments: bool = True
    enable_auto_commit: bool = False
    require_human_approval: bool = True
    dashboard_url: str = ""  # Base URL for dashboard links

    @classmethod
    def from_env(cls) -> GitIntegrationConfig:
        """Load configuration from environment variables."""
        return cls(
            enabled=os.getenv("GIT_INTEGRATION_ENABLED", "false").lower() == "true",
            provider=os.getenv("GIT_PROVIDER", "github"),
            token=os.getenv("GIT_TOKEN", ""),
            repository=os.getenv("GIT_REPOSITORY", ""),
            base_url=os.getenv("GIT_BASE_URL", ""),
            enable_pr_comments=os.getenv("ENABLE_PR_COMMENTS", "true").lower() == "true",
            enable_auto_commit=os.getenv("ENABLE_AUTO_COMMIT", "false").lower() == "true",
            require_human_approval=os.getenv("REQUIRE_HUMAN_APPROVAL", "true").lower() == "true",
            dashboard_url=os.getenv("DASHBOARD_URL", "http://localhost:8888"),
        )

    def validate(self) -> bool:
        """Validate configuration."""
        if not self.enabled:
            return True
        if not self.token:
            logger.warning("Git integration enabled but no token provided")
            return False
        if not self.repository:
            logger.warning("Git integration enabled but no repository provided")
            return False
        return True


class GitService(ABC):
    """Abstract base class for Git platform integration."""

    def __init__(self, config: GitIntegrationConfig):
        self.config = config
        self._check_run_cache: dict[str, int] = {}  # run_id -> check_run_id

    @abstractmethod
    async def create_check_run(
        self,
        commit_sha: str,
        name: str,
        status: CheckRunStatus,
        output: CheckRunOutput | None = None,
    ) -> int:
        """Create a new check run. Returns check run ID."""
        pass

    @abstractmethod
    async def update_check_run(
        self,
        check_run_id: int,
        status: CheckRunStatus,
        conclusion: CheckRunConclusion | None = None,
        output: CheckRunOutput | None = None,
    ) -> None:
        """Update an existing check run."""
        pass

    @abstractmethod
    async def post_pr_comment(
        self,
        pr_number: int,
        comment: PRComment,
    ) -> int:
        """Post a PR comment. Returns comment ID."""
        pass

    @abstractmethod
    async def get_existing_pr_comments(
        self,
        pr_number: int,
    ) -> list[PRComment]:
        """Get existing PR comments for deduplication."""
        pass

    @abstractmethod
    async def create_branch(
        self,
        branch_name: str,
        base_sha: str,
    ) -> None:
        """Create a new branch from base SHA."""
        pass

    @abstractmethod
    async def commit_changes(
        self,
        branch: str,
        message: str,
        files: dict[str, str],  # path -> content
    ) -> str:
        """Commit changes to a branch. Returns commit SHA."""
        pass

    @abstractmethod
    async def create_pull_request(
        self,
        title: str,
        body: str,
        head_branch: str,
        base_branch: str,
    ) -> int:
        """Create a new pull request. Returns PR number."""
        pass

    async def create_or_update_check_run(
        self,
        run_id: str,
        commit_sha: str,
        name: str,
        status: CheckRunStatus,
        output: CheckRunOutput | None = None,
    ) -> int:
        """Create new check run or update existing one."""
        if run_id in self._check_run_cache:
            check_run_id = self._check_run_cache[run_id]
            await self.update_check_run(check_run_id, status, output=output)
            return check_run_id
        else:
            check_run_id = await self.create_check_run(commit_sha, name, status, output)
            self._check_run_cache[run_id] = check_run_id
            return check_run_id


class GitHubService(GitService):
    """GitHub integration using REST API."""

    def __init__(self, config: GitIntegrationConfig):
        super().__init__(config)
        self.api_base = "https://api.github.com"
        self.headers = {
            "Authorization": f"Bearer {config.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    async def _api_request(
        self,
        method: str,
        endpoint: str,
        data: dict | None = None,
    ) -> dict:
        """Make GitHub API request."""
        import aiohttp

        url = f"{self.api_base}/repos/{self.config.repository}{endpoint}"

        async with (
            aiohttp.ClientSession() as session,
            session.request(method, url, headers=self.headers, json=data) as response,
        ):
            if response.status >= 400:
                text = await response.text()
                raise RuntimeError(f"GitHub API error {response.status}: {text}")
            return await response.json()

    async def create_check_run(
        self,
        commit_sha: str,
        name: str,
        status: CheckRunStatus,
        output: CheckRunOutput | None = None,
    ) -> int:
        """Create a new GitHub check run."""
        data = {
            "name": name,
            "head_sha": commit_sha,
            "status": status.value,
        }
        if output:
            data["output"] = output.to_dict()

        result = await self._api_request("POST", "/check-runs", data)
        check_run_id = result["id"]
        logger.info(f"Created GitHub check run {check_run_id} for {commit_sha[:7]}")
        return check_run_id

    async def update_check_run(
        self,
        check_run_id: int,
        status: CheckRunStatus,
        conclusion: CheckRunConclusion | None = None,
        output: CheckRunOutput | None = None,
    ) -> None:
        """Update existing GitHub check run."""
        data = {"status": status.value}

        if conclusion:
            data["conclusion"] = conclusion.value
        if output:
            data["output"] = output.to_dict()

        await self._api_request("PATCH", f"/check-runs/{check_run_id}", data)
        logger.debug(f"Updated GitHub check run {check_run_id}: {status.value}")

    async def post_pr_comment(
        self,
        pr_number: int,
        comment: PRComment,
    ) -> int:
        """Post a PR review comment (inline on code)."""
        # Create a review with comment
        data = {
            "commit_id": comment.commit_id,
            "body": comment.body,
            "event": "COMMENT",
            "comments": [
                {
                    "path": comment.path,
                    "line": comment.line,
                    "body": comment.body,
                }
            ],
        }

        if comment.start_line and comment.start_line != comment.line:
            data["comments"][0]["start_line"] = comment.start_line
            data["comments"][0]["start_side"] = "RIGHT"

        result = await self._api_request("POST", f"/pulls/{pr_number}/reviews", data)
        comment_id = result["id"]
        logger.info(f"Posted GitHub PR comment {comment_id} on PR #{pr_number}")
        return comment_id

    async def get_existing_pr_comments(
        self,
        pr_number: int,
    ) -> list[PRComment]:
        """Get existing PR review comments for deduplication."""
        result = await self._api_request("GET", f"/pulls/{pr_number}/comments")

        comments = []
        for item in result:
            comment = PRComment(
                path=item["path"],
                line=item.get("line"),
                body=item["body"],
                commit_id=item.get("commit_id"),
            )
            comments.append(comment)

        return comments

    async def create_branch(
        self,
        branch_name: str,
        base_sha: str,
    ) -> None:
        """Create a new branch."""
        data = {
            "ref": f"refs/heads/{branch_name}",
            "sha": base_sha,
        }
        await self._api_request("POST", "/git/refs", data)
        logger.info(f"Created branch {branch_name} from {base_sha[:7]}")

    async def commit_changes(
        self,
        branch: str,
        message: str,
        files: dict[str, str],
    ) -> str:
        """Commit multiple file changes."""
        # Get current commit on branch
        ref_data = await self._api_request("GET", f"/git/ref/heads/{branch}")
        base_sha = ref_data["object"]["sha"]

        # Get base tree
        commit_data = await self._api_request("GET", f"/git/commits/{base_sha}")
        base_tree_sha = commit_data["tree"]["sha"]

        # Create blobs for each file
        tree_entries = []
        for path, content in files.items():
            blob_data = await self._api_request(
                "POST", "/git/blobs", {"content": content, "encoding": "utf-8"}
            )
            tree_entries.append(
                {
                    "path": path,
                    "mode": "100644",
                    "type": "blob",
                    "sha": blob_data["sha"],
                }
            )

        # Create tree
        tree_data = await self._api_request(
            "POST", "/git/trees", {"base_tree": base_tree_sha, "tree": tree_entries}
        )

        # Create commit
        commit_data = await self._api_request(
            "POST",
            "/git/commits",
            {
                "message": message,
                "tree": tree_data["sha"],
                "parents": [base_sha],
            },
        )

        # Update branch reference
        await self._api_request(
            "PATCH",
            f"/git/refs/heads/{branch}",
            {
                "sha": commit_data["sha"],
            },
        )

        logger.info(f"Committed changes to {branch}: {commit_data['sha'][:7]}")
        return commit_data["sha"]

    async def create_pull_request(
        self,
        title: str,
        body: str,
        head_branch: str,
        base_branch: str,
    ) -> int:
        """Create a new pull request."""
        data = {
            "title": title,
            "body": body,
            "head": head_branch,
            "base": base_branch,
        }
        result = await self._api_request("POST", "/pulls", data)
        pr_number = result["number"]
        logger.info(f"Created PR #{pr_number}: {title}")
        return pr_number


class GitLabService(GitService):
    """GitLab integration (stub for future implementation)."""

    def __init__(self, config: GitIntegrationConfig):
        super().__init__(config)
        self.api_base = config.base_url or "https://gitlab.com/api/v4"
        self.headers = {"PRIVATE-TOKEN": config.token}

    async def create_check_run(
        self,
        commit_sha: str,
        name: str,
        status: CheckRunStatus,
        output: CheckRunOutput | None = None,
    ) -> int:
        # GitLab uses commit status API (simpler than GitHub check runs)
        raise NotImplementedError("GitLab integration coming soon")

    async def update_check_run(
        self,
        check_run_id: int,
        status: CheckRunStatus,
        conclusion: CheckRunConclusion | None = None,
        output: CheckRunOutput | None = None,
    ) -> None:
        raise NotImplementedError("GitLab integration coming soon")

    async def post_pr_comment(self, pr_number: int, comment: PRComment) -> int:
        raise NotImplementedError("GitLab integration coming soon")

    async def get_existing_pr_comments(self, pr_number: int) -> list[PRComment]:
        raise NotImplementedError("GitLab integration coming soon")

    async def create_branch(self, branch_name: str, base_sha: str) -> None:
        raise NotImplementedError("GitLab integration coming soon")

    async def commit_changes(self, branch: str, message: str, files: dict[str, str]) -> str:
        raise NotImplementedError("GitLab integration coming soon")

    async def create_pull_request(
        self, title: str, body: str, head_branch: str, base_branch: str
    ) -> int:
        raise NotImplementedError("GitLab integration coming soon")
