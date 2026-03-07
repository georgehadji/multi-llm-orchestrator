"""
Git Integration - Auto-commit and version control for orchestrator outputs
=========================================================================
Features:
- Auto-commit after each task completion
- Branch-per-project workflow
- Meaningful commit messages with task details
- Git history preservation for audit trails
- Configurable commit behavior

Inspired by bmalph's Ralph loop with TDD commits.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from enum import Enum

logger = logging.getLogger("orchestrator.git")


class GitError(Exception):
    """Git operation error."""
    pass


class CommitStrategy(Enum):
    """Git commit strategies."""
    AFTER_EACH_TASK = "after_each_task"      # Commit after every task
    AFTER_PHASE = "after_phase"              # Commit after each phase
    AFTER_PROJECT = "after_project"          # Single commit at end
    MANUAL = "manual"                        # No auto-commits


@dataclass
class GitConfig:
    """Git integration configuration."""
    enabled: bool = True
    strategy: CommitStrategy = CommitStrategy.AFTER_EACH_TASK
    branch_prefix: str = "orchestrator"
    commit_message_template: str = "{task_id}: {status} - {model} (${cost:.4f})"
    include_diff_stats: bool = True
    auto_push: bool = False  # Push to remote (use with caution)
    user_name: Optional[str] = None
    user_email: Optional[str] = None


class GitIntegration:
    """Manages Git operations for orchestrator outputs."""
    
    def __init__(self, repo_path: Path, config: Optional[GitConfig] = None):
        self.repo_path = Path(repo_path)
        self.config = config or GitConfig()
        self._git_available = self._check_git()
        self._original_branch: Optional[str] = None
        self._project_branch: Optional[str] = None
    
    def _check_git(self) -> bool:
        """Check if Git is available and repo exists."""
        try:
            result = subprocess.run(
                ["git", "--version"],
                capture_output=True,
                check=True,
            )
            logger.debug(f"Git available: {result.stdout.decode().strip()}")
            
            # Check if we're in a git repo
            git_dir = self.repo_path / ".git"
            if not git_dir.exists():
                logger.warning(f"Not a git repository: {self.repo_path}")
                return False
            
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("Git not available")
            return False
    
    def _run_git(self, args: List[str], check: bool = True) -> subprocess.CompletedProcess:
        """Run a git command."""
        if not self._git_available:
            raise GitError("Git not available")
        
        cmd = ["git", "-C", str(self.repo_path)] + args
        logger.debug(f"Running: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        
        if check and result.returncode != 0:
            raise GitError(f"Git command failed: {result.stderr}")
        
        return result
    
    def is_available(self) -> bool:
        """Check if Git integration is available."""
        return self._git_available and self.config.enabled
    
    # ─────────────────────────────────────────
    # Repository Setup
    # ─────────────────────────────────────────
    
    def setup_project_branch(self, project_id: str, project_name: str) -> str:
        """Create and switch to a project-specific branch."""
        from .secure_execution import InputValidator
        
        if not self.is_available():
            logger.info("Git integration disabled or unavailable")
            return ""
        
        # Save original branch
        result = self._run_git(["rev-parse", "--abbrev-ref", "HEAD"], check=False)
        self._original_branch = result.stdout.strip() if result.returncode == 0 else "main"
        
        # SECURITY FIX: Use proper branch name sanitization
        # Create branch name with validated/sanitized components
        safe_name = InputValidator.sanitize_branch_name(project_name)[:30]
        # Validate project_id is alphanumeric (prevents injection)
        safe_project_id = InputValidator.sanitize_filename(project_id)[:8]
        timestamp = datetime.now().strftime("%Y%m%d")
        safe_prefix = InputValidator.sanitize_branch_name(self.config.branch_prefix)
        branch_name = f"{safe_prefix}/{safe_name}_{timestamp}_{safe_project_id}"
        
        # Create and switch to branch
        try:
            self._run_git(["checkout", "-b", branch_name])
            self._project_branch = branch_name
            logger.info(f"Created and switched to branch: {branch_name}")
            
            # Configure git user if provided
            if self.config.user_name:
                self._run_git(["config", "user.name", self.config.user_name], check=False)
            if self.config.user_email:
                self._run_git(["config", "user.email", self.config.user_email], check=False)
            
            return branch_name
        except GitError as e:
            logger.error(f"Failed to create branch: {e}")
            return ""
    
    def restore_original_branch(self) -> bool:
        """Switch back to original branch."""
        if not self.is_available() or not self._original_branch:
            return False
        
        try:
            self._run_git(["checkout", self._original_branch])
            logger.info(f"Restored original branch: {self._original_branch}")
            return True
        except GitError as e:
            logger.error(f"Failed to restore branch: {e}")
            return False
    
    # ─────────────────────────────────────────
    # Staging and Committing
    # ─────────────────────────────────────────
    
    def stage_all(self) -> bool:
        """Stage all changes."""
        if not self.is_available():
            return False
        
        try:
            self._run_git(["add", "-A"])
            return True
        except GitError as e:
            logger.error(f"Failed to stage: {e}")
            return False
    
    def stage_paths(self, paths: List[Path]) -> bool:
        """Stage specific paths."""
        if not self.is_available():
            return False
        
        try:
            for path in paths:
                if path.exists():
                    rel_path = path.relative_to(self.repo_path)
                    self._run_git(["add", str(rel_path)])
            return True
        except GitError as e:
            logger.error(f"Failed to stage paths: {e}")
            return False
    
    def has_changes(self) -> bool:
        """Check if there are uncommitted changes."""
        if not self.is_available():
            return False
        
        result = self._run_git(["status", "--porcelain"], check=False)
        return bool(result.stdout.strip())
    
    def commit(
        self,
        message: str,
        description: Optional[str] = None,
    ) -> Optional[str]:
        """Create a commit with the given message."""
        if not self.is_available():
            return None
        
        if not self.has_changes():
            logger.debug("No changes to commit")
            return None
        
        try:
            # Build full message
            full_message = message
            if description:
                full_message += f"\n\n{description}"
            
            # Commit
            self._run_git(["commit", "-m", full_message])
            
            # Get commit hash
            result = self._run_git(["rev-parse", "--short", "HEAD"])
            commit_hash = result.stdout.strip()
            
            logger.info(f"Created commit: {commit_hash} - {message[:50]}...")
            
            # Auto-push if enabled
            if self.config.auto_push and self._project_branch:
                self._push_branch()
            
            return commit_hash
        except GitError as e:
            logger.error(f"Failed to commit: {e}")
            return None
    
    # ─────────────────────────────────────────
    # Task-specific Commits
    # ─────────────────────────────────────────
    
    def commit_task_completion(
        self,
        task_id: str,
        status: str,
        model: str,
        cost: float,
        score: float,
        files: List[Path],
    ) -> Optional[str]:
        """Commit after task completion (bmalph-style)."""
        if not self.is_available():
            return None
        
        if self.config.strategy != CommitStrategy.AFTER_EACH_TASK:
            return None
        
        # Stage specific files
        if files:
            self.stage_paths(files)
        else:
            self.stage_all()
        
        if not self.has_changes():
            return None
        
        # Build commit message
        message = self.config.commit_message_template.format(
            task_id=task_id,
            status=status.upper(),
            model=model,
            cost=cost,
            score=f"{score:.2f}" if score > 0 else "N/A",
        )
        
        # Add description
        description_lines = [
            f"Task: {task_id}",
            f"Status: {status}",
            f"Model: {model}",
            f"Cost: ${cost:.4f}",
        ]
        if score > 0:
            description_lines.append(f"Score: {score:.2f}")
        
        if self.config.include_diff_stats:
            stats = self._get_diff_stats()
            if stats:
                description_lines.append(f"\n{stats}")
        
        description = "\n".join(description_lines)
        
        return self.commit(message, description)
    
    def commit_phase(
        self,
        phase: str,
        task_count: int,
        total_cost: float,
    ) -> Optional[str]:
        """Commit after phase completion."""
        if not self.is_available():
            return None
        
        if self.config.strategy not in (CommitStrategy.AFTER_PHASE, CommitStrategy.AFTER_EACH_TASK):
            return None
        
        self.stage_all()
        
        if not self.has_changes():
            return None
        
        message = f"Phase: {phase} - {task_count} tasks (${total_cost:.4f})"
        description = f"Completed phase: {phase}\nTasks: {task_count}\nCost: ${total_cost:.4f}"
        
        return self.commit(message, description)
    
    def commit_project(
        self,
        project_name: str,
        total_tasks: int,
        total_cost: float,
        elapsed_time: float,
    ) -> Optional[str]:
        """Final project commit."""
        if not self.is_available():
            return None
        
        self.stage_all()
        
        if not self.has_changes():
            return None
        
        message = f"Project: {project_name} - Complete"
        description = f"""Project completed: {project_name}

Summary:
- Tasks: {total_tasks}
- Cost: ${total_cost:.4f}
- Time: {elapsed_time:.1f}s
- Branch: {self._project_branch or 'N/A'}

Generated by Multi-LLM Orchestrator"""
        
        return self.commit(message, description)
    
    # ─────────────────────────────────────────
    # Utility Methods
    # ─────────────────────────────────────────
    
    def _get_diff_stats(self) -> str:
        """Get diff statistics for commit message."""
        try:
            result = self._run_git(["diff", "--cached", "--stat"], check=False)
            return result.stdout.strip()
        except GitError:
            return ""
    
    def _push_branch(self) -> bool:
        """Push current branch to remote."""
        try:
            if not self._project_branch:
                return False
            
            self._run_git(["push", "-u", "origin", self._project_branch])
            logger.info(f"Pushed branch: {self._project_branch}")
            return True
        except GitError as e:
            logger.warning(f"Failed to push branch: {e}")
            return False
    
    def get_commit_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent commit history."""
        if not self.is_available():
            return []
        
        try:
            format_str = "%H|%s|%ci|%an"
            result = self._run_git(
                ["log", f"-{limit}", f"--format={format_str}"],
                check=False
            )
            
            commits = []
            for line in result.stdout.strip().split("\n"):
                if "|" in line:
                    parts = line.split("|", 3)
                    commits.append({
                        "hash": parts[0][:8],
                        "message": parts[1],
                        "date": parts[2],
                        "author": parts[3] if len(parts) > 3 else "Unknown",
                    })
            
            return commits
        except GitError:
            return []
    
    def create_tag(self, tag_name: str, message: str) -> bool:
        """Create a git tag."""
        if not self.is_available():
            return False
        
        try:
            self._run_git(["tag", "-a", tag_name, "-m", message])
            logger.info(f"Created tag: {tag_name}")
            return True
        except GitError as e:
            logger.error(f"Failed to create tag: {e}")
            return False
    
    def get_branch_name(self) -> str:
        """Get current branch name."""
        if not self.is_available():
            return ""
        
        try:
            result = self._run_git(["rev-parse", "--abbrev-ref", "HEAD"])
            return result.stdout.strip()
        except GitError:
            return ""


# Global git config
def get_default_git_config() -> GitConfig:
    """Get default git configuration from environment."""
    import os
    
    return GitConfig(
        enabled=os.environ.get("ORCHESTRATOR_GIT_ENABLED", "true").lower() == "true",
        strategy=CommitStrategy(
            os.environ.get("ORCHESTRATOR_GIT_STRATEGY", "after_each_task")
        ),
        branch_prefix=os.environ.get("ORCHESTRATOR_GIT_BRANCH_PREFIX", "orchestrator"),
        auto_push=os.environ.get("ORCHESTRATOR_GIT_AUTO_PUSH", "false").lower() == "true",
        user_name=os.environ.get("ORCHESTRATOR_GIT_USER_NAME"),
        user_email=os.environ.get("ORCHESTRATOR_GIT_USER_EMAIL"),
    )
