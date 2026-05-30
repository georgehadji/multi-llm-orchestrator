"""
Null-safe bridge to the optional git integration.

P3-5 of REFACTORING_PLAN_V7.md — extracted from engine.run_project git block.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class GitBridge:
    """Wraps the optional git integration with null-safety and error suppression.

    A no-op when the underlying ``git`` object is ``None`` or unavailable.
    Failures are logged as warnings but never propagate into orchestration logic.
    """

    def __init__(self, git: Any) -> None:
        self._git = git

    def commit_project(
        self,
        project_name: str,
        total_tasks: int,
        total_cost: float,
        elapsed_time: float,
    ) -> str | None:
        """Create a final commit for a completed project.

        Returns the commit hash, or ``None`` if git is unavailable.
        """
        if self._git is None:
            return None
        try:
            if not self._git.is_available():
                return None
            commit_hash = self._git.commit_project(
                project_name=project_name,
                total_tasks=total_tasks,
                total_cost=total_cost,
                elapsed_time=elapsed_time,
            )
            if commit_hash:
                logger.info("Final git commit: %s", commit_hash)
                branch = self._git.get_branch_name()
                logger.info("Branch: %s", branch)
            return commit_hash  # type: ignore[no-any-return]
        except Exception as exc:
            logger.warning("Final git commit failed: %s", exc)
            return None
