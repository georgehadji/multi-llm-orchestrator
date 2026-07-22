"""
Unit tests for orchestrator.application.git_bridge.GitBridge

P3-5 of REFACTORING_PLAN_V7.md.
"""

from unittest.mock import MagicMock
import pytest

pytestmark = pytest.mark.unit

from orchestrator.application.git_bridge import GitBridge

# ─────────────────────────────────────────────────────────────────────────────
# Null git — returns None, no raises
# ─────────────────────────────────────────────────────────────────────────────


def test_null_git_returns_none():
    bridge = GitBridge(None)
    result = bridge.commit_project("proj", total_tasks=3, total_cost=0.5, elapsed_time=60.0)
    assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# Live git — calls forwarded, hash returned
# ─────────────────────────────────────────────────────────────────────────────


def test_commit_project_calls_git():
    git = MagicMock()
    git.is_available.return_value = True
    git.commit_project.return_value = "abc123"
    git.get_branch_name.return_value = "main"

    bridge = GitBridge(git)
    result = bridge.commit_project("My Project", total_tasks=5, total_cost=1.2, elapsed_time=120.0)

    git.commit_project.assert_called_once_with(
        project_name="My Project",
        total_tasks=5,
        total_cost=1.2,
        elapsed_time=120.0,
    )
    assert result == "abc123"


def test_commit_project_unavailable_returns_none():
    git = MagicMock()
    git.is_available.return_value = False

    bridge = GitBridge(git)
    result = bridge.commit_project("proj", total_tasks=1, total_cost=0.0, elapsed_time=10.0)

    git.commit_project.assert_not_called()
    assert result is None


def test_commit_project_exception_suppressed():
    git = MagicMock()
    git.is_available.return_value = True
    git.commit_project.side_effect = RuntimeError("git not initialized")

    bridge = GitBridge(git)
    result = bridge.commit_project("proj", total_tasks=1, total_cost=0.0, elapsed_time=10.0)

    assert result is None  # Exception swallowed


def test_commit_project_no_hash_still_returns_none():
    git = MagicMock()
    git.is_available.return_value = True
    git.commit_project.return_value = None

    bridge = GitBridge(git)
    result = bridge.commit_project("proj", total_tasks=1, total_cost=0.0, elapsed_time=10.0)

    assert result is None
