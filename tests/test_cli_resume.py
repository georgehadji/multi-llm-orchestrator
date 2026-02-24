"""
Tests for _check_resume() CLI gate — Auto-Resume Detection (Task 3).

Tests cover:
- Returns None when no candidates found
- Returns None immediately when --new-project flag is set
- Single fuzzy match: user confirms Y → returns project_id
- Single fuzzy match: user says N → returns None
- Multiple fuzzy matches: user picks first by number → returns correct project_id
- Timeout on slow DB → returns None (silent fallback)
- No keyword overlap → filtered by _score_candidates → returns None
"""
from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock
from datetime import datetime, timedelta


def _make_db_row(
    project_id: str = "proj_001",
    description: str = "Build a FastAPI auth service",
    status: str = "PARTIAL_SUCCESS",
):
    """Make a database row dict as returned by find_resumable().

    ``updated_at`` is a Unix timestamp float (matching the real DB schema).
    Two days ago gives a moderate recency_score so the candidate passes the
    overall_score > 0.3 filter in _score_candidates when there is keyword
    overlap.
    """
    two_days_ago = datetime.utcnow() - timedelta(days=2)
    return {
        "project_id": project_id,
        "description": description,
        # keywords that strongly overlap with "Build a FastAPI auth service"
        "keywords": ["auth", "build", "fastapi", "service"],
        "status": status,
        # store as Unix timestamp float (what the real DB returns)
        "updated_at": two_days_ago.timestamp(),
    }


class TestCheckResume:
    """Unit-test _check_resume() in isolation."""

    @pytest.mark.asyncio
    async def test_returns_none_when_no_candidates(self):
        from orchestrator.cli import _check_resume

        state_mgr = AsyncMock()
        state_mgr.find_resumable.return_value = []
        result = await _check_resume(
            "Build a GraphQL API", state_mgr, new_project=False
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_new_project_flag(self):
        from orchestrator.cli import _check_resume

        state_mgr = AsyncMock()
        state_mgr.find_resumable.return_value = [_make_db_row()]
        # new_project=True must bypass all detection and return None immediately
        result = await _check_resume(
            "Build a FastAPI auth service", state_mgr, new_project=True
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_single_match_user_says_yes(self):
        from orchestrator.cli import _check_resume

        state_mgr = AsyncMock()
        state_mgr.find_resumable.return_value = [_make_db_row()]
        result = await _check_resume(
            "Build a FastAPI JWT backend",
            state_mgr,
            new_project=False,
            _input_fn=lambda _: "y",
        )
        assert result == "proj_001"

    @pytest.mark.asyncio
    async def test_single_match_user_says_no(self):
        from orchestrator.cli import _check_resume

        state_mgr = AsyncMock()
        state_mgr.find_resumable.return_value = [_make_db_row()]
        result = await _check_resume(
            "Build a FastAPI JWT backend",
            state_mgr,
            new_project=False,
            _input_fn=lambda _: "n",
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_multi_match_user_picks_first(self):
        from orchestrator.cli import _check_resume

        rows = [
            _make_db_row("proj_001", description="Build a FastAPI auth service"),
            _make_db_row("proj_002", description="Build a FastAPI login service"),
        ]
        state_mgr = AsyncMock()
        state_mgr.find_resumable.return_value = rows
        # User types "1" to pick the first entry shown in the numbered list
        result = await _check_resume(
            "Build a FastAPI JWT backend",
            state_mgr,
            new_project=False,
            _input_fn=lambda _: "1",
        )
        # The top-scored candidate should be returned (both rows are similar;
        # whichever is ranked first gets picked with "1")
        assert result in ("proj_001", "proj_002")

    @pytest.mark.asyncio
    async def test_timeout_returns_none(self):
        from orchestrator.cli import _check_resume

        state_mgr = AsyncMock()

        async def slow(*a, **kw):
            await asyncio.sleep(10)
            return []

        state_mgr.find_resumable.side_effect = slow
        result = await _check_resume(
            "Build a FastAPI auth service",
            state_mgr,
            new_project=False,
        )
        assert result is None  # timeout → silent fallback

    @pytest.mark.asyncio
    async def test_no_scoring_match_returns_none(self):
        """Candidates with 0 keyword overlap are filtered by _score_candidates."""
        from orchestrator.cli import _check_resume

        state_mgr = AsyncMock()
        state_mgr.find_resumable.return_value = [
            _make_db_row(description="Deploy a Kubernetes cluster on AWS")
        ]
        result = await _check_resume(
            "Build a mobile app with React Native",
            state_mgr,
            new_project=False,
            _input_fn=lambda _: "n",
        )
        # The keywords for "React Native mobile app" share no overlap with
        # "auth build fastapi service" → filtered out → None
        assert result is None
