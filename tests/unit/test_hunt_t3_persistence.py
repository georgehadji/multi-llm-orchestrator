"""
T3 (persistence & resume) proof-of-defect and no-regression tests.

Two VERIFIED DEFECTs from docs/hunts/t3-persistence/inventory.md:

C1 — orchestrator/state_mgmt/checkpoints.py was an independent fork of
     orchestrator/checkpoints.py that silently lost ContentCheckpointManager
     (real content-preserving rollback via a SnapshotPort) — the fork only
     had hash-metadata-only rollback. Neither had a live importer found, but
     the divergence meant picking the "wrong" (subpackage) import path would
     silently give you a strictly weaker checkpoint manager.

C2 — ResumeDetector.find_resumable_project() was a stub that always
     returned None (its own comment: "For now, return None (would be
     implemented with async/await)"), despite the file's own scoring
     machinery (_extract_keywords, _recency_factor, _score_candidates) being
     fully implemented and already used correctly elsewhere — the live
     resume path in entrypoints/cli_dispatch.py::_check_resume() calls
     StateManager.find_resumable() + _score_candidates() directly, bypassing
     this class entirely.
"""

from __future__ import annotations

import time

import pytest

pytestmark = pytest.mark.unit


# --- C1 -----------------------------------------------------------------


@pytest.mark.unit
def test_c1_state_mgmt_checkpoints_content_manager_is_canonical():
    from orchestrator.checkpoints import ContentCheckpointManager as canonical
    from orchestrator.state_mgmt.checkpoints import ContentCheckpointManager as via_state_mgmt

    assert via_state_mgmt is canonical


@pytest.mark.unit
def test_c1_state_mgmt_checkpoints_exposes_all_manager_classes():
    """No-regression: the shim must not drop any of the pre-existing classes
    the subpackage already had (Checkpoint, CheckpointManager, NamedCheckpoint,
    NamedCheckpointManager)."""
    import orchestrator.state_mgmt.checkpoints as mod

    for name in (
        "Checkpoint",
        "CheckpointManager",
        "NamedCheckpoint",
        "NamedCheckpointManager",
        "ContentCheckpointManager",
    ):
        assert hasattr(mod, name), f"{name} missing from state_mgmt.checkpoints shim"


# --- C2 -------------------------------------------------------------------


class _FakeStateManager:
    """Fake at the StateManager seam — not a mock of the unit under test."""

    def __init__(self, rows):
        self._rows = rows

    async def find_resumable(self, keywords):
        return self._rows


@pytest.mark.asyncio
async def test_c2_find_resumable_project_no_longer_always_none():
    """The core defect: given a real matching candidate, this must return it,
    not unconditionally None."""
    from orchestrator.state_mgmt.resume_detector import ResumeDetector

    rows = [
        {
            "project_id": "proj-1",
            "description": "build a rest api with fastapi",
            "keywords": ["build", "rest", "api", "fastapi"],
            "status": "PARTIAL_SUCCESS",
            "updated_at": time.time(),
        }
    ]
    detector = ResumeDetector(state_manager=_FakeStateManager(rows))

    result = await detector.find_resumable_project("build a rest api with fastapi", "tests pass")

    assert result is not None
    assert result["project_id"] == "proj-1"
    assert result["match_score"] > 0.0


@pytest.mark.asyncio
async def test_c2_find_resumable_project_returns_none_below_threshold():
    """No false positives: an unrelated candidate must not be returned."""
    from orchestrator.state_mgmt.resume_detector import ResumeDetector

    rows = [
        {
            "project_id": "proj-unrelated",
            "description": "completely unrelated topic about gardening",
            "keywords": ["gardening", "unrelated", "topic"],
            "status": "PARTIAL_SUCCESS",
            "updated_at": time.time() - 40 * 24 * 3600,  # 40 days old — recency = 0
        }
    ]
    detector = ResumeDetector(state_manager=_FakeStateManager(rows))

    result = await detector.find_resumable_project("build a rest api with fastapi", "tests pass")

    assert result is None


@pytest.mark.asyncio
async def test_c2_find_resumable_project_returns_none_without_state_manager():
    """No-regression: the original preconditions (no keywords, no state_manager)
    must still short-circuit correctly."""
    from orchestrator.state_mgmt.resume_detector import ResumeDetector

    detector = ResumeDetector(state_manager=None)
    result = await detector.find_resumable_project("build a rest api", "tests pass")

    assert result is None


@pytest.mark.asyncio
async def test_c2_find_resumable_project_handles_no_candidates():
    """No-regression: an empty result set from the state manager must return
    None cleanly, not raise."""
    from orchestrator.state_mgmt.resume_detector import ResumeDetector

    detector = ResumeDetector(state_manager=_FakeStateManager([]))
    result = await detector.find_resumable_project("build a rest api", "tests pass")

    assert result is None
