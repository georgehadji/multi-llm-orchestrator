"""
Gradual rollout stage decisions
===============================
The default last stage (100%) has max_failures=0, and the failure check was
``failures >= max_failures``, so 0 >= 0 rolled back every rollout that reached
it, with no failures at all. max_failures=0 means "no failure tolerated", not
"roll back unconditionally".
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

from orchestrator.meta.orchestrator import StrategyProposal, StrategyType
from orchestrator.operations.gradual_rollout import GradualRolloutManager, RolloutConfig


async def _rollout_at(tmp_path, stage_index: int):
    mgr = GradualRolloutManager(archive=None, config=RolloutConfig(storage_path=tmp_path))
    proposal = StrategyProposal(
        proposal_id="p1",
        strategy_type=StrategyType.MODEL_ROUTING,
        description="d",
        current_config={},
        proposed_config={},
        expected_improvement=0.1,
        confidence=0.9,
        evidence=[],
    )
    rollout = await mgr.start_rollout(proposal)
    rollout.current_stage_index = stage_index
    return mgr, rollout.rollout_id


async def _record(mgr, rid, successes: int, failures: int) -> None:
    for ok in [True] * successes + [False] * failures:
        await mgr.record_execution(rid, ok, 0.9, 0.01, 10.0, "proj")


async def test_last_stage_with_no_failures_advances(tmp_path):
    mgr, rid = await _rollout_at(tmp_path, stage_index=3)

    decision = await mgr.check_stage_progress(rid)

    assert decision.decision == "advance"


async def test_last_stage_rolls_back_on_any_failure(tmp_path):
    mgr, rid = await _rollout_at(tmp_path, stage_index=3)
    await _record(mgr, rid, successes=0, failures=1)

    decision = await mgr.check_stage_progress(rid)

    assert decision.decision == "rollback"


async def test_early_stage_rolls_back_at_its_failure_limit(tmp_path):
    mgr, rid = await _rollout_at(tmp_path, stage_index=0)  # max_failures=3
    await _record(mgr, rid, successes=10, failures=3)

    decision = await mgr.check_stage_progress(rid)

    assert decision.decision == "rollback"


async def test_early_stage_under_its_failure_limit_advances(tmp_path):
    mgr, rid = await _rollout_at(tmp_path, stage_index=0)  # min_successes=10
    await _record(mgr, rid, successes=10, failures=2)

    decision = await mgr.check_stage_progress(rid)

    assert decision.decision == "advance"
