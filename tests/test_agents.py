"""
Tests for AgentPool and TaskChannel (Improvement 5).
Covers: TaskChannel async operations, peek_all non-destructiveness,
        AgentPool.run_parallel with mock orchestrators, best_result,
        merge_telemetry, get_channel wiring on engine.

No external async plugin required — all async tests use asyncio.run() wrappers.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.agents import AgentPool, TaskChannel
from orchestrator.models import (
    Budget,
    Model,
    ProjectState,
    ProjectStatus,
    TaskResult,
    TaskType,
    build_default_profiles,
)
from orchestrator.policy import JobSpec, PolicySet


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def run(coro):
    """Run a coroutine synchronously — no pytest-asyncio needed."""
    return asyncio.run(coro)


def _make_project_state(status: ProjectStatus = ProjectStatus.SUCCESS) -> ProjectState:
    return ProjectState(
        project_description="test",
        success_criteria="pass",
        budget=Budget(max_usd=1.0),
        status=status,
    )


def _make_spec() -> JobSpec:
    return JobSpec(
        project_description="test",
        success_criteria="pass",
        budget=Budget(max_usd=1.0),
    )


def _state_with_scores(*scores: float) -> ProjectState:
    results = {}
    for i, score in enumerate(scores):
        results[f"t_{i}"] = TaskResult(
            task_id=f"t_{i}",
            output="ok",
            score=score,
            model_used=Model.KIMI_K2_5,
        )
    return ProjectState(
        project_description="test",
        success_criteria="pass",
        budget=Budget(max_usd=1.0),
        status=ProjectStatus.SUCCESS,
        results=results,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TaskChannel — basic async operations
# ─────────────────────────────────────────────────────────────────────────────

def test_put_and_get_message():
    async def _run():
        ch = TaskChannel()
        await ch.put({"type": "schema", "value": 42})
        msg = await ch.get()
        return msg
    assert run(_run()) == {"type": "schema", "value": 42}


def test_qsize_reflects_queue_depth():
    async def _run():
        ch = TaskChannel()
        assert ch.qsize() == 0
        await ch.put({"a": 1})
        assert ch.qsize() == 1
        await ch.put({"b": 2})
        assert ch.qsize() == 2
    run(_run())


def test_peek_all_nondestructive():
    """Items returned by peek_all must still be in the queue afterwards."""
    async def _run():
        ch = TaskChannel()
        msgs = [{"i": i} for i in range(3)]
        for m in msgs:
            await ch.put(m)

        peeked = ch.peek_all()
        assert peeked == msgs
        # Queue still has the same items
        assert ch.qsize() == 3
        # Get them all out
        retrieved = [await ch.get() for _ in range(3)]
        assert retrieved == msgs
    run(_run())


def test_peek_all_preserves_order():
    async def _run():
        ch = TaskChannel()
        for i in range(5):
            await ch.put({"seq": i})
        peeked = ch.peek_all()
        assert [m["seq"] for m in peeked] == list(range(5))
    run(_run())


def test_peek_all_empty_queue():
    ch = TaskChannel()
    assert ch.peek_all() == []


def test_channel_with_maxsize():
    async def _run():
        ch = TaskChannel(maxsize=2)
        await ch.put({"a": 1})
        await ch.put({"b": 2})
        assert ch.qsize() == 2
    run(_run())


def test_get_blocks_until_message():
    """get() must await a put() from a concurrent coroutine."""
    async def _run():
        ch = TaskChannel()

        async def producer():
            await asyncio.sleep(0.01)
            await ch.put({"hello": "world"})

        asyncio.create_task(producer())
        msg = await asyncio.wait_for(ch.get(), timeout=2.0)
        return msg

    assert run(_run()) == {"hello": "world"}


def test_multiple_puts_and_gets_fifo_order():
    """Messages should come out in FIFO order."""
    async def _run():
        ch = TaskChannel()
        for i in range(4):
            await ch.put({"n": i})
        results = [await ch.get() for _ in range(4)]
        return [r["n"] for r in results]

    assert run(_run()) == [0, 1, 2, 3]


# ─────────────────────────────────────────────────────────────────────────────
# AgentPool — registration
# ─────────────────────────────────────────────────────────────────────────────

def test_add_agent_and_agents_view():
    pool = AgentPool()
    mock_orch = MagicMock()
    pool.add_agent("alpha", mock_orch)
    agents = pool.agents()
    assert "alpha" in agents
    assert agents["alpha"] is mock_orch


def test_agents_returns_copy_not_reference():
    """Mutations of the returned dict should not affect the pool's internal dict."""
    pool = AgentPool()
    pool.add_agent("alpha", MagicMock())
    agents_copy = pool.agents()
    agents_copy["beta"] = MagicMock()
    assert "beta" not in pool.agents()


def test_multiple_agents_registered():
    pool = AgentPool()
    pool.add_agent("a", MagicMock())
    pool.add_agent("b", MagicMock())
    pool.add_agent("c", MagicMock())
    assert len(pool.agents()) == 3


# ─────────────────────────────────────────────────────────────────────────────
# AgentPool — run_parallel
# ─────────────────────────────────────────────────────────────────────────────

def test_run_parallel_all_succeed():
    async def _run():
        pool = AgentPool()
        state_a = _make_project_state()
        state_b = _make_project_state()

        mock_a = MagicMock()
        mock_a.run_job = AsyncMock(return_value=state_a)
        mock_b = MagicMock()
        mock_b.run_job = AsyncMock(return_value=state_b)

        pool.add_agent("a", mock_a)
        pool.add_agent("b", mock_b)

        spec = _make_spec()
        return await pool.run_parallel({"a": spec, "b": spec})

    results = run(_run())
    assert "a" in results
    assert "b" in results


def test_run_parallel_one_agent_fails_others_succeed():
    """An exception from one agent must not cancel the others."""
    async def _run():
        pool = AgentPool()
        state_ok = _make_project_state()

        mock_ok = MagicMock()
        mock_ok.run_job = AsyncMock(return_value=state_ok)

        mock_fail = MagicMock()
        mock_fail.run_job = AsyncMock(side_effect=RuntimeError("agent exploded"))

        pool.add_agent("ok", mock_ok)
        pool.add_agent("fail", mock_fail)

        spec = _make_spec()
        return await pool.run_parallel({"ok": spec, "fail": spec})

    results = run(_run())
    # Only the successful agent should be in the results
    assert "ok" in results
    assert "fail" not in results


def test_run_parallel_empty_assignments():
    async def _run():
        pool = AgentPool()
        return await pool.run_parallel({})

    assert run(_run()) == {}


def test_run_parallel_correct_spec_passed_to_each_agent():
    """Each agent's run_job must be called with the spec assigned to it."""
    async def _run():
        pool = AgentPool()
        spec_a = _make_spec()
        spec_b = _make_spec()

        mock_a = MagicMock()
        mock_a.run_job = AsyncMock(return_value=_make_project_state())
        mock_b = MagicMock()
        mock_b.run_job = AsyncMock(return_value=_make_project_state())

        pool.add_agent("a", mock_a)
        pool.add_agent("b", mock_b)

        await pool.run_parallel({"a": spec_a, "b": spec_b})
        mock_a.run_job.assert_called_once_with(spec_a)
        mock_b.run_job.assert_called_once_with(spec_b)

    run(_run())


# ─────────────────────────────────────────────────────────────────────────────
# AgentPool — best_result
# ─────────────────────────────────────────────────────────────────────────────

def test_best_result_picks_highest_mean_score():
    pool = AgentPool()
    state_low = _state_with_scores(0.5, 0.6)       # mean 0.55
    state_high = _state_with_scores(0.9, 0.95)     # mean 0.925
    results = {"low": state_low, "high": state_high}
    best = pool.best_result(results)
    assert best is state_high


def test_best_result_empty_results_returns_none():
    pool = AgentPool()
    assert pool.best_result({}) is None


def test_best_result_single_agent():
    pool = AgentPool()
    state = _state_with_scores(0.8)
    assert pool.best_result({"only": state}) is state


def test_best_result_skips_states_with_no_scored_results():
    pool = AgentPool()
    empty_state = ProjectState(project_description="test", success_criteria="pass", budget=Budget(max_usd=1.0), status=ProjectStatus.SUCCESS)
    good_state = _state_with_scores(0.75)
    best = pool.best_result({"empty": empty_state, "good": good_state})
    assert best is good_state


def test_best_result_all_empty_returns_none():
    pool = AgentPool()
    s1 = ProjectState(project_description="test", success_criteria="pass", budget=Budget(max_usd=1.0), status=ProjectStatus.SUCCESS)
    s2 = ProjectState(project_description="test2", success_criteria="pass2", budget=Budget(max_usd=1.0), status=ProjectStatus.SUCCESS)
    assert pool.best_result({"a": s1, "b": s2}) is None


# ─────────────────────────────────────────────────────────────────────────────
# AgentPool — merge_telemetry
# ─────────────────────────────────────────────────────────────────────────────

def test_merge_telemetry_empty_pool():
    pool = AgentPool()
    result = pool.merge_telemetry()
    assert result == {}


def test_merge_telemetry_sums_counters():
    pool = AgentPool()
    profiles_a = build_default_profiles()
    profiles_b = build_default_profiles()

    profiles_a[Model.KIMI_K2_5].call_count = 10
    profiles_b[Model.KIMI_K2_5].call_count = 5

    mock_a = MagicMock()
    mock_a._profiles = profiles_a
    mock_b = MagicMock()
    mock_b._profiles = profiles_b

    pool.add_agent("a", mock_a)
    pool.add_agent("b", mock_b)

    merged = pool.merge_telemetry()
    assert merged[Model.KIMI_K2_5].call_count == 15


def test_merge_telemetry_averages_ema_fields():
    pool = AgentPool()
    profiles_a = build_default_profiles()
    profiles_b = build_default_profiles()

    profiles_a[Model.KIMI_K2_5].quality_score = 0.8
    profiles_b[Model.KIMI_K2_5].quality_score = 0.6

    mock_a = MagicMock()
    mock_a._profiles = profiles_a
    mock_b = MagicMock()
    mock_b._profiles = profiles_b

    pool.add_agent("a", mock_a)
    pool.add_agent("b", mock_b)

    merged = pool.merge_telemetry()
    assert merged[Model.KIMI_K2_5].quality_score == pytest.approx(0.7)


def test_merge_telemetry_does_not_mutate_source_profiles():
    pool = AgentPool()
    profiles_a = build_default_profiles()
    original_call_count = profiles_a[Model.KIMI_K2_5].call_count

    mock_a = MagicMock()
    mock_a._profiles = profiles_a
    pool.add_agent("a", mock_a)

    pool.merge_telemetry()
    # Source profiles must not be mutated
    assert profiles_a[Model.KIMI_K2_5].call_count == original_call_count


def test_merge_telemetry_re_derives_success_rate():
    """success_rate in merged profile must be re-computed from merged counters."""
    pool = AgentPool()
    profiles_a = build_default_profiles()
    profiles_b = build_default_profiles()

    profiles_a[Model.KIMI_K2_5].call_count = 10
    profiles_a[Model.KIMI_K2_5].failure_count = 2   # 80% success
    profiles_b[Model.KIMI_K2_5].call_count = 10
    profiles_b[Model.KIMI_K2_5].failure_count = 0   # 100% success

    mock_a = MagicMock()
    mock_a._profiles = profiles_a
    mock_b = MagicMock()
    mock_b._profiles = profiles_b

    pool.add_agent("a", mock_a)
    pool.add_agent("b", mock_b)

    merged = pool.merge_telemetry()
    # 20 calls, 2 failures → 90% success
    assert merged[Model.KIMI_K2_5].success_rate == pytest.approx(0.9)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator.get_channel integration (engine wiring)
# ─────────────────────────────────────────────────────────────────────────────

def test_get_channel_returns_task_channel():
    from orchestrator.engine import Orchestrator
    orch = Orchestrator(budget=Budget(max_usd=1.0))
    ch = orch.get_channel("artifacts")
    assert isinstance(ch, TaskChannel)


def test_get_channel_lazy_creates_and_caches():
    from orchestrator.engine import Orchestrator
    orch = Orchestrator(budget=Budget(max_usd=1.0))
    ch1 = orch.get_channel("my_channel")
    ch2 = orch.get_channel("my_channel")
    assert ch1 is ch2


def test_get_channel_different_names_are_different():
    from orchestrator.engine import Orchestrator
    orch = Orchestrator(budget=Budget(max_usd=1.0))
    ch_a = orch.get_channel("channel_a")
    ch_b = orch.get_channel("channel_b")
    assert ch_a is not ch_b
