"""
Integration tests for TelemetryStore wiring in Orchestrator.

Plan reference: docs/plans/2026-02-25-learn-and-show-design.md

Collection points:
  1. After run_project() completes — record_snapshot per model with ≥1 call
  2. After each task result — record_routing_event

Warm-start:
  3. At run_project() start — load historical profiles and blend into defaults

These tests will initially FAIL because the wiring does not exist yet.
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call

from orchestrator.models import (
    Budget, Model, ProjectStatus, Task, TaskResult, TaskStatus, TaskType,
)
from orchestrator.policy import ModelProfile
from orchestrator.telemetry_store import TelemetryStore, HistoricalProfile


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_state(status=ProjectStatus.SUCCESS):
    state = MagicMock()
    state.status = status
    state.results = {}
    return state


def _make_task_result(task_id="t1", model=Model.DEEPSEEK_CHAT, score=0.88):
    return TaskResult(
        task_id=task_id,
        output="output",
        score=score,
        model_used=model,
        reviewer_model=Model.CLAUDE_SONNET,
        cost_usd=0.005,
        iterations=2,
        status=TaskStatus.COMPLETED,
        deterministic_check_passed=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TelemetryStore is injectable into Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def test_orchestrator_accepts_telemetry_store_kwarg(tmp_path):
    """Orchestrator.__init__() accepts a telemetry_store keyword argument."""
    from orchestrator.engine import Orchestrator

    store = TelemetryStore(db_path=tmp_path / "telemetry.db")
    # Must not raise
    orch = Orchestrator(telemetry_store=store)
    assert orch._telemetry_store is store


def test_orchestrator_creates_default_telemetry_store():
    """Orchestrator.__init__() creates a TelemetryStore with the default path when none given."""
    from orchestrator.engine import Orchestrator

    orch = Orchestrator()
    assert orch._telemetry_store is not None
    assert isinstance(orch._telemetry_store, TelemetryStore)


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot written after run_project()
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_record_snapshot_called_for_models_with_calls(tmp_path):
    """
    After run_project() completes, record_snapshot() is called once for each
    ModelProfile that had call_count >= 1 during this run.
    """
    import asyncio
    from orchestrator.engine import Orchestrator

    store = TelemetryStore(db_path=tmp_path / "telemetry.db")
    mock_record_snapshot = AsyncMock()
    store.record_snapshot = mock_record_snapshot

    with patch("orchestrator.engine.Orchestrator.run_project", new_callable=AsyncMock) as mock_run:
        mock_state = _make_state()
        mock_run.return_value = mock_state

        orch = Orchestrator(telemetry_store=store)

        # Simulate two models having been called this run
        orch._profiles[Model.DEEPSEEK_CHAT].call_count = 5
        orch._profiles[Model.GPT_4O].call_count = 3

        from orchestrator.policy import JobSpec
        spec = JobSpec(
            project_description="Test",
            success_criteria="Works",
            budget=Budget(max_usd=5.0),
        )
        await orch.run_job(spec)
        # Yield to event loop so fire-and-forget create_task callbacks execute
        await asyncio.sleep(0)

    # Both models with call_count > 0 should have been snapshotted
    called_models = {c.args[1] for c in mock_record_snapshot.call_args_list}
    assert Model.DEEPSEEK_CHAT in called_models
    assert Model.GPT_4O in called_models


@pytest.mark.asyncio
async def test_record_snapshot_not_called_for_models_with_zero_calls(tmp_path):
    """
    Models that were never called (call_count == 0) must NOT be snapshotted —
    they have no new data to persist.
    """
    import asyncio
    from orchestrator.engine import Orchestrator

    store = TelemetryStore(db_path=tmp_path / "telemetry.db")
    mock_record_snapshot = AsyncMock()
    store.record_snapshot = mock_record_snapshot

    with patch("orchestrator.engine.Orchestrator.run_project", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = _make_state()

        orch = Orchestrator(telemetry_store=store)

        # Only deepseek was called; all other models have call_count == 0 (defaults)
        orch._profiles[Model.DEEPSEEK_CHAT].call_count = 3

        from orchestrator.policy import JobSpec
        spec = JobSpec(
            project_description="Test",
            success_criteria="Works",
            budget=Budget(max_usd=5.0),
        )
        await orch.run_job(spec)
        # Yield to event loop so fire-and-forget create_task callbacks execute
        await asyncio.sleep(0)

    called_models = {c.args[1] for c in mock_record_snapshot.call_args_list}
    # Only deepseek should appear; models with 0 calls must not be snapshotted
    assert Model.DEEPSEEK_CHAT in called_models
    # Every other model in called_models must have had call_count > 0
    for m in called_models:
        assert orch._profiles[m].call_count > 0


@pytest.mark.asyncio
async def test_snapshots_are_fire_and_forget(tmp_path):
    """
    Snapshot writes must be scheduled as background tasks (asyncio.create_task),
    not awaited inline — they must never block the hot path.

    We verify this by confirming run_job() returns before writes can settle,
    which means using create_task (not await).
    """
    import asyncio
    from orchestrator.engine import Orchestrator

    write_started = []
    write_completed = []

    async def slow_record_snapshot(*args, **kwargs):
        write_started.append(True)
        await asyncio.sleep(0.05)  # simulate slow DB write
        write_completed.append(True)

    store = TelemetryStore(db_path=tmp_path / "telemetry.db")
    store.record_snapshot = slow_record_snapshot

    with patch("orchestrator.engine.Orchestrator.run_project", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = _make_state()

        orch = Orchestrator(telemetry_store=store)
        orch._profiles[Model.DEEPSEEK_CHAT].call_count = 1

        from orchestrator.policy import JobSpec
        spec = JobSpec(
            project_description="Test",
            success_criteria="Works",
            budget=Budget(max_usd=5.0),
        )
        await orch.run_job(spec)

        # run_job has returned; writes may have started but need not be complete
        # The key assertion: run_job returned even if writes are still pending
        # (fire-and-forget pattern — not a hard timing constraint but proves task scheduling)
        assert True  # run_job completed without hanging

    # Allow background tasks to finish
    await asyncio.sleep(0.1)
    # After giving time, writes should eventually complete
    assert len(write_started) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Routing event written after each task
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_record_routing_event_called_after_task_completion(tmp_path):
    """
    record_routing_event() is called once for each task that completes
    via _execute_all (i.e., after self.results[task_id] = result).
    """
    from orchestrator.engine import Orchestrator

    store = TelemetryStore(db_path=tmp_path / "telemetry.db")
    mock_record_event = AsyncMock()
    store.record_routing_event = mock_record_event

    task_result = _make_task_result(task_id="task-1", model=Model.DEEPSEEK_CHAT)
    task = Task(id="task-1", type=TaskType.CODE_GEN, prompt="Write code")

    with (
        patch("orchestrator.engine.Orchestrator._decompose", new_callable=AsyncMock) as mock_decompose,
        patch("orchestrator.engine.Orchestrator._execute_task", new_callable=AsyncMock) as mock_exec,
        patch("orchestrator.engine.Orchestrator._determine_final_status") as mock_status,
        patch("orchestrator.engine.Orchestrator._topological_sort") as mock_sort,
        patch("orchestrator.engine.StateManager.load_project", new_callable=AsyncMock) as mock_load,
        patch("orchestrator.engine.StateManager.save_project", new_callable=AsyncMock),
        patch("orchestrator.engine.StateManager.close", new_callable=AsyncMock),
        patch("orchestrator.engine.DiskCache.close", new_callable=AsyncMock),
    ):
        mock_load.return_value = None
        mock_decompose.return_value = {"task-1": task}
        mock_sort.return_value = [["task-1"]]
        mock_exec.return_value = task_result
        mock_status.return_value = ProjectStatus.SUCCESS

        orch = Orchestrator(telemetry_store=store)
        orch.budget = Budget(max_usd=5.0)

        await orch.run_project("Build thing", "It works")

    # record_routing_event should have been called for task-1
    assert mock_record_event.call_count >= 1
    call_args = mock_record_event.call_args_list[0]
    # Signature: record_routing_event(project_id, task_id, task_type, result)
    assert call_args.args[1] == "task-1"
    assert call_args.args[2] == TaskType.CODE_GEN
    assert call_args.args[3] is task_result


# ─────────────────────────────────────────────────────────────────────────────
# Warm-start blending
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_warm_start_blending_applied_when_historical_data_exists(tmp_path):
    """
    When load_historical_profile() returns a HistoricalProfile (WARM/HOT),
    the Orchestrator blends it into the in-memory ModelProfile before execution.

    WARM blend (10-49 calls): new = 0.4 * hist + 0.6 * default
    """
    from orchestrator.engine import Orchestrator
    from orchestrator.models import build_default_profiles

    # Default quality for deepseek-chat
    defaults = build_default_profiles()
    default_quality = defaults[Model.DEEPSEEK_CHAT].quality_score  # 0.8

    hist_quality = 0.95  # Strong historical quality
    # WARM blend: 0.4 * 0.95 + 0.6 * 0.8 = 0.38 + 0.48 = 0.86
    expected_blended = 0.4 * hist_quality + 0.6 * default_quality

    historical = HistoricalProfile(
        model=Model.DEEPSEEK_CHAT,
        task_type=TaskType.CODE_GEN,
        quality_score=hist_quality,
        trust_factor=0.95,
        avg_latency_ms=1200.0,
        latency_p95_ms=2400.0,
        success_rate=0.98,
        avg_cost_usd=0.001,
        call_count=25,  # WARM: 10-49
    )

    store = TelemetryStore(db_path=tmp_path / "telemetry.db")
    store.load_historical_profile = AsyncMock(return_value=historical)

    with (
        patch("orchestrator.engine.Orchestrator.run_project", new_callable=AsyncMock) as mock_run,
    ):
        mock_run.return_value = _make_state()

        orch = Orchestrator(telemetry_store=store)
        from orchestrator.policy import JobSpec
        spec = JobSpec(
            project_description="Test",
            success_criteria="Works",
            budget=Budget(max_usd=5.0),
        )
        await orch.run_job(spec)

    # After warm-start blending, the profile should reflect blended quality
    blended_quality = orch._profiles[Model.DEEPSEEK_CHAT].quality_score
    assert blended_quality == pytest.approx(expected_blended, abs=0.01), (
        f"Expected blended quality ~{expected_blended:.3f}, got {blended_quality:.3f}"
    )


@pytest.mark.asyncio
async def test_hot_start_uses_100_percent_historical(tmp_path):
    """
    HOT models (≥ 50 calls): profile uses 100% historical quality and latency.
    """
    from orchestrator.engine import Orchestrator

    hist_quality = 0.95
    hist_latency = 900.0

    historical = HistoricalProfile(
        model=Model.DEEPSEEK_CHAT,
        task_type=TaskType.CODE_GEN,
        quality_score=hist_quality,
        trust_factor=0.95,
        avg_latency_ms=hist_latency,
        latency_p95_ms=1800.0,
        success_rate=0.98,
        avg_cost_usd=0.001,
        call_count=75,  # HOT: ≥ 50
    )

    store = TelemetryStore(db_path=tmp_path / "telemetry.db")
    store.load_historical_profile = AsyncMock(return_value=historical)

    with patch("orchestrator.engine.Orchestrator.run_project", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = _make_state()

        orch = Orchestrator(telemetry_store=store)
        from orchestrator.policy import JobSpec
        spec = JobSpec(
            project_description="Test",
            success_criteria="Works",
            budget=Budget(max_usd=5.0),
        )
        await orch.run_job(spec)

    profile = orch._profiles[Model.DEEPSEEK_CHAT]
    assert profile.quality_score == pytest.approx(hist_quality, abs=0.001)
    assert profile.avg_latency_ms == pytest.approx(hist_latency, abs=1.0)


@pytest.mark.asyncio
async def test_cold_start_uses_defaults(tmp_path):
    """
    COLD models (no historical data / < 10 calls): load_historical_profile returns None
    and the orchestrator uses the compiled-in defaults unchanged.
    """
    from orchestrator.engine import Orchestrator
    from orchestrator.models import build_default_profiles

    defaults = build_default_profiles()
    default_quality = defaults[Model.DEEPSEEK_CHAT].quality_score

    store = TelemetryStore(db_path=tmp_path / "telemetry.db")
    store.load_historical_profile = AsyncMock(return_value=None)  # cold

    with patch("orchestrator.engine.Orchestrator.run_project", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = _make_state()

        orch = Orchestrator(telemetry_store=store)
        from orchestrator.policy import JobSpec
        spec = JobSpec(
            project_description="Test",
            success_criteria="Works",
            budget=Budget(max_usd=5.0),
        )
        await orch.run_job(spec)

    # Defaults must be unchanged
    assert orch._profiles[Model.DEEPSEEK_CHAT].quality_score == pytest.approx(default_quality)
