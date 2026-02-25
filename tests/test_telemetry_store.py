"""
Tests for TelemetryStore — persistent cross-run learning.

Plan reference: docs/plans/2026-02-25-learn-and-show-design.md

All tests use a tmp_path fixture to avoid touching ~/.orchestrator_cache/.
These tests will initially FAIL (ImportError) because telemetry_store.py
does not exist yet — proving RED phase before implementation.
"""
from __future__ import annotations

import pytest

from orchestrator.models import Model, TaskType, TaskResult, TaskStatus
from orchestrator.policy import ModelProfile
from orchestrator.telemetry_store import (
    TelemetryStore,
    HistoricalProfile,
    ModelRanking,
    Recommendation,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_profile(
    *,
    model: Model = Model.DEEPSEEK_CHAT,
    quality_score: float = 0.85,
    trust_factor: float = 0.90,
    avg_latency_ms: float = 1500.0,
    latency_p95_ms: float = 3000.0,
    success_rate: float = 0.95,
    avg_cost_usd: float = 0.002,
    call_count: int = 1,
    failure_count: int = 0,
    validator_fail_count: int = 0,
) -> ModelProfile:
    return ModelProfile(
        model=model,
        provider="deepseek",
        cost_per_1m_input=0.27,
        cost_per_1m_output=1.10,
        quality_score=quality_score,
        trust_factor=trust_factor,
        avg_latency_ms=avg_latency_ms,
        latency_p95_ms=latency_p95_ms,
        success_rate=success_rate,
        avg_cost_usd=avg_cost_usd,
        call_count=call_count,
        failure_count=failure_count,
        validator_fail_count=validator_fail_count,
    )


def _make_result(
    *,
    task_id: str = "task-1",
    score: float = 0.87,
    model: Model = Model.DEEPSEEK_CHAT,
    reviewer: Model | None = Model.CLAUDE_SONNET,
    cost_usd: float = 0.003,
    iterations: int = 2,
    det_passed: bool = True,
) -> TaskResult:
    return TaskResult(
        task_id=task_id,
        output="def foo(): pass",
        score=score,
        model_used=model,
        reviewer_model=reviewer,
        cost_usd=cost_usd,
        iterations=iterations,
        status=TaskStatus.COMPLETED,
        deterministic_check_passed=det_passed,
    )


# ─────────────────────────────────────────────────────────────────────────────
# record_snapshot
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_record_snapshot_inserts_row(tmp_path):
    """record_snapshot() inserts exactly one row into model_snapshots."""
    store = TelemetryStore(db_path=tmp_path / "telemetry.db")
    await store.record_snapshot("proj-1", Model.DEEPSEEK_CHAT, TaskType.CODE_GEN,
                                _make_profile(call_count=5))

    import aiosqlite
    async with aiosqlite.connect(tmp_path / "telemetry.db") as db:
        async with db.execute("SELECT COUNT(*) FROM model_snapshots") as cur:
            row = await cur.fetchone()
    assert row[0] == 1


@pytest.mark.asyncio
async def test_record_snapshot_stores_correct_fields(tmp_path):
    """record_snapshot() persists model, task_type, quality_score, call_count, cost."""
    store = TelemetryStore(db_path=tmp_path / "telemetry.db")
    await store.record_snapshot("proj-2", Model.DEEPSEEK_CHAT, TaskType.CODE_GEN,
                                _make_profile(quality_score=0.92, call_count=12, avg_cost_usd=0.005))

    import aiosqlite
    async with aiosqlite.connect(tmp_path / "telemetry.db") as db:
        async with db.execute(
            "SELECT model, task_type, quality_score, call_count, avg_cost_usd FROM model_snapshots"
        ) as cur:
            row = await cur.fetchone()

    assert row[0] == Model.DEEPSEEK_CHAT.value
    assert row[1] == TaskType.CODE_GEN.value
    assert row[2] == pytest.approx(0.92)
    assert row[3] == 12
    assert row[4] == pytest.approx(0.005)


# ─────────────────────────────────────────────────────────────────────────────
# record_routing_event
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_record_routing_event_inserts_row(tmp_path):
    """record_routing_event() inserts one row into routing_events."""
    store = TelemetryStore(db_path=tmp_path / "telemetry.db")
    await store.record_routing_event("proj-1", "task-1", TaskType.CODE_GEN, _make_result())

    import aiosqlite
    async with aiosqlite.connect(tmp_path / "telemetry.db") as db:
        async with db.execute("SELECT COUNT(*) FROM routing_events") as cur:
            row = await cur.fetchone()
    assert row[0] == 1


@pytest.mark.asyncio
async def test_record_routing_event_stores_correct_fields(tmp_path):
    """record_routing_event() persists model, task_type, score, det_passed."""
    store = TelemetryStore(db_path=tmp_path / "telemetry.db")
    await store.record_routing_event("proj-1", "task-1", TaskType.CODE_GEN,
                                     _make_result(score=0.91, det_passed=True))

    import aiosqlite
    async with aiosqlite.connect(tmp_path / "telemetry.db") as db:
        async with db.execute(
            "SELECT model_chosen, task_type, score, det_passed FROM routing_events"
        ) as cur:
            row = await cur.fetchone()

    assert row[0] == Model.DEEPSEEK_CHAT.value
    assert row[1] == TaskType.CODE_GEN.value
    assert row[2] == pytest.approx(0.91)
    assert row[3] == 1  # True stored as integer 1


# ─────────────────────────────────────────────────────────────────────────────
# load_historical_profile — cold / warm / hot thresholds
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_load_historical_profile_returns_none_when_no_data(tmp_path):
    """load_historical_profile() returns None with an empty DB (cold start)."""
    store = TelemetryStore(db_path=tmp_path / "telemetry.db")
    result = await store.load_historical_profile(Model.DEEPSEEK_CHAT, TaskType.CODE_GEN)
    assert result is None


@pytest.mark.asyncio
async def test_load_historical_profile_returns_none_below_10_calls(tmp_path):
    """Cold start: < 10 total calls → None (orchestrator uses defaults only)."""
    store = TelemetryStore(db_path=tmp_path / "telemetry.db")

    for i in range(9):
        await store.record_snapshot(f"proj-{i}", Model.DEEPSEEK_CHAT, TaskType.CODE_GEN,
                                    _make_profile(call_count=1))

    result = await store.load_historical_profile(Model.DEEPSEEK_CHAT, TaskType.CODE_GEN)
    assert result is None


@pytest.mark.asyncio
async def test_load_historical_profile_returns_data_at_10_calls(tmp_path):
    """Warm start: exactly 10 calls → HistoricalProfile returned (40% blend threshold)."""
    store = TelemetryStore(db_path=tmp_path / "telemetry.db")

    for i in range(10):
        await store.record_snapshot(f"proj-{i}", Model.DEEPSEEK_CHAT, TaskType.CODE_GEN,
                                    _make_profile(quality_score=0.88, call_count=1))

    result = await store.load_historical_profile(Model.DEEPSEEK_CHAT, TaskType.CODE_GEN)
    assert result is not None
    assert isinstance(result, HistoricalProfile)
    assert result.model == Model.DEEPSEEK_CHAT
    assert result.task_type == TaskType.CODE_GEN
    assert result.call_count == 10
    assert result.quality_score == pytest.approx(0.88, abs=0.01)


@pytest.mark.asyncio
async def test_load_historical_profile_aggregates_across_projects(tmp_path):
    """load_historical_profile() averages quality across all projects for the model+task."""
    store = TelemetryStore(db_path=tmp_path / "telemetry.db")

    # 5 snapshots at 0.80, 5 at 0.90 → mean = 0.85
    for i in range(5):
        await store.record_snapshot(f"pa-{i}", Model.DEEPSEEK_CHAT, TaskType.CODE_GEN,
                                    _make_profile(quality_score=0.80, call_count=1))
    for i in range(5):
        await store.record_snapshot(f"pb-{i}", Model.DEEPSEEK_CHAT, TaskType.CODE_GEN,
                                    _make_profile(quality_score=0.90, call_count=1))

    result = await store.load_historical_profile(Model.DEEPSEEK_CHAT, TaskType.CODE_GEN)
    assert result is not None
    assert result.quality_score == pytest.approx(0.85, abs=0.01)


@pytest.mark.asyncio
async def test_load_historical_profile_separates_by_task_type(tmp_path):
    """load_historical_profile() isolates by (model, task_type) — no cross-contamination."""
    store = TelemetryStore(db_path=tmp_path / "telemetry.db")

    for i in range(10):
        await store.record_snapshot(f"cg-{i}", Model.DEEPSEEK_CHAT, TaskType.CODE_GEN,
                                    _make_profile(quality_score=0.90, call_count=1))
    for i in range(10):
        await store.record_snapshot(f"sm-{i}", Model.DEEPSEEK_CHAT, TaskType.SUMMARIZE,
                                    _make_profile(quality_score=0.70, call_count=1))

    code_gen = await store.load_historical_profile(Model.DEEPSEEK_CHAT, TaskType.CODE_GEN)
    summarize = await store.load_historical_profile(Model.DEEPSEEK_CHAT, TaskType.SUMMARIZE)

    assert code_gen.quality_score == pytest.approx(0.90, abs=0.01)
    assert summarize.quality_score == pytest.approx(0.70, abs=0.01)


# ─────────────────────────────────────────────────────────────────────────────
# model_rankings
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_model_rankings_returns_empty_when_no_data(tmp_path):
    """model_rankings() returns [] when there are no snapshots."""
    store = TelemetryStore(db_path=tmp_path / "telemetry.db")
    rankings = await store.model_rankings(days=30)
    assert rankings == []


@pytest.mark.asyncio
async def test_model_rankings_sorted_by_value_score(tmp_path):
    """model_rankings() sorts by quality/cost descending (higher value = better)."""
    store = TelemetryStore(db_path=tmp_path / "telemetry.db")

    # deepseek: quality 0.90, cost $0.001 → value_score ≈ 900
    for i in range(10):
        await store.record_snapshot(f"d{i}", Model.DEEPSEEK_CHAT, TaskType.CODE_GEN,
                                    _make_profile(quality_score=0.90, avg_cost_usd=0.001, call_count=1))

    # gpt-4o: quality 0.85, cost $0.010 → value_score ≈ 85
    for i in range(10):
        await store.record_snapshot(f"g{i}", Model.GPT_4O, TaskType.CODE_GEN,
                                    _make_profile(model=Model.GPT_4O, quality_score=0.85,
                                                  avg_cost_usd=0.010, call_count=1))

    rankings = await store.model_rankings(days=30)
    assert len(rankings) == 2
    assert rankings[0].model == Model.DEEPSEEK_CHAT
    assert rankings[1].model == Model.GPT_4O
    assert rankings[0].value_score > rankings[1].value_score


@pytest.mark.asyncio
async def test_model_rankings_confidence_levels(tmp_path):
    """model_rankings() assigns HOT/WARM/COLD based on total call count."""
    store = TelemetryStore(db_path=tmp_path / "telemetry.db")

    # HOT: 50 calls
    for i in range(50):
        await store.record_snapshot(f"h{i}", Model.DEEPSEEK_CHAT, TaskType.CODE_GEN,
                                    _make_profile(call_count=1))

    # WARM: 10 calls
    for i in range(10):
        await store.record_snapshot(f"w{i}", Model.GPT_4O, TaskType.CODE_GEN,
                                    _make_profile(model=Model.GPT_4O, quality_score=0.85,
                                                  avg_cost_usd=0.010, call_count=1))

    # COLD: 5 calls
    for i in range(5):
        await store.record_snapshot(f"c{i}", Model.MINIMAX_3, TaskType.CODE_GEN,
                                    _make_profile(model=Model.MINIMAX_3, quality_score=0.91,
                                                  avg_cost_usd=0.030, call_count=1))

    rankings = await store.model_rankings(days=30)
    by_model = {r.model: r for r in rankings}

    assert by_model[Model.DEEPSEEK_CHAT].confidence == "HOT"
    assert by_model[Model.GPT_4O].confidence == "WARM"
    assert by_model[Model.MINIMAX_3].confidence == "COLD"


@pytest.mark.asyncio
async def test_model_rankings_returns_modelranking_instances(tmp_path):
    """model_rankings() returns a list of ModelRanking dataclass instances."""
    store = TelemetryStore(db_path=tmp_path / "telemetry.db")

    for i in range(10):
        await store.record_snapshot(f"p{i}", Model.DEEPSEEK_CHAT, TaskType.CODE_GEN,
                                    _make_profile(call_count=1))

    rankings = await store.model_rankings(days=30)
    assert len(rankings) >= 1
    r = rankings[0]
    assert isinstance(r, ModelRanking)
    assert hasattr(r, "model")
    assert hasattr(r, "value_score")
    assert hasattr(r, "call_count")
    assert hasattr(r, "quality_score")
    assert hasattr(r, "confidence")


# ─────────────────────────────────────────────────────────────────────────────
# task_type_leaders
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_task_type_leaders_returns_empty_when_no_data(tmp_path):
    """task_type_leaders() returns {} with an empty DB."""
    store = TelemetryStore(db_path=tmp_path / "telemetry.db")
    leaders = await store.task_type_leaders(days=30)
    assert leaders == {}


@pytest.mark.asyncio
async def test_task_type_leaders_returns_best_per_type(tmp_path):
    """task_type_leaders() returns the highest value_score model for each task type."""
    store = TelemetryStore(db_path=tmp_path / "telemetry.db")

    # CODE_GEN: deepseek wins (0.92 quality, cheap)
    for i in range(10):
        await store.record_snapshot(f"cd{i}", Model.DEEPSEEK_CHAT, TaskType.CODE_GEN,
                                    _make_profile(quality_score=0.92, avg_cost_usd=0.001, call_count=1))
    for i in range(10):
        await store.record_snapshot(f"cg{i}", Model.GPT_4O, TaskType.CODE_GEN,
                                    _make_profile(model=Model.GPT_4O, quality_score=0.80,
                                                  avg_cost_usd=0.010, call_count=1))

    # SUMMARIZE: gemini wins
    for i in range(10):
        await store.record_snapshot(f"sm{i}", Model.GEMINI_FLASH, TaskType.SUMMARIZE,
                                    _make_profile(model=Model.GEMINI_FLASH, quality_score=0.88,
                                                  avg_cost_usd=0.0005, call_count=1))

    leaders = await store.task_type_leaders(days=30)

    assert TaskType.CODE_GEN in leaders
    assert leaders[TaskType.CODE_GEN].model == Model.DEEPSEEK_CHAT
    assert TaskType.SUMMARIZE in leaders
    assert leaders[TaskType.SUMMARIZE].model == Model.GEMINI_FLASH


# ─────────────────────────────────────────────────────────────────────────────
# recommendations
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_recommendations_returns_list(tmp_path):
    """recommendations() always returns a list (may be empty)."""
    store = TelemetryStore(db_path=tmp_path / "telemetry.db")
    recs = await store.recommendations(days=30)
    assert isinstance(recs, list)


@pytest.mark.asyncio
async def test_recommendations_identifies_cheaper_equivalent(tmp_path):
    """
    When a cheaper model has equivalent quality, recommendations() flags it.
    deepseek (0.90 quality, $0.001) vs gpt-4o (0.90 quality, $0.010):
    should recommend switching gpt-4o traffic to deepseek.
    """
    store = TelemetryStore(db_path=tmp_path / "telemetry.db")

    # Both at HOT confidence (≥50 calls) so recommendation fires
    for i in range(50):
        await store.record_snapshot(f"hd{i}", Model.DEEPSEEK_CHAT, TaskType.CODE_GEN,
                                    _make_profile(quality_score=0.90, avg_cost_usd=0.001, call_count=1))
    for i in range(50):
        await store.record_snapshot(f"hg{i}", Model.GPT_4O, TaskType.CODE_GEN,
                                    _make_profile(model=Model.GPT_4O, quality_score=0.90,
                                                  avg_cost_usd=0.010, call_count=1))

    recs = await store.recommendations(days=30)
    messages = " ".join(r.message.lower() for r in recs)
    # Should mention the expensive model or the cheaper alternative
    assert "gpt-4o" in messages or "deepseek" in messages


@pytest.mark.asyncio
async def test_recommendations_returns_recommendation_instances(tmp_path):
    """recommendations() returns Recommendation dataclass instances with a message field."""
    store = TelemetryStore(db_path=tmp_path / "telemetry.db")

    for i in range(50):
        await store.record_snapshot(f"d{i}", Model.DEEPSEEK_CHAT, TaskType.CODE_GEN,
                                    _make_profile(quality_score=0.90, avg_cost_usd=0.001, call_count=1))
    for i in range(50):
        await store.record_snapshot(f"g{i}", Model.GPT_4O, TaskType.CODE_GEN,
                                    _make_profile(model=Model.GPT_4O, quality_score=0.90,
                                                  avg_cost_usd=0.010, call_count=1))

    recs = await store.recommendations(days=30)
    for r in recs:
        assert isinstance(r, Recommendation)
        assert isinstance(r.message, str)
        assert len(r.message) > 0
