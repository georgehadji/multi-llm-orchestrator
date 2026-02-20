from orchestrator.aggregator import ProfileAggregator, RunRecord
from orchestrator.models import Model, TaskType


def _make_run(model, task_type, score, cost, latency):
    return RunRecord(
        project_id="proj-1",
        task_type=task_type,
        model=model,
        score=score,
        cost_usd=cost,
        latency_ms=latency,
    )


def test_best_model_for_task_type():
    agg = ProfileAggregator()
    agg.record(_make_run(Model.DEEPSEEK_CHAT, TaskType.CODE_GEN, 0.92, 0.001, 800))
    agg.record(_make_run(Model.KIMI_K2_5,     TaskType.CODE_GEN, 0.88, 0.002, 1200))
    agg.record(_make_run(Model.DEEPSEEK_CHAT, TaskType.CODE_GEN, 0.90, 0.001, 850))
    best = agg.best_model(TaskType.CODE_GEN)
    assert best == Model.DEEPSEEK_CHAT   # higher avg score


def test_cost_efficiency_ranking():
    agg = ProfileAggregator()
    agg.record(_make_run(Model.DEEPSEEK_CHAT, TaskType.CODE_GEN, 0.90, 0.001, 800))
    agg.record(_make_run(Model.KIMI_K2_5,     TaskType.CODE_GEN, 0.90, 0.002, 800))
    ranking = agg.cost_efficiency_ranking(TaskType.CODE_GEN)
    # Same score, deepseek cheaper → deepseek ranks first
    assert ranking[0][0] == Model.DEEPSEEK_CHAT


def test_summary_table_includes_all_recorded_types():
    agg = ProfileAggregator()
    agg.record(_make_run(Model.DEEPSEEK_CHAT, TaskType.CODE_GEN,    0.9, 0.001, 800))
    agg.record(_make_run(Model.KIMI_K2_5,     TaskType.CODE_REVIEW, 0.8, 0.002, 900))
    table = agg.summary_table()
    assert TaskType.CODE_GEN    in table
    assert TaskType.CODE_REVIEW in table


def test_empty_aggregator_returns_none():
    agg = ProfileAggregator()
    assert agg.best_model(TaskType.CODE_GEN) is None


def test_stats_for_multiple_runs():
    agg = ProfileAggregator()
    for i in range(5):
        agg.record(_make_run(Model.DEEPSEEK_CHAT, TaskType.WRITING,
                             0.80 + i * 0.02, 0.001, 1000))
    stats = agg.stats_for(Model.DEEPSEEK_CHAT, TaskType.WRITING)
    assert stats["count"] == 5
    assert abs(stats["avg_score"] - 0.84) < 0.01


def test_stats_for_empty_returns_zeros():
    agg = ProfileAggregator()
    stats = agg.stats_for(Model.DEEPSEEK_CHAT, TaskType.CODE_GEN)
    assert stats["count"] == 0
    assert stats["avg_score"] == 0.0
