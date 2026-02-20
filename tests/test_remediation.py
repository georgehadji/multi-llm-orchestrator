# tests/test_remediation.py
from orchestrator.remediation import (
    RemediationStrategy, RemediationPlan, RemediationEngine,
)
from orchestrator.models import TaskResult, Model, TaskStatus


def _failed_result(task_id="t1"):
    return TaskResult(
        task_id=task_id, output="", score=0.0,
        model_used=Model.DEEPSEEK_CHAT,
        status=TaskStatus.FAILED,
    )


def _low_score_result(task_id="t1", score=0.55):
    return TaskResult(
        task_id=task_id, output="some output", score=score,
        model_used=Model.DEEPSEEK_CHAT,
        status=TaskStatus.DEGRADED,
    )


def test_plan_with_single_strategy():
    plan = RemediationPlan([RemediationStrategy.AUTO_RETRY])
    assert plan.next_strategy() == RemediationStrategy.AUTO_RETRY
    plan.advance()
    assert plan.next_strategy() is None
    assert plan.exhausted()


def test_plan_with_multiple_strategies():
    plan = RemediationPlan([
        RemediationStrategy.AUTO_RETRY,
        RemediationStrategy.FALLBACK_MODEL,
        RemediationStrategy.DEGRADE_QUALITY,
    ])
    assert plan.next_strategy() == RemediationStrategy.AUTO_RETRY
    plan.advance()
    assert plan.next_strategy() == RemediationStrategy.FALLBACK_MODEL
    plan.advance()
    assert plan.next_strategy() == RemediationStrategy.DEGRADE_QUALITY
    plan.advance()
    assert plan.exhausted()


def test_should_remediate_on_failure():
    engine = RemediationEngine()
    assert engine.should_remediate(_failed_result(), threshold=0.85) is True


def test_should_remediate_on_low_score():
    engine = RemediationEngine()
    assert engine.should_remediate(_low_score_result(score=0.60), threshold=0.85) is True


def test_no_remediation_on_success():
    engine = RemediationEngine()
    result = TaskResult(
        task_id="t1", output="good", score=0.90,
        model_used=Model.DEEPSEEK_CHAT,
        status=TaskStatus.COMPLETED,
    )
    assert engine.should_remediate(result, threshold=0.85) is False


def test_degrade_quality_lowers_threshold():
    engine = RemediationEngine()
    adjusted = engine.adjusted_threshold(RemediationStrategy.DEGRADE_QUALITY, 0.85)
    assert adjusted < 0.85
    assert adjusted >= 0.0


def test_default_plan_not_exhausted():
    engine = RemediationEngine()
    plan = engine.default_plan()
    assert not plan.exhausted()
    assert plan.next_strategy() in list(RemediationStrategy)


def test_plan_from_list():
    engine = RemediationEngine()
    plan = engine.plan_from_list(["auto_retry", "fallback_model"])
    assert plan.next_strategy() == RemediationStrategy.AUTO_RETRY
    plan.advance()
    assert plan.next_strategy() == RemediationStrategy.FALLBACK_MODEL
