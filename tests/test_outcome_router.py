"""
Tests for Outcome-Weighted Router
"""

import pytest
from datetime import datetime

from orchestrator.outcome_router import (
    OutcomeWeightedRouter,
    ModelScore,
    RoutingContext,
    RoutingStrategy,
    get_outcome_router,
    reset_outcome_router,
    create_routing_context,
)
from orchestrator.feedback_loop import (
    FeedbackLoop,
    ProductionOutcome,
    CodebaseFingerprint,
    OutcomeStatus,
)
from orchestrator.models import Model, TaskType, Task


class TestModelScore:
    def test_composite_score_calculation(self):
        score = ModelScore(
            model=Model.GPT_4O,
            cost_score=0.8,
            quality_score=0.9,
            latency_score=0.7,
            reliability_score=0.95,
            production_score=0.85,
            sample_size=50,
        )

        composite = score.composite_score

        # Should be between 0 and 1
        assert 0.0 <= composite <= 1.0

        # With high scores, composite should be good
        assert composite > 0.5

    def test_composite_with_few_samples(self):
        score = ModelScore(
            model=Model.GPT_4O,
            production_score=0.9,
            sample_size=2,  # Few samples
        )

        # Lower confidence with few samples
        assert score.confidence < 1.0


class TestRoutingContext:
    def test_context_creation(self):
        task = Task(id="test", task_type=TaskType.CODE_GEN, prompt="test")

        context = RoutingContext(
            task=task,
            task_type=TaskType.CODE_GEN,
            budget_remaining=5.0,
            budget_total=10.0,
            strategy=RoutingStrategy.BALANCED,
        )

        assert context.budget_remaining == 5.0
        assert context.strategy == RoutingStrategy.BALANCED


class TestOutcomeWeightedRouter:
    def setup_method(self):
        reset_outcome_router()
        self.router = OutcomeWeightedRouter()

    @pytest.mark.asyncio
    async def test_select_model_basic(self):
        task = Task(id="test", task_type=TaskType.CODE_GEN, prompt="test")

        context = create_routing_context(
            task=task,
            budget_remaining=5.0,
            budget_total=10.0,
        )

        model, metadata = await self.router.select_model(context)

        assert model is not None
        assert isinstance(model, Model)
        assert "strategy" in metadata

    def test_score_model(self):
        task = Task(id="test", task_type=TaskType.CODE_GEN, prompt="test")
        context = create_routing_context(
            task=task,
            budget_remaining=5.0,
            budget_total=10.0,
        )

        score = self.router._score_model(Model.GPT_4O, context)

        assert score.model == Model.GPT_4O
        assert 0.0 <= score.composite_score <= 1.0

    def test_get_production_score_unknown(self):
        task = Task(id="test", task_type=TaskType.CODE_GEN, prompt="test")
        context = create_routing_context(
            task=task,
            budget_remaining=5.0,
            budget_total=10.0,
        )

        score, samples = self.router._get_production_score(Model.GPT_4O, context)

        # Unknown model should return default
        assert score == 0.5
        assert samples == 0

    @pytest.mark.asyncio
    async def test_select_model_with_production_data(self):
        # Add production data
        await self.router.feedback.record_outcome(
            ProductionOutcome(
                project_id="test",
                deployment_id="dep-1",
                task_type=TaskType.CODE_GEN,
                model_used=Model.GPT_4O,
                generated_code_hash="abc123",
                status=OutcomeStatus.SUCCESS,
            )
        )

        task = Task(id="test", task_type=TaskType.CODE_GEN, prompt="test")
        context = create_routing_context(
            task=task,
            budget_remaining=5.0,
            budget_total=10.0,
        )

        model, metadata = await self.router.select_model(context)

        # Should prefer model with success history
        assert metadata["production_score"] > 0.5
        assert metadata["sample_size"] >= 1

    def test_exploration_candidate_selection(self):
        # Create scores with varying sample sizes
        scores = {
            Model.GPT_4O: ModelScore(model=Model.GPT_4O, sample_size=100),
            Model.GPT_4O_MINI: ModelScore(model=Model.GPT_4O_MINI, sample_size=5),
        }

        context = RoutingContext(
            task=Task(id="test", task_type=TaskType.CODE_GEN, prompt="test"),
            task_type=TaskType.CODE_GEN,
            budget_remaining=5.0,
            budget_total=10.0,
        )

        # May return None or the under-sampled model
        candidate = self.router._select_exploration_candidate(scores, context)

        if candidate:
            assert candidate in scores

    def test_nash_stability_report(self):
        report = self.router.get_nash_stability_report()

        assert "total_production_samples" in report
        assert "unique_codebases_learned" in report
        assert "information_advantage" in report
        assert "exploration_status" in report


class TestCreateRoutingContext:
    def test_factory_function(self):
        task = Task(id="test", task_type=TaskType.CODE_GEN, prompt="test")

        context = create_routing_context(
            task=task,
            budget_remaining=3.0,
            budget_total=10.0,
            strategy=RoutingStrategy.QUALITY_OPTIMIZED,
        )

        assert context.task == task
        assert context.budget_remaining == 3.0
        assert context.strategy == RoutingStrategy.QUALITY_OPTIMIZED


class TestGlobalOutcomeRouter:
    def setup_method(self):
        reset_outcome_router()

    def test_get_outcome_router_singleton(self):
        router1 = get_outcome_router()
        router2 = get_outcome_router()

        assert router1 is router2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
