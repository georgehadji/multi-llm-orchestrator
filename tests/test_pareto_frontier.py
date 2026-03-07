"""
Tests for Predictive Cost-Quality Frontier API.
"""

import pytest
import asyncio
from pathlib import Path
import tempfile
import shutil

from orchestrator.pareto_frontier import (
    CostQualityFrontier,
    Objective,
    ModelPrediction,
    ParetoPoint,
)
from orchestrator.models import Model, TaskType
from orchestrator.feedback_loop import CodebaseFingerprint


class TestCostQualityFrontier:
    """Test suite for cost-quality frontier."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp = tempfile.mkdtemp()
        yield Path(temp)
        shutil.rmtree(temp)
    
    @pytest.fixture
    def frontier(self, temp_dir):
        """Create frontier instance."""
        return CostQualityFrontier(storage_path=temp_dir)
    
    def test_initialization(self, frontier):
        """Test frontier initialization."""
        assert frontier is not None
        assert frontier.storage_path.exists()
    
    @pytest.mark.asyncio
    async def test_predict_model(self, frontier):
        """Test single model prediction."""
        pred = await frontier._predict_model(
            model=Model.DEEPSEEK_CHAT,
            task_type=TaskType.CODE_GEN,
            fingerprint=None,
            objectives=[Objective.QUALITY, Objective.COST],
        )
        
        assert pred is not None
        assert isinstance(pred, ModelPrediction)
        assert 0 <= pred.quality <= 1
        assert pred.cost >= 0
        assert pred.confidence >= 0
    
    @pytest.mark.asyncio
    async def test_get_pareto_frontier(self, frontier):
        """Test getting Pareto frontier."""
        recommendations = await frontier.get_pareto_frontier(
            task_type=TaskType.CODE_GEN,
            objectives=[Objective.QUALITY, Objective.COST],
        )
        
        assert isinstance(recommendations, list)
        # Should have recommendations for multiple models
        assert len(recommendations) > 0
    
    @pytest.mark.asyncio
    async def test_pareto_optimality(self, frontier):
        """Test that frontier contains Pareto-optimal points."""
        recommendations = await frontier.get_pareto_frontier(
            task_type=TaskType.CODE_GEN,
            objectives=[Objective.QUALITY, Objective.COST],
        )
        
        if recommendations:
            # At least one should be Pareto-optimal
            pareto_optimal = [r for r in recommendations if r.is_pareto_optimal]
            assert len(pareto_optimal) > 0
    
    @pytest.mark.asyncio
    async def test_budget_constraint(self, frontier):
        """Test budget constraint filtering."""
        budget = 0.01  # $0.01 per 1K tokens
        
        recommendations = await frontier.get_pareto_frontier(
            task_type=TaskType.CODE_GEN,
            objectives=[Objective.QUALITY, Objective.COST],
            budget_constraint=budget,
        )
        
        # All recommendations should be within budget
        for rec in recommendations:
            assert rec.cost <= budget
    
    @pytest.mark.asyncio
    async def test_confidence_intervals(self, frontier):
        """Test confidence interval calculation."""
        pred = await frontier._predict_model(
            model=Model.DEEPSEEK_CHAT,
            task_type=TaskType.CODE_GEN,
            fingerprint=None,
            objectives=[Objective.QUALITY],
        )
        
        # Should have confidence interval
        assert pred.quality_ci[0] <= pred.quality <= pred.quality_ci[1]
        # Lower bound should be less than upper bound
        assert pred.quality_ci[0] < pred.quality_ci[1]
    
    @pytest.mark.asyncio
    async def test_compare_models(self, frontier):
        """Test model comparison."""
        comparison = frontier.compare_models(
            model_a=Model.DEEPSEEK_CHAT,
            model_b=Model.GPT_4O,
            task_type=TaskType.CODE_GEN,
        )
        
        assert "model_a" in comparison
        assert "model_b" in comparison
        assert "differences" in comparison
        assert "winners" in comparison
    
    def test_dominance_checking(self, frontier):
        """Test Pareto dominance checking."""
        # Create test points
        point_a = ParetoPoint(
            prediction=None,
            objective_values={Objective.QUALITY: 0.8, Objective.COST: 0.2},
        )
        point_b = ParetoPoint(
            prediction=None,
            objective_values={Objective.QUALITY: 0.6, Objective.COST: 0.4},
        )
        point_c = ParetoPoint(
            prediction=None,
            objective_values={Objective.QUALITY: 0.8, Objective.COST: 0.4},
        )
        
        # A dominates B (better on both)
        assert frontier._dominates(point_a, point_b, [Objective.QUALITY, Objective.COST])
        
        # A does not dominate C (equal quality, better cost)
        assert not frontier._dominates(point_a, point_c, [Objective.QUALITY, Objective.COST])
        
        # C does not dominate A (better cost, equal quality)
        assert not frontier._dominates(point_c, point_a, [Objective.QUALITY, Objective.COST])
    
    def test_normalization(self, frontier):
        """Test objective normalization."""
        predictions = [
            ModelPrediction(
                model=Model.DEEPSEEK_CHAT,
                quality=0.9,
                cost=0.001,
                latency_ms=1000,
                reliability=0.95,
                efficiency=900,
            ),
            ModelPrediction(
                model=Model.GPT_4O,
                quality=0.95,
                cost=0.01,
                latency_ms=2000,
                reliability=0.98,
                efficiency=95,
            ),
        ]
        
        normalized = frontier._normalize_objectives(
            predictions,
            [Objective.QUALITY, Objective.COST],
        )
        
        assert len(normalized) == 2
        for pred, obj_vals in normalized:
            for obj in [Objective.QUALITY, Objective.COST]:
                assert 0 <= obj_vals[obj] <= 1
    
    def test_frontier_stats(self, frontier):
        """Test getting frontier statistics."""
        stats = frontier.get_frontier_stats()
        
        assert "cache_size" in stats
        assert "history_entries" in stats
    
    def test_model_prediction_to_dict(self):
        """Test ModelPrediction serialization."""
        pred = ModelPrediction(
            model=Model.DEEPSEEK_CHAT,
            quality=0.85,
            cost=0.002,
            latency_ms=1200,
            reliability=0.9,
            efficiency=425,
            confidence=0.8,
            uncertainty=0.1,
            sample_size=10,
        )
        
        data = pred.to_dict()
        assert data["model"] == "deepseek-chat"
        assert "quality" in data
        assert "cost" in data
        assert "confidence" in data


class TestObjective:
    """Test Objective enum."""
    
    def test_objective_directions(self):
        """Test that objectives have correct directions."""
        from orchestrator.pareto_frontier import OBJECTIVE_DIRECTIONS, OptimizationDirection
        
        # Quality should be maximized
        assert OBJECTIVE_DIRECTIONS[Objective.QUALITY] == OptimizationDirection.MAXIMIZE
        
        # Cost should be minimized
        assert OBJECTIVE_DIRECTIONS[Objective.COST] == OptimizationDirection.MINIMIZE
        
        # Latency should be minimized
        assert OBJECTIVE_DIRECTIONS[Objective.LATENCY] == OptimizationDirection.MINIMIZE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
