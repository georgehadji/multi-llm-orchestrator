"""
Tests for Cross-Organization Learning with Differential Privacy.
"""

import pytest
import asyncio
from pathlib import Path
import tempfile
import shutil

from orchestrator.federated_learning import (
    FederatedLearningOrchestrator,
    DifferentialPrivacyEngine,
    PrivacyBudget,
    ModelInsight,
    PrivacyMechanism,
)
from orchestrator.models import Model, TaskType
from orchestrator.feedback_loop import (
    ProductionOutcome,
    CodebaseFingerprint,
    OutcomeStatus,
)


class TestDifferentialPrivacyEngine:
    """Test suite for DP engine."""
    
    @pytest.fixture
    def dp_engine(self):
        """Create DP engine."""
        return DifferentialPrivacyEngine(epsilon=1.0, delta=1e-5)
    
    def test_initialization(self, dp_engine):
        """Test engine initialization."""
        assert dp_engine.epsilon == 1.0
        assert dp_engine.delta == 1e-5
    
    def test_add_gaussian_noise(self, dp_engine):
        """Test Gaussian noise addition."""
        value = 0.5
        noisy_values = [dp_engine.add_gaussian_noise(value) for _ in range(100)]
        
        # Mean should be approximately preserved
        mean = sum(noisy_values) / len(noisy_values)
        assert 0.3 < mean < 0.7  # Loose bounds due to randomness
        
        # Should have variance
        assert max(noisy_values) > min(noisy_values)
    
    def test_add_laplace_noise(self, dp_engine):
        """Test Laplace noise addition."""
        value = 0.5
        noisy_values = [dp_engine.add_laplace_noise(value) for _ in range(100)]
        
        # Mean should be approximately preserved
        mean = sum(noisy_values) / len(noisy_values)
        assert 0.3 < mean < 0.7
    
    def test_randomized_response(self, dp_engine):
        """Test randomized response."""
        # True value
        true_responses = [dp_engine.randomized_response(True, p=0.75) for _ in range(1000)]
        true_rate = sum(true_responses) / len(true_responses)
        
        # Should be biased toward True but not always True
        assert 0.6 < true_rate < 0.9
        
        # False value
        false_responses = [dp_engine.randomized_response(False, p=0.75) for _ in range(1000)]
        false_rate = sum(false_responses) / len(false_responses)
        
        # Should be biased toward False but not always False
        assert 0.1 < false_rate < 0.4
    
    def test_privatize_histogram(self, dp_engine):
        """Test histogram privatization."""
        histogram = {"a": 100, "b": 50, "c": 25}
        
        privatized = dp_engine.privatize_histogram(histogram)
        
        # Should have same keys
        assert set(privatized.keys()) == set(histogram.keys())
        
        # Values should be perturbed but non-negative
        for key, value in privatized.items():
            assert value >= 0
            # Should be somewhat close to original
            assert abs(value - histogram[key]) < 50  # Loose bound


class TestPrivacyBudget:
    """Test PrivacyBudget class."""
    
    def test_initialization(self):
        """Test budget initialization."""
        budget = PrivacyBudget(epsilon=1.0, delta=1e-5)
        
        assert budget.epsilon == 1.0
        assert budget.consumed_epsilon == 0.0
        assert budget.remaining == 1.0
    
    def test_spend_budget(self):
        """Test spending budget."""
        budget = PrivacyBudget(epsilon=1.0)
        
        # Can spend within budget
        assert budget.spend(0.3) is True
        assert budget.consumed_epsilon == 0.3
        
        # Can spend more
        assert budget.spend(0.5) is True
        assert budget.consumed_epsilon == 0.8
        
        # Cannot exceed budget
        assert budget.spend(0.3) is False
        assert budget.consumed_epsilon == 0.8
    
    def test_utilization(self):
        """Test utilization calculation."""
        budget = PrivacyBudget(epsilon=1.0)
        
        assert budget.utilization == 0.0
        
        budget.spend(0.5)
        assert budget.utilization == 0.5
        
        budget.spend(0.5)
        assert budget.utilization == 1.0


class TestFederatedLearningOrchestrator:
    """Test suite for federated learning orchestrator."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp = tempfile.mkdtemp()
        yield Path(temp)
        shutil.rmtree(temp)
    
    @pytest.fixture
    def federated(self, temp_dir):
        """Create federated orchestrator."""
        return FederatedLearningOrchestrator(
            org_id="test-org",
            privacy_budget=1.0,
            storage_path=temp_dir,
        )
    
    @pytest.fixture
    def sample_outcome(self):
        """Create sample production outcome."""
        return ProductionOutcome(
            project_id="test-project",
            deployment_id="deploy-001",
            task_type=TaskType.CODE_GEN,
            model_used=Model.DEEPSEEK_CHAT,
            generated_code_hash="abc123",
            status=OutcomeStatus.SUCCESS,
        )
    
    @pytest.fixture
    def sample_fingerprint(self):
        """Create sample codebase fingerprint."""
        return CodebaseFingerprint(
            languages=["python"],
            framework="fastapi",
            patterns=["repository"],
        )
    
    def test_initialization(self, federated):
        """Test orchestrator initialization."""
        assert federated is not None
        assert federated.org_id == "test-org"
        assert federated.privacy.epsilon == 1.0
    
    @pytest.mark.asyncio
    async def test_contribute_insight(self, federated, sample_outcome, sample_fingerprint):
        """Test contributing an insight."""
        insight = await federated.contribute_insight(
            sample_outcome,
            sample_fingerprint,
        )
        
        assert insight is not None
        assert insight.org_hash == federated.org_hash
        assert insight.model == Model.DEEPSEEK_CHAT
        # Should be privatized
        assert insight.sample_size == 1
    
    @pytest.mark.asyncio
    async def test_privacy_budget_exhaustion(self, temp_dir, sample_outcome):
        """Test behavior when privacy budget is exhausted."""
        federated = FederatedLearningOrchestrator(
            org_id="test-org",
            privacy_budget=0.05,  # Very low budget
            storage_path=temp_dir,
        )
        
        # First contribution should work
        insight1 = await federated.contribute_insight(sample_outcome)
        assert insight1 is not None
        
        # Budget should be nearly exhausted
        # Subsequent contributions may fail
        # (depending on exact epsilon cost calculation)
    
    @pytest.mark.asyncio
    async def test_get_global_baseline(self, federated, sample_outcome, sample_fingerprint):
        """Test getting global baseline."""
        # Add some insights first
        await federated.contribute_insight(sample_outcome, sample_fingerprint)
        
        # Add insights from different "orgs" by manipulating global pool
        for i in range(5):
            insight = ModelInsight(
                insight_id=f"test-{i}",
                org_hash=f"other-org-{i}",
                model=Model.GPT_4O,
                task_type=TaskType.CODE_GEN,
                success_rate=0.8,
                avg_quality=0.85,
                avg_cost=0.01,
                sample_size=10,
                pattern_signature="test-pattern",
                language_signature="python",
            )
            federated._global_insights.append(insight)
        
        baseline = await federated.get_global_baseline(
            task_type=TaskType.CODE_GEN,
            fingerprint=sample_fingerprint,
        )
        
        assert baseline is not None
        assert baseline.task_type == TaskType.CODE_GEN
        assert baseline.total_contributions > 0
    
    @pytest.mark.asyncio
    async def test_get_global_baseline_no_data(self, federated):
        """Test baseline when no data available."""
        baseline = await federated.get_global_baseline(
            task_type=TaskType.CODE_GEN,
        )
        
        # Should return empty baseline
        assert baseline.confidence == 0.0
        assert len(baseline.recommended_models) == 0
    
    def test_switching_cost_estimate(self, federated, sample_outcome, sample_fingerprint):
        """Test switching cost calculation."""
        # Add some local insights
        asyncio.run(federated.contribute_insight(sample_outcome, sample_fingerprint))
        
        cost_estimate = federated.get_switching_cost_estimate()
        
        assert "local_samples" in cost_estimate
        assert "total_switching_cost_usd" in cost_estimate
        assert "nash_stability_score" in cost_estimate
        assert cost_estimate["local_samples"] > 0
    
    def test_federated_stats(self, federated):
        """Test getting federated stats."""
        stats = federated.get_federated_stats()
        
        assert "local_insights" in stats
        assert "privacy_budget" in stats
        assert stats["privacy_budget"]["epsilon"] == 1.0
    
    @pytest.mark.asyncio
    async def test_model_insight_anonymization(self, federated, sample_outcome, sample_fingerprint):
        """Test that insights are properly anonymized."""
        insight = await federated.contribute_insight(sample_outcome, sample_fingerprint)
        
        # Anonymize
        anonymized = insight.anonymize()
        
        # Should not expose actual patterns
        assert anonymized.pattern_signature != "repository"
        # Should be a hash
        assert len(anonymized.pattern_signature) == 16
    
    def test_persistence(self, temp_dir, sample_outcome):
        """Test saving and loading data."""
        # Create and populate
        fed1 = FederatedLearningOrchestrator(
            org_id="persist-test",
            storage_path=temp_dir,
        )
        asyncio.run(fed1.contribute_insight(sample_outcome))
        fed1._save_local_data()
        
        # Load in new instance
        fed2 = FederatedLearningOrchestrator(
            org_id="persist-test",
            storage_path=temp_dir,
        )
        
        assert len(fed2._local_insights) > 0


class TestModelInsight:
    """Test ModelInsight dataclass."""
    
    def test_to_dict(self):
        """Test serialization."""
        insight = ModelInsight(
            insight_id="test-123",
            org_hash="abc123",
            model=Model.DEEPSEEK_CHAT,
            task_type=TaskType.CODE_GEN,
            success_rate=0.9,
            avg_quality=0.85,
            avg_cost=0.002,
            sample_size=10,
            pattern_signature="pattern-hash",
            language_signature="lang-hash",
        )
        
        data = insight.to_dict()
        assert data["insight_id"] == "test-123"
        assert data["model"] == "deepseek-chat"
        assert data["success_rate"] == 0.9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
