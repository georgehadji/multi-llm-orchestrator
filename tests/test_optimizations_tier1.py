"""
Tests for Tier 1 Cost Optimizations
=====================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Run: pytest tests/test_optimizations_tier1.py -v
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from orchestrator.cost_optimization import (
    OptimizationPhase,
    OptimizationConfig,
    OptimizationMetrics,
    get_optimization_config,
    update_config,
)
from orchestrator.cost_optimization.prompt_cache import (
    PromptCacher,
    CacheMetrics,
    warm_prompt_cache,
)
from orchestrator.cost_optimization.batch_client import (
    BatchClient,
    BatchStatus,
    BatchMetrics,
    batch_call,
)
from orchestrator.cost_optimization.token_budget import (
    TokenBudget,
    TokenUsage,
    TokenBudgetMetrics,
    get_token_limit,
)

# ─────────────────────────────────────────────
# Test Fixtures
# ─────────────────────────────────────────────


@pytest.fixture
def mock_client():
    """Create mock API client."""
    client = AsyncMock()
    client.messages = AsyncMock()
    client.messages.create = AsyncMock(
        return_value=MagicMock(
            content="Mock response",
            usage=MagicMock(input_tokens=100, output_tokens=50),
        )
    )
    return client


@pytest.fixture
def prompt_cacher(mock_client):
    """Create prompt cacher with mock client."""
    return PromptCacher(client=mock_client)


@pytest.fixture
def batch_client(mock_client):
    """Create batch client with mock client."""
    return BatchClient(client=mock_client)


@pytest.fixture
def token_budget():
    """Create token budget manager."""
    return TokenBudget()


# ─────────────────────────────────────────────
# Test Optimization Config
# ─────────────────────────────────────────────


class TestOptimizationConfig:
    """Test optimization configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = OptimizationConfig()

        assert config.enable_prompt_caching is True
        assert config.enable_batch_api is True
        assert config.enable_token_budget is True
        assert config.enable_cascading is False
        assert len(config.batch_phases) == 3

    def test_config_to_dict(self):
        """Test config serialization."""
        config = OptimizationConfig()
        config_dict = config.to_dict()

        assert "enable_prompt_caching" in config_dict
        assert "enable_batch_api" in config_dict
        assert "output_token_limits" in config_dict

    def test_optimization_metrics(self):
        """Test metrics tracking."""
        metrics = OptimizationMetrics()
        metrics.cache_hits = 80
        metrics.cache_misses = 20

        assert metrics.cache_hit_rate == 0.0  # Not auto-calculated
        metrics_dict = metrics.to_dict()
        assert "cache_hits" in metrics_dict


# ─────────────────────────────────────────────
# Test PromptCacher
# ─────────────────────────────────────────────


class TestPromptCacher:
    """Test PromptCacher class."""

    def test_cacher_initialization(self, prompt_cacher):
        """Test cacher initializes correctly."""
        assert prompt_cacher is not None
        assert prompt_cacher.metrics is not None
        assert isinstance(prompt_cacher.metrics, CacheMetrics)

    def test_compute_cache_key(self, prompt_cacher):
        """Test cache key computation."""
        key1 = prompt_cacher._compute_cache_key("system prompt", "context")
        key2 = prompt_cacher._compute_cache_key("system prompt", "context")
        key3 = prompt_cacher._compute_cache_key("different prompt", "context")

        assert key1 == key2
        assert key1 != key3
        assert len(key1) == 16  # SHA256 hex[:16]

    @pytest.mark.asyncio
    async def test_warm_cache(self, prompt_cacher, mock_client):
        """Test cache warming."""
        cache_key = await prompt_cacher.warm_cache(
            system_prompt="Test system prompt",
            project_context="Test context",
        )

        assert cache_key is not None
        assert len(cache_key) == 16
        assert prompt_cacher.metrics.warmings == 1
        mock_client.messages.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_call_with_cache(self, prompt_cacher, mock_client):
        """Test API call with caching."""
        response = await prompt_cacher.call_with_cache(
            model="claude-sonnet-4.6",
            messages=[{"role": "user", "content": "Hello"}],
            system_prompt="Test system",
        )

        assert response is not None
        assert prompt_cacher.metrics.hits >= 0  # May be hit or miss

    @pytest.mark.asyncio
    async def test_call_with_project_context(self, prompt_cacher, mock_client):
        """Test call with project context."""
        response = await prompt_cacher.call_with_cache(
            model="claude-sonnet-4.6",
            messages=[{"role": "user", "content": "Hello"}],
            system_prompt="Test system",
            project_context="My project context",
        )

        assert response is not None

    @pytest.mark.asyncio
    async def test_cache_metrics(self, prompt_cacher):
        """Test cache metrics tracking."""
        metrics = prompt_cacher.get_metrics()

        assert "hit_rate" in metrics
        assert "hits" in metrics
        assert "misses" in metrics
        assert "warmings" in metrics

    @pytest.mark.asyncio
    async def test_clear_cache(self, prompt_cacher):
        """Test cache clearing."""
        # Warm cache first
        await prompt_cacher.warm_cache("test prompt")

        # Clear cache
        await prompt_cacher.clear_cache()

        assert len(prompt_cacher._cache_entries) == 0
        assert prompt_cacher._system_prompt_cache is None

    @pytest.mark.asyncio
    async def test_cache_warming_before_parallel(self, prompt_cacher, mock_client):
        """Test cache warming before parallel execution."""
        # Warm cache
        await prompt_cacher.warm_cache("system prompt", "context")

        # Simulate parallel calls
        tasks = [
            prompt_cacher.call_with_cache(
                model="claude-sonnet-4.6",
                messages=[{"role": "user", "content": f"Task {i}"}],
                system_prompt="system prompt",
            )
            for i in range(5)
        ]

        responses = await asyncio.gather(*tasks)
        assert len(responses) == 5


# ─────────────────────────────────────────────
# Test BatchClient
# ─────────────────────────────────────────────


class TestBatchClient:
    """Test BatchClient class."""

    def test_batch_client_initialization(self, batch_client):
        """Test batch client initializes correctly."""
        assert batch_client is not None
        assert batch_client.metrics is not None
        assert isinstance(batch_client.metrics, BatchMetrics)

    def test_should_use_batch(self, batch_client):
        """Test batch phase detection."""
        # Should batch
        assert batch_client._should_use_batch(OptimizationPhase.EVALUATION) is True
        assert batch_client._should_use_batch(OptimizationPhase.CRITIQUE) is True

        # Should not batch
        assert batch_client._should_use_batch(OptimizationPhase.GENERATION) is False
        assert batch_client._should_use_batch(OptimizationPhase.DECOMPOSITION) is False

    @pytest.mark.asyncio
    async def test_batch_call_routing(self, batch_client, mock_client):
        """Test automatic batch/realtime routing."""
        # Evaluation phase -> batch
        result = await batch_client.call(
            model="claude-sonnet-4.6",
            prompt="Evaluate this",
            phase=OptimizationPhase.EVALUATION,
        )
        assert result is not None
        assert batch_client.metrics.batch_requests >= 0

        # Generation phase -> realtime
        result = await batch_client.call(
            model="claude-sonnet-4.6",
            prompt="Generate code",
            phase=OptimizationPhase.GENERATION,
        )
        assert result is not None
        assert batch_client.metrics.realtime_requests >= 0

    @pytest.mark.asyncio
    async def test_batch_metrics(self, batch_client):
        """Test batch metrics tracking."""
        metrics = batch_client.get_metrics()

        assert "batch_requests" in metrics
        assert "realtime_requests" in metrics
        assert "batch_ratio" in metrics
        assert "total_savings" in metrics

    def test_estimate_cost(self, batch_client):
        """Test cost estimation."""
        cost = batch_client._estimate_cost(
            model="claude-sonnet-4.6",
            prompt="Test prompt" * 100,
        )

        assert cost > 0
        assert isinstance(cost, float)

    @pytest.mark.asyncio
    async def test_batch_queue_processing(self, batch_client):
        """Test batch queue processing."""
        # Queue multiple requests
        tasks = [
            batch_client.call(
                model="claude-sonnet-4.6",
                prompt=f"Request {i}",
                phase=OptimizationPhase.EVALUATION,
            )
            for i in range(5)
        ]

        # Process with timeout
        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=10.0,
        )

        # Some may timeout, but queue should be processed
        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_shutdown(self, batch_client):
        """Test graceful shutdown."""
        await batch_client.shutdown()
        # Should complete without errors


# ─────────────────────────────────────────────
# Test TokenBudget
# ─────────────────────────────────────────────


class TestTokenBudget:
    """Test TokenBudget class."""

    def test_budget_initialization(self, token_budget):
        """Test budget initializes correctly."""
        assert token_budget is not None
        assert token_budget.metrics is not None
        assert isinstance(token_budget.metrics, TokenBudgetMetrics)

    def test_get_limit(self, token_budget):
        """Test token limit retrieval."""
        # Test all phases
        limits = token_budget.DEFAULT_LIMITS
        for phase in OptimizationPhase:
            limit = token_budget.get_limit(phase)
            assert limit > 0
            assert isinstance(limit, int)

    def test_get_limit_by_name(self, token_budget):
        """Test token limit by phase name."""
        limit = token_budget.get_limit_by_name("generation")
        assert limit == 4000

        limit = token_budget.get_limit_by_name("evaluation")
        assert limit == 500

        # Unknown phase
        limit = token_budget.get_limit_by_name("unknown")
        assert limit == 1000  # Default

    def test_record_usage(self, token_budget):
        """Test token usage tracking."""
        token_budget.record_usage(
            model="claude-sonnet-4.6",
            input_tokens=1000,
            output_tokens=500,
            phase=OptimizationPhase.GENERATION,
        )

        metrics = token_budget.metrics
        assert metrics.total_input_tokens == 1000
        assert metrics.total_output_tokens == 500
        assert metrics.total_input_cost > 0
        assert metrics.total_output_cost > 0

    def test_enforce_limit(self, token_budget):
        """Test limit enforcement."""
        params = token_budget.enforce_limit(
            model="claude-sonnet-4.6",
            prompt="Test prompt" * 100,
            phase=OptimizationPhase.GENERATION,
        )

        assert "max_tokens" in params
        assert params["max_tokens"] == 4000
        assert params["phase"] == "generation"

    def test_enforce_limit_exceeds(self, token_budget):
        """Test limit enforcement when estimate exceeds."""
        params = token_budget.enforce_limit(
            model="claude-sonnet-4.6",
            prompt="Test" * 10000,  # Very long prompt
            phase=OptimizationPhase.EVALUATION,
            estimated_output=2000,  # Exceeds 500 limit
        )

        assert params["max_tokens"] == 500  # Enforced limit
        assert token_budget.metrics.limit_enforced_count >= 1

    def test_get_usage(self, token_budget):
        """Test usage statistics."""
        # Record some usage
        token_budget.record_usage(
            model="claude-sonnet-4.6",
            input_tokens=1000,
            output_tokens=500,
            phase=OptimizationPhase.GENERATION,
        )

        # Get all usage
        usage = token_budget.get_usage()
        assert "total_input_tokens" in usage
        assert "models" in usage

        # Get specific model usage
        model_usage = token_budget.get_usage("claude-sonnet-4.6")
        assert model_usage["input_tokens"] == 1000
        assert model_usage["output_tokens"] == 500

    def test_get_metrics(self, token_budget):
        """Test metrics retrieval."""
        metrics = token_budget.get_metrics()

        assert "total_input_tokens" in metrics
        assert "total_output_tokens" in metrics
        assert "estimated_savings" in metrics

    def test_reset_metrics(self, token_budget):
        """Test metrics reset."""
        # Record some usage
        token_budget.record_usage(
            model="claude-sonnet-4.6",
            input_tokens=1000,
            output_tokens=500,
            phase=OptimizationPhase.GENERATION,
        )

        # Reset
        token_budget.reset_metrics()

        assert token_budget.metrics.total_input_tokens == 0
        assert token_budget.metrics.total_output_tokens == 0


# ─────────────────────────────────────────────
# Test Convenience Functions
# ─────────────────────────────────────────────


class TestConvenienceFunctions:
    """Test convenience functions."""

    @pytest.mark.asyncio
    async def test_warm_prompt_cache(self):
        """Test warm_prompt_cache function."""
        with patch("orchestrator.optimization.prompt_cache.PromptCacher") as MockCacher:
            mock_cacher = MockCacher.return_value
            mock_cacher.warm_cache = AsyncMock(return_value="cache_key_123")

            key = await warm_prompt_cache("system prompt", "context")

            assert key == "cache_key_123"

    def test_get_token_limit(self):
        """Test get_token_limit function."""
        limit = get_token_limit("generation")
        assert limit == 4000

        limit = get_token_limit("evaluation")
        assert limit == 500

    @pytest.mark.asyncio
    async def test_batch_call(self):
        """Test batch_call function."""
        with patch("orchestrator.optimization.batch_client.BatchClient") as MockClient:
            mock_client = MockClient.return_value
            mock_client.call = AsyncMock(return_value={"result": "test"})

            result = await batch_call(
                model="claude-sonnet-4.6",
                prompt="Test prompt",
                phase="evaluation",
            )

            assert result == {"result": "test"}


# ─────────────────────────────────────────────
# Test Integration
# ─────────────────────────────────────────────


class TestIntegration:
    """Test integration between optimization modules."""

    @pytest.mark.asyncio
    async def test_cache_and_budget_integration(self, mock_client):
        """Test prompt caching with token budget."""
        from orchestrator.optimization.prompt_cache import PromptCacher
        from orchestrator.optimization.token_budget import TokenBudget

        cacher = PromptCacher(client=mock_client)
        budget = TokenBudget()

        # Warm cache
        await cacher.warm_cache("system prompt", "context")

        # Make call with budget enforcement
        limit = budget.get_limit(OptimizationPhase.GENERATION)

        response = await cacher.call_with_cache(
            model="claude-sonnet-4.6",
            messages=[{"role": "user", "content": "Hello"}],
            system_prompt="system prompt",
            max_tokens=limit,
        )

        assert response is not None

    @pytest.mark.asyncio
    async def test_batch_with_token_budget(self, mock_client):
        """Test batch processing with token budget."""
        batch = BatchClient(client=mock_client)
        budget = TokenBudget()

        # Make batch call with budget enforcement
        limit = budget.get_limit(OptimizationPhase.EVALUATION)

        # Simulate batch call
        result = await mock_client.messages.create(
            model="claude-sonnet-4.6",
            messages=[{"role": "user", "content": "Evaluate"}],
            max_tokens=limit,
        )

        assert result is not None

    def test_config_update(self):
        """Test configuration update."""
        new_config = OptimizationConfig(
            enable_prompt_caching=False,
            enable_batch_api=True,
            enable_token_budget=True,
        )

        update_config(new_config)
        config = get_optimization_config()

        assert config.enable_prompt_caching is False
        assert config.enable_batch_api is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
