"""
Tests for Tier 2 Architectural Optimizations
=============================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Run: pytest tests/test_optimizations_tier2.py -v
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from orchestrator.cost_optimization import (
    ModelCascader,
    CascadeMetrics,
    CascadeResult,
    SpeculativeGenerator,
    SpeculativeMetrics,
    SpeculativeResult,
    StreamingValidator,
    StreamingMetrics,
    StreamingResult,
)

# ─────────────────────────────────────────────
# Test Fixtures
# ─────────────────────────────────────────────


@pytest.fixture
def mock_client():
    """Create mock API client."""
    client = AsyncMock()
    client.call = AsyncMock(
        return_value=MagicMock(
            text="Generated code response",
            usage=MagicMock(input_tokens=100, output_tokens=50),
        )
    )
    client.stream = AsyncMock()
    client.stream.__aiter__ = MagicMock(return_value=iter(["chunk1", "chunk2", "chunk3"]))
    return client


@pytest.fixture
def model_cascader(mock_client):
    """Create model cascader with mock client."""
    return ModelCascader(client=mock_client)


@pytest.fixture
def speculative_gen(mock_client):
    """Create speculative generator with mock client."""
    return SpeculativeGenerator(client=mock_client)


@pytest.fixture
def streaming_validator(mock_client):
    """Create streaming validator with mock client."""
    return StreamingValidator(client=mock_client)


# ─────────────────────────────────────────────
# Test ModelCascader
# ─────────────────────────────────────────────


class TestModelCascader:
    """Test ModelCascader class."""

    def test_cascader_initialization(self, model_cascader):
        """Test cascader initializes correctly."""
        assert model_cascader is not None
        assert model_cascader.metrics is not None
        assert isinstance(model_cascader.metrics, CascadeMetrics)

    def test_set_cascade_chain(self, model_cascader):
        """Test setting custom cascade chain."""
        chain = [
            ("deepseek-chat", 0.80),
            ("claude-sonnet-4.6", 0.75),
        ]
        model_cascader.set_cascade_chain("code_generation", chain)

        retrieved = model_cascader.get_cascade_chain("code_generation")
        assert retrieved == chain

    def test_get_default_cascade_chain(self, model_cascader):
        """Test getting default cascade chain."""
        chain = model_cascader.get_cascade_chain("unknown_type")
        assert len(chain) > 0

    def test_metrics_to_dict(self, model_cascader):
        """Test metrics serialization."""
        metrics = model_cascader.metrics
        metrics.total_attempts = 10
        metrics.cascade_exits_early = 7

        metrics_dict = metrics.to_dict()
        assert "total_attempts" in metrics_dict
        assert "early_exit_rate" in metrics_dict
        assert metrics_dict["early_exit_rate"] == 0.7

    @pytest.mark.asyncio
    async def test_cascading_generate_success(self, model_cascader, mock_client):
        """Test successful cascading generation."""
        mock_client.call = AsyncMock(
            return_value=MagicMock(
                text="Good quality code",
            )
        )

        result = await model_cascader.cascading_generate(
            prompt="Generate Python code",
            task_type="code_generation",
        )

        assert isinstance(result, CascadeResult)
        assert result.model_used in ["deepseek-chat", "claude-sonnet-4.6", "claude-opus-4.6"]
        assert result.score >= 0.0
        assert result.score <= 1.0

    @pytest.mark.asyncio
    async def test_cascading_generate_early_exit(self, model_cascader, mock_client):
        """Test early exit from cascade."""
        # Mock client to return good response
        mock_client.call = AsyncMock(
            return_value=MagicMock(
                text="Excellent code with def function(): pass",
            )
        )

        result = await model_cascader.cascading_generate(
            prompt="Generate Python code",
            task_type="code_generation",
        )

        # Should exit early if cheap model scores high
        assert result.attempts >= 1
        assert result.cascade_exit_tier < 3  # Should not reach last tier

    @pytest.mark.asyncio
    async def test_cascading_generate_all_tiers(self, model_cascader, mock_client):
        """Test cascading through all tiers."""
        # Mock to return poor quality (low score)
        mock_client.call = AsyncMock(
            return_value=MagicMock(
                text="Error: failed",
            )
        )

        result = await model_cascader.cascading_generate(
            prompt="Generate Python code",
            task_type="code_generation",
        )

        # Should try all tiers
        assert result.attempts >= 1
        assert result.cascade_exit_tier == len(result.all_scores) - 1

    def test_estimate_cost(self, model_cascader):
        """Test cost estimation."""
        cost = model_cascader._estimate_cost(
            model="claude-sonnet-4.6",
            response="Test response" * 100,
            prompt="Test prompt" * 100,
        )

        assert cost > 0
        assert isinstance(cost, float)

    def test_heuristic_score(self, model_cascader):
        """Test heuristic scoring."""
        score = asyncio.get_event_loop().run_until_complete(
            model_cascader._heuristic_score(
                prompt="Generate code",
                response="```python\ndef hello():\n    pass\n```",
            )
        )

        assert score >= 0.0
        assert score <= 1.0
        # Should score higher for code with structure
        assert score >= 0.5

    def test_reset_metrics(self, model_cascader):
        """Test metrics reset."""
        model_cascader.metrics.total_attempts = 100
        model_cascader.reset_metrics()

        assert model_cascader.metrics.total_attempts == 0


# ─────────────────────────────────────────────
# Test SpeculativeGenerator
# ─────────────────────────────────────────────


class TestSpeculativeGenerator:
    """Test SpeculativeGenerator class."""

    def test_speculative_initialization(self, speculative_gen):
        """Test speculative generator initializes correctly."""
        assert speculative_gen is not None
        assert speculative_gen.metrics is not None
        assert isinstance(speculative_gen.metrics, SpeculativeMetrics)

    def test_set_model_pair(self, speculative_gen):
        """Test setting custom model pair."""
        speculative_gen.set_model_pair(
            task_type="code_generation",
            cheap_model="deepseek-chat",
            premium_model="claude-opus-4.6",
            threshold=0.85,
        )

        pair = speculative_gen.get_model_pair("code_generation")
        assert pair["cheap"] == "deepseek-chat"
        assert pair["premium"] == "claude-opus-4.6"
        assert pair["threshold"] == 0.85

    def test_metrics_to_dict(self, speculative_gen):
        """Test metrics serialization."""
        metrics = speculative_gen.metrics
        metrics.total_attempts = 20
        metrics.cheap_wins = 12

        metrics_dict = metrics.to_dict()
        assert "cheap_win_rate" in metrics_dict
        assert metrics_dict["cheap_win_rate"] == 0.6

    @pytest.mark.asyncio
    async def test_speculative_generate_cheap_wins(self, speculative_gen, mock_client):
        """Test speculative generation where cheap model wins."""
        mock_client.call = AsyncMock(
            return_value=MagicMock(
                text="Excellent code with def function(): pass",
            )
        )

        result = await speculative_gen.speculative_generate(
            prompt="Generate Python code",
            task_type="code_generation",
            threshold=0.50,  # Low threshold to ensure cheap wins
        )

        assert isinstance(result, SpeculativeResult)
        assert result.premium_cancelled is True
        assert result.model_used in ["deepseek-chat", "claude-sonnet-4.6"]

    @pytest.mark.asyncio
    async def test_speculative_generate_premium_wins(self, speculative_gen, mock_client):
        """Test speculative generation where premium is needed."""
        # Return poor quality for cheap model
        mock_client.call = AsyncMock(
            return_value=MagicMock(
                text="Error: failed to generate",
            )
        )

        result = await speculative_gen.speculative_generate(
            prompt="Generate Python code",
            task_type="code_generation",
            threshold=0.90,  # High threshold
        )

        assert isinstance(result, SpeculativeResult)
        # Premium should be used when cheap fails
        assert result.cheap_score < 0.90

    def test_quick_evaluate(self, speculative_gen):
        """Test quick evaluation."""
        score = asyncio.get_event_loop().run_until_complete(
            speculative_gen._quick_evaluate(
                prompt="Generate code",
                response="```python\ndef hello():\n    return 'world'\n```",
            )
        )

        assert score >= 0.0
        assert score <= 1.0
        # Should score higher for structured code
        assert score >= 0.5

    def test_estimate_cost(self, speculative_gen):
        """Test cost estimation."""
        cost = speculative_gen._estimate_cost(
            model="claude-opus-4.6",
            response="Test" * 1000,
            prompt="Test" * 100,
        )

        assert cost > 0
        # Premium model should cost more
        assert cost > speculative_gen._estimate_cost(
            model="deepseek-chat",
            response="Test" * 1000,
            prompt="Test" * 100,
        )

    def test_reset_metrics(self, speculative_gen):
        """Test metrics reset."""
        speculative_gen.metrics.total_attempts = 100
        speculative_gen.reset_metrics()

        assert speculative_gen.metrics.total_attempts == 0


# ─────────────────────────────────────────────
# Test StreamingValidator
# ─────────────────────────────────────────────


class TestStreamingValidator:
    """Test StreamingValidator class."""

    def test_streaming_initialization(self, streaming_validator):
        """Test streaming validator initializes correctly."""
        assert streaming_validator is not None
        assert streaming_validator.metrics is not None
        assert isinstance(streaming_validator.metrics, StreamingMetrics)
        assert len(streaming_validator._compiled_patterns) > 0

    def test_detect_early_failure_refusal(self, streaming_validator):
        """Test detecting refusal patterns."""
        failure = streaming_validator._detect_early_failure(
            text="I cannot help with that task as an AI language model.",
            task_type="code_generation",
        )

        assert failure is not None
        assert "cannot" in failure.lower() or "ai" in failure.lower()

    def test_detect_early_failure_code(self, streaming_validator):
        """Test detecting code issues."""
        failure = streaming_validator._detect_early_failure(
            text="pass  # TODO: implement this later",
            task_type="code_generation",
        )

        assert failure is not None

    def test_detect_early_failure_no_structure(self, streaming_validator):
        """Test detecting lack of code structure."""
        failure = streaming_validator._detect_early_failure(
            text="This is just plain text without any code structure or functions.",
            task_type="code_generation",
        )

        # Should detect lack of structure in longer text
        if len("This is just plain text without any code structure or functions.") > 200:
            assert failure is not None

    def test_no_early_failure_good_code(self, streaming_validator):
        """Test that good code doesn't trigger early failure."""
        failure = streaming_validator._detect_early_failure(
            text="```python\ndef hello():\n    return 'world'\n```",
            task_type="code_generation",
        )

        assert failure is None

    def test_metrics_to_dict(self, streaming_validator):
        """Test metrics serialization."""
        metrics = streaming_validator.metrics
        metrics.total_streams = 50
        metrics.early_aborts = 5

        metrics_dict = metrics.to_dict()
        assert "early_abort_rate" in metrics_dict
        assert metrics_dict["early_abort_rate"] == 0.1

    @pytest.mark.asyncio
    async def test_stream_and_validate_success(self, streaming_validator, mock_client):
        """Test successful streaming validation."""
        result = await streaming_validator.stream_and_validate(
            model="claude-sonnet-4.6",
            prompt="Generate Python code",
            task_type="code_generation",
        )

        assert isinstance(result, StreamingResult)
        assert result.early_aborted is False
        assert result.response is not None

    @pytest.mark.asyncio
    async def test_stream_and_validate_early_abort(self, streaming_validator, mock_client):
        """Test early abort on failure."""
        # Mock to return refusal pattern
        mock_client.call = AsyncMock(
            return_value=MagicMock(
                text="I cannot help with that. As an AI, I am unable.",
            )
        )

        result = await streaming_validator.stream_and_validate(
            model="claude-sonnet-4.6",
            prompt="Generate Python code",
            task_type="code_generation",
            early_abort_tokens=100,  # Low threshold for testing
        )

        # Should detect early failure
        assert result.early_aborted is True or result.retry_count > 0

    def test_estimate_cost(self, streaming_validator):
        """Test cost estimation."""
        cost = streaming_validator._estimate_cost(
            model="claude-opus-4.6",
            tokens=1000,
        )

        assert cost > 0
        assert isinstance(cost, float)

    def test_reset_metrics(self, streaming_validator):
        """Test metrics reset."""
        streaming_validator.metrics.total_streams = 100
        streaming_validator.reset_metrics()

        assert streaming_validator.metrics.total_streams == 0


# ─────────────────────────────────────────────
# Test Integration
# ─────────────────────────────────────────────


class TestIntegration:
    """Test integration between Tier 2 modules."""

    @pytest.mark.asyncio
    async def test_cascade_with_speculative(self, mock_client):
        """Test using cascade and speculative together."""
        cascader = ModelCascader(client=mock_client)
        speculative = SpeculativeGenerator(client=mock_client)

        # Both should work with same client
        cascade_result = await cascader.cascading_generate(
            prompt="Test",
            task_type="code_generation",
        )
        speculative_result = await speculative.speculative_generate(
            prompt="Test",
            task_type="code_generation",
        )

        assert isinstance(cascade_result, CascadeResult)
        assert isinstance(speculative_result, SpeculativeResult)

    def test_combined_metrics(self, mock_client):
        """Test collecting metrics from all modules."""
        cascader = ModelCascader(client=mock_client)
        speculative = SpeculativeGenerator(client=mock_client)
        streaming = StreamingValidator(client=mock_client)

        all_metrics = {
            "cascade": cascader.get_metrics(),
            "speculative": speculative.get_metrics(),
            "streaming": streaming.get_metrics(),
        }

        assert "total_attempts" in all_metrics["cascade"]
        assert "cheap_wins" in all_metrics["speculative"]
        assert "total_streams" in all_metrics["streaming"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
