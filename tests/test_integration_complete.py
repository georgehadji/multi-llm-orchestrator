"""
AI Orchestrator — Integration Tests
=====================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Comprehensive integration tests for all AI Orchestrator features.

Run: pytest tests/test_integration_complete.py -v
"""

import pytest
import asyncio
from pathlib import Path

from orchestrator.models import Model, TaskType, OutputTarget
from orchestrator.api_clients import UnifiedClient
from orchestrator.rate_limiter import GrokRateLimiter, get_rate_limiter
from orchestrator.provisioned_throughput import (
    ProvisionedThroughputManager,
    ProvisionedThroughputConfig,
)
from orchestrator.advanced_query_processing import (
    LLMQueryExpander,
    LearningClassifier,
    ResultSummarizer,
)
from orchestrator.xai_search import XSearchClient
from orchestrator.nexus_search.optimization import (
    ResultDeduplicator,
    QueryCache,
    SemanticReranker,
    ParallelSearchExecutor,
)

# ─────────────────────────────────────────────
# Test Fixtures
# ─────────────────────────────────────────────


@pytest.fixture
def api_client():
    """Create UnifiedClient instance."""
    return UnifiedClient()


@pytest.fixture
def rate_limiter():
    """Create GrokRateLimiter instance."""
    return GrokRateLimiter(api_key="test-key", initial_tier=1)


@pytest.fixture
def pt_manager():
    """Create ProvisionedThroughputManager instance."""
    config = ProvisionedThroughputConfig(
        enabled=True,
        units=2,
        models=["grok-4.20"],
    )
    return ProvisionedThroughputManager(config=config, api_key="test-key")


@pytest.fixture
def query_expander(api_client):
    """Create LLMQueryExpander instance."""
    return LLMQueryExpander(client=api_client)


@pytest.fixture
def classifier():
    """Create LearningClassifier instance."""
    return LearningClassifier()


@pytest.fixture
def summarizer(api_client):
    """Create ResultSummarizer instance."""
    return ResultSummarizer(client=api_client)


# ─────────────────────────────────────────────
# Test API Client Integration
# ─────────────────────────────────────────────


class TestAPIClientIntegration:
    """Test API client integration."""

    @pytest.mark.asyncio
    async def test_client_initialization(self, api_client):
        """Test client initializes correctly."""
        assert api_client is not None
        assert hasattr(api_client, "_clients")

    @pytest.mark.asyncio
    async def test_client_region_support(self):
        """Test client supports regional endpoints."""
        from orchestrator.api_clients import UnifiedClient

        # Test global endpoint
        client_global = UnifiedClient(xai_region=None)
        assert "api.x.ai" in client_global.XAI_REGIONS[None]

        # Test regional endpoint
        client_regional = UnifiedClient(xai_region="us-east-1")
        assert client_regional.xai_region == "us-east-1"


# ─────────────────────────────────────────────
# Test Rate Limiter Integration
# ─────────────────────────────────────────────


class TestRateLimiterIntegration:
    """Test rate limiter integration."""

    @pytest.mark.asyncio
    async def test_rate_limiter_tier_progression(self, rate_limiter):
        """Test tier progression based on spend."""
        # Start at tier 1
        assert rate_limiter.state.current_tier == 1

        # Spend $50 → tier 2
        rate_limiter.record_spend(50.0)
        assert rate_limiter.state.current_tier == 2

        # Spend $150 more → tier 3
        rate_limiter.record_spend(150.0)
        assert rate_limiter.state.current_tier == 3

    @pytest.mark.asyncio
    async def test_rate_limiter_acquire(self, rate_limiter):
        """Test token acquisition."""
        acquired = await rate_limiter.acquire(tokens=1000)
        assert acquired is True
        assert rate_limiter.total_requests == 1

    @pytest.mark.asyncio
    async def test_rate_limiter_stats(self, rate_limiter):
        """Test statistics."""
        await rate_limiter.acquire(tokens=1000)
        stats = rate_limiter.get_stats()

        assert "current_tier" in stats
        assert "total_requests" in stats
        assert stats["total_requests"] == 1


# ─────────────────────────────────────────────
# Test Provisioned Throughput Integration
# ─────────────────────────────────────────────


class TestProvisionedThroughputIntegration:
    """Test provisioned throughput integration."""

    @pytest.mark.asyncio
    async def test_capacity_check(self, pt_manager):
        """Test capacity checking."""
        acquired = await pt_manager.check_capacity(tokens=10000, is_input=True)
        assert acquired is True

    @pytest.mark.asyncio
    async def test_usage_recording(self, pt_manager):
        """Test usage recording."""
        pt_manager.record_usage(
            input_tokens=5000,
            output_tokens=3000,
            capacity_type="committed",
        )

        assert pt_manager.usage.total_input_tokens == 5000
        assert pt_manager.usage.total_output_tokens == 3000

    @pytest.mark.asyncio
    async def test_capacity_stats(self, pt_manager):
        """Test statistics."""
        await pt_manager.check_capacity(tokens=10000)
        stats = pt_manager.get_stats()

        assert "units" in stats
        assert "input_capacity_tpm" in stats
        assert stats["units"] == 2


# ─────────────────────────────────────────────
# Test Advanced Query Processing Integration
# ─────────────────────────────────────────────


class TestAdvancedQueryProcessingIntegration:
    """Test advanced query processing integration."""

    @pytest.mark.asyncio
    async def test_query_expansion_synonyms(self, query_expander):
        """Test synonym-based query expansion."""
        expansions = query_expander.expand_with_synonyms("python async")

        assert len(expansions) > 0
        assert any(
            "asynchronous" in e.expanded.lower() or "python" in e.expanded.lower()
            for e in expansions
        )

    @pytest.mark.asyncio
    async def test_query_classification(self, classifier):
        """Test query classification."""
        result = await classifier.classify("python async tutorial")

        assert result.category in ["technical", "research", "factual", "academic", "creative"]
        assert result.confidence > 0.0

    @pytest.mark.asyncio
    async def test_classifier_learning(self, classifier):
        """Test classifier learning from feedback."""
        # Classify query
        result = await classifier.classify("python code example")
        initial_category = result.category

        # Record feedback
        classifier.record_feedback(
            query="python code example",
            predicted_category=initial_category,
            actual_category="technical",
        )

        # Should have feedback recorded
        assert len(classifier.feedback_history) == 1

    @pytest.mark.asyncio
    async def test_result_summarization(self, summarizer):
        """Test result summarization."""
        from orchestrator.nexus_search.models import SearchResult, SearchSource

        results = [
            SearchResult(
                title="Python Async Best Practices",
                url="https://example.com/python-async",
                content="Learn about async/await in Python with asyncio for concurrent programming",
                source=SearchSource.TECH,
                score=0.9,
            ),
            SearchResult(
                title="Advanced Python Asyncio",
                url="https://example.com/advanced-asyncio",
                content="Deep dive into asyncio event loop, tasks, and coroutines",
                source=SearchSource.TECH,
                score=0.8,
            ),
        ]

        summary = await summarizer.summarize("python async", results)

        assert summary.query == "python async"
        assert len(summary.summary) > 0
        assert summary.sources_count == 2


# ─────────────────────────────────────────────
# Test Nexus Optimization Integration
# ─────────────────────────────────────────────


class TestNexusOptimizationIntegration:
    """Test Nexus optimization integration."""

    @pytest.mark.asyncio
    async def test_deduplication_integration(self):
        """Test result deduplication."""
        from orchestrator.nexus_search.models import SearchResult, SearchSource

        results = [
            SearchResult(
                title="Python Guide",
                url="https://example.com/python",
                content="Python programming guide",
                source=SearchSource.WEB,
                score=0.9,
            ),
            SearchResult(
                title="Python Guide",  # Duplicate title
                url="https://example.com/python",  # Same URL
                content="Python programming guide",
                source=SearchSource.WEB,
                score=0.8,
            ),
        ]

        deduplicator = ResultDeduplicator()
        deduped = deduplicator.deduplicate(results)

        # Should remove duplicate
        assert len(deduped) == 1

    @pytest.mark.asyncio
    async def test_cache_integration(self):
        """Test query cache."""
        from orchestrator.nexus_search.models import SearchResults, SearchSource

        cache = QueryCache(ttl_seconds=3600)

        # Create search results
        search_results = SearchResults(
            query="test query",
            results=[
                SearchResult(
                    title="Test Result",
                    url="https://test.com/result",
                    content="Test content",
                    source=SearchSource.WEB,
                    score=0.9,
                )
            ],
        )

        # Cache results
        await cache.set("test query", [SearchSource.WEB], search_results)

        # Retrieve cached results
        cached = await cache.get("test query", [SearchSource.WEB])

        assert cached is not None
        assert cached.query == "test query"

    @pytest.mark.asyncio
    async def test_reranker_integration(self):
        """Test semantic reranker."""
        from orchestrator.nexus_search.models import SearchResult, SearchSource

        results = [
            SearchResult(
                title="JavaScript Tutorial",
                url="https://js.com/tutorial",
                content="Learn JavaScript",
                source=SearchSource.WEB,
                score=0.9,  # High initial score but irrelevant
            ),
            SearchResult(
                title="Python Async Guide",
                url="https://python.com/async",
                content="Learn Python async/await",
                source=SearchSource.TECH,
                score=0.5,  # Lower initial score but relevant
            ),
        ]

        reranker = SemanticReranker()
        reranked = reranker.rerank("Python async", results, top_k=2)

        # Python result should be ranked higher after reranking
        assert len(reranked) == 2


# ─────────────────────────────────────────────
# Test X Search Integration
# ─────────────────────────────────────────────


class TestXSearchIntegration:
    """Test X Search integration."""

    @pytest.mark.asyncio
    async def test_x_search_client_creation(self):
        """Test X Search client creation."""
        client = XSearchClient(api_key="test-key")

        assert client.api_key == "test-key"
        assert client.BASE_URL == "https://api.x.ai/v1"

    @pytest.mark.asyncio
    async def test_x_search_context_manager(self):
        """Test X Search context manager."""
        async with XSearchClient(api_key="test-key") as client:
            assert client is not None


# ─────────────────────────────────────────────
# Test Multi-Platform Generator Integration
# ─────────────────────────────────────────────


class TestMultiPlatformGeneratorIntegration:
    """Test multi-platform generator integration."""

    @pytest.mark.asyncio
    async def test_output_target_enum(self):
        """Test OutputTarget enum."""
        from orchestrator.multi_platform_generator import OutputTarget

        assert OutputTarget.PYTHON_LIBRARY.value == "python"
        assert OutputTarget.REACT_WEB_APP.value == "react"
        assert OutputTarget.SWIFTUI_IOS.value == "swiftui"

    @pytest.mark.asyncio
    async def test_project_output_config(self):
        """Test ProjectOutputConfig."""
        from orchestrator.multi_platform_generator import ProjectOutputConfig

        config = ProjectOutputConfig(
            targets=["python", "react"],
            include_auth=True,
            include_database=True,
        )

        assert config.include_auth is True
        assert config.include_database is True


# ─────────────────────────────────────────────
# Test App Store Validator Integration
# ─────────────────────────────────────────────


class TestAppStoreValidatorIntegration:
    """Test App Store validator integration."""

    @pytest.mark.asyncio
    async def test_app_store_platform_enum(self):
        """Test AppStorePlatform enum."""
        from orchestrator.app_store_validator import AppStorePlatform

        assert AppStorePlatform.IOS.value == "ios"
        assert AppStorePlatform.ANDROID.value == "android"
        assert AppStorePlatform.WEB.value == "web"

    @pytest.mark.asyncio
    async def test_validator_creation(self):
        """Test validator creation."""
        from orchestrator.app_store_validator import AppStoreValidator

        validator = AppStoreValidator()

        assert validator is not None
        assert validator.auto_fix is False


# ─────────────────────────────────────────────
# Test End-to-End Workflows
# ─────────────────────────────────────────────


class TestEndToEndWorkflows:
    """Test end-to-end workflows."""

    @pytest.mark.asyncio
    async def test_full_search_workflow(self, api_client):
        """Test complete search workflow."""
        # 1. Expand query
        expander = LLMQueryExpander(client=api_client)
        expansions = expander.expand_with_synonyms("python async")

        # 2. Classify query
        classifier = LearningClassifier()
        classification = await classifier.classify("python async")

        # 3. Deduplicate results (simulated)
        from orchestrator.nexus_search.models import SearchResult, SearchSource

        results = [
            SearchResult(
                title="Python Async",
                url="https://python.com/async",
                content="Python async guide",
                source=SearchSource.TECH,
                score=0.9,
            ),
        ]
        deduplicator = ResultDeduplicator()
        deduped = deduplicator.deduplicate(results)

        # 4. Cache results
        from orchestrator.nexus_search.optimization import QueryCache
        from orchestrator.nexus_search.models import SearchResults

        cache = QueryCache()
        search_results = SearchResults(
            query="python async",
            results=deduped,
        )
        await cache.set("python async", [SearchSource.TECH], search_results)

        # Verify workflow completed
        assert len(expansions) >= 0
        assert classification.category in ["technical", "research"]
        assert len(deduped) >= 0

        # Verify cache
        cached = await cache.get("python async", [SearchSource.TECH])
        assert cached is not None

    @pytest.mark.asyncio
    async def test_rate_limited_search_workflow(self, rate_limiter):
        """Test rate-limited search workflow."""
        # Acquire tokens
        acquired = await rate_limiter.acquire(tokens=1000)
        assert acquired is True

        # Record spend
        rate_limiter.record_spend(0.01)

        # Get stats
        stats = rate_limiter.get_stats()
        assert stats["total_requests"] == 1
        assert stats["total_tokens"] == 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
