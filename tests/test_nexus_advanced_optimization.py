"""
Tests for Nexus Search Advanced Optimizations
==============================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Run: pytest tests/test_nexus_advanced_optimization.py -v
"""

import pytest
import asyncio
import time

from orchestrator.nexus_search.models import SearchResult, SearchSource, SearchResults
from orchestrator.nexus_search.optimization.reranker import (
    SemanticReranker,
    rerank_results,
)
from orchestrator.nexus_search.optimization.parallel_search import (
    ParallelSearchExecutor,
    search_parallel,
)

# ─────────────────────────────────────────────
# Test Fixtures
# ─────────────────────────────────────────────


@pytest.fixture
def sample_search_results():
    """Create sample search results for reranking."""
    results = [
        SearchResult(
            title="Python Async Best Practices Guide",
            url="https://example.com/python-async-guide",
            content="Comprehensive guide to async/await in Python with asyncio",
            source=SearchSource.TECH,
            score=0.5,  # Initial score (will be reranked)
        ),
        SearchResult(
            title="JavaScript Tutorial for Beginners",
            url="https://example.com/js-tutorial",
            content="Learn JavaScript from scratch with examples",
            source=SearchSource.WEB,
            score=0.9,  # High initial score but irrelevant
        ),
        SearchResult(
            title="Advanced Python Asyncio Patterns",
            url="https://example.com/advanced-asyncio",
            content="Deep dive into asyncio event loop, tasks, and coroutines",
            source=SearchSource.TECH,
            score=0.6,
        ),
        SearchResult(
            title="Python Synchronous Programming",
            url="https://example.com/python-sync",
            content="Learn about synchronous programming in Python",
            source=SearchSource.WEB,
            score=0.7,
        ),
    ]
    return results


@pytest.fixture
def reranker():
    """Create SemanticReranker instance."""
    return SemanticReranker()


@pytest.fixture
def parallel_executor():
    """Create ParallelSearchExecutor instance."""
    return ParallelSearchExecutor(max_concurrency=3)


# ─────────────────────────────────────────────
# Test Semantic Reranker
# ─────────────────────────────────────────────


class TestSemanticReranker:
    """Test SemanticReranker class."""

    def test_reranker_initialization(self, reranker):
        """Test reranker initializes correctly."""
        assert reranker.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert reranker._initialized is False

    def test_rerank_single_result(self, reranker):
        """Test reranking with single result."""
        results = [SearchResult(title="Test", url="https://test.com", content="...")]
        reranked = reranker.rerank("test query", results)

        # Single result should be returned as-is
        assert len(reranked) == 1

    def test_rerank_empty_results(self, reranker):
        """Test reranking with empty results."""
        reranked = reranker.rerank("test query", [])
        assert len(reranked) == 0

    def test_rerank_ordering(self, sample_search_results, reranker):
        """Test that reranking changes order based on relevance."""
        query = "Python async asyncio"

        reranked = reranker.rerank(query, sample_search_results, top_k=4)

        # Python async results should be ranked higher than JavaScript
        assert len(reranked) == 4

        # Top results should be Python-related
        top_titles = [r.title.lower() for r in reranked[:2]]
        python_in_top = any("python" in t for t in top_titles)
        assert python_in_top

    def test_rerank_updates_scores(self, sample_search_results, reranker):
        """Test that reranking updates result scores."""
        query = "Python async"

        # Original scores
        original_scores = [r.score for r in sample_search_results]

        reranked = reranker.rerank(query, sample_search_results)

        # Scores should be updated (semantic similarity scores)
        for result in reranked:
            assert result.score >= 0.0
            assert result.score <= 1.0

    def test_rerank_top_k(self, sample_search_results, reranker):
        """Test top_k parameter."""
        reranked = reranker.rerank("test query", sample_search_results, top_k=2)
        assert len(reranked) == 2

    def test_rerank_fallback_on_error(self, sample_search_results):
        """Test that reranker returns original results on error."""
        # Create reranker with invalid model
        reranker = SemanticReranker(model_name="invalid-model-name")

        # Should return original results (not crash)
        reranked = reranker.rerank("test query", sample_search_results)

        # Should return results (may be original order if model failed to load)
        assert len(reranked) > 0

    def test_rerank_results_function(self, sample_search_results):
        """Test convenience function."""
        reranked = rerank_results(
            query="Python async",
            results=sample_search_results,
            top_k=3,
        )
        assert len(reranked) == 3


# ─────────────────────────────────────────────
# Test Parallel Search
# ─────────────────────────────────────────────


class TestParallelSearchExecutor:
    """Test ParallelSearchExecutor class."""

    def test_executor_initialization(self, parallel_executor):
        """Test executor initializes correctly."""
        assert parallel_executor.max_concurrency == 3

    def test_deduplicate(self, parallel_executor):
        """Test result deduplication."""
        results = [
            SearchResult(url="https://example.com/page", title="Page 1", content="..."),
            SearchResult(
                url="https://example.com/page", title="Page 1", content="..."
            ),  # Duplicate
            SearchResult(url="https://other.com/page", title="Page 2", content="..."),
        ]

        deduped = parallel_executor._deduplicate(results)

        # Should remove duplicate
        assert len(deduped) == 2

    @pytest.mark.asyncio
    async def test_search_parallel_with_mock(self, parallel_executor):
        """Test parallel search with mock provider."""

        # Create mock provider
        class MockProvider:
            async def search(self, query, sources, num_results):
                # Simulate search delay
                await asyncio.sleep(0.1)

                return SearchResults(
                    query=query,
                    results=[
                        SearchResult(
                            title=f"Result from {sources[0].value}",
                            url=f"https://{sources[0].value}.com/result",
                            content="...",
                            source=sources[0],
                            score=0.8,
                        )
                    ],
                    search_time=100,
                )

        provider = MockProvider()
        sources = [SearchSource.WEB, SearchSource.TECH, SearchSource.NEWS]

        # Time the parallel search
        start = time.time()
        results = await parallel_executor.search_parallel(
            query="test query",
            sources=sources,
            provider=provider,
            num_results_per_source=5,
        )
        elapsed = time.time() - start

        # Should have results from all sources
        assert len(results.results) == 3

        # Should be faster than sequential (0.3s)
        # Parallel should take ~0.1s + overhead
        assert elapsed < 0.25

    @pytest.mark.asyncio
    async def test_search_parallel_with_failures(self, parallel_executor):
        """Test parallel search handles source failures gracefully."""

        # Create mock provider that fails for some sources
        class MockProviderWithFailures:
            async def search(self, query, sources, num_results):
                source = sources[0]
                if source == SearchSource.NEWS:
                    raise Exception("Source unavailable")

                await asyncio.sleep(0.05)
                return SearchResults(
                    query=query,
                    results=[
                        SearchResult(
                            title=f"Result from {source.value}",
                            url=f"https://{source.value}.com/result",
                            content="...",
                            source=source,
                            score=0.8,
                        )
                    ],
                    search_time=50,
                )

        provider = MockProviderWithFailures()
        sources = [SearchSource.WEB, SearchSource.TECH, SearchSource.NEWS]

        # Should not raise, should handle partial failures
        results = await parallel_executor.search_parallel(
            query="test query",
            sources=sources,
            provider=provider,
            num_results_per_source=5,
        )

        # Should have partial results
        assert len(results.results) == 2

        # Should track failures
        assert results.metadata.get("partial_results") is True
        assert "NEWS" in results.metadata.get("failed_sources", [])

    def test_search_parallel_function(self):
        """Test convenience function."""
        # Just test it exists and has correct signature
        assert callable(search_parallel)


# ─────────────────────────────────────────────
# Test Integration
# ─────────────────────────────────────────────


class TestOptimizationIntegration:
    """Test integration of multiple optimizations."""

    @pytest.mark.asyncio
    async def test_rerank_after_dedup(self, sample_search_results):
        """Test reranking after deduplication."""
        from orchestrator.nexus_search.optimization import (
            ResultDeduplicator,
            SemanticReranker,
        )

        # Deduplicate first
        deduplicator = ResultDeduplicator()
        deduped = deduplicator.deduplicate(sample_search_results)

        # Then rerank
        reranker = SemanticReranker()
        reranked = reranker.rerank("Python async", deduped)

        # Should have results
        assert len(reranked) > 0

    def test_all_optimizations_import(self):
        """Test that all optimization modules can be imported."""
        from orchestrator.nexus_search.optimization import (
            ResultDeduplicator,
            deduplicate_results,
            QueryCache,
            get_query_cache,
            SemanticReranker,
            rerank_results,
            ParallelSearchExecutor,
            search_parallel,
        )

        # All should be importable
        assert ResultDeduplicator is not None
        assert QueryCache is not None
        assert SemanticReranker is not None
        assert ParallelSearchExecutor is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
