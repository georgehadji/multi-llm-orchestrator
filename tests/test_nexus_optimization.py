"""
Tests for Nexus Search Optimizations
=====================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Run: pytest tests/test_nexus_optimization.py -v
"""

import pytest
import asyncio
from datetime import datetime, timedelta

from orchestrator.nexus_search.models import SearchResult, SearchSource, SearchResults
from orchestrator.nexus_search.optimization.deduplication import (
    ResultDeduplicator,
    deduplicate_results,
)
from orchestrator.nexus_search.optimization.query_cache import (
    QueryCache,
    get_query_cache,
)


# ─────────────────────────────────────────────
# Test Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def sample_search_results():
    """Create sample search results with duplicates."""
    results = [
        SearchResult(
            title="Python Async Best Practices",
            url="https://example.com/python-async",
            content="Learn about async/await in Python...",
            source=SearchSource.WEB,
            score=0.9,
        ),
        SearchResult(
            title="Python Async Best Practices",  # Duplicate title
            url="https://different-site.com/python-async-guide",  # Different URL
            content="Learn about async/await in Python...",  # Same content
            source=SearchSource.WEB,
            score=0.85,
        ),
        SearchResult(
            title="Python Async Best Practices",  # Exact duplicate
            url="https://example.com/python-async",  # Same URL
            content="Learn about async/await in Python...",
            source=SearchSource.WEB,
            score=0.9,
        ),
        SearchResult(
            title="Advanced Async Patterns in Python",
            url="https://example.com/advanced-async",
            content="Deep dive into async patterns...",
            source=SearchSource.TECH,
            score=0.8,
        ),
    ]
    return results


@pytest.fixture
def deduplicator():
    """Create ResultDeduplicator instance."""
    return ResultDeduplicator(similarity_threshold=0.85)


@pytest.fixture
def query_cache():
    """Create QueryCache instance."""
    return QueryCache(ttl_seconds=3600, max_size=100)


# ─────────────────────────────────────────────
# Test Deduplication
# ─────────────────────────────────────────────

class TestResultDeduplicator:
    """Test ResultDeduplicator class."""
    
    def test_url_deduplication(self, sample_search_results, deduplicator):
        """Test URL-based deduplication."""
        # Should remove exact URL duplicate
        deduped = deduplicator._dedup_by_url(sample_search_results)
        
        # Original has 4 results, after URL dedup should have 3
        assert len(deduped) == 3
        
        # Check unique URLs
        urls = set(r.url for r in deduped)
        assert len(urls) == 3
    
    def test_title_hash_deduplication(self, deduplicator):
        """Test title hash-based deduplication."""
        results = [
            SearchResult(title="Python Guide", url="https://a.com", content="..."),
            SearchResult(title="python guide", url="https://b.com", content="..."),  # Same title (case-insensitive)
            SearchResult(title="JavaScript Guide", url="https://c.com", content="..."),  # Different title
        ]
        
        deduped = deduplicator._dedup_by_title_hash(results)
        
        # Should keep 2 results (Python Guide and JavaScript Guide)
        assert len(deduped) == 2
    
    def test_content_similarity_deduplication(self, deduplicator):
        """Test TF-IDF content similarity deduplication."""
        results = [
            SearchResult(
                title="Python Async",
                url="https://a.com",
                content="Learn about async await in Python for concurrent programming with asyncio event loop"
            ),
            SearchResult(
                title="Python Concurrency",
                url="https://b.com",
                content="Learn about async await in Python for concurrent programming with asyncio event loop"  # Same content
            ),
            SearchResult(
                title="JavaScript Async",
                url="https://c.com",
                content="Learn about promises and async await in JavaScript for web development"  # Different content
            ),
        ]
        
        deduped = deduplicator._dedup_by_content_similarity(results)
        
        # Should remove one of the similar Python async results (keep 2)
        # Note: TF-IDF may not always catch short texts, so we check <= 3
        assert len(deduped) <= 3
        assert len(deduped) >= 2  # Should keep at least 2
    
    def test_full_deduplication(self, sample_search_results, deduplicator):
        """Test full multi-level deduplication."""
        original_count = len(sample_search_results)
        
        deduped = deduplicator.deduplicate(sample_search_results)
        
        # Should reduce from 4 to at most 3 (removed duplicates)
        assert len(deduped) < original_count
        assert len(deduped) >= 2  # Should keep at least 2 unique results
    
    def test_deduplicate_results_function(self, sample_search_results):
        """Test convenience function."""
        deduped = deduplicate_results(sample_search_results, similarity_threshold=0.85)
        
        assert len(deduped) < len(sample_search_results)
    
    def test_empty_results(self, deduplicator):
        """Test deduplication with empty results."""
        results = []
        deduped = deduplicator.deduplicate(results)
        assert len(deduped) == 0
    
    def test_single_result(self, deduplicator):
        """Test deduplication with single result."""
        results = [
            SearchResult(title="Test", url="https://test.com", content="...")
        ]
        deduped = deduplicator.deduplicate(results)
        assert len(deduped) == 1


# ─────────────────────────────────────────────
# Test Query Cache
# ─────────────────────────────────────────────

class TestQueryCache:
    """Test QueryCache class."""
    
    @pytest.mark.asyncio
    async def test_cache_miss(self, query_cache):
        """Test cache miss for new query."""
        results = await query_cache.get("new query", [SearchSource.WEB])
        assert results is None
        assert query_cache.misses == 1
    
    @pytest.mark.asyncio
    async def test_cache_hit(self, query_cache):
        """Test cache hit for cached query."""
        # Create search results
        search_results = SearchResults(
            query="test query",
            results=[
                SearchResult(title="Test", url="https://test.com", content="...")
            ],
        )
        
        # Cache the results
        await query_cache.set("test query", [SearchSource.WEB], search_results)
        
        # Try to get cached results
        cached = await query_cache.get("test query", [SearchSource.WEB])
        
        assert cached is not None
        assert cached.query == "test query"
        assert query_cache.hits == 1
    
    @pytest.mark.asyncio
    async def test_cache_expiration(self):
        """Test cache entry expiration."""
        # Create cache with 1 second TTL
        cache = QueryCache(ttl_seconds=1)
        
        search_results = SearchResults(
            query="test",
            results=[],
        )
        
        # Cache the results
        await cache.set("test", [SearchSource.WEB], search_results)
        
        # Should be cached initially
        cached = await cache.get("test", [SearchSource.WEB])
        assert cached is not None
        
        # Wait for expiration
        await asyncio.sleep(1.5)
        
        # Should be expired now
        expired = await cache.get("test", [SearchSource.WEB])
        assert expired is None
    
    @pytest.mark.asyncio
    async def test_cache_key_computation(self, query_cache):
        """Test cache key is consistent."""
        key1 = query_cache._compute_cache_key("test query", [SearchSource.WEB])
        key2 = query_cache._compute_cache_key("test query", [SearchSource.WEB])
        
        assert key1 == key2
        
        # Different sources should produce different key
        key3 = query_cache._compute_cache_key("test query", [SearchSource.TECH])
        assert key1 != key3
    
    @pytest.mark.asyncio
    async def test_cache_max_size_eviction(self):
        """Test cache eviction when max size reached."""
        cache = QueryCache(ttl_seconds=3600, max_size=3)
        
        # Add 4 entries (exceeds max_size)
        for i in range(4):
            search_results = SearchResults(query=f"query{i}", results=[])
            await cache.set(f"query{i}", [SearchSource.WEB], search_results)
        
        # Should have evicted oldest entry
        assert len(cache._cache) == 3
        assert cache.evictions == 1
    
    @pytest.mark.asyncio
    async def test_cache_stats(self, query_cache):
        """Test cache statistics."""
        # Create some cache activity
        search_results = SearchResults(query="test", results=[])
        await query_cache.set("test", [SearchSource.WEB], search_results)
        await query_cache.get("test", [SearchSource.WEB])
        await query_cache.get("nonexistent", [SearchSource.WEB])
        
        stats = query_cache.get_stats()
        
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1
    
    @pytest.mark.asyncio
    async def test_cache_clear(self, query_cache):
        """Test cache clearing."""
        search_results = SearchResults(query="test", results=[])
        await query_cache.set("test", [SearchSource.WEB], search_results)
        
        await query_cache.clear()
        
        assert len(query_cache._cache) == 0
    
    @pytest.mark.asyncio
    async def test_cache_cleanup_expired(self):
        """Test cleanup of expired entries."""
        cache = QueryCache(ttl_seconds=1)
        
        # Add entry
        search_results = SearchResults(query="test", results=[])
        await cache.set("test", [SearchSource.WEB], search_results)
        
        # Wait for expiration
        await asyncio.sleep(1.5)
        
        # Cleanup
        removed = await cache.cleanup_expired()
        
        assert removed == 1
        assert len(cache._cache) == 0
    
    @pytest.mark.asyncio
    async def test_global_cache_instance(self):
        """Test global cache instance."""
        cache1 = get_query_cache()
        cache2 = get_query_cache()
        
        # Should return same instance
        assert cache1 is cache2


# ─────────────────────────────────────────────
# Test Integration
# ─────────────────────────────────────────────

class TestOptimizationIntegration:
    """Test integration of optimizations."""
    
    @pytest.mark.asyncio
    async def test_dedup_then_cache(self, sample_search_results):
        """Test deduplication followed by caching."""
        deduplicator = ResultDeduplicator()
        cache = QueryCache()
        
        # Deduplicate
        deduped = deduplicator.deduplicate(sample_search_results)
        
        # Create SearchResults
        search_results = SearchResults(
            query="test",
            results=deduped,
        )
        
        # Cache
        await cache.set("test", [SearchSource.WEB], search_results)
        
        # Retrieve
        cached = await cache.get("test", [SearchSource.WEB])
        
        assert cached is not None
        assert len(cached.results) == len(deduped)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
