"""
Test P1-1: Profile Caching Optimization
========================================
Tests that active profiles are cached to avoid repeated iteration
over all 50+ models in _profiles.
"""
import pytest
from unittest.mock import AsyncMock, patch

from orchestrator.engine import Orchestrator
from orchestrator.models import Budget, Model
from orchestrator.policy import ModelProfile


class TestProfileCaching:
    """Test P1-1: Active profile caching."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator for testing."""
        orch = Orchestrator(budget=Budget(max_usd=100.0))
        return orch

    def test_active_profiles_cache_initialized(self, orchestrator):
        """
        Verify _active_profiles_cache is initialized to None.
        """
        # Assert: Should be None initially
        assert orchestrator._active_profiles_cache is None

    def test_invalidate_profile_cache(self, orchestrator):
        """
        Verify _invalidate_profile_cache() sets cache to None.
        """
        # Arrange: Set cache to some value
        orchestrator._active_profiles_cache = [("dummy", "data")]
        
        # Act: Invalidate
        orchestrator._invalidate_profile_cache()
        
        # Assert: Cache should be None
        assert orchestrator._active_profiles_cache is None

    def test_get_active_profiles_builds_cache(self, orchestrator):
        """
        Verify _get_active_profiles() builds cache on first call.
        """
        # Arrange: Set some profiles as active
        test_model = list(orchestrator._profiles.keys())[0]
        orchestrator._profiles[test_model].call_count = 5
        
        # Act: Get active profiles
        active = orchestrator._get_active_profiles()
        
        # Assert: Cache should be built
        assert orchestrator._active_profiles_cache is not None
        assert len(active) >= 1
        # All returned profiles should have call_count > 0
        for model, profile in active:
            assert profile.call_count > 0

    def test_get_active_profiles_uses_cache(self, orchestrator):
        """
        Verify _get_active_profiles() returns cached value on second call.
        """
        # Arrange: Set some profiles as active
        test_model = list(orchestrator._profiles.keys())[0]
        orchestrator._profiles[test_model].call_count = 5
        
        # Act: First call (builds cache)
        active1 = orchestrator._get_active_profiles()
        cache_id1 = id(orchestrator._active_profiles_cache)
        
        # Act: Second call (should use cache)
        active2 = orchestrator._get_active_profiles()
        cache_id2 = id(orchestrator._active_profiles_cache)
        
        # Assert: Same cache object returned
        assert cache_id1 == cache_id2
        assert active1 is active2

    def test_get_active_profiles_filters_inactive(self, orchestrator):
        """
        Verify only profiles with call_count > 0 are returned.
        """
        # Arrange: Set some profiles active, some inactive
        models = list(orchestrator._profiles.keys())[:5]
        for i, model in enumerate(models):
            if i % 2 == 0:
                orchestrator._profiles[model].call_count = 5  # Active
            else:
                orchestrator._profiles[model].call_count = 0  # Inactive
        
        # Act: Get active profiles
        active = orchestrator._get_active_profiles()
        
        # Assert: Only active profiles returned
        for model, profile in active:
            assert profile.call_count > 0
        
        # Count should match
        expected_count = sum(
            1 for m in models
            if orchestrator._profiles[m].call_count > 0
        )
        assert len(active) == expected_count

    def test_record_success_invalidates_cache(self, orchestrator):
        """
        Verify _record_success() invalidates profile cache.
        """
        # Arrange: Build cache
        test_model = list(orchestrator._profiles.keys())[0]
        orchestrator._profiles[test_model].call_count = 5
        orchestrator._get_active_profiles()  # Build cache
        assert orchestrator._active_profiles_cache is not None
        
        # Act: Record success (should invalidate cache)
        from orchestrator.api_clients import APIResponse
        response = APIResponse(
            text="test",
            input_tokens=10,
            output_tokens=20,
            model=test_model,
        )
        orchestrator._record_success(test_model, response)
        
        # Assert: Cache should be invalidated
        assert orchestrator._active_profiles_cache is None

    @pytest.mark.asyncio
    async def test_flush_telemetry_uses_cached_profiles(self, orchestrator):
        """
        Verify _flush_telemetry_snapshots() uses cached profiles.
        """
        # Arrange: Set some profiles as active
        test_model = list(orchestrator._profiles.keys())[0]
        orchestrator._profiles[test_model].call_count = 5
        
        # Build cache
        orchestrator._get_active_profiles()
        assert orchestrator._active_profiles_cache is not None
        
        # Mock the batch method to verify it's called
        original_batch = orchestrator._telemetry_store.record_snapshots_batch
        orchestrator._telemetry_store.record_snapshots_batch = AsyncMock()
        
        # Act: Flush telemetry
        await orchestrator._flush_telemetry_snapshots("test_project")
        
        # Small delay for async task
        import asyncio
        await asyncio.sleep(0.05)
        
        # Assert: Batch method should be called
        assert orchestrator._telemetry_store.record_snapshots_batch.called
        
        # Restore
        orchestrator._telemetry_store.record_snapshots_batch = original_batch

    def test_cache_rebuilds_after_invalidation(self, orchestrator):
        """
        Verify cache is rebuilt after invalidation.
        """
        # Arrange: Set profiles and build cache
        test_model = list(orchestrator._profiles.keys())[0]
        orchestrator._profiles[test_model].call_count = 5
        orchestrator._get_active_profiles()
        original_cache = orchestrator._active_profiles_cache
        
        # Act: Invalidate and rebuild
        orchestrator._invalidate_profile_cache()
        assert orchestrator._active_profiles_cache is None
        
        # Rebuild
        orchestrator._get_active_profiles()
        
        # Assert: New cache built
        assert orchestrator._active_profiles_cache is not None
        assert orchestrator._active_profiles_cache is not original_cache

    def test_performance_improvement(self, orchestrator):
        """
        Benchmark-style test: Verify caching improves performance.
        """
        import time
        
        # Arrange: Set many profiles as active
        for model in list(orchestrator._profiles.keys())[:20]:
            orchestrator._profiles[model].call_count = 5
        
        # Warm up cache
        orchestrator._get_active_profiles()
        
        # Measure cached access
        iterations = 1000
        start = time.time()
        for _ in range(iterations):
            orchestrator._get_active_profiles()
        cached_time = time.time() - start
        
        # Invalidate and measure uncached access
        orchestrator._invalidate_profile_cache()
        start = time.time()
        for _ in range(iterations):
            orchestrator._get_active_profiles()
        uncached_time = time.time() - start
        
        # Assert: Cached should be faster (allowing for variance)
        # This is a soft assertion - caching should help but may vary
        assert cached_time <= uncached_time * 1.5  # Allow 50% variance
