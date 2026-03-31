"""
Tests for Grok Rate Limiter
============================
Author: Georgios-Chrysovalantis Chatzivantsidis

Run: pytest tests/test_rate_limiter.py -v
"""

import pytest
import asyncio
import time

from orchestrator.rate_limiter import (
    GrokRateLimiter,
    TierLimits,
    RateLimitState,
    get_rate_limiter,
)

# ─────────────────────────────────────────────
# Test Fixtures
# ─────────────────────────────────────────────


@pytest.fixture
def rate_limiter():
    """Create GrokRateLimiter instance."""
    return GrokRateLimiter(api_key="test-key", initial_tier=1)


@pytest.fixture
def tier_limits():
    """Get tier limits for reference."""
    return GrokRateLimiter.TIER_LIMITS


# ─────────────────────────────────────────────
# Test TierLimits
# ─────────────────────────────────────────────


class TestTierLimits:
    """Test TierLimits dataclass."""

    def test_tier_limits_creation(self, tier_limits):
        """Test tier limits are defined."""
        assert 1 in tier_limits
        assert 6 in tier_limits

        # Tier 1 should have lowest limits
        assert tier_limits[1].rpm <= tier_limits[6].rpm
        assert tier_limits[1].tpm <= tier_limits[6].tpm

    def test_tier_limits_to_dict(self):
        """Test tier limits to_dict method."""
        limits = TierLimits(rpm=10, tpm=1000, tpd=10000)
        limits_dict = limits.to_dict()

        assert limits_dict["rpm"] == 10
        assert limits_dict["tpm"] == 1000
        assert limits_dict["tpd"] == 10000


# ─────────────────────────────────────────────
# Test RateLimitState
# ─────────────────────────────────────────────


class TestRateLimitState:
    """Test RateLimitState dataclass."""

    def test_state_initialization(self):
        """Test state initializes correctly."""
        state = RateLimitState()

        assert state.current_rpm == 0
        assert state.current_tpm == 0
        assert state.current_tier == 1
        assert state.cumulative_spend == 0.0

    def test_state_reset(self):
        """Test state reset functionality."""
        state = RateLimitState()
        state.current_rpm = 10
        state.current_tpm = 1000

        # Reset manually
        state.current_rpm = 0
        state.current_tpm = 0

        assert state.current_rpm == 0
        assert state.current_tpm == 0


# ─────────────────────────────────────────────
# Test GrokRateLimiter
# ─────────────────────────────────────────────


class TestGrokRateLimiter:
    """Test GrokRateLimiter class."""

    def test_limiter_initialization(self, rate_limiter):
        """Test limiter initializes correctly."""
        assert rate_limiter.state.current_tier == 1
        assert rate_limiter.api_key == "test-key"
        assert rate_limiter.total_requests == 0
        assert rate_limiter.total_tokens == 0

    def test_get_current_limits(self, rate_limiter, tier_limits):
        """Test getting current tier limits."""
        limits = rate_limiter._get_current_limits()

        assert limits.rpm == tier_limits[1].rpm
        assert limits.tpm == tier_limits[1].tpm

    def test_record_spend(self, rate_limiter):
        """Test recording spend."""
        rate_limiter.record_spend(50.0)

        assert rate_limiter.state.cumulative_spend == 50.0
        assert rate_limiter.state.current_tier == 2  # Should upgrade to tier 2

    def test_tier_progression(self, rate_limiter):
        """Test tier progression based on spend."""
        # Start at tier 1
        assert rate_limiter.state.current_tier == 1

        # Spend $50 → tier 2
        rate_limiter.record_spend(50.0)
        assert rate_limiter.state.current_tier == 2

        # Spend $150 more → tier 3
        rate_limiter.record_spend(150.0)
        assert rate_limiter.state.current_tier == 3

        # Spend $300 more → tier 4
        rate_limiter.record_spend(300.0)
        assert rate_limiter.state.current_tier == 4

    @pytest.mark.asyncio
    async def test_acquire_success(self, rate_limiter):
        """Test successful token acquisition."""
        acquired = await rate_limiter.acquire(tokens=1000)

        assert acquired is True
        assert rate_limiter.total_requests == 1
        assert rate_limiter.total_tokens == 1000

    @pytest.mark.asyncio
    async def test_acquire_multiple(self, rate_limiter):
        """Test multiple acquisitions."""
        for i in range(5):
            acquired = await rate_limiter.acquire(tokens=100)
            assert acquired is True

        assert rate_limiter.total_requests == 5
        assert rate_limiter.total_tokens == 500

    @pytest.mark.asyncio
    async def test_acquire_with_semaphore(self, rate_limiter):
        """Test acquisition respects semaphore."""
        # Create limiter with small semaphore
        limiter = GrokRateLimiter(api_key="test", max_concurrency=2)

        # Try to acquire multiple concurrently
        tasks = [limiter.acquire(tokens=100) for _ in range(5)]
        results = await asyncio.gather(*tasks)

        # All should succeed (semaphore limits concurrency, not total)
        assert all(results)
        assert limiter.total_requests == 5

    @pytest.mark.asyncio
    async def test_acquire_timeout(self):
        """Test acquisition timeout."""
        # Create limiter at tier 1 with very low limits
        limiter = GrokRateLimiter(api_key="test", initial_tier=1)
        limiter.TIER_LIMITS[1] = TierLimits(rpm=1, tpm=100, tpd=1000)

        # First acquire should succeed
        acquired = await limiter.acquire(tokens=100)
        assert acquired is True

        # Second should timeout (RPM limit)
        acquired = await limiter.acquire(tokens=100, timeout=0.1)
        assert acquired is False

    def test_get_stats(self, rate_limiter):
        """Test getting statistics."""
        # Make some requests
        asyncio.run(rate_limiter.acquire(tokens=1000))
        asyncio.run(rate_limiter.acquire(tokens=500))

        stats = rate_limiter.get_stats()

        assert "current_tier" in stats
        assert "cumulative_spend" in stats
        assert "total_requests" in stats
        assert "total_tokens" in stats
        assert stats["total_requests"] == 2
        assert stats["total_tokens"] == 1500

    def test_reset_stats(self, rate_limiter):
        """Test resetting statistics."""
        # Make some requests
        asyncio.run(rate_limiter.acquire(tokens=1000))

        # Reset stats
        rate_limiter.reset_stats()

        assert rate_limiter.total_requests == 0
        assert rate_limiter.total_tokens == 0
        # Spend should NOT be reset
        assert rate_limiter.state.cumulative_spend == 0.0

    @pytest.mark.asyncio
    async def test_close(self, rate_limiter):
        """Test closing limiter."""
        await rate_limiter.close()

        assert rate_limiter._session is None


# ─────────────────────────────────────────────
# Test Global Instance
# ─────────────────────────────────────────────


class TestGlobalInstance:
    """Test global rate limiter instance."""

    def test_get_rate_limiter(self):
        """Test getting global rate limiter."""
        limiter1 = get_rate_limiter(api_key="test-key-1")
        limiter2 = get_rate_limiter(api_key="test-key-2")

        # Should return same instance
        assert limiter1 is limiter2

    @pytest.mark.asyncio
    async def test_close_rate_limiter(self):
        """Test closing global rate limiter."""
        limiter = get_rate_limiter(api_key="test-key")
        await limiter.acquire(tokens=100)

        await close_rate_limiter()

        # Should be able to create new one
        new_limiter = get_rate_limiter(api_key="new-key")
        assert new_limiter is not None


# ─────────────────────────────────────────────
# Test Integration
# ─────────────────────────────────────────────


class TestIntegration:
    """Test rate limiter integration scenarios."""

    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test handling concurrent requests."""
        limiter = GrokRateLimiter(api_key="test", initial_tier=3)

        # Simulate 10 concurrent requests
        async def make_request():
            acquired = await limiter.acquire(tokens=1000)
            return acquired

        tasks = [make_request() for _ in range(10)]
        results = await asyncio.gather(*tasks)

        # All should succeed at tier 3
        assert all(results)
        assert limiter.total_requests == 10

    @pytest.mark.asyncio
    async def test_spend_tracking_integration(self):
        """Test spend tracking with tier progression."""
        limiter = GrokRateLimiter(api_key="test", initial_tier=1)

        # Simulate spending over time
        for spend_amount in [50, 150, 300, 500, 1000]:
            limiter.record_spend(spend_amount)

        # Should be at tier 5 ($2000 total spend)
        assert limiter.state.current_tier == 5
        assert limiter.state.cumulative_spend == 2000.0

        # Should have higher limits now
        limits = limiter._get_current_limits()
        assert limits.rpm > GrokRateLimiter.TIER_LIMITS[1].rpm

    def test_stats_accuracy(self, rate_limiter):
        """Test statistics accuracy."""
        # Make several requests
        for i in range(5):
            asyncio.run(rate_limiter.acquire(tokens=100 * (i + 1)))

        stats = rate_limiter.get_stats()

        # Total tokens should be 100+200+300+400+500 = 1500
        assert stats["total_tokens"] == 1500
        assert stats["total_requests"] == 5
        assert stats["rate_limit_hits"] == 0  # No limits hit


# Import for cleanup test
from orchestrator.rate_limiter import close_rate_limiter

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
