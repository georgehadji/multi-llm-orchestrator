"""
Grok Rate Limiter — Tier-Based Rate Limiting
=============================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Tier-based rate limiter for xAI Grok API with spend tracking.

Features:
- Automatic tier progression based on cumulative spend
- RPM (requests per minute) limiting
- TPM (tokens per minute) limiting
- Spend tracking
- Async-safe with semaphore

Tiers (based on cumulative spend since Jan 1, 2026):
- Tier 1: $0 (default)
- Tier 2: $50+
- Tier 3: $200+
- Tier 4: $500+
- Tier 5: $1,000+
- Tier 6: $5,000+

Usage:
    from orchestrator.rate_limiter import GrokRateLimiter, RateLimiter
    
    limiter = GrokRateLimiter(api_key="xai-...")
    await limiter.acquire(tokens=1000)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import aiohttp

logger = logging.getLogger("orchestrator.rate_limiter")


# ─────────────────────────────────────────────
# Backward Compatibility Classes
# ─────────────────────────────────────────────

class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded."""
    
    def __init__(self, message: str, limit_type: str = "unknown"):
        super().__init__(message)
        self.limit_type = limit_type


class RateLimiter:
    """
    Backward-compatible rate limiter wrapper.
    
    Wraps GrokRateLimiter for backward compatibility.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self._limiter = GrokRateLimiter(api_key=api_key)
    
    async def acquire(self, tokens: int = 1000) -> bool:
        """Acquire tokens."""
        return await self._limiter.acquire(tokens=tokens)
    
    def record_spend(self, amount: float):
        """Record spend."""
        self._limiter.record_spend(amount)
    
    async def close(self):
        """Close limiter."""
        await self._limiter.close()


# ─────────────────────────────────────────────
# Main Implementation
# ─────────────────────────────────────────────


@dataclass
class TierLimits:
    """Rate limits for a specific tier."""
    rpm: int  # Requests per minute
    tpm: int  # Tokens per minute
    tpd: int  # Tokens per day (optional)
    
    def to_dict(self) -> Dict[str, int]:
        return {
            "rpm": self.rpm,
            "tpm": self.tpm,
            "tpd": self.tpd,
        }


@dataclass
class RateLimitState:
    """Current rate limit state."""
    current_rpm: int = 0
    current_tpm: int = 0
    current_tpd: int = 0
    last_reset: datetime = field(default_factory=datetime.now)
    cumulative_spend: float = 0.0
    current_tier: int = 1
    
    def reset_if_needed(self):
        """Reset counters if minute has passed."""
        now = datetime.now()
        if (now - self.last_reset).total_seconds() >= 60:
            self.current_rpm = 0
            self.current_tpm = 0
            self.last_reset = now


class GrokRateLimiter:
    """
    Tier-based rate limiter for xAI Grok API.
    
    Features:
    - Automatic tier progression based on spend
    - RPM and TPM limiting
    - Async-safe with semaphore
    - Spend tracking
    
    Usage:
        limiter = GrokRateLimiter(api_key="xai-...")
        await limiter.acquire(tokens=1000)
    """
    
    # Tier limits (estimated based on xAI documentation)
    TIER_LIMITS = {
        1: TierLimits(rpm=10, tpm=10_000, tpd=100_000),      # $0 spend
        2: TierLimits(rpm=60, tpm=100_000, tpd=1_000_000),   # $50+ spend
        3: TierLimits(rpm=120, tpm=500_000, tpd=5_000_000),  # $200+ spend
        4: TierLimits(rpm=300, tpm=1_000_000, tpd=10_000_000),  # $500+ spend
        5: TierLimits(rpm=600, tpm=2_000_000, tpd=20_000_000),  # $1,000+ spend
        6: TierLimits(rpm=1200, tpm=5_000_000, tpd=50_000_000),  # $5,000+ spend
    }
    
    # Spend thresholds for tier progression
    TIER_THRESHOLDS = {
        1: 0.0,
        2: 50.0,
        3: 200.0,
        4: 500.0,
        5: 1000.0,
        6: 5000.0,
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        initial_tier: int = 1,
        max_concurrency: int = 10,
    ):
        """
        Initialize Grok rate limiter.
        
        Args:
            api_key: xAI API key (for spend tracking)
            initial_tier: Starting tier (1-6)
            max_concurrency: Maximum concurrent requests
        """
        self.api_key = api_key or os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY")
        self.state = RateLimitState(current_tier=initial_tier)
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._lock = asyncio.Lock()
        self._session: Optional[aiohttp.ClientSession] = None
        
        # Metrics
        self.total_requests = 0
        self.total_tokens = 0
        self.total_wait_time = 0.0
        self.rate_limit_hits = 0
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            )
        return self._session
    
    async def close(self):
        """Close the rate limiter and cleanup resources."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
    
    def _get_current_limits(self) -> TierLimits:
        """Get current tier limits."""
        return self.TIER_LIMITS.get(self.state.current_tier, self.TIER_LIMITS[1])
    
    def _update_tier_from_spend(self):
        """Update tier based on cumulative spend."""
        spend = self.state.cumulative_spend
        
        # Find appropriate tier
        new_tier = 1
        for tier, threshold in sorted(self.TIER_THRESHOLDS.items(), key=lambda x: x[1]):
            if spend >= threshold:
                new_tier = tier
            else:
                break
        
        if new_tier != self.state.current_tier:
            logger.info(f"Tier upgraded: {self.state.current_tier} → {new_tier} (spend: ${spend:.2f})")
            self.state.current_tier = new_tier
    
    async def acquire(self, tokens: int = 1000, timeout: float = 60.0) -> bool:
        """
        Acquire permission to make a request with specified tokens.

        Args:
            tokens: Number of tokens for this request
            timeout: Maximum time to wait for acquisition (seconds)

        Returns:
            True if acquired, False if timeout
        """
        start_time = time.time()

        async with self._semaphore:
            while True:
                # BUG-NEW-001 FIX: never sleep while holding the lock — compute
                # the sleep duration inside the lock, then await outside so other
                # coroutines are not starved for the full window duration.
                # BUG-NEW-002 FIX: refresh `limits` every iteration so that a
                # tier upgrade triggered by `_update_tier_from_spend()` is
                # reflected in the very next RPM/TPM comparison.
                sleep_secs: float = 0.0
                async with self._lock:
                    self.state.reset_if_needed()
                    self._update_tier_from_spend()
                    limits = self._get_current_limits()  # refreshed every iteration

                    # Check RPM limit
                    if self.state.current_rpm >= limits.rpm:
                        wait_time = max(0.0, 60.0 - (datetime.now() - self.state.last_reset).total_seconds())
                        if wait_time > 0 and (time.time() - start_time) + wait_time <= timeout:
                            logger.debug(f"RPM limit hit, waiting {wait_time:.1f}s")
                            self.rate_limit_hits += 1
                            sleep_secs = wait_time
                        else:
                            logger.warning(f"RPM rate limit exceeded (tier {self.state.current_tier})")
                            return False

                    # Check TPM limit
                    elif self.state.current_tpm + tokens > limits.tpm:
                        wait_time = max(0.0, 60.0 - (datetime.now() - self.state.last_reset).total_seconds())
                        if wait_time > 0 and (time.time() - start_time) + wait_time <= timeout:
                            logger.debug(f"TPM limit hit, waiting {wait_time:.1f}s")
                            self.rate_limit_hits += 1
                            sleep_secs = wait_time
                        else:
                            logger.warning(f"TPM rate limit exceeded (tier {self.state.current_tier})")
                            return False

                    else:
                        # Acquire successful
                        self.state.current_rpm += 1
                        self.state.current_tpm += tokens
                        self.total_requests += 1
                        self.total_tokens += tokens

                        elapsed = time.time() - start_time
                        self.total_wait_time += elapsed

                        logger.debug(
                            f"Acquired: {tokens} tokens, tier={self.state.current_tier}, "
                            f"rpm={self.state.current_rpm}/{limits.rpm}, "
                            f"tpm={self.state.current_tpm}/{limits.tpm}"
                        )
                        return True

                # Lock released — sleep outside so other coroutines can progress
                if sleep_secs > 0:
                    await asyncio.sleep(sleep_secs)
    
    def record_spend(self, amount: float):
        """
        Record API spend for tier progression.
        
        Args:
            amount: Amount spent in USD
        """
        self.state.cumulative_spend += amount
        self._update_tier_from_spend()
        logger.debug(f"Recorded spend: ${amount:.2f}, cumulative: ${self.state.cumulative_spend:.2f}")
    
    async def fetch_current_spend(self) -> float:
        """
        Fetch current cumulative spend from xAI API.
        
        Returns:
            Current cumulative spend in USD
        """
        if not self.api_key:
            logger.warning("No API key configured, cannot fetch spend")
            return self.state.cumulative_spend
        
        try:
            session = await self._get_session()
            async with session.get("https://api.x.ai/v1/usage") as response:
                if response.status == 200:
                    data = await response.json()
                    spend = data.get("cumulative_spend", 0.0)
                    self.state.cumulative_spend = spend
                    self._update_tier_from_spend()
                    return spend
                else:
                    logger.warning(f"Failed to fetch spend: {response.status}")
                    return self.state.cumulative_spend
        except Exception as e:
            logger.error(f"Error fetching spend: {e}")
            return self.state.cumulative_spend
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get rate limiter statistics.
        
        Returns:
            Dictionary with stats
        """
        limits = self._get_current_limits()
        avg_wait = self.total_wait_time / self.total_requests if self.total_requests > 0 else 0.0
        
        return {
            "current_tier": self.state.current_tier,
            "cumulative_spend": self.state.cumulative_spend,
            "current_rpm": self.state.current_rpm,
            "max_rpm": limits.rpm,
            "current_tpm": self.state.current_tpm,
            "max_tpm": limits.tpm,
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "rate_limit_hits": self.rate_limit_hits,
            "avg_wait_time": avg_wait,
        }
    
    def reset_stats(self):
        """Reset statistics (not spend)."""
        self.total_requests = 0
        self.total_tokens = 0
        self.total_wait_time = 0.0
        self.rate_limit_hits = 0


# Global rate limiter instance
_limiter: Optional[GrokRateLimiter] = None


def get_rate_limiter(api_key: Optional[str] = None) -> GrokRateLimiter:
    """
    Get or create global rate limiter instance.
    
    Args:
        api_key: xAI API key
    
    Returns:
        GrokRateLimiter instance
    """
    global _limiter
    if _limiter is None:
        _limiter = GrokRateLimiter(api_key=api_key)
    return _limiter


async def close_rate_limiter() -> None:
    """Close global rate limiter."""
    global _limiter
    if _limiter:
        await _limiter.close()
        _limiter = None
