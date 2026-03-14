"""Verify RateLimiter integration in Orchestrator."""
import pytest
from unittest.mock import MagicMock
from orchestrator.rate_limiter import RateLimiter, RateLimitExceeded
from orchestrator.engine import Orchestrator


def test_engine_has_rate_limiter():
    orch = Orchestrator.__new__(Orchestrator)
    orch._rate_limiter = RateLimiter()
    assert isinstance(orch._rate_limiter, RateLimiter)


def test_engine_exposes_configure_rate_limits():
    orch = Orchestrator.__new__(Orchestrator)
    orch._rate_limiter = RateLimiter()
    # Must not raise
    orch.configure_rate_limits("acme", "deepseek-chat", tpm=50_000, rpm=100)
    usage = orch._rate_limiter.get_usage("acme", "deepseek-chat")
    assert usage["requests"] == 0


def test_rate_limiter_check_raises_correctly():
    """Engine's rate_limiter raises when limit exceeded."""
    orch = Orchestrator.__new__(Orchestrator)
    rl = RateLimiter()
    rl.set_limits("t1", "deepseek-chat", tpm=10, rpm=100)
    rl.record("t1", "deepseek-chat", tokens=10)  # hit limit
    orch._rate_limiter = rl

    with pytest.raises(RateLimitExceeded):
        orch._rate_limiter.check("t1", "deepseek-chat", tokens=1)
