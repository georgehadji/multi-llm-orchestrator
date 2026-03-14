"""Tests for TPM/RPM sliding window rate limiter."""
import time
import pytest
from orchestrator.rate_limiter import RateLimiter, RateLimitExceeded


def test_no_limits_set_always_passes():
    """Without set_limits(), check() never raises."""
    rl = RateLimiter()
    rl.check("tenant1", "deepseek-chat", tokens=1_000_000)  # no limits → no raise


def test_tpm_limit_raises_when_exceeded():
    rl = RateLimiter()
    rl.set_limits("tenant1", "deepseek-chat", tpm=100, rpm=1000)
    rl.record("tenant1", "deepseek-chat", tokens=90)   # 90 used
    with pytest.raises(RateLimitExceeded) as exc_info:
        rl.check("tenant1", "deepseek-chat", tokens=20)  # 90+20=110 > 100
    assert exc_info.value.limit_type == "tpm"
    assert exc_info.value.retry_after > 0


def test_rpm_limit_raises_when_exceeded():
    rl = RateLimiter()
    rl.set_limits("tenant1", "deepseek-chat", tpm=1_000_000, rpm=2)
    rl.record("tenant1", "deepseek-chat", tokens=1)
    rl.record("tenant1", "deepseek-chat", tokens=1)
    with pytest.raises(RateLimitExceeded) as exc_info:
        rl.check("tenant1", "deepseek-chat", tokens=1)  # 3rd request > rpm=2
    assert exc_info.value.limit_type == "rpm"


def test_check_passes_when_under_limit():
    rl = RateLimiter()
    rl.set_limits("tenant1", "deepseek-chat", tpm=1000, rpm=10)
    rl.record("tenant1", "deepseek-chat", tokens=500)
    rl.check("tenant1", "deepseek-chat", tokens=499)  # 500+499=999 < 1000 → no raise


def test_sliding_window_evicts_old_entries():
    rl = RateLimiter()
    rl.set_limits("tenant1", "deepseek-chat", tpm=100, rpm=10)

    # Manually inject a stale entry (61 seconds ago)
    key = ("tenant1", "deepseek-chat")
    rl._windows[key].append((time.monotonic() - 61, 90))  # stale

    # Should not count stale entry; 10 tokens is fine
    rl.check("tenant1", "deepseek-chat", tokens=10)  # no raise


def test_get_usage_returns_current_window_stats():
    rl = RateLimiter()
    rl.set_limits("tenant1", "deepseek-chat", tpm=1000, rpm=10)
    rl.record("tenant1", "deepseek-chat", tokens=300)
    rl.record("tenant1", "deepseek-chat", tokens=200)

    usage = rl.get_usage("tenant1", "deepseek-chat")
    assert usage["tokens_used"] == 500
    assert usage["requests"] == 2


def test_separate_tenants_have_independent_limits():
    rl = RateLimiter()
    rl.set_limits("tenant_a", "deepseek-chat", tpm=100, rpm=10)
    rl.set_limits("tenant_b", "deepseek-chat", tpm=100, rpm=10)
    rl.record("tenant_a", "deepseek-chat", tokens=90)
    # tenant_b unaffected — should not raise
    rl.check("tenant_b", "deepseek-chat", tokens=90)
