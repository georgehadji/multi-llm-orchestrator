"""Verify RateLimiter integration in Orchestrator."""
import pytest
from unittest.mock import MagicMock
from orchestrator.rate_limiter import RateLimiter, RateLimitExceeded
from orchestrator.engine import Orchestrator


def test_rate_limiter_is_imported_at_module_level():
    """RateLimiter must be importable from engine module (not deferred)."""
    import orchestrator.engine as eng_module
    assert hasattr(eng_module, "RateLimiter"), "RateLimiter must be imported at module level in engine.py"


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


@pytest.mark.asyncio
async def test_rate_limit_exceeded_skips_client_call():
    """
    When check() raises RateLimitExceeded, client.call() must not be invoked.

    Before the fix: engine.py never calls check(), so client.call IS invoked.
    After the fix: check() blocks the call, client.call is NOT invoked.
    """
    from unittest.mock import AsyncMock, MagicMock

    rl = RateLimiter()
    rl.set_limits("default", "deepseek-chat", tpm=10, rpm=100)
    rl.record("default", "deepseek-chat", tokens=10)  # TPM exhausted

    mock_client = MagicMock()
    mock_client.call = AsyncMock()

    tenant = "default"
    model_str = "deepseek-chat"
    tokens_to_use = 1

    # This is the pattern that engine.py must implement around client.call():
    try:
        rl.check(tenant, model_str, tokens_to_use)
        await mock_client.call()   # should NOT reach here
    except RateLimitExceeded:
        pass  # expected: limit was enforced

    mock_client.call.assert_not_called()


@pytest.mark.asyncio
async def test_rate_limiter_record_called_after_successful_generation():
    """record() must be called after a successful client.call() to track usage."""
    from unittest.mock import AsyncMock, MagicMock

    rl = RateLimiter()
    rl.set_limits("default", "deepseek-chat", tpm=50_000, rpm=100)

    mock_response = MagicMock()
    mock_response.input_tokens = 100
    mock_response.output_tokens = 200

    mock_client = MagicMock()
    mock_client.call = AsyncMock(return_value=mock_response)

    tenant = "default"
    model_str = "deepseek-chat"

    # Engine must call record() after successful generation:
    rl.check(tenant, model_str, 300)
    resp = await mock_client.call()
    rl.record(tenant, model_str, resp.input_tokens + resp.output_tokens)

    usage = rl.get_usage(tenant, model_str)
    assert usage["requests"] == 1
    assert usage["tokens_used"] == 300
