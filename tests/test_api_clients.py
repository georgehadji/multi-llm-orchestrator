"""
Unit tests for orchestrator/api_clients.py (UnifiedClient).
All tests use unittest.mock — no real API calls are made.
"""
import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from orchestrator.api_clients import APIResponse, UnifiedClient, _is_rate_limit_error
from orchestrator.models import Model, estimate_cost


# ─────────────────────────────────────────────────────────────────
# TestAPIResponse
# ─────────────────────────────────────────────────────────────────

class TestAPIResponse:
    def test_cached_response_has_zero_cost(self):
        r = APIResponse("hello", 100, 50, Model.GPT_4O_MINI, cached=True)
        assert r.cost_usd == 0.0

    def test_non_cached_response_has_positive_cost(self):
        r = APIResponse("hello", 1000, 500, Model.GPT_4O_MINI, cached=False)
        expected = estimate_cost(Model.GPT_4O_MINI, 1000, 500)
        assert r.cost_usd == pytest.approx(expected)

    def test_default_latency_is_zero(self):
        r = APIResponse("x", 10, 5, Model.GPT_4O_MINI)
        assert r.latency_ms == 0.0

    def test_text_stored_correctly(self):
        r = APIResponse("test output", 0, 0, Model.GPT_4O_MINI)
        assert r.text == "test output"


# ─────────────────────────────────────────────────────────────────
# TestRateLimitDetection
# ─────────────────────────────────────────────────────────────────

class TestRateLimitDetection:
    @pytest.mark.parametrize("msg", [
        "rate_limit exceeded",
        "Rate Limit hit",
        "429 Too Many Requests",
        "resource_exhausted",
        "quota exceeded",
        "model is overloaded",
    ])
    def test_detects_rate_limit_phrases(self, msg):
        assert _is_rate_limit_error(Exception(msg)) is True

    @pytest.mark.parametrize("msg", [
        "connection refused",
        "invalid api key",
        "model not found",
        "internal server error",
    ])
    def test_ignores_non_rate_limit_errors(self, msg):
        assert _is_rate_limit_error(Exception(msg)) is False


# ─────────────────────────────────────────────────────────────────
# TestIsAvailable
# ─────────────────────────────────────────────────────────────────

class TestIsAvailable:
    def test_returns_true_when_provider_client_present(self, tmp_path):
        client = _make_client(tmp_path, MagicMock())
        # GPT_4O_MINI → provider "openai" → client injected → True
        assert client.is_available(Model.GPT_4O_MINI) is True

    def test_returns_false_when_no_clients(self, tmp_path):
        client = _make_client(tmp_path)  # no mock injected
        assert client.is_available(Model.GPT_4O_MINI) is False


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _make_client(tmp_path: Path, provider_mock=None) -> UnifiedClient:
    """
    Build a UnifiedClient and bypass _init_clients by directly injecting
    the mock into _clients.  GPT_4O_MINI dispatches via 'openai', so the
    mock lives under that key.  Tests that need a different provider can
    call client._clients[provider] = mock directly after construction.
    """
    from orchestrator.cache import DiskCache
    cache = DiskCache(tmp_path / "cache.db")

    # Construct without going through _init_clients to keep tests isolated
    client = object.__new__(UnifiedClient)
    client.cache = cache
    client.semaphore = asyncio.Semaphore(2)
    client._connect_timeout = UnifiedClient.DEFAULT_CONNECT_TIMEOUT
    client._read_timeout = UnifiedClient.DEFAULT_READ_TIMEOUT
    client._clients = {}
    if provider_mock is not None:
        # GPT_4O_MINI → provider "openai"; inject mock under that key
        client._clients["openai"] = provider_mock
    return client


def _fake_openai_response(text: str = "ok", in_tok: int = 10, out_tok: int = 5):
    """Build a mock chat completion response."""
    choice = MagicMock()
    choice.message.content = text
    usage = MagicMock()
    usage.prompt_tokens = in_tok
    usage.completion_tokens = out_tok

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


# ─────────────────────────────────────────────────────────────────
# TestCacheBehavior
# ─────────────────────────────────────────────────────────────────

class TestCacheBehavior:
    @pytest.mark.asyncio
    async def test_cache_hit_skips_dispatch(self, tmp_path):
        or_client = AsyncMock()
        client = _make_client(tmp_path, or_client)

        # Pre-populate cache
        await client.cache.put(
            Model.GPT_4O_MINI.value, "hello", 100,
            "cached response", 10, 5, "", 0.3
        )

        result = await client.call(Model.GPT_4O_MINI, "hello", max_tokens=100)

        assert result.cached is True
        assert result.text == "cached response"
        or_client.chat.completions.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_bypass_cache_forces_fresh_call(self, tmp_path):
        or_client = AsyncMock()
        or_client.chat.completions.create = AsyncMock(
            return_value=_fake_openai_response("fresh")
        )
        client = _make_client(tmp_path, or_client)

        # Pre-populate cache
        await client.cache.put(
            Model.GPT_4O_MINI.value, "hello", 100,
            "cached response", 10, 5, "", 0.3
        )

        result = await client.call(
            Model.GPT_4O_MINI, "hello", max_tokens=100, bypass_cache=True
        )

        assert result.cached is False
        assert result.text == "fresh"
        or_client.chat.completions.create.assert_called_once()


# ─────────────────────────────────────────────────────────────────
# TestSemaphoreLimitsConcurrency
# ─────────────────────────────────────────────────────────────────

class TestSemaphoreLimitsConcurrency:
    @pytest.mark.asyncio
    async def test_peak_concurrent_calls_bounded(self, tmp_path):
        max_concurrency = 2
        peak = [0]
        current = [0]

        async def slow_dispatch(*args, **kwargs):
            current[0] += 1
            peak[0] = max(peak[0], current[0])
            await asyncio.sleep(0.05)
            current[0] -= 1
            return _fake_openai_response()

        or_client = AsyncMock()
        or_client.chat.completions.create = slow_dispatch
        client = _make_client(tmp_path, or_client)
        client.semaphore = asyncio.Semaphore(max_concurrency)

        tasks = [
            client.call(Model.GPT_4O_MINI, f"q{i}", bypass_cache=True)
            for i in range(6)
        ]
        await asyncio.gather(*tasks)

        assert peak[0] <= max_concurrency


# ─────────────────────────────────────────────────────────────────
# TestRetryLogic
# ─────────────────────────────────────────────────────────────────

class TestRetryLogic:
    @pytest.mark.asyncio
    async def test_retries_on_generic_error_and_succeeds(self, tmp_path):
        call_count = [0]

        async def flaky(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] < 2:
                raise RuntimeError("transient error")
            return _fake_openai_response("retry worked")

        or_client = AsyncMock()
        or_client.chat.completions.create = flaky
        client = _make_client(tmp_path, or_client)

        result = await client.call(Model.GPT_4O_MINI, "test", bypass_cache=True, retries=2)
        assert result.text == "retry worked"
        assert call_count[0] == 2

    @pytest.mark.asyncio
    async def test_raises_after_max_retries(self, tmp_path):
        or_client = AsyncMock()
        or_client.chat.completions.create = AsyncMock(
            side_effect=RuntimeError("always fails")
        )
        client = _make_client(tmp_path, or_client)

        with pytest.raises(RuntimeError, match="always fails"):
            await client.call(Model.GPT_4O_MINI, "test", bypass_cache=True, retries=1)

    @pytest.mark.asyncio
    async def test_rate_limit_triggers_backoff(self, tmp_path):
        call_count = [0]

        async def rate_limited(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] < 2:
                raise RuntimeError("429 Too Many Requests")
            return _fake_openai_response("ok after backoff")

        or_client = AsyncMock()
        or_client.chat.completions.create = rate_limited
        client = _make_client(tmp_path, or_client)

        with patch("orchestrator.api_clients.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await client.call(
                Model.GPT_4O_MINI, "test", bypass_cache=True, retries=2
            )

        assert result.text == "ok after backoff"
        mock_sleep.assert_called_once()  # backoff sleep happened


# ─────────────────────────────────────────────────────────────────
# TestTimeoutHandling
# ─────────────────────────────────────────────────────────────────

class TestTimeoutHandling:
    @pytest.mark.asyncio
    async def test_timeout_raises_after_deadline(self, tmp_path):
        async def very_slow(*args, **kwargs):
            await asyncio.sleep(10)
            return _fake_openai_response()

        or_client = AsyncMock()
        or_client.chat.completions.create = very_slow
        client = _make_client(tmp_path, or_client)

        with pytest.raises((TimeoutError, asyncio.TimeoutError)):
            await client.call(
                Model.GPT_4O_MINI, "test", bypass_cache=True,
                timeout=1, retries=0
            )


# ─────────────────────────────────────────────────────────────────
# TestDispatchRouting
# ─────────────────────────────────────────────────────────────────

class TestDispatchRouting:
    @pytest.mark.asyncio
    async def test_standard_model_calls_chat_completions(self, tmp_path):
        or_client = AsyncMock()
        or_client.chat.completions.create = AsyncMock(
            return_value=_fake_openai_response("standard")
        )
        client = _make_client(tmp_path, or_client)

        result = await client.call(Model.GPT_4O_MINI, "hi", bypass_cache=True)
        assert result.text == "standard"
        or_client.chat.completions.create.assert_called_once()

    def test_is_reasoning_model_detects_o_series(self, tmp_path):
        client = _make_client(tmp_path)
        # Create mock models with known reasoning values
        for val in ["o1", "o3", "o3-mini", "o4-mini", "deepseek-reasoner"]:
            m = MagicMock()
            m.value = val
            assert client._is_reasoning_model(m) is True

    def test_is_reasoning_model_false_for_standard(self, tmp_path):
        client = _make_client(tmp_path)
        m = MagicMock()
        m.value = "gpt-4o-mini"
        assert client._is_reasoning_model(m) is False
