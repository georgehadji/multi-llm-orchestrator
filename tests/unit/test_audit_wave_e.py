"""Wave E of the 2026-10-29 audit remediation plan: routing and the quality loop."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.models import Model

pytestmark = pytest.mark.unit


def _client():
    from orchestrator.infrastructure.llm_client import UnifiedClient

    with patch.object(UnifiedClient, "_init_clients", lambda self: None):
        client = UnifiedClient(cache=MagicMock(get=AsyncMock(return_value=None), put=AsyncMock()))
    client._get_client_for_model = AsyncMock(return_value=object())
    return client


def _ok_response():
    from types import SimpleNamespace

    return SimpleNamespace(text="ok", input_tokens=1, output_tokens=1, cost_usd=0.0, latency_ms=0.0)


# ── E1: one breaker per model, cancellation is not a failure ────────────────


class TestPerModelBreaker:
    @pytest.mark.asyncio
    async def test_failures_on_one_model_do_not_block_another(self) -> None:
        from orchestrator.circuit_breaker import CircuitBreakerOpen

        client = _client()

        async def dispatch(*, model_id, **_):
            if model_id == Model.GPT_4O_MINI.value:
                raise RuntimeError("provider down")
            return _ok_response()

        client._dispatch = dispatch
        for _ in range(5):
            with pytest.raises(RuntimeError):
                await client.call(Model.GPT_4O_MINI, "p", bypass_cache=True)
        with pytest.raises(CircuitBreakerOpen):
            await client.call(Model.GPT_4O_MINI, "p", bypass_cache=True)

        resp = await client.call(Model.GPT_4O, "p", bypass_cache=True)
        assert resp.text == "ok"

    @pytest.mark.asyncio
    async def test_cancellation_is_not_counted_as_failure(self) -> None:
        from orchestrator.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(name="t", failure_threshold=1, reset_timeout=60.0)
        with pytest.raises(asyncio.CancelledError):
            async with cb.context():
                raise asyncio.CancelledError
        assert cb.total_failures == 0
        assert cb.is_closed
