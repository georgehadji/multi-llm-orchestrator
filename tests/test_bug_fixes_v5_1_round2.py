"""
Regression tests for bugs fixed in v5.1 round-2.

BUG-004  cost.py     BudgetHierarchy.remaining(level="team") ignored _team_reserved,
                     allowing callers to see inflated available budget.
BUG-005  rate_limiter.py  TOCTOU between check() and record(): concurrent async
                     callers both passed the check before either recorded.
OPENAI-T api_clients.py   temperature was forwarded to OpenAI models, which fix it
                     at 1 and reject the parameter.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# BUG-004 — BudgetHierarchy.remaining(level="team") ignores pending reservations
# ---------------------------------------------------------------------------

from orchestrator.cost import BudgetHierarchy


class TestTeamRemainingIncludesReservations:
    """remaining(level='team') must deduct pending reservations."""

    def _h(self) -> BudgetHierarchy:
        return BudgetHierarchy(org_max_usd=200.0, team_budgets={"eng": 100.0})

    def test_remaining_team_reflects_reservation(self):
        """After can_afford_job(), remaining(team) must show reduced budget."""
        h = self._h()
        h.can_afford_job("job-1", "eng", 60.0)

        remaining = h.remaining("team", key="eng")
        assert remaining == pytest.approx(
            40.0
        ), f"Expected 40.0 after reserving 60.0, got {remaining}"

    def test_remaining_team_restored_after_release(self):
        """After release_reservation(), remaining(team) must be fully restored."""
        h = self._h()
        h.can_afford_job("job-2", "eng", 60.0)
        h.release_reservation("job-2", "eng")

        remaining = h.remaining("team", key="eng")
        assert remaining == pytest.approx(
            100.0
        ), "Team budget must be fully restored after reservation release"

    def test_remaining_team_reflects_charge(self):
        """After charge_job(), remaining(team) must reflect actual spend."""
        h = self._h()
        h.can_afford_job("job-3", "eng", 40.0)
        h.charge_job("job-3", "eng", 40.0)

        remaining = h.remaining("team", key="eng")
        assert remaining == pytest.approx(60.0)

    def test_two_concurrent_reservations_visible(self):
        """Two reservations must both be deducted from remaining(team)."""
        h = self._h()
        h.can_afford_job("job-4", "eng", 30.0)
        h.can_afford_job("job-5", "eng", 20.0)

        remaining = h.remaining("team", key="eng")
        assert remaining == pytest.approx(
            50.0
        ), f"Expected 50.0 after two reservations totalling 50.0, got {remaining}"


# ---------------------------------------------------------------------------
# BUG-005 — RateLimiter TOCTOU: concurrent check() callers bypass the limit
# ---------------------------------------------------------------------------

from orchestrator.rate_limiter import RateLimiter, RateLimitExceeded


class TestRateLimiterNoConcurrentBypass:
    """Second check() must see the in-flight reservation made by the first."""

    def test_second_check_sees_inflight_tpm(self):
        """Two concurrent callers must not both pass a tpm limit."""
        rl = RateLimiter()
        rl.set_limits("tenant", "model", tpm=100, rpm=1000)

        # First caller reserves 60 tokens (passes, not yet recorded)
        rl.check("tenant", "model", 60)

        # Second caller wants 50 — 60 in-flight + 50 > 100 → must be blocked
        with pytest.raises(RateLimitExceeded) as exc_info:
            rl.check("tenant", "model", 50)
        assert exc_info.value.limit_type == "tpm"

    def test_second_check_sees_inflight_rpm(self):
        """Two concurrent callers must not both pass an rpm=1 limit."""
        rl = RateLimiter()
        rl.set_limits("tenant", "model", tpm=1_000_000, rpm=1)

        # First caller reserves one request slot
        rl.check("tenant", "model", 10)

        # Second caller — rpm=1 already consumed by in-flight → must be blocked
        with pytest.raises(RateLimitExceeded) as exc_info:
            rl.check("tenant", "model", 10)
        assert exc_info.value.limit_type == "rpm"

    def test_release_restores_inflight_slot(self):
        """release() must free the in-flight slot so the next caller can proceed."""
        rl = RateLimiter()
        rl.set_limits("tenant", "model", tpm=100, rpm=1000)

        rl.check("tenant", "model", 60)  # reserves 60 in-flight

        # Simulate API call failure — release the reservation
        rl.release("tenant", "model", 60)

        # Now a fresh caller with 60 tokens must pass again
        rl.check("tenant", "model", 60)  # no raise

    @pytest.mark.asyncio
    async def test_concurrent_asyncio_callers_respect_limit(self):
        """Two async tasks must not both bypass tpm=100."""
        rl = RateLimiter()
        rl.set_limits("acme", "deepseek-chat", tpm=100, rpm=1000)

        passed: list[str] = []
        blocked: list[str] = []

        async def try_call(name: str, tokens: int) -> None:
            try:
                rl.check("acme", "deepseek-chat", tokens)
                await asyncio.sleep(0)  # yield — other coroutine runs here
                rl.record("acme", "deepseek-chat", tokens)
                passed.append(name)
            except RateLimitExceeded:
                blocked.append(name)

        await asyncio.gather(
            try_call("A", 70),
            try_call("B", 70),
        )

        # Exactly one must pass (70) and one must be blocked (70+70=140 > 100)
        assert len(passed) == 1, f"Expected 1 to pass, got: {passed}"
        assert len(blocked) == 1, f"Expected 1 to be blocked, got: {blocked}"

    def test_record_after_check_clears_inflight(self):
        """record() must settle the in-flight reservation so limits reset correctly."""
        rl = RateLimiter()
        rl.set_limits("tenant", "model", tpm=100, rpm=1000)

        rl.check("tenant", "model", 60)  # reserves 60 in-flight
        rl.record("tenant", "model", 60)  # settles — in-flight goes to window

        # Now 60 tokens are in the window; 40 more should pass
        rl.check("tenant", "model", 40)  # no raise — 60+40=100 == limit
        with pytest.raises(RateLimitExceeded):
            rl.check("tenant", "model", 1)  # now 60+40 in-flight/window + 1 > 100


# ---------------------------------------------------------------------------
# OPENAI-T — temperature must NOT be forwarded to OpenAI models
# ---------------------------------------------------------------------------


class TestOpenAINoTemperature:
    """_call_openai must omit the temperature parameter."""

    @pytest.mark.asyncio
    async def test_openai_call_omits_temperature(self):
        """completions.create must be called without a temperature kwarg."""
        from orchestrator.api_clients import UnifiedClient
        from orchestrator.models import Model

        clients = UnifiedClient.__new__(UnifiedClient)
        clients.semaphore = asyncio.Semaphore(10)

        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(content="hello"))]
        mock_completion.usage = MagicMock(prompt_tokens=10, completion_tokens=5)

        mock_openai = AsyncMock()
        mock_openai.chat.completions.create = AsyncMock(return_value=mock_completion)
        clients._clients = {"openai": mock_openai}

        await clients._call_openai(Model.GPT_4O, "prompt", "system", 100, 0.7)

        call_kwargs = mock_openai.chat.completions.create.call_args.kwargs
        assert (
            "temperature" not in call_kwargs
        ), f"temperature must not be forwarded to OpenAI; got kwargs: {call_kwargs}"
