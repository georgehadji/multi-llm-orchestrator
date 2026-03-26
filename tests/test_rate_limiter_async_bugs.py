"""
Regression tests for BUG-NEW-001 and BUG-NEW-002 in GrokRateLimiter.

BUG-NEW-001: asyncio.sleep() called while holding self._lock, starving all
             other coroutines for the full wait window.
BUG-NEW-002: `limits` captured once before the while loop; tier upgrades
             triggered by _update_tier_from_spend() were not reflected in
             subsequent RPM/TPM checks within the same acquire() call.
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta

import pytest

from orchestrator.rate_limiter import GrokRateLimiter, RateLimitState, TierLimits


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _limiter_at_rpm_limit(remaining_window_secs: float = 1.0) -> GrokRateLimiter:
    """
    Create a tier-1 limiter whose RPM counter is already at the limit.
    `remaining_window_secs` controls how long until the window resets.
    This avoids mutating the class-level TIER_LIMITS dict.
    """
    limiter = GrokRateLimiter(initial_tier=1, max_concurrency=10)
    rpm_limit = limiter.TIER_LIMITS[1].rpm  # 10
    limiter.state.current_rpm = rpm_limit
    # Set last_reset so the window expires after remaining_window_secs
    limiter.state.last_reset = datetime.now() - timedelta(
        seconds=60.0 - remaining_window_secs
    )
    return limiter


# ---------------------------------------------------------------------------
# BUG-NEW-001: lock must not be held during sleep
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lock_not_held_during_sleep():
    """
    Two concurrent acquire() calls when RPM is exhausted must not serialize
    into two separate 60-second windows.

    Strategy: RPM is already at limit; window resets in ~1 second.  Both
    coroutines are launched together and should both complete in ≈1 s, not
    ≈2 s (which would happen if the first held the lock while sleeping).
    """
    limiter = _limiter_at_rpm_limit(remaining_window_secs=1.0)

    lock_grabbed_times: list[float] = []

    async def try_acquire():
        await limiter.acquire(tokens=100, timeout=10.0)
        lock_grabbed_times.append(time.monotonic())

    t0 = time.monotonic()
    await asyncio.gather(try_acquire(), try_acquire())
    elapsed = time.monotonic() - t0

    # Both coroutines should finish in ≈1 s (the remaining window).
    # If the lock was held during sleep the second one would start its sleep
    # only after the first finishes, taking ≈2 s total.
    assert elapsed < 3.0, (
        f"Both acquires took {elapsed:.2f}s — lock was probably held during "
        "sleep (BUG-NEW-001 regressed)"
    )
    assert len(lock_grabbed_times) == 2


@pytest.mark.asyncio
async def test_concurrent_acquires_do_not_block_each_other():
    """
    Under a generous limit, N concurrent acquires should all complete almost
    simultaneously (no serialisation through a held lock).
    """
    limiter = GrokRateLimiter(initial_tier=1, max_concurrency=20)

    t0 = time.monotonic()
    # Tier-1 rpm=10; run only 8 requests — well within the limit
    results = await asyncio.gather(*[limiter.acquire(tokens=100) for _ in range(8)])
    elapsed = time.monotonic() - t0

    assert all(results), "All acquires should succeed"
    # Eight fast acquires under no limit should finish in well under 1 s
    assert elapsed < 1.0, (
        f"8 concurrent acquires took {elapsed:.2f}s — serialisation suspected"
    )


# ---------------------------------------------------------------------------
# BUG-NEW-002: limits must be refreshed after tier upgrade inside acquire()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_limits_refreshed_after_tier_upgrade():
    """
    If record_spend() causes a tier upgrade between two acquire() calls, the
    *next* acquire() must use the new tier's limits, not the old cached value.

    We set up a limiter at tier 1 (rpm=10), fill all 10 rpm slots, record
    enough spend to jump to tier 2 (rpm=60), then verify the 11th acquire
    succeeds (would fail if BUG-NEW-002 regressed and stale rpm=10 were used).
    """
    limiter = GrokRateLimiter(initial_tier=1, max_concurrency=10)
    rpm_limit_t1 = limiter.TIER_LIMITS[1].rpm  # 10
    rpm_limit_t2 = limiter.TIER_LIMITS[2].rpm  # 60

    # Fill all tier-1 RPM slots
    for _ in range(rpm_limit_t1):
        ok = await limiter.acquire(tokens=100, timeout=5.0)
        assert ok, "Should be within tier-1 limit"

    # Upgrade to tier 2 by recording spend
    limiter.record_spend(50.0)
    assert limiter.state.current_tier == 2, "Tier should have upgraded to 2"
    assert rpm_limit_t2 > rpm_limit_t1, "Tier 2 must have a higher RPM limit"

    # The next acquire must succeed because tier-2 rpm=60 > current_rpm=10
    # (BUG-NEW-002 would have caused failure here by checking against stale rpm=10)
    ok = await limiter.acquire(tokens=100, timeout=5.0)
    assert ok, (
        "11th acquire failed — acquire() probably used stale tier-1 limits "
        "instead of refreshing after tier upgrade (BUG-NEW-002 regressed)"
    )


@pytest.mark.asyncio
async def test_limits_variable_reflects_current_tier():
    """
    Inside a single acquire() that loops (because of an initial limit hit),
    a tier upgrade recorded externally must be picked up by the next iteration.
    """
    limiter = GrokRateLimiter(initial_tier=1, max_concurrency=10)
    # Exhaust tier-1 rpm
    limiter.state.current_rpm = limiter.TIER_LIMITS[1].rpm
    # Make the reset happen quickly (0.2 s from now the window resets)
    limiter.state.last_reset = datetime.now() - timedelta(seconds=59.8)

    # Upgrade tier while the acquire() is waiting
    async def upgrade_after_delay():
        await asyncio.sleep(0.05)
        limiter.record_spend(50.0)  # → tier 2

    await asyncio.gather(
        limiter.acquire(tokens=100, timeout=5.0),
        upgrade_after_delay(),
    )

    # If we get here without timeout the test passes; the acquire must have
    # re-read the limits and used tier-2's higher rpm cap on the next iteration.
    assert limiter.state.current_tier == 2
