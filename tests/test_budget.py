"""Unit tests for orchestrator.budget.Budget."""

from __future__ import annotations

import asyncio

import pytest

from orchestrator.budget import Budget


# ─────────────────────────────────────────────────────────────────────────────
# BUG-002 regression: commit_reservation atomicity
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_commit_reservation_never_releases_lock_mid_operation(monkeypatch):
    """
    BUG-002 regression: commit_reservation must hold the budget lock across
    both the reservation release and the spend update.

    Before the fix, the lock was released after reducing _reserved_usd and
    then self.charge() was awaited, which re-acquired the lock.  In that
    window a concurrent reserve() could observe a transiently inflated
    remaining budget and over-commit.
    """
    budget = Budget(max_usd=1.0)

    lock_acquisitions = 0
    original_acquire = budget._get_lock().acquire

    async def counting_acquire():
        nonlocal lock_acquisitions
        lock_acquisitions += 1
        return await original_acquire()

    monkeypatch.setattr(budget._get_lock(), "acquire", counting_acquire)

    await budget.reserve(0.5)
    lock_acquisitions = 0
    await budget.commit_reservation(0.5, 0.3, phase="generation")

    # One continuous critical section -> exactly one acquire (no re-acquire).
    assert lock_acquisitions == 1
    assert budget._reserved_usd == 0.0
    assert budget.spent_usd == 0.3
    assert budget.remaining_usd == pytest.approx(0.7)


@pytest.mark.asyncio
async def test_concurrent_commit_and_reserve_no_over_commit():
    """
    BUG-002 regression: Under concurrent load, commit_reservation must not
    allow reserve() to see a transient state where _reserved_usd has been
    reduced but spent_usd has not yet been increased.
    """
    budget = Budget(max_usd=1.0)
    assert await budget.reserve(0.5) is True

    async def committer():
        await budget.commit_reservation(0.5, 0.3)

    async def reserver():
        # Small sleep to land inside the race window of the buggy code.
        await asyncio.sleep(0)
        return await budget.reserve(0.3)

    # Run many iterations to stress the race window.
    for _ in range(100):
        budget = Budget(max_usd=1.0)
        assert await budget.reserve(0.5) is True
        await asyncio.gather(committer(), reserver())
        # Final invariant: we can never have reserved+spent more than max.
        assert budget.spent_usd + budget._reserved_usd <= budget.max_usd + 1e-9


@pytest.mark.asyncio
async def test_commit_reservation_updates_phase_spent():
    """commit_reservation should charge to the correct phase bucket."""
    budget = Budget(max_usd=1.0)
    await budget.reserve(0.2)
    await budget.commit_reservation(0.2, 0.15, phase="evaluation")
    assert budget.phase_spent["evaluation"] == pytest.approx(0.15)
    assert budget.phase_spent["generation"] == 0.0
