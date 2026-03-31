"""
Budget — Async budget tracking with atomic reserve pattern
===========================================================
Extracted from models.py to satisfy the "models.py = pure data" rule.
models.py re-exports Budget from here for backward compatibility.

Pattern: Dataclass + async Lock
Async: Yes — charge/reserve/commit/release are coroutines
Layer: L1 Infrastructure
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from .models import BUDGET_PARTITIONS


@dataclass
class Budget:
    """
    Budget tracking with atomic reserve pattern for concurrent execution.

    FIX-001a: Added reserve/commit/release pattern to prevent race conditions
    when multiple concurrent tasks check budget simultaneously.
    """

    max_usd: float = 8.0
    max_time_seconds: float = 5400.0  # 90 min
    spent_usd: float = 0.0
    start_time: float = field(default_factory=time.time)
    phase_spent: dict[str, float] = field(
        default_factory=lambda: {
            "decomposition": 0.0,
            "generation": 0.0,
            "cross_review": 0.0,
            "evaluation": 0.0,
            "reserve": 0.0,
        }
    )
    # FIX-001a: Track reserved but not-yet-charged budget
    _reserved_usd: float = field(default=0.0, repr=False)
    # FIX-001a: Async lock for atomic operations (lazy initialized)
    _lock: asyncio.Lock | None = field(default=None, repr=False)

    def _get_lock(self) -> asyncio.Lock:
        """Get or create asyncio.Lock lazily (must be called from async context)."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    @property
    def remaining_usd(self) -> float:
        """Get remaining budget excluding reserved amounts."""
        return max(0.0, self.max_usd - self.spent_usd - self._reserved_usd)

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.max_time_seconds - self.elapsed_seconds)

    def can_afford(self, estimated_cost: float) -> bool:
        """Check if budget can afford estimated cost (non-atomic, for non-concurrent use)."""
        return self.remaining_usd >= estimated_cost

    def time_remaining(self) -> bool:
        return self.elapsed_seconds < self.max_time_seconds

    def phase_budget(self, phase: str) -> float:
        return self.max_usd * BUDGET_PARTITIONS.get(phase, 0.0)

    def phase_remaining(self, phase: str) -> float:
        return max(0.0, self.phase_budget(phase) - self.phase_spent.get(phase, 0.0))

    async def charge(self, amount: float, phase: str = "generation"):
        """
        Charge actual spend to budget (thread-safe).

        FIX-BUG-001: Made async with lock to prevent race conditions when
        multiple concurrent tasks charge simultaneously via asyncio.gather().
        """
        async with self._get_lock():
            self.spent_usd += amount
            if phase in self.phase_spent:
                self.phase_spent[phase] += amount

    async def reserve(self, amount: float) -> bool:
        """
        FIX-001a: Atomically reserve budget amount.

        Returns True if reservation succeeded, False if insufficient budget.
        Must be called from async context.
        """
        if amount < 0:
            raise ValueError("Reservation amount must be non-negative")

        async with self._get_lock():
            available = self.max_usd - self.spent_usd - self._reserved_usd
            if available >= amount:
                self._reserved_usd += amount
                return True
            return False

    async def commit_reservation(
        self, reserved_amount: float, actual_amount: float, phase: str = "generation"
    ):
        """
        FIX-001a: Convert reservation to actual charge.

        Should be called after successful task execution.

        Args:
            reserved_amount: The amount originally reserved (to release from _reserved_usd)
            actual_amount: The actual cost incurred (may differ from reserved_amount)
            phase: Budget phase to charge

        BUG-FIX: Previously set _reserved_usd = 0.0 unconditionally, zeroing all
        concurrent reservations. Now releases only this task's reserved_amount.
        Also fixes a leak where actual_amount == 0 (cached) left the reservation
        permanently held.
        """
        async with self._get_lock():
            self._reserved_usd = max(0.0, self._reserved_usd - reserved_amount)
        # Charge the actual amount (not the reserved amount)
        await self.charge(actual_amount, phase)

    async def release_reservation(self, amount: float):
        """
        FIX-001a: Release unused reservation.

        Should be called when task fails or is skipped.
        """
        async with self._get_lock():
            self._reserved_usd = max(0.0, self._reserved_usd - amount)

    def to_dict(self) -> dict:
        return {
            "max_usd": self.max_usd,
            "spent_usd": round(self.spent_usd, 4),
            "remaining_usd": round(self.remaining_usd, 4),
            "reserved_usd": round(self._reserved_usd, 4),
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "remaining_seconds": round(self.remaining_seconds, 1),
            "phase_spent": {k: round(v, 4) for k, v in self.phase_spent.items()},
        }


__all__ = ["Budget"]
