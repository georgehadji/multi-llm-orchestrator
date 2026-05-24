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

# NOTE: Lazy import to avoid circular dependency with models.py
# BUDGET_PARTITIONS will be imported when first needed
_budget_partitions_cache = None


def _get_budget_partitions():
    """Lazy-load BUDGET_PARTITIONS to avoid circular imports."""
    global _budget_partitions_cache
    if _budget_partitions_cache is None:
        from .models import BUDGET_PARTITIONS  # noqa: PLC0415

        _budget_partitions_cache = BUDGET_PARTITIONS
    return _budget_partitions_cache


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
    # FIX-RESUME-001: Track original start time for elapsed time calculation when resuming
    original_start_time: float = field(default_factory=time.time)
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
        # FIX-RESUME-001: Use original_start_time for elapsed time when resuming
        return time.time() - self.original_start_time

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.max_time_seconds - self.elapsed_seconds)

    @property
    def phase_limits(self) -> dict[str, float]:
        """Return phase budget limits (compatibility with BudgetEnforcer)."""
        return _get_budget_partitions()

    def can_afford(self, estimated_cost: float) -> bool:
        """Check if budget can afford estimated cost (non-atomic, for non-concurrent use)."""
        return self.remaining_usd >= estimated_cost

    def validate_sufficient_for_tasks(
        self, task_count: int, min_cost_per_task: float = 0.15
    ) -> tuple[bool, str]:
        """Validate if budget is sufficient for estimated task count.

        Args:
            task_count: Number of tasks to execute
            min_cost_per_task: Minimum cost per task (default $0.15 for cheap models)

        Returns:
            (is_sufficient, warning_message)
        """
        min_required = task_count * min_cost_per_task
        if self.max_usd < min_required:
            return (
                False,
                f"Budget ${self.max_usd:.2f} insufficient for {task_count} tasks. "
                f"Minimum required: ${min_required:.2f} (${min_cost_per_task:.2f}/task)",
            )
        return True, ""

    def time_remaining(self) -> bool:
        return self.elapsed_seconds < self.max_time_seconds

    def phase_budget(self, phase: str) -> float:
        budget_partitions = _get_budget_partitions()
        return self.max_usd * budget_partitions.get(phase, 0.0)

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
        # BUG-002: Hold lock across both reservation release and charge to prevent
        # transient budget inflation visible to concurrent reserve() calls.
        async with self._get_lock():
            self._reserved_usd = max(0.0, self._reserved_usd - reserved_amount)
            self.spent_usd += actual_amount
            if phase in self.phase_spent:
                self.phase_spent[phase] += actual_amount

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
