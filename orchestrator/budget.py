"""
Budget — Async budget tracker with atomic reserve pattern
==========================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

The Budget class lives here because it has async behavior (reserve/commit/release
pattern with asyncio.Lock). Pure data models belong in models.py — async behavior
belongs in the application layer.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

# ─────────────────────────────────────────────
# Budget partitioning (soft caps)
# ─────────────────────────────────────────────

BUDGET_PARTITIONS: dict[str, float] = {
    "decomposition": 0.05,
    "generation": 0.45,
    "cross_review": 0.25,
    "evaluation": 0.15,
    "reserve": 0.10,
}


# ─────────────────────────────────────────────
# Budget — async budget tracker
# ─────────────────────────────────────────────


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
    # BUG-001 FIX: Eagerly initialized async lock prevents TOCTOU race
    _lock: asyncio.Lock = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """BUG-001 FIX: Initialize eagerly to prevent TOCTOU race on lock creation."""
        self._lock = asyncio.Lock()

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
        return BUDGET_PARTITIONS

    def can_afford(self, estimated_cost: float) -> bool:
        """Check if budget can afford estimated cost (non-atomic, for non-concurrent use)."""
        return self.remaining_usd >= estimated_cost

    def validate_sufficient_for_tasks(
        self, task_count: int, min_cost_per_task: float = 0.15
    ) -> tuple[bool, str]:
        """Validate if budget is sufficient for estimated task count."""
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
        return self.max_usd * BUDGET_PARTITIONS.get(phase, 0.0)

    def phase_remaining(self, phase: str) -> float:
        return max(0.0, self.phase_budget(phase) - self.phase_spent.get(phase, 0.0))

    async def charge(self, amount: float, phase: str = "generation"):
        """Charge actual spend to budget (thread-safe)."""
        async with self._lock:
            self.spent_usd += amount
            if phase in self.phase_spent:
                self.phase_spent[phase] += amount

    async def reserve(self, amount: float) -> bool:
        """Atomically reserve budget amount. Returns True if succeeded."""
        if amount < 0:
            raise ValueError("Reservation amount must be non-negative")
        async with self._lock:
            available = self.max_usd - self.spent_usd - self._reserved_usd
            if available >= amount:
                self._reserved_usd += amount
                return True
            return False

    async def commit_reservation(
        self, reserved_amount: float, actual_amount: float, phase: str = "generation"
    ):
        """Convert reservation to actual charge."""
        async with self._lock:
            self._reserved_usd = max(0.0, self._reserved_usd - reserved_amount)
            self.spent_usd += actual_amount
            if phase in self.phase_spent:
                self.phase_spent[phase] += actual_amount

    async def release_reservation(self, amount: float):
        """Release unused reservation."""
        async with self._lock:
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


__all__ = ["Budget", "BUDGET_PARTITIONS"]
