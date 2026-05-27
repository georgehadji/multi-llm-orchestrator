"""
AgentRateLimiter — Per-agent request rate limiting
=====================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Optimization C-8: Sliding-window rate limiter per agent role
to prevent runaway costs and API abuse.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger("orchestrator.agents.rate_limiter")


@dataclass
class RateLimit:
    """Rate limit configuration for an agent role."""

    max_calls: int = 60  # Max calls per window
    window_seconds: int = 60  # Window duration
    max_cost_usd: float = 5.0  # Max cost per window


class WindowCounter:
    """Sliding window counter for rate limiting."""

    def __init__(self, window_seconds: int = 60) -> None:
        self.window = window_seconds
        self.entries: list[tuple[float, float]] = []  # (timestamp, cost)

    def add(self, cost: float = 0.0) -> None:
        cutoff = time.time() - self.window
        self.entries = [e for e in self.entries if e[0] > cutoff]
        self.entries.append((time.time(), cost))

    def count(self) -> int:
        cutoff = time.time() - self.window
        return sum(1 for e in self.entries if e[0] > cutoff)

    def total_cost(self) -> float:
        cutoff = time.time() - self.window
        return sum(e[1] for e in self.entries if e[0] > cutoff)


class AgentRateLimiter:
    """Per-agent rate limiting."""

    def __init__(self) -> None:
        self._limits: dict[str, RateLimit] = {}
        self._counters: dict[str, WindowCounter] = defaultdict(WindowCounter)

    def set_limit(self, role: str, limit: RateLimit) -> None:
        self._limits[role] = limit

    def check(self, role: str, cost: float = 0.0) -> bool:
        """Check if a call is allowed. Returns True if allowed."""
        limit = self._limits.get(role)
        if limit is None:
            return True  # No limit configured = allowed

        counter = self._counters[role]
        current_count = counter.count()
        current_cost = counter.total_cost()

        if current_count >= limit.max_calls:
            logger.warning(
                "Rate limit reached for %s: %d calls in %ds window",
                role,
                current_count,
                limit.window_seconds,
            )
            return False
        if current_cost + cost >= limit.max_cost_usd:
            logger.warning(
                "Cost limit reached for %s: $%.2f in %ds window",
                role,
                current_cost,
                limit.window_seconds,
            )
            return False

        counter.add(cost)
        return True
