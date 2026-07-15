"""
TabuSearchManager — Retry Diversity for SelfConsistencyStage
=============================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Prevents ``SelfConsistencyStage`` from retrying the same
``(model, revision_context)`` pair within a configurable tenure window.
Inspired by Tabu Search from metaheuristic optimisation.

Usage:
    tabu = TabuSearchManager(tenure=3)
    tabu.add(("gpt4", "fix import error"))
    assert tabu.is_tabu(("gpt4", "fix import error"))  # True (tenure >= 1)
    tabu.add(("claude", "refactor loop"))
    assert not tabu.is_tabu(("claude", "add tests"))    # Different pair
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TabuSearchManager:
    """Manages a tabu list of (model, context) pairs.

    Each pair has a tenure (number of evaluations before it expires).
    When the tenure expires, the pair is removed from the tabu list
    and becomes eligible for retry.

    Args:
        tenure: Number of evaluations a pair stays tabu (default 3).
    """

    tenure: int = 3
    _tabu: dict[str, int] = field(default_factory=dict)  # pair_key -> remaining_tenure

    # ═══════════════════════════════════════════════════════════════
    # Public API
    # ═══════════════════════════════════════════════════════════════

    def add(self, pair: tuple[str, str]) -> None:
        """Mark a (model, context) pair as tabu.

        Args:
            pair: ``(model_name, revision_context)`` tuple.
        """
        key = self._key(pair)
        self._tabu[key] = self.tenure

    def is_tabu(self, pair: tuple[str, str]) -> bool:
        """Check whether a (model, context) pair is currently tabu.

        Args:
            pair: ``(model_name, revision_context)`` tuple.

        Returns:
            True if the pair is tabu (should not be retried).
        """
        return self._key(pair) in self._tabu

    def evict(self) -> list[tuple[str, str]]:
        """Decrement all tenures and remove expired entries.

        Call this once per evaluation cycle.

        Returns:
            List of pairs that just expired (were evicted this cycle).
        """
        expired: list[tuple[str, str]] = []
        to_delete: list[str] = []

        for key, remaining in self._tabu.items():
            if remaining <= 1:
                to_delete.append(key)
                model, context = self._parse_key(key)
                expired.append((model, context))
            else:
                self._tabu[key] = remaining - 1

        for key in to_delete:
            del self._tabu[key]

        return expired

    def clear(self) -> None:
        """Remove all entries from the tabu list."""
        self._tabu.clear()

    @property
    def size(self) -> int:
        """Number of currently tabu pairs."""
        return len(self._tabu)

    @property
    def is_empty(self) -> bool:
        """True if no pairs are tabu."""
        return len(self._tabu) == 0

    # ═══════════════════════════════════════════════════════════════
    # Internal helpers
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _key(pair: tuple[str, str]) -> str:
        """Normalise a (model, context) pair into a dictionary key."""
        model, context = pair
        return f"{model}:::{context[:200].strip()}"  # truncate to avoid noise

    @staticmethod
    def _parse_key(key: str) -> tuple[str, str]:
        """Reverse of ``_key``."""
        parts = key.split(":::", 1)
        if len(parts) == 2:
            return (parts[0], parts[1])
        return (key, "")
