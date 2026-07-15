"""
MechanismRegistry — Active Mechanism Tracking for Bilevel Autoresearch
=======================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Tracks active search mechanisms with activation history, backup/restore,
and performance attribution. Each mechanism can be independently
activated, deactivated, and rolled back.

Usage:
    registry = MechanismRegistry()
    registry.register("tabu_search", version="1.0", code="...")
    registry.activate("tabu_search")
    # ... run pipeline ...
    registry.record_outcome("tabu_search", score_delta=0.05)
    registry.deactivate("tabu_search")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger("orchestrator.bilevel.mechanism_registry")


@dataclass
class MechanismEntry:
    """Metadata for a single registered mechanism."""

    name: str
    version: str = "1.0"
    description: str = ""
    code: str = ""
    active: bool = False
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    activated_at: str | None = None
    deactivated_at: str | None = None
    outcome_history: list[dict[str, Any]] = field(default_factory=list)
    backup_state: dict[str, Any] | None = None

    @property
    def avg_score_delta(self) -> float:
        """Average score delta across all recorded outcomes."""
        delta = 0.0
        count = 0
        for entry in self.outcome_history:
            if entry.get("event") == "outcome":
                delta += entry.get("score_delta", 0.0)
                count += 1
        return delta / count if count > 0 else 0.0

    @property
    def activation_count(self) -> int:
        """Number of times this mechanism was activated."""
        count = 0
        for entry in self.outcome_history:
            if entry.get("event") == "activate":
                count += 1
        return count


class MechanismRegistry:
    """Registry for active mechanisms with backup/restore capability.

    Mechanisms are identified by name. Each can store a backup of
    the state it modified so deactivation restores the original behaviour.
    """

    def __init__(self) -> None:
        self._mechanisms: dict[str, MechanismEntry] = {}

    # ── Registration ──────────────────────────────────────────────────

    def register(
        self,
        name: str,
        version: str = "1.0",
        description: str = "",
        code: str = "",
    ) -> MechanismEntry:
        """Register a new mechanism.

        Args:
            name: Unique mechanism name.
            version: Semantic version string.
            description: Human-readable description.
            code: Python source code for the mechanism.

        Returns:
            The newly created ``MechanismEntry``.

        Raises:
            ValueError: If a mechanism with this name is already registered.
        """
        if name in self._mechanisms:
            raise ValueError(f"Mechanism '{name}' is already registered")
        entry = MechanismEntry(name=name, version=version, description=description, code=code)
        self._mechanisms[name] = entry
        logger.info("Registered mechanism '%s' v%s", name, version)
        return entry

    def unregister(self, name: str) -> None:
        """Remove a mechanism from the registry."""
        if name in self._mechanisms:
            del self._mechanisms[name]
            logger.info("Unregistered mechanism '%s'", name)

    # ── Lifecycle ─────────────────────────────────────────────────────

    def activate(self, name: str, backup_state: dict[str, Any] | None = None) -> MechanismEntry:
        """Activate a mechanism, optionally saving a backup of the state it replaces.

        Args:
            name: Mechanism name.
            backup_state: Snapshot of the state this mechanism modifies,
                          so deactivation can restore the original.

        Returns:
            The ``MechanismEntry`` now marked active.

        Raises:
            KeyError: If the mechanism is not registered.
        """
        entry = self._get(name)
        entry.active = True
        entry.activated_at = datetime.now().isoformat()
        entry.backup_state = backup_state
        entry.outcome_history.append({"event": "activate", "timestamp": entry.activated_at})
        logger.info("Activated mechanism '%s'", name)
        return entry

    def deactivate(self, name: str) -> MechanismEntry:
        """Deactivate a mechanism and restore its backup state.

        Args:
            name: Mechanism name.

        Returns:
            The ``MechanismEntry`` now marked inactive (with backup).

        Raises:
            KeyError: If the mechanism is not registered.
        """
        entry = self._get(name)
        entry.active = False
        entry.deactivated_at = datetime.now().isoformat()
        entry.outcome_history.append({"event": "deactivate", "timestamp": entry.deactivated_at})
        logger.info("Deactivated mechanism '%s'", name)
        return entry

    def rollback(self, name: str) -> dict[str, Any] | None:
        """Deactivate and return the backup state for restoration.

        Args:
            name: Mechanism name.

        Returns:
            The backup state that was saved at activation, or ``None``.
        """
        entry = self.deactivate(name)
        backup = entry.backup_state
        entry.backup_state = None
        logger.info("Rolled back mechanism '%s'", name)
        return backup

    # ── Performance tracking ──────────────────────────────────────────

    def record_outcome(
        self,
        name: str,
        score_delta: float = 0.0,
        cost_delta: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record an outcome for a mechanism.

        Args:
            name: Mechanism name.
            score_delta: Change in score (positive = improvement).
            cost_delta: Change in cost (negative = saving).
            metadata: Additional structured data.
        """
        try:
            entry = self._get(name)
        except KeyError:
            logger.warning("Cannot record outcome for unknown mechanism '%s'", name)
            return
        entry.outcome_history.append(
            {
                "event": "outcome",
                "timestamp": datetime.now().isoformat(),
                "score_delta": score_delta,
                "cost_delta": cost_delta,
                "metadata": metadata or {},
            }
        )

    # ── Query ─────────────────────────────────────────────────────────

    @property
    def active_names(self) -> list[str]:
        """Names of all currently active mechanisms."""
        return [n for n, e in self._mechanisms.items() if e.active]

    @property
    def all_names(self) -> list[str]:
        """Names of all registered mechanisms."""
        return list(self._mechanisms.keys())

    def get(self, name: str) -> MechanismEntry | None:
        """Get a mechanism entry, or ``None`` if not found."""
        return self._mechanisms.get(name)

    def _get(self, name: str) -> MechanismEntry:
        if name not in self._mechanisms:
            raise KeyError(f"Mechanism '{name}' not found in registry")
        return self._mechanisms[name]
