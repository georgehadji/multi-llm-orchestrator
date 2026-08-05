"""FlakeTracker — flake observation accounting and quarantine (E-5).

Records flaky node ids across runs; three observations of the same node id
quarantine it (the test is excluded from future repair decisions). The
tracker is pure state — persistence to project state (StateManager) is
wired by the caller, keeping this module I/O-free and testable.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


@dataclass
class FlakeTracker:
    """Counts flake observations per node id and reports quarantined ids."""

    quarantine_threshold: int = 3
    _observations: Counter[str] = field(default_factory=Counter)

    def record(self, flaky_node_ids: tuple[str, ...]) -> tuple[str, ...]:
        """Record flake observations; return newly-quarantined node ids.

        Args:
            flaky_node_ids: Node ids observed flaky in this run.

        Returns:
            Node ids that crossed the quarantine threshold in this call.
        """
        newly_quarantined: list[str] = []
        for nid in flaky_node_ids:
            self._observations[nid] += 1
            if self._observations[nid] == self.quarantine_threshold:
                newly_quarantined.append(nid)
        return tuple(newly_quarantined)

    @property
    def quarantined(self) -> tuple[str, ...]:
        """Node ids at or above the quarantine threshold."""
        return tuple(
            nid for nid, count in self._observations.items() if count >= self.quarantine_threshold
        )

    def observation_count(self, node_id: str) -> int:
        """Number of flake observations for a node id."""
        return self._observations.get(node_id, 0)
