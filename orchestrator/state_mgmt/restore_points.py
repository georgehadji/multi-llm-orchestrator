"""
RestorePointManager — Chat-level restore points.
==================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Part of Category 2, Phase N2 (Newly-inspired): Save and restore conversation
state at any point. Each restore point captures the full conversation history
and project state at that moment, enabling undo to any previous prompt.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RestorePoint:
    """A save point in the conversation timeline."""

    point_id: str
    label: str
    timestamp: float = field(default_factory=time.time)
    prompt_index: int = 0  # Which prompt in the history this corresponds to
    state_snapshot: dict[str, Any] = field(default_factory=dict)
    conversation_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "point_id": self.point_id,
            "label": self.label,
            "timestamp": self.timestamp,
            "prompt_index": self.prompt_index,
            "state_snapshot": self.state_snapshot,
            "conversation_summary": self.conversation_summary,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RestorePoint:
        return cls(
            point_id=d["point_id"],
            label=d.get("label", ""),
            timestamp=d.get("timestamp", time.time()),
            prompt_index=d.get("prompt_index", 0),
            state_snapshot=d.get("state_snapshot", {}),
            conversation_summary=d.get("conversation_summary", ""),
        )


class RestorePointManager:
    """Manages chat-level restore points for the orchestrator.

    Each point captures the conversation state at a specific prompt,
    enabling rollback to any previous point in the interaction history.
    """

    def __init__(self, storage_dir: str | None = None):
        self._dir = Path(storage_dir or Path.home() / ".orchestrator_cache" / "restore_points")
        self._dir.mkdir(parents=True, exist_ok=True)
        self._points: list[RestorePoint] = []
        self._load()

    def _load(self) -> None:
        """Load restore points from disk."""
        fp = self._dir / "restore_points.json"
        if fp.exists():
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
                self._points = [RestorePoint.from_dict(d) for d in data]
            except Exception as e:
                logger.warning(f"Failed to load restore points: {e}")

    def _save(self) -> None:
        """Save restore points to disk."""
        fp = self._dir / "restore_points.json"
        fp.write_text(
            json.dumps([p.to_dict() for p in self._points], indent=2),
            encoding="utf-8",
        )

    async def capture(
        self,
        label: str,
        prompt_index: int = 0,
        state: dict[str, Any] | None = None,
        conversation_summary: str = "",
    ) -> RestorePoint:
        """Capture the current conversation state as a restore point.

        Args:
            label: Human-readable label (e.g., "before refactoring")
            prompt_index: Index in the conversation history
            state: Full project state snapshot
            conversation_summary: Summary of conversation up to this point

        Returns:
            RestorePoint that can be used to roll back
        """
        point_id = f"rp_{int(time.time())}"
        rp = RestorePoint(
            point_id=point_id,
            label=label,
            prompt_index=prompt_index,
            state_snapshot=state or {},
            conversation_summary=conversation_summary,
        )
        self._points.append(rp)
        self._save()
        logger.info(f"Restore point '{label}' captured at prompt {prompt_index}")
        return rp

    async def restore(self, point_id: str) -> RestorePoint | None:
        """Restore to a specific point.

        Args:
            point_id: ID of the restore point

        Returns:
            RestorePoint if found, None otherwise
        """
        for p in self._points:
            if p.point_id == point_id:
                logger.info(f"Restored to point '{p.label}'")
                return p
        return None

    async def restore_latest(self) -> RestorePoint | None:
        """Restore to the most recent restore point."""
        if self._points:
            return self._points[-1]
        return None

    async def list_points(self) -> list[dict[str, Any]]:
        """List all restore points for display."""
        return [
            {
                "id": p.point_id,
                "label": p.label,
                "prompt": p.prompt_index,
                "timestamp": time.strftime("%H:%M:%S", time.localtime(p.timestamp)),
            }
            for p in self._points
        ]

    async def discard_from(self, point_id: str) -> int:
        """Discard all restore points from (and including) the given point.

        This represents "undoing" the conversation from that point forward.

        Args:
            point_id: ID to discard from

        Returns:
            Number of points discarded
        """
        idx = None
        for i, p in enumerate(self._points):
            if p.point_id == point_id:
                idx = i
                break
        if idx is None:
            return 0

        discarded = len(self._points) - idx
        self._points = self._points[:idx]
        self._save()
        logger.info(f"Discarded {discarded} restore points from '{point_id}'")
        return discarded

    @property
    def point_count(self) -> int:
        return len(self._points)