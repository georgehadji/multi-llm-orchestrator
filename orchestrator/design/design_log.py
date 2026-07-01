"""
Design Log — Cross-output tracking for diversification enforcement.
==================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Tracks previous frontend outputs per project to ensure no two consecutive
outputs share the same macrostructure, nav, footer, or theme.

Source: Hallmark design skill (diversification rule)
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DesignLogEntry:
    """A single recorded design output."""

    timestamp: str
    macrostructure: str
    theme: str
    genre: str
    nav_archetype: str
    footer_archetype: str
    pre_emit_scores: dict[str, int] = field(default_factory=dict)


class DesignLog:
    """Persistent log of design outputs for a project.

    Usage:
        log = DesignLog(Path("./my-project"))
        if log.is_recently_used("bento_grid", within=3):
            # pick a different macrostructure
        log.append(DesignLogEntry(...))
    """

    def __init__(self, project_dir: Path | str) -> None:
        self._dir = Path(project_dir)
        self._path = self._dir / ".orchestrator" / "design_log.json"
        self.entries: list[DesignLogEntry] = []
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            logger.debug("design_log: no existing log at %s", self._path)
            return
        try:
            text = self._path.read_text(encoding="utf-8")
            data = json.loads(text)
            self.entries = [DesignLogEntry(**e) for e in data.get("entries", [])]
            logger.debug("design_log: loaded %d entries", len(self.entries))
        except (json.JSONDecodeError, OSError, TypeError) as exc:
            logger.warning("design_log: failed to load %s: %s", self._path, exc)
            self.entries = []

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"entries": [asdict(e) for e in self.entries]}
            self._path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError as exc:
            logger.warning("design_log: failed to save %s: %s", self._path, exc)

    # ── Public API ────────────────────────────────────────────────────────────

    def is_recently_used(self, macrostructure: str, within: int = 3) -> bool:
        """Return True if *macrostructure* appears in the last *within* entries."""
        recent = self.entries[-within:] if len(self.entries) >= within else self.entries
        return any(e.macrostructure == macrostructure for e in recent)

    def is_nav_recently_used(self, nav_archetype: str, within: int = 3) -> bool:
        """Return True if *nav_archetype* appears in the last *within* entries."""
        recent = self.entries[-within:] if len(self.entries) >= within else self.entries
        return any(e.nav_archetype == nav_archetype for e in recent)

    def is_footer_recently_used(self, footer_archetype: str, within: int = 3) -> bool:
        """Return True if *footer_archetype* appears in the last *within* entries."""
        recent = self.entries[-within:] if len(self.entries) >= within else self.entries
        return any(e.footer_archetype == footer_archetype for e in recent)

    def is_theme_recently_used(self, theme: str, within: int = 3) -> bool:
        """Return True if *theme* appears in the last *within* entries."""
        recent = self.entries[-within:] if len(self.entries) >= within else self.entries
        return any(e.theme == theme for e in recent)

    def append(self, entry: DesignLogEntry) -> None:
        """Append an entry and persist."""
        self.entries.append(entry)
        self._save()
        logger.debug(
            "design_log: appended %s / %s / %s",
            entry.macrostructure,
            entry.theme,
            entry.nav_archetype,
        )

    def last_n(self, n: int = 1) -> list[DesignLogEntry]:
        """Return the last *n* entries."""
        return self.entries[-n:] if self.entries else []

    def to_dict(self) -> dict[str, Any]:
        """Serialize for debugging/telemetry."""
        return {
            "path": str(self._path),
            "entry_count": len(self.entries),
            "last_macrostructures": [e.macrostructure for e in self.entries[-5:]],
            "last_themes": [e.theme for e in self.entries[-5:]],
        }
