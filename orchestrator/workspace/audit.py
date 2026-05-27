"""
AuditTrail — Immutable append-only log of agent actions
==========================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Optimization C-9: Records every agent action to an append-only
log for compliance, debugging, and attribution.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger("orchestrator.workspace.audit")


@dataclass
class AuditEntry:
    """A single audit log entry."""

    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    agent: str = ""
    action: str = ""  # FILE_CREATED, FILE_MODIFIED, MODEL_CALL, TOOL_EXECUTED, DECISION_MADE
    detail: str = ""
    success: bool = True
    duration_ms: float = 0.0


class AuditTrail:
    """Append-only audit trail for agent actions."""

    def __init__(self) -> None:
        self._entries: list[AuditEntry] = []

    def record(
        self,
        agent: str,
        action: str,
        detail: str = "",
        success: bool = True,
        duration_ms: float = 0.0,
    ) -> AuditEntry:
        entry = AuditEntry(
            agent=agent, action=action, detail=detail, success=success, duration_ms=duration_ms
        )
        self._entries.append(entry)
        return entry

    def get_recent(self, limit: int = 20) -> list[AuditEntry]:
        return self._entries[-limit:]

    def export_json(self) -> str:
        return json.dumps(
            [
                {
                    "timestamp": e.timestamp,
                    "agent": e.agent,
                    "action": e.action,
                    "detail": e.detail,
                    "success": e.success,
                    "duration_ms": e.duration_ms,
                }
                for e in self._entries
            ],
            indent=2,
        )

    def clear(self) -> None:
        self._entries.clear()
