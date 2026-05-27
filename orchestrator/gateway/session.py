"""
Gateway Session Management — Track Active Project Sessions
=============================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Tracks active project sessions per user, handles timeouts and cleanup.
Follows the pattern from state.py but scoped to gateway sessions
(in-memory, not persisted).
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("orchestrator.gateway.session")


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class GatewaySession:
    """A single gateway session bound to a user + platform."""

    session_id: str
    platform: str
    user_id: str
    project_id: str = ""
    status: str = "idle"  # idle | running | completed | failed
    created_at: float = field(default_factory=time.time)
    last_active_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def idle_seconds(self) -> float:
        return time.time() - self.last_active_at


# ─────────────────────────────────────────────────────────────────────────────
# GatewaySessionManager
# ─────────────────────────────────────────────────────────────────────────────


class GatewaySessionManager:
    """Manages gateway sessions with timeout and cleanup.

    Sessions are tracked in-memory (not persisted). Cleanup runs
    periodically to remove stale sessions.

    Usage:
        mgr = GatewaySessionManager(idle_timeout=3600)
        session = mgr.create_session("telegram", "user123")
        mgr.touch(session.session_id)
        mgr.cleanup_stale()
    """

    def __init__(self, idle_timeout: float = 3600) -> None:
        """Initialize session manager.

        Args:
            idle_timeout: Seconds before an idle session is eligible
                for cleanup. Default 3600 (1 hour).
        """
        self._sessions: dict[str, GatewaySession] = {}
        self._idle_timeout = idle_timeout

    def create_session(
        self,
        platform: str,
        user_id: str,
    ) -> GatewaySession:
        """Create a new session for a user.

        Args:
            platform: Platform name.
            user_id: User identifier.

        Returns:
            The newly created GatewaySession.
        """
        session = GatewaySession(
            session_id=str(uuid.uuid4())[:8],
            platform=platform,
            user_id=user_id,
        )
        self._sessions[session.session_id] = session
        logger.debug("Session %s created for %s/%s", session.session_id, platform, user_id)
        return session

    def get_session(self, session_id: str) -> GatewaySession | None:
        """Get a session by ID."""
        return self._sessions.get(session_id)

    def find_session(
        self,
        platform: str,
        user_id: str,
    ) -> GatewaySession | None:
        """Find an active session for a user+platform pair."""
        for session in self._sessions.values():
            if session.platform == platform and session.user_id == user_id:
                return session
        return None

    def touch(self, session_id: str) -> None:
        """Update the last_active_at timestamp."""
        session = self._sessions.get(session_id)
        if session:
            session.last_active_at = time.time()

    def remove_session(self, session_id: str) -> bool:
        """Remove a session. Returns True if removed."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.debug("Session %s removed", session_id)
            return True
        return False

    def cleanup_stale(self) -> int:
        """Remove sessions that have exceeded idle_timeout.

        Returns:
            Count of removed sessions.
        """
        now = time.time()
        stale = [
            sid
            for sid, session in self._sessions.items()
            if now - session.last_active_at > self._idle_timeout
        ]
        for sid in stale:
            del self._sessions[sid]
        if stale:
            logger.debug("Cleaned up %d stale session(s)", len(stale))
        return len(stale)

    @property
    def active_count(self) -> int:
        return len(self._sessions)

    def stats(self) -> dict[str, Any]:
        return {
            "active_sessions": self.active_count,
            "idle_timeout_seconds": self._idle_timeout,
        }
