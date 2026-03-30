"""
Session Watcher — Auto-capture Conversations in Real-time
==========================================================

Implements Mnemo Cortex session watcher:
- Auto-captures task input + output pairs
- Stores with session metadata
- Supports HOT/WARM/COLD memory tiers

Usage:
    from orchestrator.session_watcher import SessionWatcher, SessionRecord

    watcher = SessionWatcher()

    # Start a session
    session_id = watcher.start_session("project_001")

    # Record an interaction
    watcher.record_interaction(
        session_id=session_id,
        task_input="Write a function to calculate fibonacci",
        task_output="def fibonacci(n): ...",
        task_type="code_generation",
        metadata={"model": "gpt-4o", "tokens": 1500},
    )

    # Get session context
    context = await watcher.get_context(session_id, limit=5)

    # Archive session
    await watcher.archive_session(session_id)
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from .log_config import get_logger

logger = get_logger(__name__)


class MemoryTier(Enum):
    """Memory tier based on age."""
    HOT = "hot"     # Days 1-3: Raw JSONL, instant search
    WARM = "warm"   # Days 4-30: Summarized + embedded
    COLD = "cold"   # Day 30+: Compressed archive


class SessionStatus(Enum):
    """Session status."""
    ACTIVE = "active"
    ARCHIVED = "archived"
    SUMMARIZED = "summarized"
    COLD = "cold"


@dataclass
class InteractionRecord:
    """A single interaction in a session."""
    id: str
    timestamp: datetime
    task_input: str
    task_output: str
    task_type: str
    model: str | None = None
    tokens_used: int | None = None
    duration_ms: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "task_input": self.task_input,
            "task_output": self.task_output,
            "task_type": self.task_type,
            "model": self.model,
            "tokens_used": self.tokens_used,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> InteractionRecord:
        return cls(
            id=d["id"],
            timestamp=datetime.fromisoformat(d["timestamp"]),
            task_input=d["task_input"],
            task_output=d["task_output"],
            task_type=d["task_type"],
            model=d.get("model"),
            tokens_used=d.get("tokens_used"),
            duration_ms=d.get("duration_ms"),
            metadata=d.get("metadata", {}),
        )


@dataclass
class SessionRecord:
    """A complete session record."""
    id: str
    project_id: str
    created_at: datetime
    last_activity: datetime
    status: SessionStatus
    interactions: list[InteractionRecord] = field(default_factory=list)
    summary: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def tier(self) -> MemoryTier:
        """Determine memory tier based on age."""
        age = datetime.utcnow() - self.created_at
        if age.days < 3:
            return MemoryTier.HOT
        elif age.days < 30:
            return MemoryTier.WARM
        else:
            return MemoryTier.COLD

    @property
    def interaction_count(self) -> int:
        return len(self.interactions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "project_id": self.project_id,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "status": self.status.value,
            "interactions": [i.to_dict() for i in self.interactions],
            "summary": self.summary,
            "metadata": self.metadata,
            "tier": self.tier.value,
        }


class SessionWatcher:
    """
    Watches and captures sessions in real-time.

    Implements the "Live Wire" concept from Mnemo Cortex:
    - Auto-captures task input + output pairs
    - Stores in HOT tier (raw JSONL)
    - Supports tier migration (HOT -> WARM -> COLD)
    """

    def __init__(
        self,
        storage_path: Path | None = None,
        hot_ttl_days: int = 3,
        warm_ttl_days: int = 30,
    ):
        self.storage_path = storage_path or Path.home() / ".orchestrator_cache" / "sessions"
        self.hot_ttl_days = hot_ttl_days
        self.warm_ttl_days = warm_ttl_days

        # In-memory session storage
        self._sessions: dict[str, SessionRecord] = {}
        self._active_sessions: dict[str, str] = {}  # project_id -> session_id

        # Ensure storage directory exists
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Load existing sessions
        self._load_sessions()

    def _session_file_path(self, session_id: str) -> Path:
        """Get file path for session storage."""
        return self.storage_path / f"{session_id}.jsonl"

    def _load_sessions(self) -> None:
        """Load existing sessions from disk."""
        if not self.storage_path.exists():
            return

        for file_path in self.storage_path.glob("*.jsonl"):
            try:
                session_id = file_path.stem
                with open(file_path, encoding='utf-8') as f:
                    data = json.loads(f.readline())
                    session = SessionRecord(
                        id=data["id"],
                        project_id=data["project_id"],
                        created_at=datetime.fromisoformat(data["created_at"]),
                        last_activity=datetime.fromisoformat(data["last_activity"]),
                        status=SessionStatus(data["status"]),
                        summary=data.get("summary"),
                        metadata=data.get("metadata", {}),
                    )
                    # Load interactions
                    for line in f:
                        if line.strip():
                            interaction = InteractionRecord.from_dict(json.loads(line))
                            session.interactions.append(interaction)

                    self._sessions[session_id] = session
            except Exception as e:
                logger.warning(f"Failed to load session {file_path}: {e}")

        logger.info(f"Loaded {len(self._sessions)} sessions from disk")

    def _save_session(self, session: SessionRecord) -> None:
        """Save session to disk."""
        file_path = self._session_file_path(session.id)

        # Write as JSONL: first line is session metadata, rest are interactions
        with open(file_path, 'w', encoding='utf-8') as f:
            # Session header
            f.write(json.dumps({
                "id": session.id,
                "project_id": session.project_id,
                "created_at": session.created_at.isoformat(),
                "last_activity": session.last_activity.isoformat(),
                "status": session.status.value,
                "summary": session.summary,
                "metadata": session.metadata,
            }) + "\n")

            # Interactions
            for interaction in session.interactions:
                f.write(json.dumps(interaction.to_dict()) + "\n")

    def start_session(self, project_id: str, metadata: dict[str, Any] | None = None) -> str:
        """Start a new session for a project."""
        # Check if there's already an active session
        if project_id in self._active_sessions:
            return self._active_sessions[project_id]

        session_id = str(uuid.uuid4())
        now = datetime.utcnow()

        session = SessionRecord(
            id=session_id,
            project_id=project_id,
            created_at=now,
            last_activity=now,
            status=SessionStatus.ACTIVE,
            metadata=metadata or {},
        )

        self._sessions[session_id] = session
        self._active_sessions[project_id] = session_id

        self._save_session(session)
        logger.info(f"Started session {session_id} for project {project_id}")

        return session_id

    def end_session(self, project_id: str) -> str | None:
        """End the active session for a project."""
        session_id = self._active_sessions.pop(project_id, None)

        if session_id:
            session = self._sessions.get(session_id)
            if session:
                session.status = SessionStatus.ARCHIVED
                self._save_session(session)
                logger.info(f"Ended session {session_id}")

        return session_id

    def get_active_session(self, project_id: str) -> str | None:
        """Get the active session ID for a project."""
        return self._active_sessions.get(project_id)

    def record_interaction(
        self,
        session_id: str,
        task_input: str,
        task_output: str,
        task_type: str,
        model: str | None = None,
        tokens_used: int | None = None,
        duration_ms: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Record an interaction in a session.

        Returns the interaction ID.
        """
        session = self._sessions.get(session_id)
        if not session:
            logger.warning(f"Session {session_id} not found")
            return ""

        interaction = InteractionRecord(
            id=str(uuid.uuid4()),
            timestamp=datetime.utcnow(),
            task_input=task_input,
            task_output=task_output,
            task_type=task_type,
            model=model,
            tokens_used=tokens_used,
            duration_ms=duration_ms,
            metadata=metadata or {},
        )

        session.interactions.append(interaction)
        session.last_activity = datetime.utcnow()

        # Save to disk
        self._save_session(session)

        logger.debug(f"Recorded interaction {interaction.id} in session {session_id}")

        return interaction.id

    async def get_context(
        self,
        session_id: str,
        limit: int = 5,
        include_outputs: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Get recent context from a session.

        Args:
            session_id: The session to query
            limit: Maximum number of interactions to return
            include_outputs: Whether to include task outputs

        Returns:
            List of interaction dicts
        """
        session = self._sessions.get(session_id)
        if not session:
            return []

        # Get last N interactions
        recent = session.interactions[-limit:] if session.interactions else []

        result = []
        for interaction in recent:
            item = {
                "timestamp": interaction.timestamp.isoformat(),
                "task_type": interaction.task_type,
                "task_input": interaction.task_input[:500],  # Truncate for context
            }

            if include_outputs:
                item["task_output"] = interaction.task_output[:500]

            if interaction.model:
                item["model"] = interaction.model

            result.append(item)

        return result

    async def search_sessions(
        self,
        project_id: str | None = None,
        query: str | None = None,
        task_type: str | None = None,
        limit: int = 10,
    ) -> list[SessionRecord]:
        """
        Search sessions by various criteria.

        Note: Full-text search requires embedding/semantic search.
        This is a basic keyword search.
        """
        results = []

        for session in self._sessions.values():
            # Filter by project
            if project_id and session.project_id != project_id:
                continue

            # Filter by task type
            if task_type:
                has_type = any(i.task_type == task_type for i in session.interactions)
                if not has_type:
                    continue

            # Filter by query (simple keyword match)
            if query:
                query_lower = query.lower()
                matches = any(
                    query_lower in i.task_input.lower() or query_lower in i.task_output.lower()
                    for i in session.interactions
                )
                if not matches:
                    continue

            results.append(session)

        # Sort by last activity (most recent first)
        results.sort(key=lambda s: s.last_activity, reverse=True)

        return results[:limit]

    async def summarize_session(self, session_id: str) -> str | None:
        """
        Summarize a session (placeholder for LLM summarization).

        In production, this would use an LLM to generate a summary.
        """
        session = self._sessions.get(session_id)
        if not session:
            return None

        # Simple summary generation
        task_types = {}
        for interaction in session.interactions:
            task_types[interaction.task_type] = task_types.get(interaction.task_type, 0) + 1

        total_tokens = sum(i.tokens_used or 0 for i in session.interactions)

        summary = f"Session with {len(session.interactions)} interactions. "
        summary += f"Task types: {', '.join(f'{k}({v})' for k, v in task_types.items())}. "
        summary += f"Total tokens: {total_tokens}."

        session.summary = summary
        session.status = SessionStatus.SUMMARIZED

        self._save_session(session)

        return summary

    async def archive_session(self, session_id: str) -> bool:
        """Archive a session to cold storage."""
        session = self._sessions.get(session_id)
        if not session:
            return False

        # Remove from active sessions
        self._active_sessions = {
            pid: sid for pid, sid in self._active_sessions.items()
            if sid != session_id
        }

        session.status = SessionStatus.ARCHIVED
        self._save_session(session)

        logger.info(f"Archived session {session_id}")
        return True

    def get_session_stats(self, project_id: str | None = None) -> dict[str, Any]:
        """Get statistics about sessions."""
        sessions = self._sessions.values()

        if project_id:
            sessions = [s for s in sessions if s.project_id == project_id]

        total = len(sessions)
        active = sum(1 for s in sessions if s.status == SessionStatus.ACTIVE)
        archived = sum(1 for s in sessions if s.status == SessionStatus.ARCHIVED)

        total_interactions = sum(len(s.interactions) for s in sessions)
        total_tokens = sum(
            sum(i.tokens_used or 0 for i in s.interactions)
            for s in sessions
        )

        return {
            "total_sessions": total,
            "active_sessions": active,
            "archived_sessions": archived,
            "total_interactions": total_interactions,
            "total_tokens": total_tokens,
            "by_tier": {
                "hot": sum(1 for s in sessions if s.tier == MemoryTier.HOT),
                "warm": sum(1 for s in sessions if s.tier == MemoryTier.WARM),
                "cold": sum(1 for s in sessions if s.tier == MemoryTier.COLD),
            },
        }

    def cleanup_old_sessions(self, days: int = 90) -> int:
        """Clean up sessions older than specified days."""
        cutoff = datetime.utcnow() - timedelta(days=days)
        removed = 0

        for session_id, session in list(self._sessions.items()):
            if session.last_activity < cutoff:
                # Remove from active if present
                if session.project_id in self._active_sessions:
                    del self._active_sessions[session.project_id]

                # Remove file
                file_path = self._session_file_path(session_id)
                if file_path.exists():
                    file_path.unlink()

                # Remove from memory
                del self._sessions[session_id]
                removed += 1

        if removed:
            logger.info(f"Cleaned up {removed} old sessions")

        return removed


# Global watcher instance
_default_watcher: SessionWatcher | None = None


def get_session_watcher() -> SessionWatcher:
    """Get the default session watcher instance."""
    global _default_watcher
    if _default_watcher is None:
        _default_watcher = SessionWatcher()
    return _default_watcher
