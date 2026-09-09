"""Re-export shim.

SessionWatcher's canonical implementation lives in
orchestrator.state_mgmt.session_watcher (timezone-aware, with the
SESSION-1/2/3 durability fixes). This used to be an independent full copy
that could silently diverge from the fixed one; it is now just a re-export
so there is a single source of truth.

Usage:
    from orchestrator.session_watcher import SessionWatcher, SessionRecord

    watcher = SessionWatcher()
    session_id = watcher.start_session("project_001")
    watcher.record_interaction(
        session_id=session_id,
        task_input="Write a function to calculate fibonacci",
        task_output="def fibonacci(n): ...",
        task_type="code_generation",
        metadata={"model": "gpt-4o", "tokens": 1500},
    )
    context = await watcher.get_context(session_id, limit=5)
    await watcher.archive_session(session_id)
"""

from __future__ import annotations

from .state_mgmt.session_watcher import (
    InteractionRecord,
    MemoryTier,
    SessionRecord,
    SessionStatus,
    SessionWatcher,
    get_session_watcher,
)

__all__ = [
    "InteractionRecord",
    "MemoryTier",
    "SessionRecord",
    "SessionStatus",
    "SessionWatcher",
    "get_session_watcher",
]
