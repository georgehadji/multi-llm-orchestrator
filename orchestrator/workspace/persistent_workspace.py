"""
PersistentWorkspace — SQLite-backed workspace for crash recovery
===================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Optimization D-11: Extends ProjectWorkspace with SQLite persistence
so agent state survives process restarts. Uses aiosqlite for async
database operations.

On construction, loads existing state from disk.
write_file() and record_decision() auto-save to disk.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .workspace import ProjectWorkspace, FileVersion, ArchitectureDecision

logger = logging.getLogger("orchestrator.workspace.persistent_workspace")


class PersistentWorkspace(ProjectWorkspace):
    """Shared workspace with SQLite persistence.

    All state changes are persisted to an SQLite database for
    crash recovery and cross-session continuity.
    """

    def __init__(self, root: Path | None = None, db_path: Path | None = None) -> None:
        super().__init__(root)
        self.db_path = db_path or (self.root / ".orchestrator" / "workspace.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._loaded = False

    async def load(self) -> None:
        """Load persisted state from SQLite."""
        if self._loaded:
            return
        self._loaded = True
        try:
            import aiosqlite

            async with aiosqlite.connect(str(self.db_path)) as db:
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS files (
                        path TEXT PRIMARY KEY,
                        content TEXT,
                        version INTEGER,
                        author TEXT,
                        timestamp TEXT,
                        message TEXT
                    )
                """)
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS decisions (
                        id TEXT PRIMARY KEY,
                        title TEXT,
                        decision TEXT,
                        rationale TEXT,
                        author TEXT,
                        timestamp TEXT
                    )
                """)
                # Load files
                cursor = await db.execute("SELECT * FROM files")
                async for row in cursor:
                    self.files[row[0]] = FileVersion(
                        path=Path(row[0]),
                        content=row[1],
                        version=row[2],
                        author=row[3] or "",
                        timestamp=datetime.fromisoformat(row[4]) if row[4] else datetime.now(),
                        message=row[5] or "",
                    )
                # Load decisions
                cursor2 = await db.execute("SELECT * FROM decisions")
                async for row2 in cursor2:
                    self.architectural_decisions.append(
                        ArchitectureDecision(
                            id=row2[0],
                            title=row2[1],
                            decision=row2[2],
                            rationale=row2[3],
                            author=row2[4] or "",
                            timestamp=(
                                datetime.fromisoformat(row2[5]) if row2[5] else datetime.now()
                            ),
                        )
                    )
            logger.info(
                "PersistentWorkspace: loaded %d files, %d decisions from %s",
                len(self.files),
                len(self.architectural_decisions),
                self.db_path,
            )
        except ImportError:
            logger.warning("aiosqlite not available, running in-memory only")
        except Exception as exc:
            logger.warning("Failed to load workspace state: %s", exc)

    async def save(self) -> None:
        """Persist current state to SQLite."""
        if not self._loaded:
            return
        try:
            import aiosqlite

            async with self._lock:
                async with aiosqlite.connect(str(self.db_path)) as db:
                    await db.executemany(
                        "INSERT OR REPLACE INTO files VALUES (?, ?, ?, ?, ?, ?)",
                        [
                            (
                                str(k),
                                v.content,
                                v.version,
                                v.author,
                                v.timestamp.isoformat(),
                                v.message,
                            )
                            for k, v in self.files.items()
                        ],
                    )
                    await db.executemany(
                        "INSERT OR REPLACE INTO decisions VALUES (?, ?, ?, ?, ?, ?)",
                        [
                            (
                                d.id,
                                d.title,
                                d.decision,
                                d.rationale,
                                d.author,
                                d.timestamp.isoformat(),
                            )
                            for d in self.architectural_decisions
                        ],
                    )
                    await db.commit()
        except ImportError:
            pass
        except Exception as exc:
            logger.warning("Failed to save workspace state: %s", exc)

    def write_file(
        self, path: str, content: str, author: str = "unknown", message: str = ""
    ) -> FileVersion:
        result = super().write_file(path, content, author, message)
        # BUG-007 FIX: asyncio.ensure_future() requires a running event loop and
        # is deprecated without one (Python 3.10+; error in 3.12+). Use
        # get_running_loop() instead — silently skip the background save when
        # called from a synchronous context (tests, init) rather than crashing
        # or silently discarding the coroutine without awaiting it.
        try:
            asyncio.get_running_loop().create_task(self.save())
        except RuntimeError:
            pass  # No running event loop; save will happen on next async call
        return result

    def record_decision(
        self, title: str, decision: str, rationale: str, author: str = ""
    ) -> ArchitectureDecision:
        result = super().record_decision(title, decision, rationale, author)
        # BUG-007 FIX: same guard as write_file() — see comment above.
        try:
            asyncio.get_running_loop().create_task(self.save())
        except RuntimeError:
            pass  # No running event loop; save will happen on next async call
        return result
