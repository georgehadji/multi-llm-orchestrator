"""
MultipleContexts - Separate AI conversations, shared codebase.
================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis
Part of Category 3, Phase D1 (Dyad-inspired).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import json, logging, time, uuid

logger = logging.getLogger(__name__)


@dataclass
class AIThread:
    thread_id: str
    name: str
    created_at: float = 0.0
    last_active: float = 0.0
    context_summary: str = ""
    active: bool = True
    metadata: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "thread_id": self.thread_id,
            "name": self.name,
            "created_at": self.created_at,
            "last_active": self.last_active,
            "context_summary": self.context_summary,
            "active": self.active,
            "metadata": self.metadata,
        }


class MultiContextManager:
    """Manages multiple AI conversation threads sharing a codebase."""

    def __init__(self, project_dir="."):
        self._dir = Path(project_dir) / ".ai_threads"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._threads: dict[str, AIThread] = {}
        self._active_thread: str = ""
        self._load()

    def _load(self):
        fp = self._dir / "threads.json"
        if fp.exists():
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
                self._threads = {t["thread_id"]: AIThread(**t) for t in data.get("threads", [])}
                self._active_thread = data.get("active_thread", "")
            except Exception:
                pass

    def _save(self):
        (self._dir / "threads.json").write_text(
            json.dumps(
                {
                    "threads": [t.to_dict() for t in self._threads.values()],
                    "active_thread": self._active_thread,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def create_thread(self, name, context_summary=""):
        """Create a new AI conversation thread."""
        tid = f"thread_{uuid.uuid4().hex[:8]}"
        thread = AIThread(
            thread_id=tid,
            name=name,
            created_at=time.time(),
            last_active=time.time(),
            context_summary=context_summary,
        )
        self._threads[tid] = thread
        self._save()
        return thread

    def switch_to(self, thread_id):
        """Switch active conversation to another thread."""
        if thread_id in self._threads:
            self._threads[thread_id].last_active = time.time()
            self._active_thread = thread_id
            self._save()
            return self._threads[thread_id]
        return None

    def get_active(self):
        return self._threads.get(self._active_thread) if self._active_thread else None

    def summarize_all(self):
        """Get a summary of all threads for context injection."""
        lines = ["## Active AI Threads", ""]
        for t in self._threads.values():
            marker = " [ACTIVE]" if t.thread_id == self._active_thread else ""
            lines.append(f"- **{t.name}**{marker}: {t.context_summary}")
        return "\n".join(lines)

    def list_threads(self):
        return [
            {
                "id": t.thread_id,
                "name": t.name,
                "active": t.thread_id == self._active_thread,
                "last_active": time.strftime("%H:%M", time.localtime(t.last_active)),
                "summary": t.context_summary[:80],
            }
            for t in self._threads.values()
        ]

    def merge_threads(self, thread_ids, new_name):
        """Merge multiple threads into one."""
        merged_context = []
        for tid in thread_ids:
            if tid in self._threads:
                merged_context.append(
                    f"[{self._threads[tid].name}] {self._threads[tid].context_summary}"
                )
        thread = self.create_thread(new_name, " | ".join(merged_context))
        return thread
