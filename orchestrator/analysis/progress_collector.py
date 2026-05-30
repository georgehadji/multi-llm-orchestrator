"""
ProgressCollector - Streaming progress for dashboard UI.
===========================================================
Author: Georgios-Chrysovalantis Chatzivantsidis
Part of Cat 7, Phase U6.
"""

from __future__ import annotations
from dataclasses import dataclass
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class ProgressEvent:
    timestamp: float = 0.0
    event_type: str = ""
    task_id: str = ""
    progress: float = 0.0
    status: str = ""
    details: str = ""
    model: str = ""
    cost_usd: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0


class ProgressCollector:
    """Collects streaming progress events for dashboard status bar."""

    def __init__(self):
        self._events: list[ProgressEvent] = []
        self._task_progress: dict[str, float] = {}
        self._total_cost: float = 0.0
        self._total_tokens: int = 0

    def task_start(self, task_id, task_name=""):
        e = ProgressEvent(
            timestamp=time.time(), event_type="task_start", task_id=task_id, details=task_name
        )
        self._events.append(e)
        return e

    def task_progress(self, task_id, progress, status="", details=""):
        self._task_progress[task_id] = progress
        e = ProgressEvent(
            timestamp=time.time(),
            event_type="task_progress",
            task_id=task_id,
            progress=progress,
            status=status,
            details=details,
        )
        self._events.append(e)
        return e

    def task_complete(self, task_id, score=0.0, cost=0.0, model=""):
        self._task_progress[task_id] = 1.0
        self._total_cost += cost
        e = ProgressEvent(
            timestamp=time.time(),
            event_type="task_complete",
            task_id=task_id,
            progress=1.0,
            cost_usd=cost,
            model=model,
        )
        self._events.append(e)
        return e

    def record_cost(self, model, cost, tokens_in, tokens_out, latency_ms):
        self._total_cost += cost
        self._total_tokens += tokens_in + tokens_out
        e = ProgressEvent(
            timestamp=time.time(),
            event_type="cost_update",
            model=model,
            cost_usd=cost,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
        )
        self._events.append(e)
        return e

    def get_status(self):
        return {
            "cost_usd": self._total_cost,
            "total_tokens": self._total_tokens,
            "tasks": dict(self._task_progress),
            "completed": sum(1 for v in self._task_progress.values() if v >= 1.0),
            "total": len(self._task_progress),
            "last_event": self._events[-1].event_type if self._events else "none",
        }

    def clear(self):
        self._events.clear()
        self._task_progress.clear()
