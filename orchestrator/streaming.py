"""
Streaming event types for run_project_streaming().

Events flow through ProjectEventBus — an asyncio-based pub-sub hub that
fans out to all subscribers. Each subscriber gets an independent async
generator (AsyncIterator) over the event stream.
"""
from __future__ import annotations
import asyncio
from dataclasses import dataclass
from typing import AsyncIterator, Optional, Union
from .models import TaskStatus


# ── Event dataclasses ─────────────────────────────────────────────────────────

@dataclass
class ProjectStarted:
    project_id: str
    total_tasks: int
    budget_usd: float


@dataclass
class TaskStarted:
    task_id: str
    task_type: str
    model: str


@dataclass
class TaskProgressUpdate:
    task_id: str
    iteration: int
    score: float
    best_score: float


@dataclass
class TaskCompleted:
    task_id: str
    score: float
    status: TaskStatus
    model: str
    cost_usd: float
    iterations: int


@dataclass
class TaskFailed:
    task_id: str
    reason: str
    model: str


@dataclass
class BudgetWarning:
    phase: str
    spent_usd: float
    cap_usd: float
    ratio: float


@dataclass
class ProjectCompleted:
    project_id: str
    status: str          # ProjectStatus.value
    total_cost_usd: float
    elapsed_seconds: float
    tasks_completed: int
    tasks_failed: int


StreamEvent = Union[
    ProjectStarted, TaskStarted, TaskProgressUpdate,
    TaskCompleted, TaskFailed, BudgetWarning, ProjectCompleted,
]


# ── Event bus ─────────────────────────────────────────────────────────────────

_SENTINEL = object()   # marks end-of-stream


class ProjectEventBus:
    """
    Fan-out pub-sub hub.  Each call to subscribe() returns an independent
    AsyncIterator that yields every event published after the subscription.
    Call close() to signal end-of-stream to all subscribers.
    """

    def __init__(self) -> None:
        self._queues: list[asyncio.Queue] = []
        self._closed = False

    def subscribe(self) -> AsyncIterator[StreamEvent]:
        q: asyncio.Queue = asyncio.Queue()
        self._queues.append(q)
        return self._drain(q)

    async def _drain(self, q: asyncio.Queue) -> AsyncIterator[StreamEvent]:
        while True:
            item = await q.get()
            if item is _SENTINEL:
                return
            yield item

    async def publish(self, event: StreamEvent) -> None:
        for q in self._queues:
            await q.put(event)

    async def close(self) -> None:
        self._closed = True
        for q in self._queues:
            await q.put(_SENTINEL)
