"""
Failure tap — map execution events to structured lessons.

Subscribes to the unified event stream and turns failure/degradation events
into append-only ``Lesson`` records.

Author: Georgios-Chrysovalantis Chatzivantsidis
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from orchestrator.unified_events.core import (
    BudgetWarningEvent,
    DomainEvent,
    EventType,
    ProjectCompletedEvent,
    TaskFailedEvent,
)

from .models import Lesson, LessonKind

logger = logging.getLogger("orchestrator.supervisor.failure_tap")

_KIND_FOR_EVENT_TYPE: dict[EventType, LessonKind] = {
    EventType.TASK_FAILED: "task_failed",
    EventType.VALIDATION_FAILED: "validation_failed",
    EventType.ERROR: "error",
    EventType.BUDGET_WARNING: "budget",
}


def event_to_lesson(
    event: Any,
    session_id: str,
    project_id: str,
    task_type: str = "general",
) -> Lesson | None:
    """Convert a domain event into a Lesson, if applicable.

    Handles task failures, degraded completions, validation failures, budget
    warnings, and generic error events.
    """
    kind: LessonKind | None = None
    signal = ""
    detail = ""

    if isinstance(event, TaskFailedEvent):
        kind = "task_failed"
        signal = str(event.metadata.get("task_id", "unknown"))[:200]
        detail = str(event.metadata.get("error", ""))[:2000]
    elif isinstance(event, ProjectCompletedEvent):
        status = event.metadata.get("status", "")
        tasks_failed = event.metadata.get("tasks_failed", 0)
        if status in ("failed", "degraded") or tasks_failed:
            kind = "degraded"
            signal = f"status={status}"
            detail = (
                f"tasks_failed={tasks_failed}, "
                f"tasks_completed={event.metadata.get('tasks_completed', 0)}, "
                f"total_cost={event.metadata.get('total_cost', 0)}"
            )
    elif isinstance(event, BudgetWarningEvent):
        kind = "budget"
        signal = f"phase={event.metadata.get('phase', 'unknown')}"
        detail = (
            f"spent={event.metadata.get('spent', 0)} "
            f"cap={event.metadata.get('cap', 0)} "
            f"ratio={event.metadata.get('ratio', 0)}"
        )
    elif isinstance(event, DomainEvent):
        event_kind = _KIND_FOR_EVENT_TYPE.get(event.event_type)
        if event_kind:
            kind = event_kind
            signal = event.aggregate_id
            detail = str(event.metadata)

    if kind is None:
        return None

    return Lesson(
        id=_new_id(),
        session_id=session_id,
        project_id=project_id,
        task_type=task_type,
        kind=kind,
        signal=signal,
        detail=detail,
        created_at=time.time(),
    )


def _new_id() -> str:
    return uuid.uuid4().hex[:16]
