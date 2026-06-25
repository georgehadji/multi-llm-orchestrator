"""
Supervisor service — persistent identity, lesson injection, engine delegation.

The Supervisor sits above ``engine.py`` and treats the existing ``Orchestrator``
as its single worker.  It wires intake, memory, and execution without adding
business logic to the engine.

Author: Georgios-Chrysovalantis Chatzivantsidis
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Any

from orchestrator.engine import Orchestrator
from orchestrator.unified_events.core import DomainEvent, EventType

from .failure_tap import event_to_lesson
from .intake import normalize
from .models import Directive, JobArgs, Lesson, SupervisorResult
from .store import SupervisorStore

logger = logging.getLogger("orchestrator.supervisor.service")

_DEFAULT_LESSON_CAP = 5


class _RunState:
    """Mutable state carried through an execution run."""

    def __init__(self) -> None:
        self.lessons_recorded = 0
        self.project_status = "unknown"
        self.output_dir: str | None = None
        self.result: SupervisorResult | None = None


class Supervisor:
    """Persistent supervisor that delegates work to the engine and learns."""

    def __init__(
        self,
        store: SupervisorStore,
        orchestrator_factory: Callable[[float | None], Orchestrator],
        lesson_cap: int = _DEFAULT_LESSON_CAP,
    ) -> None:
        self._store = store
        self._orchestrator_factory = orchestrator_factory
        self._lesson_cap = lesson_cap

    async def handle(
        self,
        directive: Directive,
        on_event: Callable[[Any], None] | None = None,
    ) -> SupervisorResult:
        """Execute a directive and return a summary result.

        An optional ``on_event`` callback receives every streamed event, letting
        CLI/WebSocket adapters render progress without consuming the async
        generator themselves.
        """
        state = _RunState()
        async for _event in self._run(directive, state):
            if on_event is not None:
                on_event(_event)
        assert state.result is not None
        return state.result

    async def stream(self, directive: Directive) -> AsyncIterator[Any]:
        """Stream execution events for a directive."""
        state = _RunState()
        async for event in self._run(directive, state):
            yield event

    async def _run(
        self,
        directive: Directive,
        state: _RunState,
    ) -> AsyncIterator[Any]:
        session = await self._store.create_session()
        await self._store.increment_directive_count(session.id)

        job = normalize(directive)
        project_id = job.project_id or session.id

        lessons = await self._recent_lessons(job)
        description = self._inject_lessons(job.project_description, lessons)

        try:
            orch = self._orchestrator_factory(job.budget)
            async for event in orch.run_project_streaming(
                project_description=description,
                success_criteria=job.success_criteria,
                project_id=project_id,
            ):
                yield event
                if await self._maybe_record_lesson(event, session.id, project_id):
                    state.lessons_recorded += 1
                if (
                    isinstance(event, DomainEvent)
                    and event.event_type == EventType.PROJECT_COMPLETED
                ):
                    state.project_status = event.metadata.get("status", "unknown")

        except Exception as exc:
            logger.exception("Supervisor engine execution failed")
            state.project_status = "failed"
            await self._record_engine_exception(exc, session.id, project_id, state)
        finally:
            summary = f"status={state.project_status}"
            await self._store.set_summary(session.id, summary, state.project_status)

        state.result = SupervisorResult(
            session_id=session.id,
            project_status=state.project_status,
            lessons_recorded=state.lessons_recorded,
            output_dir=state.output_dir,
        )

    async def _maybe_record_lesson(
        self,
        event: Any,
        session_id: str,
        project_id: str,
    ) -> bool:
        """Record a lesson from an event.  Returns True if a lesson was stored."""
        task_type = _task_type_from_event(event)
        lesson = event_to_lesson(event, session_id, project_id, task_type)
        if lesson is None:
            return False
        try:
            await self._store.record_lesson(lesson)
            return True
        except Exception:
            logger.exception("Failed to record lesson; continuing")
            return False

    async def _record_engine_exception(
        self,
        exc: Exception,
        session_id: str,
        project_id: str,
        state: _RunState,
    ) -> None:
        try:
            await self._store.record_lesson(
                Lesson(
                    id=_new_id(),
                    session_id=session_id,
                    project_id=project_id,
                    task_type="general",
                    kind="error",
                    signal="engine_exception",
                    detail=str(exc)[:2000],
                    created_at=time.time(),
                )
            )
            state.lessons_recorded += 1
        except Exception:
            logger.exception("Failed to record engine-exception lesson")

    async def _recent_lessons(self, job: JobArgs) -> list[Lesson]:
        # Phase 1: task_type is not yet known at intake, so retrieve general
        # recent lessons.  Phase 3 will classify the directive and use semantic
        # retrieval.
        try:
            return await self._store.recent_lessons(task_type=None, limit=self._lesson_cap)
        except Exception:
            logger.exception("Failed to retrieve lessons; running without injection")
            return []

    def _inject_lessons(self, description: str, lessons: list[Lesson]) -> str:
        """Prepend a capped, formatted block of relevant lessons."""
        if not lessons:
            return description
        lines = ["## Known pitfalls (learned from prior runs) — avoid repeating:"]
        for lesson in lessons:
            lines.append(f"- [{lesson.task_type}] {lesson.signal}: {lesson.detail}")
        block = "\n".join(lines)
        return f"{block}\n\n{description}"


def _task_type_from_event(event: Any) -> str:
    if isinstance(event, DomainEvent):
        return str(event.metadata.get("task_type", "general"))
    return "general"


def _new_id() -> str:
    return uuid.uuid4().hex[:16]
