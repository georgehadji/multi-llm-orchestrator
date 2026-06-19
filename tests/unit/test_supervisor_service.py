"""
Unit tests for orchestrator.supervisor.service.Supervisor.

Uses a fake Orchestrator to verify delegation, lesson capture, and re-injection
without calling any LLM or doing real I/O.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from orchestrator.supervisor.models import Directive, Lesson
from orchestrator.supervisor.service import Supervisor
from orchestrator.supervisor.store import SupervisorStore
from orchestrator.unified_events.core import (
    ProjectCompletedEvent,
    TaskFailedEvent,
)

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


class CaptureOrchestrator:
    """Fake orchestrator that records run_project_streaming arguments."""

    def __init__(self, events: list[Any]) -> None:
        self.events = events
        self.calls: list[dict[str, Any]] = []

    async def run_project_streaming(
        self,
        project_description: str,
        success_criteria: str,
        project_id: str = "",
    ) -> Any:
        self.calls.append(
            {
                "project_description": project_description,
                "success_criteria": success_criteria,
                "project_id": project_id,
            }
        )
        for event in self.events:
            yield event


@pytest.fixture
async def store(tmp_path: Path) -> SupervisorStore:
    s = SupervisorStore(db_path=tmp_path / "supervisor.db")
    await s.connect()
    yield s
    await s.close()


async def test_handle_success_records_session(store: SupervisorStore):
    orch = CaptureOrchestrator([ProjectCompletedEvent("agg", "p1", "completed", 0.1)])
    supervisor = Supervisor(store, lambda _budget=None: orch)
    result = await supervisor.handle(Directive(source="human", text="build api", budget=5.0))
    assert result.project_status == "completed"
    assert result.lessons_recorded == 0
    session = await store.get_session(result.session_id)
    assert session is not None
    assert session.status == "completed"


async def test_per_directive_budget_passed_to_factory(store: SupervisorStore):
    orch = CaptureOrchestrator([ProjectCompletedEvent("agg", "p1", "completed", 0.1)])
    captured: dict[str, Any] = {}

    def factory(budget: float | None = None) -> CaptureOrchestrator:
        captured["budget"] = budget
        return orch

    supervisor = Supervisor(store, factory)
    await supervisor.handle(Directive(source="agent", text="build it", budget=1.0))
    assert captured["budget"] == 1.0


async def test_handle_records_lesson_on_task_failure(store: SupervisorStore):
    events = [
        TaskFailedEvent("agg", "t1", "timeout", will_retry=False),
        ProjectCompletedEvent("agg", "p1", "failed", 0.2, tasks_completed=0, tasks_failed=1),
    ]
    orch = CaptureOrchestrator(events)
    supervisor = Supervisor(store, lambda _budget=None: orch)
    result = await supervisor.handle(Directive(source="human", text="build api", budget=5.0))
    assert result.project_status == "failed"
    assert result.lessons_recorded == 2  # task_failed + degraded completion
    lessons = await store.recent_lessons(task_type="general", limit=10)
    kinds = {lesson.kind for lesson in lessons}
    assert "task_failed" in kinds
    assert "degraded" in kinds


async def test_lessons_injected_into_description(store: SupervisorStore):
    await store.record_lesson(
        Lesson(
            id="l1",
            session_id="session-a",
            project_id="p1",
            task_type="general",
            kind="task_failed",
            signal="truncated",
            detail="Output truncated when section >12k chars",
            created_at=time.time(),
        )
    )
    orch = CaptureOrchestrator([ProjectCompletedEvent("agg", "p1", "completed", 0.1)])
    supervisor = Supervisor(store, lambda _budget=None: orch)
    await supervisor.handle(Directive(source="human", text="build api", budget=5.0))
    assert len(orch.calls) == 1
    description = orch.calls[0]["project_description"]
    assert "Known pitfalls" in description
    assert "truncated" in description


async def test_stream_yields_events(store: SupervisorStore):
    events = [
        TaskFailedEvent("agg", "t1", "timeout", will_retry=False),
        ProjectCompletedEvent("agg", "p1", "failed", 0.2, tasks_completed=0, tasks_failed=1),
    ]
    orch = CaptureOrchestrator(events)
    supervisor = Supervisor(store, lambda _budget=None: orch)
    seen = []
    async for event in supervisor.stream(Directive(source="agent", text="build api")):
        seen.append(event)
    assert len(seen) == 2


async def test_on_event_callback(store: SupervisorStore):
    orch = CaptureOrchestrator([ProjectCompletedEvent("agg", "p1", "completed", 0.1)])
    supervisor = Supervisor(store, lambda _budget=None: orch)
    seen = []
    result = await supervisor.handle(
        Directive(source="human", text="build api"),
        on_event=seen.append,
    )
    assert result.project_status == "completed"
    assert len(seen) == 1
