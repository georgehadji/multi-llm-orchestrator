"""
Unit tests for orchestrator.supervisor.failure_tap.

Verifies that each supported event type maps to the correct Lesson kind.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

from orchestrator.supervisor.failure_tap import event_to_lesson
from orchestrator.supervisor.models import Lesson
from orchestrator.unified_events.core import (
    BudgetWarningEvent,
    ProjectCompletedEvent,
    TaskCompletedEvent,
    TaskFailedEvent,
)

pytestmark = pytest.mark.unit


def _assert_lesson(lesson: Lesson | None, kind: str) -> None:
    assert lesson is not None
    assert lesson.kind == kind
    assert lesson.session_id == "s1"
    assert lesson.project_id == "p1"
    assert len(lesson.id) == 16


def test_task_failed_event():
    event = TaskFailedEvent("agg", "t1", "timeout", will_retry=False)
    lesson = event_to_lesson(event, "s1", "p1", "code_gen")
    _assert_lesson(lesson, "task_failed")
    assert lesson.signal == "t1"
    assert "timeout" in lesson.detail


def test_degraded_project_completed():
    event = ProjectCompletedEvent("agg", "p1", "degraded", 0.5, tasks_completed=3, tasks_failed=2)
    lesson = event_to_lesson(event, "s1", "p1", "code_gen")
    _assert_lesson(lesson, "degraded")
    assert "tasks_failed=2" in lesson.detail


def test_failed_project_completed():
    event = ProjectCompletedEvent("agg", "p1", "failed", 0.5, tasks_completed=1, tasks_failed=5)
    lesson = event_to_lesson(event, "s1", "p1", "code_gen")
    _assert_lesson(lesson, "degraded")


@pytest.mark.parametrize(
    "status",
    ["COMPLETED_DEGRADED", "PARTIAL_SUCCESS", "BUDGET_EXHAUSTED", "SYSTEM_FAILURE"],
)
def test_uppercase_degraded_status_records_lesson(status):
    # Real ProjectStatus degraded/failure values are uppercase; a run can be
    # degraded with tasks_failed == 0 (e.g. validation failure) and must still
    # produce a lesson.
    event = ProjectCompletedEvent("agg", "p1", status, 0.5, tasks_completed=3, tasks_failed=0)
    lesson = event_to_lesson(event, "s1", "p1", "code_gen")
    _assert_lesson(lesson, "degraded")
    assert status in lesson.signal


def test_budget_warning_event():
    event = BudgetWarningEvent("agg", "generation", 4.5, 5.0, 0.9)
    lesson = event_to_lesson(event, "s1", "p1", "code_gen")
    _assert_lesson(lesson, "budget")
    assert "phase=generation" in lesson.signal


def test_successful_completion_yields_no_lesson():
    event = ProjectCompletedEvent("agg", "p1", "completed", 0.1, tasks_completed=5, tasks_failed=0)
    lesson = event_to_lesson(event, "s1", "p1", "code_gen")
    assert lesson is None


def test_task_completed_yields_no_lesson():
    event = TaskCompletedEvent("agg", "t1", score=0.9, cost=0.01, duration_ms=100)
    lesson = event_to_lesson(event, "s1", "p1", "code_gen")
    assert lesson is None
