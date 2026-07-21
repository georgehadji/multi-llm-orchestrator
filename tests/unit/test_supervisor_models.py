"""
Unit tests for orchestrator.supervisor.models.

Covers construction and purity rules for the Supervisor data classes.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

from orchestrator.supervisor.models import (
    Directive,
    JobArgs,
    Lesson,
    SupervisorResult,
)

pytestmark = pytest.mark.unit


def test_directive_defaults():
    d = Directive(source="human", text="build an api")
    assert d.source == "human"
    assert d.text == "build an api"
    assert d.project_id == ""
    assert d.budget is None
    assert d.metadata == {}


def test_directive_frozen():
    d = Directive(source="human", text="x")
    with pytest.raises(AttributeError):
        d.text = "y"


def test_lesson_frozen():
    lesson = Lesson(
        id="l1",
        session_id="s1",
        project_id="p1",
        task_type="code_gen",
        kind="task_failed",
        signal="t1",
        detail="boom",
        created_at=1.0,
    )
    with pytest.raises(AttributeError):
        lesson.detail = "nope"


def test_supervisor_result_creation():
    r = SupervisorResult(
        session_id="s1",
        project_status="completed",
        lessons_recorded=2,
    )
    assert r.session_id == "s1"
    assert r.output_dir is None


def test_job_args_creation():
    j = JobArgs(
        project_description="desc",
        success_criteria="criteria",
        budget=5.0,
        project_id="p1",
    )
    assert j.project_id == "p1"
