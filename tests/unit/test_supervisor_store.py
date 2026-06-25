"""
Unit tests for orchestrator.supervisor.store.SupervisorStore.

Tests use temporary SQLite files so they are hermetic.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from orchestrator.supervisor.models import Lesson, SupervisorSession
from orchestrator.supervisor.store import SupervisorStore

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


@pytest.fixture
async def store(tmp_path: Path) -> SupervisorStore:
    s = SupervisorStore(db_path=tmp_path / "supervisor.db")
    await s.connect()
    yield s
    await s.close()


async def test_create_session(store: SupervisorStore):
    session = await store.create_session()
    assert isinstance(session, SupervisorSession)
    assert session.status == "idle"
    assert session.directive_count == 0


async def test_get_session(store: SupervisorStore):
    session = await store.create_session()
    fetched = await store.get_session(session.id)
    assert fetched is not None
    assert fetched.id == session.id


async def test_set_summary(store: SupervisorStore):
    session = await store.create_session()
    await store.set_summary(session.id, "all good", "completed")
    fetched = await store.get_session(session.id)
    assert fetched.status == "completed"
    assert fetched.summary == "all good"


async def test_increment_directive_count(store: SupervisorStore):
    session = await store.create_session()
    await store.increment_directive_count(session.id)
    fetched = await store.get_session(session.id)
    assert fetched.directive_count == 1


async def test_record_and_recent_lessons(store: SupervisorStore):
    session = await store.create_session()
    lesson = Lesson(
        id="l1",
        session_id=session.id,
        project_id="p1",
        task_type="code_gen",
        kind="task_failed",
        signal="t1",
        detail="error",
        created_at=time.time(),
    )
    await store.record_lesson(lesson)
    lessons = await store.recent_lessons(task_type="code_gen")
    assert len(lessons) == 1
    assert lessons[0].id == "l1"


async def test_recent_lessons_filter_by_task_type(store: SupervisorStore):
    session = await store.create_session()
    for idx, tt in enumerate(("code_gen", "code_gen", "review")):
        await store.record_lesson(
            Lesson(
                id=f"l-{idx}",
                session_id=session.id,
                project_id="p1",
                task_type=tt,
                kind="task_failed",
                signal="t",
                detail="d",
                created_at=time.time(),
            )
        )
    code_lessons = await store.recent_lessons(task_type="code_gen", limit=10)
    assert len(code_lessons) == 2


async def test_search_lessons(store: SupervisorStore):
    session = await store.create_session()
    await store.record_lesson(
        Lesson(
            id="l1",
            session_id=session.id,
            project_id="p1",
            task_type="code_gen",
            kind="task_failed",
            signal="truncated",
            detail="Output truncated when section >12k chars",
            created_at=time.time(),
        )
    )
    results = await store.search_lessons("truncated")
    assert len(results) == 1
    assert results[0].signal == "truncated"


async def test_secret_redaction(store: SupervisorStore):
    session = await store.create_session()
    detail = "api_key=sk-12345678901234567890 token=abc1234567890"
    await store.record_lesson(
        Lesson(
            id="l1",
            session_id=session.id,
            project_id="p1",
            task_type="general",
            kind="error",
            signal="secret",
            detail=detail,
            created_at=time.time(),
        )
    )
    lessons = await store.recent_lessons()
    assert "[REDACTED]" in lessons[0].detail
    assert "sk-12345678901234567890" not in lessons[0].detail


async def test_standalone_sk_key_redacted(store: SupervisorStore):
    # A raw OpenAI-style key with no api_key= prefix (e.g. inside an engine
    # exception message) must be fully redacted, not left as "sk-<secret>[REDACTED]".
    session = await store.create_session()
    await store.record_lesson(
        Lesson(
            id="l2",
            session_id=session.id,
            project_id="p1",
            task_type="general",
            kind="error",
            signal="secret",
            detail="RuntimeError: bad credential sk-ABCDEFGHIJKLMNOPQRSTUVWX rejected",
            created_at=time.time(),
        )
    )
    stored = (await store.recent_lessons())[0].detail
    assert "sk-ABCDEFGHIJKLMNOPQRSTUVWX" not in stored
    assert "ABCDEFGHIJKLMNOPQRSTUVWX" not in stored
    assert "[REDACTED]" in stored
