"""
Unit tests for orchestrator.application.skill_store.SkillStore

Tests use temporary in-memory-equivalent SQLite files (tmp_path) so they are
hermetic and leave no files on the developer's machine.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from orchestrator.application.skill_store import SkillStore
from orchestrator.models import TaskType
from orchestrator.models_skill import SkillPatch, Trajectory

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
async def store(tmp_path: Path) -> SkillStore:
    """Return an open SkillStore backed by temp SQLite files."""
    from orchestrator.infrastructure.skill_store_adapter import SkillDbAdapter
    db = SkillDbAdapter(
        traj_path=tmp_path / "trajectories.db",
        skill_path=tmp_path / "skills.db",
    )
    await db.connect()
    s = SkillStore(db)
    yield s
    await s.close()


def _traj(task_type: TaskType = TaskType.CODE_GEN, score: float = 0.8) -> Trajectory:
    return Trajectory(
        task_id="t1",
        task_type=task_type,
        prompt="write hello world",
        output="print('hello')",
        score=score,
        critique_text="good",
        model_used="openai/gpt-4o-mini",
        cost_usd=0.001,
        recorded_at=time.time(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory tests
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_save_and_load_trajectory(store: SkillStore):
    t = _traj()
    await store.save_trajectory(t)
    rows = await store.load_trajectories(TaskType.CODE_GEN)
    assert len(rows) == 1
    assert rows[0].task_id == "t1"
    assert rows[0].score == pytest.approx(0.8)


@pytest.mark.asyncio
async def test_load_trajectories_empty(store: SkillStore):
    rows = await store.load_trajectories(TaskType.CODE_GEN)
    assert rows == []


@pytest.mark.asyncio
async def test_load_trajectories_respects_limit(store: SkillStore):
    for i in range(20):
        t = Trajectory(
            task_id=f"t{i}",
            task_type=TaskType.CODE_GEN,
            prompt="x",
            output="y",
            score=0.5 + i * 0.01,
            critique_text="",
            model_used="m",
            cost_usd=0.0,
            recorded_at=time.time() + i,
        )
        await store.save_trajectory(t)

    rows = await store.load_trajectories(TaskType.CODE_GEN, limit=5)
    assert len(rows) == 5


@pytest.mark.asyncio
async def test_trajectories_filtered_by_task_type(store: SkillStore):
    await store.save_trajectory(_traj(TaskType.CODE_GEN))
    await store.save_trajectory(_traj(TaskType.REASONING))
    code_rows = await store.load_trajectories(TaskType.CODE_GEN)
    reason_rows = await store.load_trajectories(TaskType.REASONING)
    assert len(code_rows) == 1
    assert len(reason_rows) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Skill tests
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_load_best_skill_none_initially(store: SkillStore):
    result = await store.load_best_skill(TaskType.CODE_GEN)
    assert result is None


@pytest.mark.asyncio
async def test_save_and_load_best_skill(store: SkillStore):
    await store.save_skill(TaskType.CODE_GEN, "# Skill\nUse types.", 0.75, epoch=1)
    result = await store.load_best_skill(TaskType.CODE_GEN)
    assert result is not None
    doc, score, epoch = result
    assert "Use types" in doc
    assert score == pytest.approx(0.75)
    assert epoch == 1


@pytest.mark.asyncio
async def test_save_skill_demotes_previous_best(store: SkillStore):
    await store.save_skill(TaskType.CODE_GEN, "old skill", 0.5, epoch=1)
    await store.save_skill(TaskType.CODE_GEN, "new skill", 0.7, epoch=2)
    result = await store.load_best_skill(TaskType.CODE_GEN)
    assert result is not None
    doc, score, epoch = result
    assert "new skill" in doc
    assert score == pytest.approx(0.7)


# ─────────────────────────────────────────────────────────────────────────────
# Negative-feedback buffer tests
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_save_and_load_negative_feedback(store: SkillStore):
    patches = [SkillPatch(op="append", anchor="", content="bad idea", token_cost=5)]
    await store.save_negative_feedback(TaskType.CODE_GEN, patches, "no improvement")
    rows = await store.load_negative_feedback(TaskType.CODE_GEN)
    assert len(rows) == 1
    assert rows[0]["rejection_reason"] == "no improvement"
    assert rows[0]["patches"][0]["content"] == "bad idea"


@pytest.mark.asyncio
async def test_load_negative_feedback_empty(store: SkillStore):
    rows = await store.load_negative_feedback(TaskType.CODE_GEN)
    assert rows == []
