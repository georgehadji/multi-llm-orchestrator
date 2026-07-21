"""Unit tests for StateManager checkpoint pruning (F21)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

from orchestrator.budget import Budget
from orchestrator.infrastructure.state import StateManager
from orchestrator.models import ProjectState, ProjectStatus


@pytest.fixture
async def state_manager(tmp_path):
    db_path = tmp_path / "state.db"
    sm = StateManager(db_path=db_path)
    yield sm
    # Close any persistent connection opened during the test.
    if sm._conn is not None:
        await sm._conn.close()


@pytest.fixture
def project_state():
    return ProjectState(
        project_description="Test project",
        success_criteria="Tests pass",
        budget=Budget(max_usd=10.0, max_time_seconds=300),
        status=ProjectStatus.PARTIAL_SUCCESS,
    )


class TestCheckpointPruning:
    @pytest.mark.asyncio
    async def test_save_checkpoint_stores_state(self, state_manager, project_state):
        await state_manager.save_project("proj-1", project_state)
        await state_manager.save_checkpoint("proj-1", "task-1", project_state)
        loaded = await state_manager.load_latest_checkpoint("proj-1")
        assert loaded is not None
        assert loaded.project_description == "Test project"

    @pytest.mark.asyncio
    async def test_latest_checkpoint_is_retained_after_pruning(self, state_manager, project_state):
        await state_manager.save_project("proj-1", project_state)
        for i in range(15):
            state = ProjectState(
                project_description=f"iteration {i}",
                success_criteria="Tests pass",
                budget=Budget(max_usd=10.0, max_time_seconds=300),
                status=ProjectStatus.PARTIAL_SUCCESS,
            )
            await state_manager.save_checkpoint("proj-1", f"task-{i}", state)

        latest = await state_manager.load_latest_checkpoint("proj-1")
        assert latest is not None
        assert latest.project_description == "iteration 14"

    @pytest.mark.asyncio
    async def test_only_ten_checkpoints_retained_per_project(self, state_manager, project_state):
        await state_manager.save_project("proj-1", project_state)
        for i in range(15):
            state = ProjectState(
                project_description=f"iteration {i}",
                success_criteria="Tests pass",
                budget=Budget(max_usd=10.0, max_time_seconds=300),
                status=ProjectStatus.PARTIAL_SUCCESS,
            )
            await state_manager.save_checkpoint("proj-1", f"task-{i}", state)

        db = await state_manager._get_conn()
        async with db.execute(
            "SELECT COUNT(*) FROM checkpoints WHERE project_id = ?", ("proj-1",)
        ) as cursor:
            row = await cursor.fetchone()
        assert row[0] == 10

    @pytest.mark.asyncio
    async def test_pruning_is_isolated_per_project(self, state_manager, project_state):
        await state_manager.save_project("proj-1", project_state)
        await state_manager.save_project("proj-2", project_state)

        for i in range(12):
            await state_manager.save_checkpoint("proj-1", f"task-{i}", project_state)
            await state_manager.save_checkpoint("proj-2", f"task-{i}", project_state)

        db = await state_manager._get_conn()
        async with db.execute(
            "SELECT project_id, COUNT(*) FROM checkpoints GROUP BY project_id"
        ) as cursor:
            rows = await cursor.fetchall()
        counts = {project_id: count for project_id, count in rows}
        assert counts == {"proj-1": 10, "proj-2": 10}

    @pytest.mark.asyncio
    async def test_rollback_on_save_checkpoint_error(self, state_manager, project_state):
        await state_manager.save_project("proj-1", project_state)
        # A None state cannot be serialized, so the insert should fail and roll back.
        with pytest.raises((TypeError, AttributeError)):
            await state_manager.save_checkpoint("proj-1", "task-bad", None)

        db = await state_manager._get_conn()
        async with db.execute(
            "SELECT COUNT(*) FROM checkpoints WHERE project_id = ?", ("proj-1",)
        ) as cursor:
            row = await cursor.fetchone()
        assert row[0] == 0
