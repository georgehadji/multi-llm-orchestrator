"""
Integration tests for PipelineRunner.execute_all()

Tests level-based parallel execution with Semaphore-bounded concurrency,
dependency resolution, error propagation, and progress tracking.
"""

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.engine_core.pipeline_runner import PipelineRunner
from orchestrator.engine_core.project_planner import ProjectPlanner
from orchestrator.models import Task, TaskType, TaskStatus


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_pipeline():
    pipeline = MagicMock()
    pipeline.run = AsyncMock(return_value="mocked_result")
    return pipeline


@pytest.fixture
def planner():
    return ProjectPlanner()


@pytest.fixture
def runner(mock_pipeline, planner):
    return PipelineRunner(
        pipeline=mock_pipeline,
        planner=planner,
        max_parallel_tasks=3,
    )


@pytest.fixture
def sequential_runner(mock_pipeline, planner):
    return PipelineRunner(
        pipeline=mock_pipeline,
        planner=planner,
        max_parallel_tasks=1,
    )


def make_task(tid: str, deps: list[str] | None = None) -> Task:
    return Task(
        id=tid,
        type=TaskType.CODE_GEN,
        prompt=f"Task {tid}",
        dependencies=deps or [],
        status=TaskStatus.PENDING,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestPipelineRunnerExecution:
    """Integration tests for PipelineRunner.execute_all()."""

    @pytest.mark.asyncio
    async def test_executes_tasks_in_level_order(self, runner):
        """Tasks with no dependencies run first; dependent tasks run after."""
        tasks = {
            "a": make_task("a"),
            "b": make_task("b", deps=["a"]),
            "c": make_task("c", deps=["a"]),
            "d": make_task("d", deps=["b", "c"]),
        }
        exec_order = ["a", "b", "c", "d"]
        executed: list[str] = []

        async def execute_fn(task):
            executed.append(task.id)
            return {"task_id": task.id, "status": "ok"}

        await runner.execute_all(
            tasks=tasks,
            execution_order=exec_order,
            project_desc="Level order test",
            success_criteria="All pass",
            execute_task_fn=execute_fn,
        )

        # Level 0: a (no deps)
        # Level 1: b, c (depend on a) — order may vary
        # Level 2: d (depends on b, c)
        assert executed[0] == "a", f"First task should be 'a' but got {executed}"
        assert executed[-1] == "d", f"Last task should be 'd' but got {executed}"
        assert set(executed[1:3]) == {"b", "c"}, f"Middle tasks should be b,c but got {executed}"
        # Verify all 4 tasks ran
        assert len(executed) == 4

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrency(self, sequential_runner):
        """With max_parallel_tasks=1, independent tasks execute sequentially."""
        tasks = {"a": make_task("a"), "b": make_task("b"), "c": make_task("c")}
        executed: list[str] = []

        async def execute_fn(task):
            executed.append(task.id)
            return {"task_id": task.id, "status": "ok"}

        start = time.monotonic()
        await sequential_runner.execute_all(
            tasks=tasks,
            execution_order=["a", "b", "c"],
            project_desc="Concurrency test",
            success_criteria="All pass",
            execute_task_fn=execute_fn,
        )
        elapsed = time.monotonic() - start

        assert executed == ["a", "b", "c"], f"Sequential order expected, got {executed}"

    @pytest.mark.asyncio
    async def test_task_failure_does_not_crash_runner(self, runner):
        """A single task failure doesn't prevent other tasks from completing."""
        tasks = {
            "a": make_task("a"),
            "b": make_task("b", deps=["a"]),
            "c": make_task("c", deps=["a"]),
        }
        executed: list[str] = []

        async def execute_fn(task):
            if task.id == "b":
                raise RuntimeError(f"Simulated failure in {task.id}")
            executed.append(task.id)
            return {"task_id": task.id, "status": "ok"}

        await runner.execute_all(
            tasks=tasks,
            execution_order=["a", "b", "c"],
            project_desc="Failure test",
            success_criteria="All pass",
            execute_task_fn=execute_fn,
        )

        assert "a" in executed, "Task 'a' should have executed"
        assert "c" in executed, "Task 'c' should have executed despite 'b' failing"
        assert "b" not in executed, "Task 'b' should NOT have executed (it raises)"

    @pytest.mark.asyncio
    async def test_empty_task_list(self, runner):
        """execute_all with no tasks returns immediately."""
        executed: list[str] = []

        async def execute_fn(task):
            executed.append(task.id)
            return {"task_id": task.id}

        state = await runner.execute_all(
            tasks={},
            execution_order=[],
            project_desc="Empty test",
            success_criteria="Nothing",
            execute_task_fn=execute_fn,
        )
        assert executed == []
        assert state is not None

    @pytest.mark.asyncio
    async def test_single_task(self, runner):
        """A single task with no dependencies executes correctly."""
        tasks = {"only": make_task("only")}
        executed: list[str] = []

        async def execute_fn(task):
            executed.append(task.id)
            return {"task_id": task.id, "status": "ok"}

        await runner.execute_all(
            tasks=tasks,
            execution_order=["only"],
            project_desc="Single task",
            success_criteria="Pass",
            execute_task_fn=execute_fn,
        )
        assert executed == ["only"], f"Expected ['only'] but got {executed}"

    @pytest.mark.asyncio
    async def test_complex_dependency_chain(self, runner):
        """A diamond dependency pattern resolves correctly."""
        tasks = {
            "a": make_task("a"),
            "b": make_task("b", deps=["a"]),
            "c": make_task("c", deps=["a"]),
            "d": make_task("d", deps=["b", "c"]),
        }
        executed: list[str] = []

        async def execute_fn(task):
            executed.append(task.id)
            return {"task_id": task.id, "status": "ok"}

        await runner.execute_all(
            tasks=tasks,
            execution_order=["a", "b", "c", "d"],
            project_desc="Diamond dep chain",
            success_criteria="Pass",
            execute_task_fn=execute_fn,
        )

        assert executed[0] == "a", f"First should be a, got {executed}"
        assert executed[-1] == "d", f"Last should be d, got {executed}"
        assert set(executed[1:3]) == {"b", "c"}, f"Middle should be b,c, got {executed}"

    @pytest.mark.asyncio
    async def test_semaphore_with_sequential_levels(self, sequential_runner):
        """With max_parallel_tasks=1, level 1 runs only after level 0."""
        tasks = {
            "a": make_task("a"),
            "b": make_task("b"),
            "c": make_task("c", deps=["a"]),
            "d": make_task("d", deps=["b"]),
        }
        executed: list[str] = []

        async def execute_fn(task):
            executed.append(task.id)
            return {"task_id": task.id, "status": "ok"}

        await sequential_runner.execute_all(
            tasks=tasks,
            execution_order=["a", "b", "c", "d"],
            project_desc="Sequential levels",
            success_criteria="Pass",
            execute_task_fn=execute_fn,
        )

        # All four tasks should have run
        assert len(executed) == 4, f"Expected 4 tasks, got {executed}"
        # 'a' and 'b' in level 0 must complete before 'c' and 'd' in level 1
        a_idx = executed.index("a")
        b_idx = executed.index("b")
        c_idx = executed.index("c")
        d_idx = executed.index("d")
        assert max(a_idx, b_idx) < min(c_idx, d_idx), (
            f"Level 0 (a,b) before level 1 (c,d): executed={executed}"
        )
