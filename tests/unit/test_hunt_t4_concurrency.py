"""
T4 (concurrency & resource lifecycle) proof-of-defect and no-regression tests.

One VERIFIED DEFECT from docs/hunts/t4-concurrency/inventory.md:

C1 — PipelineRunner.execute_all()'s run_one() let a failed task's exception
     propagate to asyncio.gather(return_exceptions=True), which only logged
     it — results[tid] was never set for a failed task, so ProjectState.
     results silently had no record that the task was ever attempted (as
     opposed to recording it FAILED). Fixed by catching the exception inside
     run_one() and recording a FAILED TaskResult, mirroring the existing
     _build_failure_result() convention already used in
     application/task_executor.py for the same situation.
"""

from __future__ import annotations

import pytest

from orchestrator.engine_core.pipeline_runner import PipelineRunner
from orchestrator.models import Task, TaskStatus, TaskType

pytestmark = pytest.mark.unit


class _FakePlanner:
    """Fake planner — single level containing every task, no dependency logic
    needed for this test."""

    def get_execution_levels(self, tasks):
        return [list(tasks.keys())]


def _make_task(tid: str) -> Task:
    return Task(id=tid, type=TaskType.CODE_GEN, prompt="do the thing")


@pytest.mark.asyncio
async def test_c1_failed_task_gets_recorded_as_failed_not_dropped():
    """The core defect: a task whose execute_task_fn raises must still get
    a results[tid] entry, with status=FAILED, not be silently absent."""
    tasks = {"t1": _make_task("t1")}
    runner = PipelineRunner(pipeline=None, planner=_FakePlanner())

    async def failing_execute(task):
        raise RuntimeError("boom")

    state = await runner.execute_all(
        tasks=tasks,
        execution_order=["t1"],
        project_desc="d",
        success_criteria="c",
        execute_task_fn=failing_execute,
    )

    assert "t1" in state, "failed task must still have an entry in results"
    assert state["t1"].status == TaskStatus.FAILED
    assert "boom" in state["t1"].critique


@pytest.mark.asyncio
async def test_c1_successful_task_still_recorded_normally():
    """No-regression: a task that succeeds must be recorded exactly as
    before (this fix only changes the failure path)."""
    from orchestrator.models import Model, TaskResult

    tasks = {"t1": _make_task("t1")}
    runner = PipelineRunner(pipeline=None, planner=_FakePlanner())

    expected = TaskResult(task_id="t1", output="ok", score=1.0, model_used=Model.GPT_4O_MINI)

    async def succeeding_execute(task):
        return expected

    state = await runner.execute_all(
        tasks=tasks,
        execution_order=["t1"],
        project_desc="d",
        success_criteria="c",
        execute_task_fn=succeeding_execute,
    )

    assert state["t1"] is expected
    assert state["t1"].status == TaskStatus.COMPLETED


@pytest.mark.asyncio
async def test_c1_mixed_level_records_both_success_and_failure():
    """Real trigger with two concurrent tasks in the same level: one
    succeeds, one fails — both must end up in results."""
    from orchestrator.models import Model, TaskResult

    tasks = {"good": _make_task("good"), "bad": _make_task("bad")}
    runner = PipelineRunner(pipeline=None, planner=_FakePlanner())

    async def execute_fn(task):
        if task.id == "bad":
            raise ValueError("intentional failure")
        return TaskResult(task_id="good", output="ok", score=1.0, model_used=Model.GPT_4O_MINI)

    state = await runner.execute_all(
        tasks=tasks,
        execution_order=["good", "bad"],
        project_desc="d",
        success_criteria="c",
        execute_task_fn=execute_fn,
    )

    assert state["good"].status == TaskStatus.COMPLETED
    assert state["bad"].status == TaskStatus.FAILED
