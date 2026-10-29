"""
Audit Wave A — code-review follow-ups
=====================================
Regression tests for the code review of PR #32 (fix/wave-a-defaults).

R1: a per-call budget passed to run_project/run_job only rebound _run_ctx.budget, while
    every stage kept the container Budget, so a shared orchestrator carried spend from
    one project into the next.
R2: ProgressWriter._write_task_file prepended a SECURITY WARNING header to the stored
    TaskResult.output and built its filename from the raw task_id.
R3: a failing progress write escaped run_one after the result was stored, so a finished
    task was logged as failed; the write also ran while holding results_lock.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


async def test_r1_per_call_budget_resets_the_shared_budget_in_place() -> None:
    from orchestrator.application.run_context import RunContext
    from orchestrator.budget import Budget

    shared = Budget(max_usd=10.0)
    await shared.charge(9.0, "generation")
    ctx = RunContext(budget=shared)

    ctx.reset(project_id="p2", budget=Budget(max_usd=5.0, max_time_seconds=60.0))

    assert ctx.budget is shared  # stages hold this object; it must not be swapped out
    assert shared.max_usd == 5.0
    assert shared.max_time_seconds == 60.0
    assert shared.spent_usd == 0.0
    assert shared.remaining_usd == 5.0
    assert all(v == 0.0 for v in shared.phase_spent.values())


async def test_r1_run_job_applies_spec_budget_to_the_shared_budget(monkeypatch) -> None:
    from unittest.mock import AsyncMock

    from orchestrator.budget import Budget
    from orchestrator.engine import Orchestrator

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test-key-for-verification-only")
    orch = Orchestrator(budget=Budget(max_usd=8.0))
    shared = orch.budget
    await shared.charge(7.0, "generation")
    orch._project_runner.run_job = AsyncMock(return_value=None)
    spec = type(
        "Spec",
        (),
        {
            "budget": Budget(max_usd=0.5),
            "policy_set": None,
            "quality_mode": "standard",
            "max_parallel_tasks": 0,
        },
    )()
    monkeypatch.setattr("orchestrator.engine.validate_job_spec", lambda s: None)

    await orch.run_job(spec)

    assert orch._run_ctx.budget is shared
    assert shared.max_usd == 0.5 and shared.spent_usd == 0.0
    await orch.close()


async def test_r2_progress_writer_keeps_result_and_stays_in_output_dir(
    tmp_path, monkeypatch
) -> None:
    from types import SimpleNamespace

    import orchestrator.code_validator as cv
    from orchestrator.analysis.progress_writer import ProgressWriter
    from orchestrator.budget import Budget
    from orchestrator.models import Model, ProjectState, Task, TaskResult, TaskType

    invalid = SimpleNamespace(
        is_valid=False, errors=["bad"], warnings=[], dangerous_patterns_found=[]
    )
    monkeypatch.setattr(cv, "validate_code", lambda code, config: invalid)
    out = tmp_path / "a" / "b"
    task = Task(id="../../x", type=TaskType.CODE_GEN, prompt="p")
    result = TaskResult(task_id=task.id, output="print(1)", score=1.0, model_used=Model.GPT_4O_MINI)
    state = ProjectState(project_description="d", success_criteria="c", budget=Budget())

    await ProgressWriter(out, state).task_completed(task.id, result, task)

    assert result.output == "print(1)"
    assert not list(tmp_path.glob("x_*"))
    written = [p for p in out.iterdir() if p.name.endswith(".py")]
    assert len(written) == 1 and "SECURITY WARNING" in written[0].read_text(encoding="utf-8")


async def test_r3_progress_write_failure_does_not_fail_the_task(
    tmp_path, monkeypatch, caplog
) -> None:
    from orchestrator.analysis.progress_writer import ProgressWriter
    from orchestrator.budget import Budget
    from orchestrator.engine_core.pipeline_runner import PipelineRunner
    from orchestrator.engine_core.project_planner import ProjectPlanner
    from orchestrator.models import Model, ProjectState, Task, TaskResult, TaskType

    async def boom(self, *args):
        raise OSError("disk full")

    monkeypatch.setattr(ProgressWriter, "task_completed", boom)

    def make_state(desc, criteria, tasks, status=None, execution_order=None, results=None):
        return ProjectState(
            project_description=desc,
            success_criteria=criteria,
            budget=Budget(),
            tasks=tasks,
            results=results or {},
        )

    async def run_task(task):
        return TaskResult(task_id=task.id, output="x", score=1.0, model_used=Model.GPT_4O_MINI)

    tasks = {"t1": Task(id="t1", type=TaskType.CODE_GEN, prompt="p")}
    runner = PipelineRunner(pipeline=None, planner=ProjectPlanner(), max_parallel_tasks=1)
    state = await runner.execute_all(
        tasks,
        ["t1"],
        "d",
        "c",
        output_dir=tmp_path,
        execute_task_fn=run_task,
        make_state_fn=make_state,
    )

    assert state.results["t1"].output == "x"
    assert "failed in level" not in caplog.text
    assert "disk full" in caplog.text


async def test_r1_reset_without_budget_keeps_cumulative_spend() -> None:
    from orchestrator.application.run_context import RunContext
    from orchestrator.budget import Budget

    shared = Budget(max_usd=10.0)
    await shared.charge(3.0, "generation")
    ctx = RunContext(budget=shared)

    ctx.reset(project_id="p2")

    assert shared.spent_usd == 3.0
