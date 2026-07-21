"""
Integration tests for orchestrator.application.project_runner.ProjectRunner

M3 update: ProjectRunner no longer takes ``host=Orchestrator``.
All execution callbacks are passed via ProjectRunnerCallables; shared mutable
state is passed via ProjectRunState.  Tests verify the same observable
behaviours as before without needing an Orchestrator instance.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

pytestmark = pytest.mark.integration

from orchestrator.application.project_runner import ProjectRunner
from orchestrator.application.project_runner_deps import (
    ProjectRunnerCallables,
    ProjectRunState,
)
from orchestrator.models import ProjectState, ProjectStatus

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_state(status: ProjectStatus = ProjectStatus.SUCCESS) -> ProjectState:
    """Return a minimal but valid ProjectState."""
    return ProjectState(
        project_description="build X",
        success_criteria="tests pass",
        budget=None,
        tasks={},
        results={},
        status=status,
        api_health={},
    )


def _make_callables(
    *,
    entered: bool = True,
    execute_all_result: ProjectState | None = None,
) -> tuple[ProjectRunnerCallables, ProjectRunState]:
    """Return a (callables, run_state) pair with sensible mock defaults."""
    run_state = ProjectRunState(entered=entered)

    if execute_all_result is None:
        execute_all_result = _make_state()

    callables = ProjectRunnerCallables(
        topological_sort=MagicMock(return_value=["t1"]),
        topological_levels=MagicMock(return_value=[["t1"]]),
        make_state=MagicMock(return_value=_make_state()),
        determine_final_status=MagicMock(return_value=ProjectStatus.SUCCESS),
        log_summary=MagicMock(),
        execute_all=AsyncMock(return_value=execute_all_result),
        generate_architecture_rules=AsyncMock(return_value=""),
        analyze_completed_project=AsyncMock(),
        client=MagicMock(),
    )
    return callables, run_state


def _make_gen_result(tasks: dict | None = None, succeeded: bool = True) -> MagicMock:
    result = MagicMock()
    result.succeeded = succeeded
    result.tasks = tasks if tasks is not None else {"t1": MagicMock()}
    result.error = "" if succeeded else "decomp error"
    return result


def _make_runner(
    callables: ProjectRunnerCallables | None = None,
    run_state: ProjectRunState | None = None,
    **overrides,
) -> ProjectRunner:
    if callables is None or run_state is None:
        _c, _rs = _make_callables()
        if callables is None:
            callables = _c
        if run_state is None:
            run_state = _rs

    budget = MagicMock()
    budget.max_usd = 5.0
    budget.max_time_seconds = 3600
    budget.spent_usd = 0.1
    budget.elapsed_seconds = 10.0
    budget.validate_sufficient_for_tasks = MagicMock(return_value=(True, ""))

    state_mgr = AsyncMock()
    state_mgr.load_project = AsyncMock(return_value=None)
    state_mgr.save_project = AsyncMock()
    state_mgr.close = AsyncMock()

    cache = AsyncMock()
    cache.close = AsyncMock()

    generator = AsyncMock()
    generator.decompose = AsyncMock(return_value=_make_gen_result())

    dashboard_bridge = MagicMock()
    git_bridge = MagicMock()
    git_bridge.commit_project = MagicMock(return_value="abc123")

    resumption_svc = AsyncMock()
    resumption_svc.resume = AsyncMock(return_value=_make_state(ProjectStatus.SUCCESS))

    defaults = {
        "callables": callables,
        "run_state": run_state,
        "state_mgr": state_mgr,
        "budget": budget,
        "event_bus": None,
        "resumption_svc": resumption_svc,
        "dashboard_bridge": dashboard_bridge,
        "git_bridge": git_bridge,
        "generator": generator,
        "meta_v2": None,
        "cache": cache,
        "api_health": {},
    }
    defaults.update(overrides)
    return ProjectRunner(**defaults)


# ─────────────────────────────────────────────────────────────────────────────
# Happy path
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_project_happy_path_returns_success_state():
    runner = _make_runner()
    state = await runner.run_project("build X", "tests pass", project_id="proj1")
    assert state.status == ProjectStatus.SUCCESS


@pytest.mark.asyncio
async def test_run_project_sets_run_state_project_id():
    callables, run_state = _make_callables()
    runner = _make_runner(callables=callables, run_state=run_state)
    await runner.run_project("build X", "tests pass", project_id="explicit-id")
    assert run_state.project_id == "explicit-id"


@pytest.mark.asyncio
async def test_run_project_generates_project_id_when_none_given():
    callables, run_state = _make_callables()
    runner = _make_runner(callables=callables, run_state=run_state)
    await runner.run_project("build X", "tests pass")
    # A human-readable id "<slug>-<6-hex>" is generated from the description
    # (slug from "build X" -> "build-x", plus a 6-char md5 suffix).
    pid = run_state.project_id
    assert pid, "project_id should be auto-generated when none is given"
    slug, sep, short_hash = pid.rpartition("-")
    assert sep == "-" and slug == "build-x", f"unexpected slug in {pid!r}"
    assert len(short_hash) == 6 and all(c in "0123456789abcdef" for c in short_hash)


@pytest.mark.asyncio
async def test_run_project_calls_decompose_then_execute():
    callables, run_state = _make_callables()
    runner = _make_runner(callables=callables, run_state=run_state)
    await runner.run_project("build X", "tests pass")
    runner._generator.decompose.assert_awaited_once()
    callables.execute_all.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_project_saves_state_to_state_mgr():
    callables, run_state = _make_callables()
    runner = _make_runner(callables=callables, run_state=run_state)
    await runner.run_project("build X", "tests pass", project_id="p1")
    runner._state_mgr.save_project.assert_awaited_once_with(
        "p1", callables.execute_all.return_value
    )


@pytest.mark.asyncio
async def test_run_project_calls_git_bridge_commit():
    runner = _make_runner()
    await runner.run_project("build X", "tests pass")
    runner._git_bridge.commit_project.assert_called_once()


@pytest.mark.asyncio
async def test_run_project_calls_dashboard_bridge_on_project_start():
    runner = _make_runner()
    await runner.run_project("build X", "tests pass", project_id="p1")
    runner._dashboard_bridge.on_project_start.assert_called_once()
    call_args = runner._dashboard_bridge.on_project_start.call_args
    assert call_args.args[0] == "p1"


# ─────────────────────────────────────────────────────────────────────────────
# Resume path
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_project_resumes_partial_state():
    partial_state = _make_state(ProjectStatus.PARTIAL_SUCCESS)
    state_mgr = AsyncMock()
    state_mgr.load_project = AsyncMock(return_value=partial_state)
    state_mgr.save_project = AsyncMock()
    state_mgr.close = AsyncMock()

    callables, run_state = _make_callables()
    runner = _make_runner(callables=callables, run_state=run_state, state_mgr=state_mgr)

    result = await runner.run_project("build X", "tests pass", project_id="p1")

    # Should have gone through resumption_svc, not execute_all
    runner._resumption_svc.resume.assert_awaited_once_with(partial_state)
    callables.execute_all.assert_not_awaited()
    assert result is runner._resumption_svc.resume.return_value


@pytest.mark.asyncio
async def test_run_project_skips_resume_when_state_is_success():
    completed_state = _make_state(ProjectStatus.SUCCESS)
    state_mgr = AsyncMock()
    state_mgr.load_project = AsyncMock(return_value=completed_state)
    state_mgr.save_project = AsyncMock()
    state_mgr.close = AsyncMock()

    callables, run_state = _make_callables()
    runner = _make_runner(callables=callables, run_state=run_state, state_mgr=state_mgr)
    await runner.run_project("build X", "tests pass")

    # State exists but is SUCCESS — should execute fresh
    callables.execute_all.assert_awaited_once()


# ─────────────────────────────────────────────────────────────────────────────
# Error / failure paths
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_project_returns_system_failure_when_decompose_fails():
    failure_state = _make_state(ProjectStatus.SYSTEM_FAILURE)
    callables, run_state = _make_callables()
    callables.make_state = MagicMock(return_value=failure_state)

    generator = AsyncMock()
    generator.decompose = AsyncMock(return_value=_make_gen_result(succeeded=False))

    runner = _make_runner(callables=callables, run_state=run_state, generator=generator)
    state = await runner.run_project("build X", "tests pass")

    assert state.status == ProjectStatus.SYSTEM_FAILURE
    callables.execute_all.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_project_returns_system_failure_when_no_tasks():
    failure_state = _make_state(ProjectStatus.SYSTEM_FAILURE)
    callables, run_state = _make_callables()
    callables.make_state = MagicMock(return_value=failure_state)

    generator = AsyncMock()
    generator.decompose = AsyncMock(return_value=_make_gen_result(tasks={}, succeeded=True))

    runner = _make_runner(callables=callables, run_state=run_state, generator=generator)
    state = await runner.run_project("build X", "tests pass")

    assert state.status == ProjectStatus.SYSTEM_FAILURE


# ─────────────────────────────────────────────────────────────────────────────
# Connection lifecycle (BUG-003)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_project_closes_connections_when_not_entered():
    """BUG-003: connections must be closed when used without context manager."""
    callables, run_state = _make_callables(entered=False)
    state_mgr = AsyncMock()
    state_mgr.load_project = AsyncMock(return_value=None)
    state_mgr.save_project = AsyncMock()
    state_mgr.close = AsyncMock()

    cache = AsyncMock()
    cache.close = AsyncMock()

    runner = _make_runner(
        callables=callables, run_state=run_state, state_mgr=state_mgr, cache=cache
    )
    await runner.run_project("build X", "tests pass")

    state_mgr.close.assert_awaited_once()
    cache.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_project_skips_close_when_entered():
    """BUG-003: connections must NOT be closed inside a context manager run."""
    callables, run_state = _make_callables(entered=True)
    state_mgr = AsyncMock()
    state_mgr.load_project = AsyncMock(return_value=None)
    state_mgr.save_project = AsyncMock()
    state_mgr.close = AsyncMock()

    cache = AsyncMock()
    cache.close = AsyncMock()

    runner = _make_runner(
        callables=callables, run_state=run_state, state_mgr=state_mgr, cache=cache
    )
    await runner.run_project("build X", "tests pass")

    state_mgr.close.assert_not_awaited()
    cache.close.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_project_still_closes_connections_when_execute_raises():
    """BUG-003: finally block runs even on exception."""
    callables, run_state = _make_callables(entered=False)
    callables.execute_all = AsyncMock(side_effect=RuntimeError("execute crashed"))

    state_mgr = AsyncMock()
    state_mgr.load_project = AsyncMock(return_value=None)
    state_mgr.close = AsyncMock()

    cache = AsyncMock()
    cache.close = AsyncMock()

    runner = _make_runner(
        callables=callables, run_state=run_state, state_mgr=state_mgr, cache=cache
    )
    with pytest.raises(RuntimeError, match="execute crashed"):
        await runner.run_project("build X", "tests pass")

    state_mgr.close.assert_awaited_once()
    cache.close.assert_awaited_once()


# ─────────────────────────────────────────────────────────────────────────────
# dry_run
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_dry_run_returns_execution_plan_without_executing():
    """dry_run must never call execute_all."""
    callables, run_state = _make_callables()

    task = MagicMock()
    task.type = MagicMock()
    task.type.value = "CODE_GEN"
    task.prompt = "write a function"
    task.dependencies = []
    task.acceptance_threshold = 0.8
    task.max_iterations = 3

    generator = AsyncMock()
    generator.decompose = AsyncMock(
        return_value=_make_gen_result(tasks={"t1": task}, succeeded=True)
    )

    runner = _make_runner(callables=callables, run_state=run_state, generator=generator)
    plan = await runner.dry_run("build X", "tests pass")

    callables.execute_all.assert_not_awaited()
    assert plan is not None


@pytest.mark.asyncio
async def test_dry_run_returns_empty_plan_when_no_tasks():
    callables, run_state = _make_callables()
    generator = AsyncMock()
    generator.decompose = AsyncMock(return_value=_make_gen_result(tasks={}, succeeded=True))

    runner = _make_runner(callables=callables, run_state=run_state, generator=generator)
    plan = await runner.dry_run("build X", "tests pass")

    assert plan is not None
    assert plan.project_description == "build X"


# ─────────────────────────────────────────────────────────────────────────────
# Event bus integration
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_project_publishes_events_when_event_bus_provided():
    """Verify that both ProjectStarted and ProjectCompleted are published."""
    event_bus = AsyncMock()
    event_bus.publish = AsyncMock()

    runner = _make_runner(event_bus=event_bus)
    await runner.run_project("build X", "tests pass")

    # Two publishes: ProjectStartedEvent + ProjectCompletedEvent
    assert event_bus.publish.await_count == 2


# ─────────────────────────────────────────────────────────────────────────────
# Meta-optimisation callback
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_project_meta_v2_none_does_not_raise():
    """meta_v2=None must not trigger any import or call."""
    runner = _make_runner(meta_v2=None)
    state = await runner.run_project("build X", "tests pass")
    assert state is not None
