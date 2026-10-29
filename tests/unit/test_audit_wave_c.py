"""
Audit Wave C — durability and resume
====================================
Regression tests for docs/plans/2026-10-29-audit-remediation-plan.md, Wave C.

C3: only PARTIAL_SUCCESS runs were resumable, but runs halted by budget or time end as
    BUDGET_EXHAUSTED or TIMEOUT, so --resume re-decomposed them from scratch and
    find_resumable never offered them.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

pytestmark = pytest.mark.unit


def _state(status, description="build a weather dashboard"):
    from orchestrator.budget import Budget
    from orchestrator.models import ProjectState

    return ProjectState(
        project_description=description,
        success_criteria="c",
        budget=Budget(),
        status=status,
    )


async def test_c3_find_resumable_offers_runs_halted_by_time_or_budget(tmp_path) -> None:
    from orchestrator.infrastructure.state import StateManager
    from orchestrator.models import ProjectStatus

    sm = StateManager(db_path=tmp_path / "state.db")
    await sm.save_project("timed-out", _state(ProjectStatus.TIMEOUT))
    await sm.save_project("broke", _state(ProjectStatus.BUDGET_EXHAUSTED))
    await sm.save_project("done", _state(ProjectStatus.SUCCESS))

    found = {row["project_id"] for row in await sm.find_resumable([])}
    await sm.close()

    assert found == {"timed-out", "broke"}


async def test_c3_resume_gate_resumes_a_timed_out_run(monkeypatch) -> None:
    from orchestrator.application.project_runner import ProjectRunner
    from orchestrator.application.project_runner_deps import ProjectRunState
    from orchestrator.budget import Budget
    from orchestrator.models import ProjectStatus

    monkeypatch.setenv("ORCH_UNATTENDED_GUARD", "false")
    halted = _state(ProjectStatus.TIMEOUT)
    state_mgr = AsyncMock()
    state_mgr.load_project = AsyncMock(return_value=halted)
    resumption = AsyncMock()
    resumption.resume = AsyncMock(return_value=halted)
    runner = ProjectRunner(
        callables=MagicMock(),
        run_state=ProjectRunState(entered=True),
        state_mgr=state_mgr,
        budget=Budget(),
        event_bus=None,
        resumption_svc=resumption,
        dashboard_bridge=MagicMock(),
        git_bridge=MagicMock(),
        generator=AsyncMock(),
        meta_v2=None,
        cache=AsyncMock(),
        api_health={},
    )

    await runner.run_project(project_description="d", success_criteria="c", project_id="p1")

    resumption.resume.assert_awaited_once()
