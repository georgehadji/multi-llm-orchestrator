"""
Regression test: ProjectRunner.run_project must honour a per-run
constitution (--from-speckit path) by swapping it into the wired
ConstitutionGate before task execution — previously the parameter was
accepted and silently discarded.
"""

from unittest.mock import AsyncMock, Mock

import pytest

from orchestrator.application.project_runner import ProjectRunner
from orchestrator.application.project_runner_deps import (
    ProjectRunnerCallables,
    ProjectRunState,
)
from orchestrator.domain.constitution import ProjectConstitution


def _make_runner(constitution_gate=None) -> ProjectRunner:
    callables = ProjectRunnerCallables(
        topological_sort=Mock(),
        topological_levels=Mock(),
        make_state=Mock(),
        determine_final_status=Mock(),
        log_summary=Mock(),
        execute_all=AsyncMock(),
        generate_architecture_rules=AsyncMock(return_value=""),
        analyze_completed_project=AsyncMock(),
        client=Mock(),
        constitution_gate=constitution_gate,
    )
    state_mgr = AsyncMock()
    state_mgr.load_project.return_value = None
    return ProjectRunner(
        callables=callables,
        run_state=ProjectRunState(),
        state_mgr=state_mgr,
        budget=Mock(max_usd=1.0, max_time_seconds=60),
        event_bus=Mock(),
        resumption_svc=Mock(),
        dashboard_bridge=Mock(),
        git_bridge=Mock(),
        generator=Mock(),
        meta_v2=Mock(),
        cache=Mock(),
        api_health={},
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_project_swaps_constitution_into_gate_when_both_present():
    # Arrange
    mock_gate = Mock()
    runner = _make_runner(constitution_gate=mock_gate)
    speckit_constitution = ProjectConstitution(protect_paths=["src/domain/**"])

    # Act — run_project will likely fail/short-circuit later (heavily mocked
    # deps), but the constitution-gate swap happens early and unconditionally.
    try:
        await runner.run_project(
            project_description="test",
            success_criteria="done",
            constitution=speckit_constitution,
        )
    except Exception:
        pass

    # Assert
    mock_gate.set_constitution.assert_called_once_with(speckit_constitution)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_project_no_gate_does_not_raise(caplog):
    # Arrange — constitution provided but no gate wired (degraded container)
    runner = _make_runner(constitution_gate=None)
    speckit_constitution = ProjectConstitution(protect_paths=["src/domain/**"])

    # Act / Assert — must not raise from the constitution-wiring block itself
    try:
        await runner.run_project(
            project_description="test",
            success_criteria="done",
            constitution=speckit_constitution,
        )
    except Exception:
        pass  # unrelated downstream mock gaps are fine; we only assert no crash from our block

    assert any("no ConstitutionGate is wired" in r.message for r in caplog.records)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_run_project_no_constitution_does_not_touch_gate():
    # Arrange
    mock_gate = Mock()
    runner = _make_runner(constitution_gate=mock_gate)

    # Act
    try:
        await runner.run_project(project_description="test", success_criteria="done")
    except Exception:
        pass

    # Assert
    mock_gate.set_constitution.assert_not_called()
