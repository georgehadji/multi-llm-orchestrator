"""
Tests for CodebaseInvestigatorAgent and coordinator routing.

Unit tests — no LLM calls made. All CodebaseAnalyzer calls are mocked.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit
from unittest.mock import AsyncMock, MagicMock, patch

from orchestrator.agents.base import AgentRole, AgentTask, AgentTaskResult
from orchestrator.agents.investigator import CodebaseInvestigatorAgent
from orchestrator.agents.coordinator import AgentOrchestrator

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def agent() -> CodebaseInvestigatorAgent:
    return CodebaseInvestigatorAgent()


def _make_task(goal: str = "understand auth flow", ctx: dict | None = None) -> AgentTask:
    return AgentTask(id="test_001", goal=goal, context=ctx or {})


def _mock_report(markdown: str = "# Report\nFindings here.", files: int = 10, cost: float = 0.05):
    report = MagicMock()
    report.markdown = markdown
    report.files_analyzed = files
    report.total_cost = cost
    return report


def _make_mock_analyzer_cls(report):
    """Return a mock CodebaseAnalyzer class whose instance.analyze() returns report."""
    mock_cls = MagicMock()
    instance = mock_cls.return_value
    instance.analyze = AsyncMock(return_value=report)
    return mock_cls, instance


# ── AgentRole enum ────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_investigator_role_in_enum():
    assert AgentRole.INVESTIGATOR == "investigator"
    assert AgentRole.INVESTIGATOR in list(AgentRole)


# ── system_prompt ─────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_system_prompt_is_non_empty(agent):
    assert len(agent.system_prompt) > 20


# ── handle_task success ───────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.asyncio
@patch("orchestrator.agents.investigator.CodebaseAnalyzer")
async def test_handle_task_returns_success_on_good_report(MockAnalyzer, agent):
    report = _mock_report(markdown="## Findings\nAll good.")
    MockAnalyzer.return_value.analyze = AsyncMock(return_value=report)

    result = await agent.handle_task(_make_task())

    assert result.success is True
    assert "## Findings" in result.output
    assert result.score == 1.0


@pytest.mark.unit
@pytest.mark.asyncio
@patch("orchestrator.agents.investigator.CodebaseAnalyzer")
async def test_handle_task_passes_codebase_path_to_analyzer(MockAnalyzer, agent):
    report = _mock_report()
    MockAnalyzer.return_value.analyze = AsyncMock(return_value=report)

    await agent.handle_task(_make_task(ctx={"codebase_path": "/some/project"}))

    call_kwargs = MockAnalyzer.return_value.analyze.call_args.kwargs
    assert "path" in call_kwargs
    assert str(call_kwargs["path"]).replace("\\", "/").endswith("/some/project")


@pytest.mark.unit
@pytest.mark.asyncio
@patch("orchestrator.agents.investigator.CodebaseAnalyzer")
async def test_handle_task_passes_focus_to_analyzer(MockAnalyzer, agent):
    report = _mock_report()
    MockAnalyzer.return_value.analyze = AsyncMock(return_value=report)

    await agent.handle_task(_make_task(ctx={"focus": ["security", "performance"]}))

    call_kwargs = MockAnalyzer.return_value.analyze.call_args.kwargs
    assert call_kwargs["focus"] == ["security", "performance"]


@pytest.mark.unit
@pytest.mark.asyncio
@patch("orchestrator.agents.investigator.CodebaseAnalyzer")
async def test_handle_task_uses_default_focus_when_not_set(MockAnalyzer, agent):
    report = _mock_report()
    MockAnalyzer.return_value.analyze = AsyncMock(return_value=report)

    await agent.handle_task(_make_task())

    call_kwargs = MockAnalyzer.return_value.analyze.call_args.kwargs
    assert isinstance(call_kwargs["focus"], list)
    assert len(call_kwargs["focus"]) > 0


@pytest.mark.unit
@pytest.mark.asyncio
@patch("orchestrator.agents.investigator.CodebaseAnalyzer")
async def test_handle_task_passes_budget_usd_to_analyzer(MockAnalyzer, agent):
    report = _mock_report()
    MockAnalyzer.return_value.analyze = AsyncMock(return_value=report)

    await agent.handle_task(_make_task(ctx={"budget_usd": 2.5}))

    call_kwargs = MockAnalyzer.return_value.analyze.call_args.kwargs
    assert call_kwargs["budget_usd"] == pytest.approx(2.5)


# ── handle_task failure paths ─────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.asyncio
@patch("orchestrator.agents.investigator.CodebaseAnalyzer")
async def test_handle_task_returns_failure_when_analyzer_raises(MockAnalyzer, agent):
    MockAnalyzer.return_value.analyze = AsyncMock(side_effect=RuntimeError("disk read failed"))

    result = await agent.handle_task(_make_task())

    assert result.success is False
    assert "disk read failed" in result.output


@pytest.mark.unit
@pytest.mark.asyncio
@patch("orchestrator.agents.investigator.CodebaseAnalyzer", None)
async def test_handle_task_returns_failure_when_import_fails(agent):
    """If CodebaseAnalyzer is None (import failed), result.success is False."""
    result = await agent.handle_task(_make_task())
    assert result.success is False


# ── Coordinator routing ───────────────────────────────────────────────────────


@pytest.mark.unit
def test_coordinator_routes_understand_to_investigator():
    orch = AgentOrchestrator(agents={})
    tasks = orch._decompose_goal("understand the authentication flow")
    assert len(tasks) == 1
    assert tasks[0].target_role == AgentRole.INVESTIGATOR


@pytest.mark.unit
def test_coordinator_routes_trace_to_investigator():
    orch = AgentOrchestrator(agents={})
    tasks = orch._decompose_goal("trace the execution path for login")
    assert any(t.target_role == AgentRole.INVESTIGATOR for t in tasks)


@pytest.mark.unit
def test_coordinator_routes_explore_to_investigator():
    orch = AgentOrchestrator(agents={})
    tasks = orch._decompose_goal("explore the codebase structure")
    assert any(t.target_role == AgentRole.INVESTIGATOR for t in tasks)


@pytest.mark.unit
def test_coordinator_routes_investigate_to_investigator():
    orch = AgentOrchestrator(agents={})
    tasks = orch._decompose_goal("investigate why the cache is slow")
    assert any(t.target_role == AgentRole.INVESTIGATOR for t in tasks)


@pytest.mark.unit
def test_coordinator_investigation_returns_only_one_task():
    """Investigation goals must NOT also dispatch developer/tester tasks."""
    orch = AgentOrchestrator(agents={})
    tasks = orch._decompose_goal("understand how the pipeline works")
    assert len(tasks) == 1
    assert tasks[0].target_role == AgentRole.INVESTIGATOR


@pytest.mark.unit
def test_coordinator_does_not_route_build_goal_to_investigator():
    orch = AgentOrchestrator(agents={})
    tasks = orch._decompose_goal("build a REST API with FastAPI")
    assert not any(t.target_role == AgentRole.INVESTIGATOR for t in tasks)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_coordinator_execute_goal_calls_investigator_agent():
    mock_result = AgentTaskResult(task_id="investigate_001", success=True, output="findings")
    mock_agent = MagicMock()
    mock_agent.handle_task = AsyncMock(return_value=mock_result)

    orch = AgentOrchestrator(agents={AgentRole.INVESTIGATOR: mock_agent})
    results = await orch.execute_goal("understand the event bus implementation")

    assert mock_agent.handle_task.called
    assert any(r.success for r in results.values())
