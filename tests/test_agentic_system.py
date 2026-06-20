"""
Tests for the Agentic System implementation (Capabilities 1-4).
Covers: AgentBase, AgentOrchestrator, DeveloperAgent, Tool, Workspace.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

# ═════════════════════════════════════════════
# Test: AgentBase + Agents
# ═════════════════════════════════════════════


class TestAgentBase:
    """Agent hierarchy."""

    def test_agent_role_enum(self):
        from orchestrator.agents.base import AgentRole

        assert AgentRole.DEVELOPER.value == "developer"
        assert len(AgentRole) >= 7

    def test_agent_task_dataclass(self):
        from orchestrator.agents.base import AgentTask
        from orchestrator.models import TaskStatus

        t = AgentTask(id="t1", goal="Build login")
        assert t.id == "t1"
        assert t.status == TaskStatus.PENDING

    def test_agent_task_result_dataclass(self):
        from orchestrator.agents.base import AgentTaskResult

        r = AgentTaskResult(task_id="t1", success=True, output="done")
        assert r.success is True


class TestDeveloperAgent:
    """DeveloperAgent behavior."""

    @pytest.mark.asyncio
    async def test_handle_task_no_client(self):
        from orchestrator.agents.developer import DeveloperAgent
        from orchestrator.agents.base import AgentTask

        agent = DeveloperAgent()
        result = await agent.handle_task(AgentTask(id="t1", goal="test"))
        assert result.success is False
        assert "No LLM client" in result.output

    @pytest.mark.asyncio
    async def test_system_prompt(self):
        from orchestrator.agents.developer import DeveloperAgent

        agent = DeveloperAgent()
        assert "senior software engineer" in agent.system_prompt

    @pytest.mark.asyncio
    async def test_handle_task_success(self):
        from orchestrator.agents.developer import DeveloperAgent
        from orchestrator.agents.base import AgentTask

        mock_client = MagicMock()
        # client.call() returns a single APIResponse-like object (with .text),
        # not a tuple — match the real UnifiedClient.call contract.
        mock_client.call = AsyncMock(return_value=MagicMock(text="def hello(): pass"))

        agent = DeveloperAgent(client=mock_client)
        result = await agent.handle_task(AgentTask(id="t1", goal="Write hello"))
        assert result.success is True
        assert "def hello" in result.output


class TestArchitectAgent:
    @pytest.mark.asyncio
    async def test_system_prompt(self):
        from orchestrator.agents.developer import ArchitectAgent

        agent = ArchitectAgent()
        assert "software architect" in agent.system_prompt


class TestTesterAgent:
    @pytest.mark.asyncio
    async def test_system_prompt(self):
        from orchestrator.agents.developer import TesterAgent

        agent = TesterAgent()
        assert "QA engineer" in agent.system_prompt


# ═════════════════════════════════════════════
# Test: AgentOrchestrator
# ═════════════════════════════════════════════


class TestAgentOrchestrator:
    """Orchestrator coordination."""

    @pytest.mark.asyncio
    async def test_execute_goal_smoke(self):
        from orchestrator.agents.coordinator import AgentOrchestrator
        from orchestrator.agents.developer import DeveloperAgent
        from orchestrator.agents.base import AgentRole

        mock_client = MagicMock()
        mock_client.call = AsyncMock(return_value=(MagicMock(text="output"), {}))

        agent = DeveloperAgent(client=mock_client)
        orch = AgentOrchestrator(agents={AgentRole.DEVELOPER: agent})
        results = await orch.execute_goal("Write a function")
        assert len(results) >= 1

    def test_decompose_goal_architecture(self):
        from orchestrator.agents.coordinator import AgentOrchestrator
        from orchestrator.agents.base import AgentRole

        orch = AgentOrchestrator(agents={})
        tasks = orch._decompose_goal("Design architecture for web app")
        roles = {t.target_role for t in tasks if t.target_role}
        assert AgentRole.ARCHITECT in roles
        assert AgentRole.DEVELOPER in roles

    def test_decompose_goal_basic(self):
        from orchestrator.agents.coordinator import AgentOrchestrator
        from orchestrator.agents.base import AgentRole

        orch = AgentOrchestrator(agents={})
        tasks = orch._decompose_goal("Write a function")
        roles = {t.target_role for t in tasks if t.target_role}
        assert AgentRole.DEVELOPER in roles


# ═════════════════════════════════════════════
# Test: Tool Layer
# ═════════════════════════════════════════════


class TestToolBase:
    """Tool interface and registry."""

    def test_permission_enum(self):
        from orchestrator.tools.base import ToolPermission

        assert ToolPermission.FILE_READ.value == "file_read"

    def test_registry_register(self):
        from orchestrator.tools.base import ToolRegistry
        from unittest.mock import MagicMock

        tool = MagicMock()
        tool.name = "test"
        registry = ToolRegistry()
        registry.register(tool)
        assert registry.get("test") is tool

    def test_registry_permissions(self):
        from orchestrator.tools.base import ToolRegistry, ToolPermission
        from unittest.mock import MagicMock

        tool = MagicMock()
        tool.name = "safe"
        tool.required_permissions = [ToolPermission.FILE_WRITE]
        registry = ToolRegistry()
        registry.register(tool)
        registry.grant(ToolPermission.FILE_WRITE)
        assert registry.can_execute(tool) is True

    def test_registry_denies(self):
        from orchestrator.tools.base import ToolRegistry, ToolPermission
        from unittest.mock import MagicMock

        tool = MagicMock()
        tool.name = "danger"
        tool.required_permissions = [ToolPermission.SHELL_EXECUTE]
        registry = ToolRegistry()
        registry.register(tool)
        assert registry.can_execute(tool) is False

    @pytest.mark.asyncio
    async def test_shell_tool_no_command(self):
        from orchestrator.tools.shell_tool import ShellTool

        tool = ShellTool()
        result = await tool.execute({})
        assert result.success is False
        assert "No command" in result.output

    def test_shell_tool_validate(self):
        from orchestrator.tools.shell_tool import ShellTool

        assert ShellTool().validate_params({"command": "ls"}) is True
        assert ShellTool().validate_params({}) is False


class TestReviewerAgent:
    def test_system_prompt(self):
        from orchestrator.agents.reviewer import ReviewerAgent

        agent = ReviewerAgent()
        assert "security" in agent.system_prompt

    @pytest.mark.asyncio
    async def test_handle_task_no_client(self):
        from orchestrator.agents.reviewer import ReviewerAgent
        from orchestrator.agents.base import AgentTask

        agent = ReviewerAgent()
        result = await agent.handle_task(AgentTask(id="r1", goal="review"))
        assert result.success is False


class TestDevOpsAgent:
    def test_system_prompt(self):
        from orchestrator.agents.devops import DevOpsAgent

        agent = DevOpsAgent()
        assert "Docker" in agent.system_prompt


class TestResearcherAgent:
    def test_system_prompt(self):
        from orchestrator.agents.researcher import ResearcherAgent

        agent = ResearcherAgent()
        assert "research" in agent.system_prompt

    def test_tool_result_dataclass(self):
        from orchestrator.tools.base import ToolResult

        r = ToolResult(success=True, output="ok")
        assert r.success is True
        assert r.metrics == {}


# ═════════════════════════════════════════════
# Test: Workspace
# ═════════════════════════════════════════════


class TestWorkspace:
    """Shared workspace (blackboard)."""

    def test_write_and_read(self):
        from orchestrator.workspace.workspace import ProjectWorkspace

        ws = ProjectWorkspace()
        ws.write_file("main.py", "content", author="dev")
        assert ws.read_file("main.py") == "content"

    def test_versioning(self):
        from orchestrator.workspace.workspace import ProjectWorkspace

        ws = ProjectWorkspace()
        v1 = ws.write_file("f.py", "v1")
        assert v1.version == 1
        v2 = ws.write_file("f.py", "v2")
        assert v2.version == 2

    def test_decision_record(self):
        from orchestrator.workspace.workspace import ProjectWorkspace

        ws = ProjectWorkspace()
        ad = ws.record_decision("Use FastAPI", "Use FastAPI", "Best")
        assert ad.title == "Use FastAPI"
        assert len(ws.architectural_decisions) == 1

    def test_summary(self):
        from orchestrator.workspace.workspace import ProjectWorkspace

        ws = ProjectWorkspace()
        assert "Files modified: 0" in ws.get_summary()

    def test_read_missing(self):
        from orchestrator.workspace.workspace import ProjectWorkspace

        assert ProjectWorkspace().read_file("missing.py") is None
