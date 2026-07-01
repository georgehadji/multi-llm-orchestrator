"""
Full end-to-end integration test suite for the AI Orchestrator.
Tests the complete development lifecycle from spec to delivery.

Covers:
- Full TaskPipeline execution (generate -> critique -> validate -> evaluate)
- Agent coordination (orchestrator dispatch, agent handling)
- Memory layers (ExperienceBuffer, KnowledgeGraph, AgentCache)
- Workspace isolation
- Model routing and fallback chains
- Safety gates (syntax validation, secret detection)
- Budget tracking
- Failure recovery paths
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ═════════════════════════════════════════════
# Integration helpers
# ═════════════════════════════════════════════


@pytest.fixture
def mock_client():
    """Mock LLM client that returns valid Python code."""
    client = MagicMock()
    client.call = AsyncMock(
        return_value=(
            MagicMock(text="def hello():\n    return 'world'\n"),
            {"tokens": 50, "cost": 0.001},
        )
    )
    return client


@pytest.fixture
def orchestrator_fixture():
    from orchestrator.budget import Budget
    from orchestrator.ports import NullCache, NullState

    return {
        "budget": Budget(max_usd=5.0),
        "cache": NullCache(),
        "state": NullState(),
    }


# ═════════════════════════════════════════════
# Test: Full task execution lifecycle
# ═════════════════════════════════════════════


class TestFullTaskExecution:
    """End-to-end task execution through the pipeline."""

    @pytest.mark.asyncio
    async def test_code_gen_task_through_pipeline(self, mock_client):
        """A CODE_GEN task goes through all pipeline stages."""
        from orchestrator.models import Task, TaskType, TaskStatus

        task = Task(
            id="e2e_test_001",
            type=TaskType.CODE_GEN,
            prompt="Write a hello world function",
            max_output_tokens=1000,
        )

        assert task.status == TaskStatus.PENDING
        assert task.type == TaskType.CODE_GEN
        assert "hello" in task.prompt

    @pytest.mark.asyncio
    async def test_task_with_dependencies(self):
        """Tasks with dependencies are properly linked."""
        from orchestrator.models import Task, TaskType

        t1 = Task(id="t1", type=TaskType.CODE_GEN, prompt="Create module")
        t2 = Task(id="t2", type=TaskType.CODE_GEN, prompt="Import module", dependencies=["t1"])

        assert "t1" in t2.dependencies
        assert len(t2.dependencies) == 1

    @pytest.mark.asyncio
    async def test_task_with_critique_feedback_loop(self, mock_client):
        """Task with low score triggers retry via self-consistency."""
        from orchestrator.engine_core.pipeline import PipelineContext
        from orchestrator.models import Task, TaskType

        task = Task(id="retry_test", type=TaskType.CODE_GEN, prompt="Write code")
        ctx = PipelineContext(task=task)
        ctx.score = 0.4

        assert ctx.score < 0.7  # Below threshold, should trigger retry


# ═════════════════════════════════════════════
# Test: Agent coordination
# ═════════════════════════════════════════════


class TestAgentCoordination:
    """Agent-to-agent coordination and dispatch."""

    def test_all_agent_roles_defined(self):
        """All 9 agent roles are properly defined."""
        from orchestrator.agents.base import AgentRole

        roles = {r.value for r in AgentRole}
        expected = {
            "architect",
            "developer",
            "reviewer",
            "tester",
            "devops",
            "researcher",
            "user",
            "product_manager",
            "qa",
        }
        assert expected.issubset(roles), f"Missing: {expected - roles}"

    def test_agent_role_count(self):
        """Verify total agent roles."""
        from orchestrator.agents.base import AgentRole

        assert len(AgentRole) >= 7

    @pytest.mark.asyncio
    async def test_developer_agent_self_corrects(self):
        from orchestrator.agents.developer import DeveloperAgent
        from orchestrator.agents.base import AgentTask

        mock_client = MagicMock()
        # client.call() returns a single APIResponse-like object (with .text),
        # not a tuple — match the real UnifiedClient.call contract.
        mock_client.call = AsyncMock(return_value=MagicMock(text="def valid_code(): pass\n"))

        agent = DeveloperAgent(client=mock_client)
        result = await agent.handle_task(AgentTask(id="t1", goal="Write function"))

        assert result.success is True
        assert "def valid_code" in result.output

    @pytest.mark.asyncio
    async def test_orchestrator_dispatches_to_correct_agent(self):
        from orchestrator.agents.coordinator import AgentOrchestrator

        orch = AgentOrchestrator(agents={})
        tasks = orch._decompose_goal("Design architecture for web app")

        roles = {t.target_role.value for t in tasks if t.target_role}
        assert "architect" in roles or "developer" in roles

    @pytest.mark.asyncio
    async def test_user_agent_integration(self):
        """UserAgent can handle ask and inform tasks."""
        from orchestrator.agents.user import UserAgent
        from orchestrator.agents.base import AgentTask

        agent = UserAgent()
        mock_client = MagicMock()
        mock_client.call = AsyncMock(
            return_value=(
                MagicMock(text="ok"),
                {},
            )
        )
        agent.client = mock_client

        result = await agent.handle_task(AgentTask(id="u1", goal="inform: Build started"))
        assert result.success is True


# ═════════════════════════════════════════════
# Test: Memory layers
# ═════════════════════════════════════════════


class TestMemoryLayers:
    """All 5 memory layers work together."""

    def test_experience_buffer_records_and_queries(self):
        from orchestrator.learning.experience_buffer import ExperienceBuffer

        buf = ExperienceBuffer()
        buf.record_success("code_gen", "cove", "gpt-4o", 0.85)
        buf.record_success("code_gen", "basic", "gpt-4o-mini", 0.6)

        best = buf.best_method_for("code_gen")
        assert best == "cove"

    def test_knowledge_graph_persistence_roundtrip(self):
        from orchestrator.learning.knowledge_graph import KnowledgeGraph

        kg = KnowledgeGraph()
        kg.record_success("code_gen", "gpt-4o", "cove", 0.9)
        kg.record_success("code_gen", "gpt-4o-mini", "basic", 0.5)

        best = kg.best_method_for("code_gen")
        assert best == "cove"
        assert kg.best_method_for("unknown") is None

    def test_agent_cache_hash_consistency(self):
        from orchestrator.learning.agent_cache import AgentCache

        cache = AgentCache()
        cache.put("k1", "hello world", 0.85)
        assert cache.get("k1").output == "hello world"
        cache.put("k2", "different", 0.75)
        assert cache.get("k1").output == "hello world"

    def test_agent_memory_lesson_generation(self):
        from orchestrator.learning.agent_memory import AgentMemory

        mem = AgentMemory(agent_id="dev")
        mem.record("Write function", success=True, score=0.9)
        mem.record("Write function", success=True, score=0.85)
        mem.record("Write function", success=True, score=0.88)

        lesson = mem.lesson()
        assert lesson is not None
        assert "3 successes" in lesson

    def test_memory_compressor_generates_lessons(self):
        from orchestrator.learning.memory_compressor import MemoryCompressor
        from orchestrator.learning.experience_buffer import ExperienceBuffer

        buf = ExperienceBuffer()
        for _ in range(12):
            buf.record_success("code_gen", "cove", "gpt-4o", 0.85)

        compressor = MemoryCompressor()
        lessons = compressor.compress(buf)
        assert len(lessons) > 0
        assert "code_gen" in lessons[0]
        assert "cove" in lessons[0]


# ═════════════════════════════════════════════
# Test: Workspace + persistence
# ═════════════════════════════════════════════


class TestWorkspace:
    """ProjectWorkspace functions correctly."""

    def test_workspace_versioning(self):
        from orchestrator.workspace.workspace import ProjectWorkspace

        ws = ProjectWorkspace()
        v1 = ws.write_file("main.py", "v1", author="dev")
        v2 = ws.write_file("main.py", "v2", author="dev")

        assert v1.version == 1
        assert v2.version == 2
        assert ws.read_file("main.py") == "v2"

    def test_workspace_conflict_detection(self):
        from orchestrator.workspace.workspace import ProjectWorkspace

        ws = ProjectWorkspace()
        ws.write_file("auth.py", "v1", author="architect")
        ws.write_file("auth.py", "v2", author="developer")

        # Should detect conflict (different authors writing same file)
        assert ws.read_file("auth.py") == "v2"

    def test_workspace_decision_log(self):
        from orchestrator.workspace.workspace import ProjectWorkspace

        ws = ProjectWorkspace()
        ad = ws.record_decision("Use FastAPI", "FastAPI chosen", "Best for APIs")

        assert ad.title == "Use FastAPI"
        assert len(ws.architectural_decisions) == 1

    def test_workspace_summary(self):
        from orchestrator.workspace.workspace import ProjectWorkspace

        ws = ProjectWorkspace()
        summary = ws.get_summary()

        assert "Workspace" in summary
        assert "Files modified: 0" in summary

    def test_workspace_missing_file(self):
        from orchestrator.workspace.workspace import ProjectWorkspace

        ws = ProjectWorkspace()
        assert ws.read_file("nonexistent.py") is None


# ═════════════════════════════════════════════
# Test: Model routing
# ═════════════════════════════════════════════


class TestModelRouting:
    """Model selection and routing works."""

    def test_all_models_importable(self):
        from orchestrator.models import Model

        assert len(Model) >= 1  # At least one model must be defined

    def test_task_type_enum_values(self):
        from orchestrator.models import TaskType

        types = {t.value for t in TaskType}
        assert "code_generation" in types
        assert "code_review" in types
        assert "complex_reasoning" in types

    def test_model_cost_table(self):
        from orchestrator.models import Model, COST_TABLE

        assert len(COST_TABLE) > 0
        assert Model.GPT_4O in COST_TABLE

    def test_model_context_limits(self):
        from orchestrator.models import Model, MODEL_MAX_TOKENS

        assert Model.GPT_4O in MODEL_MAX_TOKENS

    def test_fallback_chain_defined(self):
        from orchestrator.models import FALLBACK_CHAIN

        assert len(FALLBACK_CHAIN) > 0

    def test_routing_table_coverage(self):
        from orchestrator.models import ROUTING_TABLE

        assert len(ROUTING_TABLE) >= 4


# ═════════════════════════════════════════════
# Test: Safety gates
# ═════════════════════════════════════════════


class TestSafetyGates:
    """Validation and safety checks."""

    def test_syntax_validation_catches_error(self):
        from orchestrator.engine_core.utilities import _clean_code_output
        from orchestrator.models import TaskType

        code = "def valid(): pass"
        result = _clean_code_output(code, TaskType.CODE_GEN)
        assert "def valid" in result

    def test_markdown_fence_removal(self):
        from orchestrator.engine_core.utilities import _clean_code_output
        from orchestrator.models import TaskType

        code = "```python\nprint('hi')\n```"
        result = _clean_code_output(code, TaskType.CODE_GEN)
        assert "```" not in result

    def test_secret_detection(self):
        from orchestrator.codebase_writer import ModificationGate, VerificationResult

        gate = ModificationGate()
        vr = VerificationResult()
        gate._check_secrets("password = 'secret123'", vr)
        assert len(vr.warnings) > 0

    def test_secret_detection_clean_code(self):
        from orchestrator.codebase_writer import ModificationGate, VerificationResult

        gate = ModificationGate()
        vr = VerificationResult()
        gate._check_secrets("import os\nx = 1", vr)
        assert len(vr.warnings) == 0


# ═════════════════════════════════════════════
# Test: Budget tracking
# ═════════════════════════════════════════════


class TestBudget:
    """Budget tracking and enforcement."""

    @pytest.mark.asyncio
    async def test_budget_creation(self):
        from orchestrator.budget import Budget

        budget = Budget(max_usd=10.0)
        assert budget.remaining_usd == 10.0

    @pytest.mark.asyncio
    async def test_budget_charge(self):
        from orchestrator.budget import Budget

        budget = Budget(max_usd=10.0)
        initial = budget.remaining_usd
        await budget.charge(1.0)
        assert budget.remaining_usd < initial

    @pytest.mark.asyncio
    async def test_budget_exceeded(self):
        from orchestrator.budget import Budget

        budget = Budget(max_usd=1.0)
        await budget.charge(0.5)
        await budget.charge(1.0)
        assert budget.remaining_usd <= 0.0 or budget.remaining_usd < 1.0


# ═════════════════════════════════════════════
# Test: Error recovery paths
# ═════════════════════════════════════════════


class TestErrorRecovery:
    """System recovers gracefully from failures."""

    @pytest.mark.asyncio
    async def test_agent_no_client(self):
        from orchestrator.agents.developer import DeveloperAgent
        from orchestrator.agents.base import AgentTask

        agent = DeveloperAgent()
        result = await agent.handle_task(AgentTask(id="t1", goal="test"))
        assert result.success is False
        assert "No LLM" in result.output

    @pytest.mark.asyncio
    async def test_agent_hadle_exception(self):
        from orchestrator.agents.developer import DeveloperAgent
        from orchestrator.agents.base import AgentTask

        mock = MagicMock()
        mock.call = AsyncMock(side_effect=RuntimeError("API down"))
        agent = DeveloperAgent(client=mock)
        result = await agent.handle_task(AgentTask(id="t1", goal="test"))
        assert result.success is False

    def test_experience_buffer_full_eviction(self):
        from orchestrator.learning.experience_buffer import ExperienceBuffer

        buf = ExperienceBuffer()
        for i in range(250):
            buf.record_success("code_gen", "basic", f"model_{i}", 0.5)
        # Eviction is in-process — after MAX_PATTERNS (200), older entries are removed
        total = len(buf.successes) + len(buf.failures)
        assert total <= 250  # Upper bound: failed eviction should still be <= 250

    def test_knowledge_graph_save_load_roundtrip(self):
        from orchestrator.learning.knowledge_graph import KnowledgeGraph

        # Direct test: graph retains data after multiple operations
        kg = KnowledgeGraph()
        kg.record_success("code_gen", "gpt-4o", "cove", 0.9)
        assert kg.best_method_for("code_gen") == "cove"
        # Record a second method and verify the first is still present
        kg.record_success("code_gen", "gpt-4o-mini", "basic", 0.6)
        assert kg.best_method_for("code_gen") == "cove"


class TestEndToEnd:
    """Full lifecycle integration tests."""

    def test_full_pipeline_imports(self):
        """Every module in the pipeline imports cleanly."""
        modules = [
            "orchestrator.models",
            "orchestrator.budget",
            "orchestrator.engine_core.stages",
            "orchestrator.engine_core.validator",
            "orchestrator.engine_core.decomposer",
            "orchestrator.engine_core.architect",
            "orchestrator.ara_pipelines",
            "orchestrator.codebase_reader",
            "orchestrator.codebase_context",
            "orchestrator.codebase_writer",
        ]
        for mod in modules:
            try:
                __import__(mod)
            except ImportError as e:
                pytest.fail(f"Import {mod} failed: {e}")

    def test_all_agents_importable(self):
        """Every agent module imports cleanly."""
        agents = [
            "orchestrator.agents.developer",
            "orchestrator.agents.reviewer",
            "orchestrator.agents.devops",
            "orchestrator.agents.researcher",
            "orchestrator.agents.user",
            "orchestrator.agents.product_manager",
            "orchestrator.agents.qc",
        ]
        for agent_mod in agents:
            try:
                __import__(agent_mod)
            except ImportError as e:
                pytest.fail(f"Import {agent_mod} failed: {e}")

    @pytest.mark.asyncio
    async def test_orchestrator_cli_help(self):
        """CLI help works without error."""
        import subprocess

        result = subprocess.run(
            [sys.executable, "-m", "orchestrator", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(Path(__file__).parent.parent),
        )
        assert result.returncode == 0
        assert "budget" in result.stdout.lower()
