"""
Tests for Capabilities 5-10 (Agent Communication, Learning, Runtime, CI, HITL, Scaffolds).
"""

import pytest
from datetime import datetime
from pathlib import Path


class TestAgentMessageBus:
    """Capability 5: Agent communication."""

    def test_publish_direct_message(self):
        from orchestrator.workspace.message_bus import AgentMessageBus, AgentMessage, MessageType
        bus = AgentMessageBus()
        bus.subscribe("agent_a", [MessageType.QUERY])
        bus.publish(AgentMessage(id="m1", sender="agent_b", content="hello", recipient="agent_a"))
        inbox = bus.read_inbox("agent_a")
        assert len(inbox) == 1
        assert inbox[0].content == "hello"

    def test_broadcast_matches_subscribers(self):
        from orchestrator.workspace.message_bus import AgentMessageBus, AgentMessage, MessageType
        bus = AgentMessageBus()
        bus.subscribe("agent_a", [MessageType.QUERY])
        bus.subscribe("agent_b", [MessageType.ALERT])
        bus.publish(AgentMessage(id="m1", sender="sender", content="test", msg_type=MessageType.QUERY))
        assert len(bus.read_inbox("agent_a")) == 1
        assert len(bus.read_inbox("agent_b")) == 0

    def test_message_history(self):
        from orchestrator.workspace.message_bus import AgentMessageBus, AgentMessage, MessageType
        bus = AgentMessageBus()
        bus.publish(AgentMessage(id="m1", sender="a", content="msg1"))
        bus.publish(AgentMessage(id="m2", sender="b", content="msg2"))
        assert len(bus.get_history()) == 2

    def test_agent_message_dataclass(self):
        from orchestrator.workspace.message_bus import AgentMessage, MessageType
        m = AgentMessage(id="m1", sender="dev", content="hello")
        assert m.msg_type == MessageType.QUERY
        assert m.priority == 0


class TestExperienceBuffer:
    """Capability 6: Self-reflection and learning."""

    def test_record_success(self):
        from orchestrator.learning.experience_buffer import ExperienceBuffer
        buf = ExperienceBuffer()
        buf.record_success("code_gen", "cove", "gpt-4o", 0.85)
        assert len(buf.successes) == 1

    def test_record_failure(self):
        from orchestrator.learning.experience_buffer import ExperienceBuffer
        buf = ExperienceBuffer()
        buf.record_failure("code_review", "debate", "gpt-4o", 0.3)
        assert len(buf.failures) == 1

    def test_best_method_returns_best(self):
        from orchestrator.learning.experience_buffer import ExperienceBuffer
        buf = ExperienceBuffer()
        buf.record_success("code_gen", "basic", "model_a", 0.6)
        buf.record_success("code_gen", "cove", "model_b", 0.95)
        best = buf.best_method_for("code_gen")
        assert best == "cove"

    def test_best_method_returns_none_when_empty(self):
        from orchestrator.learning.experience_buffer import ExperienceBuffer
        assert ExperienceBuffer().best_method_for("anything") is None

    def test_best_model_returns_best(self):
        from orchestrator.learning.experience_buffer import ExperienceBuffer
        buf = ExperienceBuffer()
        buf.record_success("code_gen", "cove", "gpt-4o", 0.9)
        buf.record_success("code_gen", "cove", "gpt-3.5", 0.7)
        best = buf.best_model_for("code_gen")
        assert best == "gpt-4o"


class TestSandboxExecutor:
    """Capability 7: Runtime execution."""

    @pytest.mark.asyncio
    async def test_execute_valid_python(self):
        from orchestrator.runtime.sandbox import SandboxExecutor
        executor = SandboxExecutor()
        result = await executor.execute('print("hello world")')
        assert result.success is True
        assert "hello world" in result.output

    @pytest.mark.asyncio
    async def test_execute_invalid_python(self):
        from orchestrator.runtime.sandbox import SandboxExecutor
        executor = SandboxExecutor()
        result = await executor.execute('invalid python syntax {{{')
        assert result.success is False

    @pytest.mark.asyncio
    async def test_execute_empty_string(self):
        from orchestrator.runtime.sandbox import SandboxExecutor
        executor = SandboxExecutor()
        result = await executor.execute('')
        assert result.success is True  # Empty file is valid Python

    @pytest.mark.asyncio
    async def test_unsupported_language(self):
        from orchestrator.runtime.sandbox import SandboxExecutor
        executor = SandboxExecutor()
        result = await executor.execute('println!("hi")', language="rust")
        assert result.success is False


class TestDynamicScaffoldGenerator:
    """Capability 8: Dynamic scaffold generation."""

    def test_detect_python(self):
        from orchestrator.scaffold.dynamic import DynamicScaffoldGenerator, TechStack
        gen = DynamicScaffoldGenerator()
        stack = gen._detect_stack("Build a Python CLI app")
        assert stack.language == "python"

    def test_detect_fastapi(self):
        from orchestrator.scaffold.dynamic import DynamicScaffoldGenerator
        gen = DynamicScaffoldGenerator()
        stack = gen._detect_stack("Create FastAPI backend with SQLite")
        assert stack.framework == "fastapi"
        assert stack.database == "sqlite"

    def test_detect_react(self):
        from orchestrator.scaffold.dynamic import DynamicScaffoldGenerator
        gen = DynamicScaffoldGenerator()
        stack = gen._detect_stack("Build React frontend")
        assert stack.frontend == "react"

    @pytest.mark.asyncio
    async def test_create_basic_structure(self, tmp_path):
        from orchestrator.scaffold.dynamic import DynamicScaffoldGenerator, TechStack
        gen = DynamicScaffoldGenerator()
        stack = TechStack(language="python", framework="fastapi")
        result = gen._create_basic_structure(stack, tmp_path)
        assert result is True
        assert (tmp_path / "README.md").exists()


class TestCIPipeline:
    """Capability 9: CI pipeline."""

    def test_ci_step_result(self):
        from orchestrator.ci.pipeline import CIStepResult
        r = CIStepResult(name="lint", passed=True)
        assert r.passed is True
        assert r.is_critical is False

    def test_ci_report_all_pass(self):
        from orchestrator.ci.pipeline import CIReport, CIStepResult
        report = CIReport(passed=True, steps=[
            CIStepResult(name="lint", passed=True),
            CIStepResult(name="test", passed=True),
        ])
        assert report.passed is True

    @pytest.mark.asyncio
    async def test_critical_step_stops_pipeline(self):
        from orchestrator.ci.pipeline import CIPipeline, CIStep, CIStepResult
        class Step1(CIStep):
            name = "critical"
            async def execute(self, workspace=None):
                return CIStepResult(name="critical", passed=False, is_critical=True)
        class Step2(CIStep):
            name = "never"
            async def execute(self, workspace=None):
                return CIStepResult(name="never", passed=True)

        pipeline = CIPipeline(steps=[Step1(), Step2()])
        report = await pipeline.run()
        assert len(report.steps) == 1  # Stopped after Step1 failed


class TestHumanInTheLoop:
    """Capability 10: Human-in-the-loop decisions."""

    @pytest.mark.asyncio
    async def test_request_decision_auto_approves(self):
        from orchestrator.hitl.gate import HumanInTheLoop, Decision, DecisionResult
        hitl = HumanInTheLoop()
        decision = Decision(category="architecture", title="Choose framework", description="FastAPI vs Django")
        result = await hitl.request_decision(decision)
        assert result == DecisionResult.APPROVED

    def test_pending_decisions(self):
        from orchestrator.hitl.gate import HumanInTheLoop, Decision
        hitl = HumanInTheLoop()
        d = Decision(category="security", title="Auth approach", description="JWT vs OAuth")
        import asyncio
        asyncio.run(hitl.request_decision(d))
        assert len(hitl.get_pending()) == 1
