"""
Tests for Constitutional Enforcement Gate
============================================
Author: Orchestrator core

RED → GREEN → refactor. Tests cover:
- ProjectConstitution.check_task() aggregator
- ConstitutionGate stage (protected path abort, forbidden import abort, clean pass)
- Empty constitution no-op
"""

from __future__ import annotations

import pytest

from orchestrator.domain.constitution import ProjectConstitution
from orchestrator.engine_core.pipeline import PipelineContext
from orchestrator.engine_core.stages.constitution_gate import ConstitutionGate
from orchestrator.models import Task, TaskType

# ─────────────────────────────────────────────────────────────────────────────
# Tests: ProjectConstitution.check_task
# ─────────────────────────────────────────────────────────────────────────────


class TestCheckTask:
    """Tests for the pure domain aggregator."""

    def make_constitution(
        self,
        protect_paths: list[str] | None = None,
        forbidden_imports: list[str] | None = None,
    ) -> ProjectConstitution:
        return ProjectConstitution(
            protect_paths=protect_paths or [],
            forbidden_imports=forbidden_imports or [],
            require_tests=False,
        )

    def make_task(self, target_path: str = "", **kwargs) -> Task:
        # Extract declared_imports before passing to Task (not a Task field)
        declared_imports = kwargs.pop("declared_imports", None)
        task = Task(
            id=kwargs.pop("id", "T001"),
            type=TaskType.CODE_GEN,
            prompt=kwargs.pop("prompt", "test task"),
            target_path=target_path,
            **kwargs,
        )
        if declared_imports is not None:
            task.declared_imports = declared_imports
        return task

    def test_protected_path_violation(self) -> None:
        """RED: Task targeting a protected path should report violation."""
        c = self.make_constitution(protect_paths=["src/domain/**"])
        task = self.make_task(target_path="src/domain/core.py")
        violations = c.check_task(task)
        assert any("protected" in v for v in violations)

    def test_forbidden_import_violation(self) -> None:
        """Task with declared_imports containing forbidden package should report."""
        c = self.make_constitution(forbidden_imports=["requests"])
        task = self.make_task(declared_imports=["requests", "os"])
        violations = c.check_task(task)
        assert any("forbidden" in v for v in violations)
        assert any("requests" in v for v in violations)

    def test_clean_task_no_violations(self) -> None:
        """Task that doesn't violate any rules should return empty list."""
        c = self.make_constitution(
            protect_paths=["src/domain/**"],
            forbidden_imports=["requests"],
        )
        task = self.make_task(
            target_path="src/services/api.py",
            declared_imports=["os", "json"],
        )
        violations = c.check_task(task)
        assert violations == []

    def test_empty_constitution_returns_empty(self) -> None:
        """Empty constitution should never report violations."""
        c = ProjectConstitution()
        task = self.make_task(target_path="src/domain/core.py")
        violations = c.check_task(task)
        assert violations == []

    def test_no_target_path_no_imports(self) -> None:
        """Task without target_path or declared_imports should pass."""
        c = self.make_constitution(protect_paths=["src/domain/**"])
        task = self.make_task(target_path="")
        violations = c.check_task(task)
        assert violations == []

    def test_multiple_violations(self) -> None:
        """Task violating multiple rules should return all violations."""
        c = self.make_constitution(
            protect_paths=["src/domain/**"],
            forbidden_imports=["requests", "subprocess"],
        )
        task = self.make_task(
            target_path="src/domain/secret.py",
            declared_imports=["requests", "os", "subprocess"],
        )
        violations = c.check_task(task)
        assert len(violations) >= 3  # at least 1 path + 2 imports


# ─────────────────────────────────────────────────────────────────────────────
# Tests: ConstitutionGate stage
# ─────────────────────────────────────────────────────────────────────────────


class TestConstitutionGate:
    """Tests for the pipeline stage."""

    @pytest.fixture
    def constitution(self) -> ProjectConstitution:
        return ProjectConstitution(
            protect_paths=["src/domain/**", "tests/**"],
            forbidden_imports=["requests", "subprocess"],
        )

    @pytest.fixture
    def gate(self, constitution: ProjectConstitution) -> ConstitutionGate:
        return ConstitutionGate(constitution=constitution)

    @pytest.fixture
    def empty_gate(self) -> ConstitutionGate:
        return ConstitutionGate()

    def make_ctx(self, task_id: str = "T001", target_path: str = "") -> PipelineContext:
        return PipelineContext(
            task=Task(
                id=task_id,
                type=TaskType.CODE_GEN,
                prompt="test task",
                target_path=target_path,
            )
        )

    async def test_aborts_on_protected_path(self, gate: ConstitutionGate) -> None:
        """RED: Task targeting protected path should abort with reason."""
        ctx = self.make_ctx(task_id="T001", target_path="src/domain/core.py")
        result = await gate.process(ctx)
        assert result.should_abort is True
        assert "constitution" in result.abort_reason
        assert "protected" in result.abort_reason

    async def test_passes_clean_task(self, gate: ConstitutionGate) -> None:
        """Task not targeting protected paths should pass through."""
        ctx = self.make_ctx(task_id="T002", target_path="src/services/public.py")
        result = await gate.process(ctx)
        assert result.should_abort is False
        assert result.abort_reason is None or result.abort_reason == ""

    async def test_empty_constitution_is_noop(self, empty_gate: ConstitutionGate) -> None:
        """Gate with empty constitution should never abort."""
        ctx = self.make_ctx(task_id="T003", target_path="src/domain/core.py")
        result = await empty_gate.process(ctx)
        assert result.should_abort is False

    async def test_gate_preserves_other_context(self, gate: ConstitutionGate) -> None:
        """Gate should not modify context beyond should_abort."""
        ctx = self.make_ctx(task_id="T004", target_path="src/services/api.py")
        original_task_id = ctx.task.id
        result = await gate.process(ctx)
        assert result.task.id == original_task_id
        assert result.should_abort is False

    async def test_satisfies_pipeline_stage_protocol(self, gate: ConstitutionGate) -> None:
        """RED: Should satisfy PipelineStage protocol (structural typing)."""
        from inspect import signature

        # Protocol check: must have async process(ctx) -> PipelineContext signature
        assert hasattr(gate, "process"), "ConstitutionGate must have process() method"
        sig = signature(gate.process)
        params = list(sig.parameters.keys())
        assert "ctx" in params, "process() must accept ctx parameter"
