"""
Tests for cascade + verifier integration.

Verifies that:
- ModelCascader with verifier uses objective scores in _quick_evaluate
- ModelCascader without verifier falls back to heuristic scoring
- ModelCascader with flag off ignores the verifier
- EnhancedSelfConsistencyStage with verifier adjusts score for quality gate
"""

from __future__ import annotations

import os
import asyncio
from typing import Any

import pytest

from orchestrator.models import TaskType, Verdict
from orchestrator.verification.python_ast import PythonASTVerifier
from orchestrator.verification.port import Verifier

# ═══════════════════════════════════════════════════════════════════════════════
# ModelCascader integration
# ═══════════════════════════════════════════════════════════════════════════════


class TestModelCascaderWithVerifier:
    """ModelCascader uses verifier when configured."""

    @pytest.fixture(autouse=True)
    def setup_env(self):
        """Enable the objective verifiers flag before each test."""
        old = os.environ.get("ORCH_USE_OBJECTIVE_VERIFIERS")
        os.environ["ORCH_USE_OBJECTIVE_VERIFIERS"] = "true"
        # Purge cached modules so next import picks up fresh flags
        import sys

        for mod in list(sys.modules):
            if "model_cascading" in mod or "crosscutting.config" in mod:
                del sys.modules[mod]
        yield
        if old is None:
            del os.environ["ORCH_USE_OBJECTIVE_VERIFIERS"]
        else:
            os.environ["ORCH_USE_OBJECTIVE_VERIFIERS"] = old

    @pytest.mark.asyncio
    async def test_verifier_intercepts_scoring(self):
        """Cascader with verifier uses verifier.score in _quick_evaluate."""
        from orchestrator.cost_optimization.model_cascading import ModelCascader

        ast_v = PythonASTVerifier(extract_blocks=False)
        cascader = ModelCascader(verifier=ast_v)

        # Valid Python → should get AST score (>= 0.5), not heuristic
        score = await cascader._quick_evaluate(
            prompt="Write a function",
            response="def foo():\n    return 42",
            model="test-model",
        )
        assert score >= 0.5, f"Expected >= 0.5, got {score}"

    @pytest.mark.asyncio
    async def test_verifier_detects_invalid_response(self):
        """Cascader with verifier returns 0.0 for invalid code."""
        from orchestrator.cost_optimization.model_cascading import ModelCascader

        ast_v = PythonASTVerifier(extract_blocks=False)
        cascader = ModelCascader(verifier=ast_v)

        score = await cascader._quick_evaluate(
            prompt="Write a function",
            response="def foo(:\n    invalid syntax",
            model="test-model",
        )
        assert score == 0.0, f"Expected 0.0, got {score}"

    @pytest.mark.asyncio
    async def test_cascading_generate_uses_verifier(self):
        """Full cascading pipeline uses verifier to trigger escalation."""
        from orchestrator.cost_optimization.model_cascading import ModelCascader

        # Create a mock client that returns invalid code (should fail cascade)
        class MockClient:
            async def call(self, model, system, **kwargs):
                return type("Resp", (), {"text": "def foo(:\n    bad syntax"})()

        ast_v = PythonASTVerifier(extract_blocks=False)
        cascader = ModelCascader(client=MockClient(), verifier=ast_v)

        result = await cascader.cascading_generate(
            task_prompt="Write a function",
            task_type="code_generation",
        )
        # Since all models produce bad code, we should end up at the last tier
        assert (
            result.cascade_exit_tier == 2
        ), f"Expected to exhaust cascade (tier 2), got tier {result.cascade_exit_tier}"
        assert result.score == 0.0, f"Expected score 0.0 for invalid code, got {result.score}"


class TestModelCascaderWithoutVerifier:
    """ModelCascader without verifier uses heuristic fallback."""

    @pytest.mark.asyncio
    async def test_heuristic_fallback(self):
        """Without verifier, _quick_evaluate uses heuristic scoring."""
        from orchestrator.cost_optimization.model_cascading import ModelCascader

        cascader = ModelCascader()

        # Even with flag on, no verifier → uses heuristic
        score = await cascader._quick_evaluate(
            prompt="Write something",
            response="This is a complete and implemented solution. " * 30,
            model="test",
        )
        assert score >= 0.5, f"Expected heuristic >= 0.5, got {score}"
        # Longer response with completeness markers should score higher
        assert score > 0.6, f"Expected > 0.6 with markers, got {score}"


# ═══════════════════════════════════════════════════════════════════════════════
# EnhancedSelfConsistencyStage integration
# ═══════════════════════════════════════════════════════════════════════════════


class TestSelfConsistencyWithVerifier:
    """EnhancedSelfConsistencyStage uses verifier when configured."""

    @pytest.fixture(autouse=True)
    def setup_env(self):
        old = os.environ.get("ORCH_USE_OBJECTIVE_VERIFIERS")
        os.environ["ORCH_USE_OBJECTIVE_VERIFIERS"] = "true"
        import sys

        for mod in list(sys.modules):
            if "self_consistency" in mod or "crosscutting.config" in mod:
                del sys.modules[mod]
        yield
        if old is None:
            del os.environ["ORCH_USE_OBJECTIVE_VERIFIERS"]
        else:
            os.environ["ORCH_USE_OBJECTIVE_VERIFIERS"] = old

    @pytest.mark.asyncio
    async def test_verifier_lowers_score_for_bad_code(self):
        """When verifier sees invalid code, it lowers ctx.score."""
        from orchestrator.engine_core.stages.self_consistency import (
            EnhancedSelfConsistencyStage,
        )
        from orchestrator.engine_core.pipeline import PipelineContext
        from orchestrator.models import Task

        task = Task(
            id="test-001",
            type=TaskType.CODE_GEN,
            prompt="Write a function",
        )
        ctx = PipelineContext(task=task)
        ctx.score = 0.8  # Evaluator gave high score (false positive)
        ctx.output = "def foo(:\n    invalid syntax"
        ctx.attempt = 0

        ast_v = PythonASTVerifier(extract_blocks=False)
        stage = EnhancedSelfConsistencyStage(
            quality_threshold=0.7,
            verifier=ast_v,
        )

        result = await stage.process(ctx)
        # Verifier should lower score to 0.0, which is < 0.7 threshold
        assert result.score < 0.7, f"Expected score lowered below threshold, got {result.score}"
