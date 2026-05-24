"""
Test Suite — God File Refactoring Phases 1-5
==============================================
Tests for all extracted modules: engine_deps, Decomposer, TaskValidator,
Architect, TaskPipeline, and engine.py delegation smoke tests.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── Test Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def mock_client():
    """Mock UnifiedClient that returns a configurable response."""
    client = MagicMock()
    client.call = AsyncMock()
    client.is_available = MagicMock(return_value=True)
    return client


@pytest.fixture
def mock_selector():
    """Mock ModelSelector."""
    from orchestrator.models import Model

    selector = MagicMock()
    selector.select = MagicMock(return_value=Model.GPT_4O_MINI)
    selector.decomposition_model = MagicMock(return_value=Model.GPT_4O)
    return selector


@pytest.fixture
def mock_budget():
    """Mock Budget."""
    from orchestrator.budget import Budget

    return Budget(max_usd=100.0)


@pytest.fixture
def mock_evaluator():
    """Mock EvaluatorService returning a CritiqueReport."""
    from orchestrator.feedback import CritiqueReport

    evaluator = MagicMock()
    evaluator.evaluate = AsyncMock(
        return_value=CritiqueReport(task_id="t1", score=8.5)
    )
    return evaluator


@pytest.fixture
def sample_task():
    """A sample Task for testing."""
    from orchestrator.models import Task, TaskStatus, TaskType

    return Task(
        id="task_001",
        type=TaskType.CODE_GEN,
        prompt="Write a function that adds two numbers.",
        context="",
        dependencies=[],
        acceptance_threshold=0.85,
        max_iterations=3,
        max_output_tokens=2048,
        status=TaskStatus.PENDING,
    )


@pytest.fixture
def valid_json_response():
    """Valid decomposition JSON response."""
    return json.dumps([
        {
            "id": "task_001",
            "type": "code_generation",
            "prompt": "Write authentication module",
            "dependencies": [],
            "target_path": "src/auth.py",
            "module_name": "src.auth",
            "tech_context": "fastapi, jwt",
        },
        {
            "id": "task_002",
            "type": "code_review",
            "prompt": "Review task_001 output",
            "dependencies": ["task_001"],
        },
    ])


@pytest.fixture
def truncated_json_response():
    """Truncated JSON array for testing partial recovery."""
    return '[{"id":"t1","type":"code_generation","prompt":"Add numbers"},{'


# ── Phase 1: engine_deps.py ─────────────────────────────────────────────────


class TestEngineDeps:
    """Verify all optional imports are accessible through engine_deps."""

    def test_engine_deps_imports_without_error(self):
        """engine_deps.py should import without any errors."""
        import orchestrator.engine_deps as ed
        assert ed is not None

    def test_has_test_validator_flag(self):
        """HAS_TEST_VALIDATOR should be defined."""
        import orchestrator.engine_deps as ed
        assert hasattr(ed, "HAS_TEST_VALIDATOR")

    def test_has_tdd_flag(self):
        """HAS_TDD should be defined."""
        import orchestrator.engine_deps as ed
        assert hasattr(ed, "HAS_TDD")

    def test_has_context_management_flag(self):
        """HAS_CONTEXT_MANAGEMENT should be defined."""
        import orchestrator.engine_deps as ed
        assert hasattr(ed, "HAS_CONTEXT_MANAGEMENT")

    def test_has_test_fixer_flag(self):
        """HAS_TEST_FIXER should be defined."""
        import orchestrator.engine_deps as ed
        assert hasattr(ed, "HAS_TEST_FIXER")

    def test_has_pre_submission_flag(self):
        """HAS_PRE_SUBMISSION should be defined."""
        import orchestrator.engine_deps as ed
        assert hasattr(ed, "HAS_PRE_SUBMISSION")

    def test_all_has_flags_are_boolean(self):
        """All HAS_* flags should be booleans."""
        import orchestrator.engine_deps as ed
        for name in dir(ed):
            if name.startswith("HAS_"):
                assert isinstance(getattr(ed, name), bool), f"{name} should be bool"


# ── Phase 2: Decomposer ─────────────────────────────────────────────────────


class TestDecomposer:
    """Test the extracted Decomposer class."""

    def test_decomposer_init(self, mock_client, mock_selector):
        """Decomposer should initialize without error."""
        from orchestrator.engine_core.decomposer import Decomposer

        d = Decomposer(client=mock_client, selector=mock_selector)
        assert d is not None
        assert d._client is mock_client
        assert d._selector is mock_selector

    def test_parse_valid_json(self, mock_client, mock_selector, valid_json_response):
        """_parse_decomposition should parse valid JSON."""
        from orchestrator.engine_core.decomposer import Decomposer

        d = Decomposer(client=mock_client, selector=mock_selector)
        tasks = d._parse_decomposition(valid_json_response)

        assert isinstance(tasks, dict)
        assert len(tasks) == 2
        assert "task_001" in tasks
        assert tasks["task_001"].prompt == "Write authentication module"

    def test_parse_empty_text(self, mock_client, mock_selector):
        """Empty input should return empty dict."""
        from orchestrator.engine_core.decomposer import Decomposer

        d = Decomposer(client=mock_client, selector=mock_selector)
        tasks = d._parse_decomposition("")
        assert tasks == {}

    def test_parse_malformed_json(self, mock_client, mock_selector):
        """Malformed JSON should return empty dict."""
        from orchestrator.engine_core.decomposer import Decomposer

        d = Decomposer(client=mock_client, selector=mock_selector)
        tasks = d._parse_decomposition("{bad}")
        assert tasks == {}

    def test_try_parse_partial_recovery(self, mock_client, mock_selector,
                                        truncated_json_response):
        """Truncated JSON should attempt partial recovery."""
        from orchestrator.engine_core.decomposer import Decomposer

        d = Decomposer(client=mock_client, selector=mock_selector)
        result = d._try_parse_partial_json_array(truncated_json_response)
        assert result is not None
        assert len(result) == 1

    def test_try_parse_empty_string(self, mock_client, mock_selector):
        """Empty string should return None."""
        from orchestrator.engine_core.decomposer import Decomposer

        d = Decomposer(client=mock_client, selector=mock_selector)
        assert d._try_parse_partial_json_array("") is None
        assert d._try_parse_partial_json_array("   ") is None

    def test_repair_partial_tasks(self, mock_client, mock_selector):
        """Partial objects should be converted to valid Tasks."""
        from orchestrator.engine_core.decomposer import Decomposer

        d = Decomposer(client=mock_client, selector=mock_selector)
        partial = [
            {"id": "t1", "type": "code_generation", "prompt": "Write foo"},
            {"id": "t2", "type": "code_review", "prompt": "Review foo",
             "dependencies": ["t1"]},
        ]
        tasks = d._repair_partial_tasks(partial)
        assert tasks is not None
        assert len(tasks) == 2
        assert tasks["t1"].type.value == "code_generation"
        assert "t1" in tasks["t2"].dependencies

    def test_get_decomposition_models(self, mock_client, mock_selector):
        """Model selection should return a non-empty list."""
        from orchestrator.engine_core.decomposer import Decomposer

        d = Decomposer(client=mock_client, selector=mock_selector)
        models = d._get_decomposition_models("simple project")
        assert len(models) >= 2

    @pytest.mark.asyncio
    async def test_decompose_with_mock_client(self, mock_client, mock_selector,
                                              valid_json_response):
        """Full decomposition with mocked client."""
        from orchestrator.engine_core.decomposer import Decomposer

        mock_response = MagicMock()
        mock_response.text = valid_json_response
        mock_response.cost_usd = 0.01
        mock_client.call = AsyncMock(return_value=mock_response)

        d = Decomposer(client=mock_client, selector=mock_selector)
        tasks = await d.decompose(
            project="Build a simple CLI tool",
            criteria="Must pass tests",
        )
        assert len(tasks) == 2


# ── Phase 3: TaskValidator ──────────────────────────────────────────────────


class TestTaskValidator:
    """Test the extracted TaskValidator class."""

    def test_validator_init(self, mock_client, mock_budget):
        """Validator should initialize without error."""
        from orchestrator.engine_core.validator import TaskValidator

        v = TaskValidator(client=mock_client, budget=mock_budget)
        assert v is not None

    def test_validate_syntax_valid_code(self, mock_client, mock_budget):
        """Valid Python code should pass syntax validation."""
        from orchestrator.engine_core.validator import TaskValidator

        v = TaskValidator(client=mock_client, budget=mock_budget)
        assert v.validate_syntax_streaming("def foo(): return 42")
        assert v.validate_syntax_batch("def foo():\n    return 42")

    def test_validate_syntax_invalid_code(self, mock_client, mock_budget):
        """Invalid Python code should fail syntax validation."""
        from orchestrator.engine_core.validator import TaskValidator

        v = TaskValidator(client=mock_client, budget=mock_budget)
        assert not v.validate_syntax_streaming("def : return")
        assert not v.validate_syntax_batch("def broken(")

    def test_validate_syntax_empty(self, mock_client, mock_budget):
        """Empty output should fail syntax check."""
        from orchestrator.engine_core.validator import TaskValidator

        v = TaskValidator(client=mock_client, budget=mock_budget)
        assert not v.validate_syntax_batch("")

    def test_filter_validators_python(self, mock_client, mock_budget, sample_task):
        """Python task should keep all hard validators."""
        from orchestrator.engine_core.validator import TaskValidator

        sample_task.hard_validators = ["python_syntax", "pytest", "ruff"]
        v = TaskValidator(client=mock_client, budget=mock_budget)
        result = v.filter_validators_for_task(sample_task, "def foo(): pass")
        assert len(result) == 3
        assert "python_syntax" in result

    def test_filter_validators_non_python(self, mock_client, mock_budget):
        """Non-Python task should remove Python-specific validators."""
        from orchestrator.engine_core.validator import TaskValidator
        from orchestrator.models import Task, TaskStatus, TaskType

        task = Task(
            id="t1", type=TaskType.WRITING,
            prompt="Write a poem", context="",
            hard_validators=["python_syntax", "pytest", "json_schema"],
        )
        v = TaskValidator(client=mock_client, budget=mock_budget)
        result = v.filter_validators_for_task(task, "# A poem about code")
        assert "python_syntax" not in result
        assert "pytest" not in result
        assert "json_schema" in result

    def test_bracket_balance_check(self, mock_client, mock_budget):
        """Bracket balance should detect mismatches."""
        from orchestrator.engine_core.validator import TaskValidator

        v = TaskValidator(client=mock_client, budget=mock_budget)
        assert v.validate_syntax_streaming("(a + b)")
        assert not v.validate_syntax_streaming("(a + b")  # unclosed
        assert not v.validate_syntax_streaming("a + b)")  # unmatched close

    @pytest.mark.asyncio
    async def test_run_preflight_without_validator(self, mock_client, mock_budget,
                                                    sample_task):
        """Preflight should pass when no preflight_validator is configured."""
        from orchestrator.engine_core.validator import TaskValidator
        from orchestrator.models import Model

        v = TaskValidator(client=mock_client, budget=mock_budget,
                          preflight_validator=None)
        result, score, pf = await v.run_preflight_check(
            task=sample_task, output="def foo(): pass",
            score=0.9, primary=Model.GPT_4O_MINI,
        )
        assert result == "def foo(): pass"
        assert score == 0.9


# ── Phase 4: Architect ──────────────────────────────────────────────────────


class TestArchitect:
    """Test the extracted Architect class."""

    def test_architect_init(self, mock_client):
        """Architect should initialize without error."""
        from orchestrator.engine_core.architect import Architect

        a = Architect(client=mock_client)
        assert a is not None

    @pytest.mark.asyncio
    async def test_architect_handles_failure_gracefully(self, mock_client):
        """Architect should return None on failure, not raise."""
        from orchestrator.engine_core.architect import Architect
        from unittest.mock import patch, AsyncMock

        a = Architect(client=mock_client)
        # Mock ArchitectureRulesEngine.generate_rules to raise
        with patch(
            "orchestrator.architecture_rules.ArchitectureRulesEngine.generate_rules",
            new_callable=AsyncMock,
            side_effect=Exception("API down"),
        ):
            result = await a.generate_rules(
                project_description="Build a web app",
                success_criteria="Must work",
            )
            assert result is None


# ── Phase 5: TaskPipeline ───────────────────────────────────────────────────


class TestPipeline:
    """Test the TaskPipeline and stage classes."""

    def test_pipeline_context_init(self, sample_task):
        """PipelineContext should initialize with task."""
        from orchestrator.engine_core.pipeline import PipelineContext

        ctx = PipelineContext(task=sample_task)
        assert ctx.task is sample_task
        assert ctx.attempt == 0
        assert ctx.score == 0.0
        assert ctx.output == ""

    def test_pipeline_context_reset(self, sample_task):
        """reset_for_retry should clear abort state."""
        from orchestrator.engine_core.pipeline import PipelineContext

        ctx = PipelineContext(task=sample_task)
        ctx.should_abort = True
        ctx.abort_reason = "retry_for_quality"
        ctx.reset_for_retry()
        assert not ctx.should_abort
        assert ctx.abort_reason == ""

    def test_pipeline_context_to_task_result(self, sample_task):
        """to_task_result should produce valid TaskResult."""
        from orchestrator.engine_core.pipeline import PipelineContext
        from orchestrator.models import Model

        ctx = PipelineContext(task=sample_task)
        ctx.output = "def add(a, b): return a + b"
        ctx.score = 0.9
        ctx.model = Model.GPT_4O_MINI

        result = ctx.to_task_result()
        assert result.task_id == "task_001"
        assert result.score == 0.9
        assert "def add" in result.output

    @pytest.mark.asyncio
    async def test_task_pipeline_runs_all_stages(self, mock_client, mock_budget,
                                                  mock_selector, mock_evaluator,
                                                  sample_task):
        """Pipeline should run all stages in order."""
        from orchestrator.engine_core.pipeline import (
            PipelineContext, TaskPipeline,
        )
        from orchestrator.engine_core.stages import (
            GenerateStage, CritiqueStage, EvaluateStage,
            ValidateStage,
        )

        # Setup mock client to return valid output
        mock_response = MagicMock()
        mock_response.text = "def add(a, b): return a + b"
        mock_response.cost_usd = 0.01
        mock_client.call = AsyncMock(return_value=mock_response)

        pipeline = TaskPipeline([
            GenerateStage(client=mock_client, budget=mock_budget,
                          selector=mock_selector),
            CritiqueStage(client=mock_client),
            EvaluateStage(evaluator=mock_evaluator),
            ValidateStage(),
        ])

        ctx = PipelineContext(task=sample_task)
        ctx = await pipeline.run(ctx)

        assert ctx.output
        assert ctx.score > 0
        assert not ctx.should_abort

    @pytest.mark.asyncio
    async def test_pipeline_stops_on_abort(self, mock_client, mock_budget,
                                            mock_selector, sample_task):
        """Pipeline should stop when should_abort is set."""
        from orchestrator.engine_core.pipeline import (
            PipelineContext, TaskPipeline,
        )
        from orchestrator.engine_core.stages import GenerateStage

        # A stage that immediately aborts
        class AbortStage:
            async def process(self, ctx):
                ctx.should_abort = True
                ctx.abort_reason = "test_abort"
                return ctx

        pipeline = TaskPipeline([
            AbortStage(),
            GenerateStage(client=mock_client, budget=mock_budget,
                          selector=mock_selector),
        ])

        ctx = PipelineContext(task=sample_task)
        ctx = await pipeline.run(ctx)

        assert ctx.should_abort
        assert ctx.abort_reason == "test_abort"
        # GenerateStage should NOT have been called
        assert ctx.output == ""

    @pytest.mark.asyncio
    async def test_self_consistency_signals_retry(self, sample_task):
        """SelfConsistencyStage should signal retry for low scores."""
        from orchestrator.engine_core.pipeline import PipelineContext
        from orchestrator.engine_core.stages import SelfConsistencyStage
        from orchestrator.models import Model

        stage = SelfConsistencyStage(max_attempts=2, quality_threshold=0.7)

        ctx = PipelineContext(task=sample_task)
        ctx.score = 0.3  # below threshold
        ctx.critique = "Code has bugs"
        ctx.model = Model.GPT_4O_MINI
        ctx.attempt = 0

        ctx = await stage.process(ctx)

        assert ctx.should_abort
        assert ctx.abort_reason == "retry_for_quality"
        assert ctx.attempt == 1
        assert ctx.task.revision_context == "Code has bugs"
        assert ctx.task.preferred_model is not None

    @pytest.mark.asyncio
    async def test_self_consistency_stops_at_max_attempts(self, sample_task):
        """SelfConsistency should not retry beyond max_attempts."""
        from orchestrator.engine_core.pipeline import PipelineContext
        from orchestrator.engine_core.stages import SelfConsistencyStage
        from orchestrator.models import Model

        stage = SelfConsistencyStage(max_attempts=2, quality_threshold=0.7)

        ctx = PipelineContext(task=sample_task)
        ctx.score = 0.3
        ctx.attempt = 2  # already at max
        ctx.model = Model.GPT_4O_MINI

        ctx = await stage.process(ctx)

        assert not ctx.should_abort  # should NOT retry
        assert ctx.attempt == 2

    @pytest.mark.asyncio
    async def test_self_consistency_passes_high_scores(self, sample_task):
        """Scores above threshold should not trigger retry."""
        from orchestrator.engine_core.pipeline import PipelineContext
        from orchestrator.engine_core.stages import SelfConsistencyStage

        stage = SelfConsistencyStage(max_attempts=2, quality_threshold=0.7)

        ctx = PipelineContext(task=sample_task)
        ctx.score = 0.85  # above threshold

        ctx = await stage.process(ctx)
        assert not ctx.should_abort


# ── Phase 2-4: Engine delegation smoke tests ────────────────────────────────


class TestEngineDelegation:
    """Verify engine.py methods delegate correctly to extracted modules."""

    def test_engine_imports(self):
        """Engine should import without error."""
        import orchestrator.engine

        assert orchestrator.engine is not None

    def test_engine_deps_available(self):
        """engine_deps should export to engine.py namespace."""
        from orchestrator.engine_deps import HAS_TEST_VALIDATOR
        assert HAS_TEST_VALIDATOR in (True, False)

    @pytest.mark.asyncio
    async def test_decomposer_method_accessible(self):
        """Decomposer should be importable and instantiable."""
        from orchestrator.engine_core.decomposer import Decomposer
        from unittest.mock import MagicMock

        d = Decomposer(client=MagicMock(), selector=MagicMock())
        assert d is not None

    @pytest.mark.asyncio
    async def test_validator_method_accessible(self, mock_client, mock_budget):
        """TaskValidator should be importable and usable."""
        from orchestrator.engine_core.validator import TaskValidator

        v = TaskValidator(client=mock_client, budget=mock_budget)
        assert v.validate_syntax_batch("def foo(): pass")

    def test_architect_method_accessible(self, mock_client):
        """Architect should be importable."""
        from orchestrator.engine_core.architect import Architect

        a = Architect(client=mock_client)
        assert a is not None

    def test_pipeline_method_accessible(self):
        """TaskPipeline and stages should be importable."""
        from orchestrator.engine_core.pipeline import (
            PipelineContext, TaskPipeline,
        )
        from orchestrator.engine_core.stages import (
            GenerateStage, CritiqueStage, EvaluateStage,
            ValidateStage, PreflightStage, SelfConsistencyStage,
        )
        assert PipelineContext is not None
        assert TaskPipeline is not None
        assert GenerateStage is not None


# ── Integration: Run the tests ──────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
