"""
Tests for production-grade quality mode toggle.

RED → GREEN workflow:
1. JobSpec accepts quality_mode field
2. Engine enriches system prompt when quality_mode="production"
3. Codebase analyzer can review/debug/suggest on a directory
4. OpenRouter sync fetches and returns model data
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
from orchestrator.policy import JobSpec, PolicySet
from orchestrator.models import Budget, TaskType

# ─────────────────────────────────────────────
# Quality mode on JobSpec
# ─────────────────────────────────────────────


class TestJobSpecQualityMode:
    """JobSpec.quality_mode field defaults to 'standard' and accepts 'production'."""

    def test_default_quality_mode_is_standard(self):
        spec = JobSpec(
            project_description="Build a CLI tool",
            success_criteria="Works correctly",
            budget=Budget(max_usd=1.0),
        )
        assert spec.quality_mode == "standard"

    def test_production_quality_mode_accepted(self):
        spec = JobSpec(
            project_description="Build a CLI tool",
            success_criteria="Works correctly",
            budget=Budget(max_usd=1.0),
            quality_mode="production",
        )
        assert spec.quality_mode == "production"

    def test_invalid_quality_mode_raises(self):
        with pytest.raises((ValueError, TypeError)):
            spec = JobSpec(
                project_description="Build a CLI tool",
                success_criteria="Works correctly",
                budget=Budget(max_usd=1.0),
                quality_mode="invalid",
            )
            # If dataclass doesn't validate, trigger explicit check
            from orchestrator.policy import VALID_QUALITY_MODES

            assert (
                spec.quality_mode in VALID_QUALITY_MODES
            ), f"Invalid quality_mode: {spec.quality_mode}"


# ─────────────────────────────────────────────
# Engine enriches prompt for production mode
# ─────────────────────────────────────────────


class TestProductionModePromptEnrichment:
    """Engine adds production-grade requirements to system prompt when quality_mode='production'."""

    def test_production_system_prompt_contains_requirements(self):
        from orchestrator.engine import Orchestrator

        orch = object.__new__(Orchestrator)
        prompt = orch._build_production_system_prompt(task_type="code_generation")
        assert (
            "type annotation" in prompt.lower()
            or "type hint" in prompt.lower()
            or "typed" in prompt.lower()
        )
        assert "test" in prompt.lower()
        assert "error" in prompt.lower() or "exception" in prompt.lower()

    def test_standard_system_prompt_differs_from_production(self):
        from orchestrator.engine import Orchestrator

        orch = object.__new__(Orchestrator)
        standard = orch._build_standard_system_prompt(task_type="code_generation")
        production = orch._build_production_system_prompt(task_type="code_generation")
        assert standard != production
        assert len(production) > len(standard)


# ─────────────────────────────────────────────
# Codebase analyzer
# ─────────────────────────────────────────────


class TestCodebaseAnalyzer:
    """CodebaseAnalyzer reviews, debugs, and suggests improvements for a directory."""

    def test_analyzer_can_be_instantiated(self):
        from orchestrator.codebase_analyzer import CodebaseAnalyzer

        analyzer = CodebaseAnalyzer(client=None)
        assert analyzer is not None

    def test_collect_files_returns_python_files(self, tmp_path):
        from orchestrator.codebase_analyzer import CodebaseAnalyzer

        (tmp_path / "main.py").write_text("def main(): pass")
        (tmp_path / "utils.py").write_text("def helper(): pass")
        (tmp_path / "README.md").write_text("# Project")

        analyzer = CodebaseAnalyzer(client=None)
        files = analyzer.collect_files(tmp_path)
        names = [f.name for f in files]
        assert "main.py" in names
        assert "utils.py" in names
        assert "README.md" not in names  # only code files

    def test_collect_files_respects_gitignore_patterns(self, tmp_path):
        from orchestrator.codebase_analyzer import CodebaseAnalyzer

        (tmp_path / "main.py").write_text("x = 1")
        venv = tmp_path / ".venv"
        venv.mkdir()
        (venv / "lib.py").write_text("x = 2")

        analyzer = CodebaseAnalyzer(client=None)
        files = analyzer.collect_files(tmp_path)
        paths = [str(f) for f in files]
        assert not any(".venv" in p for p in paths)

    def test_build_summary_returns_string(self, tmp_path):
        from orchestrator.codebase_analyzer import CodebaseAnalyzer

        (tmp_path / "app.py").write_text("class App:\n    pass\n")
        analyzer = CodebaseAnalyzer(client=None)
        files = analyzer.collect_files(tmp_path)
        summary = analyzer.build_summary(files, tmp_path)
        assert isinstance(summary, str)
        assert "app.py" in summary

    @pytest.mark.asyncio
    async def test_review_returns_report(self, tmp_path):
        from orchestrator.codebase_analyzer import CodebaseAnalyzer
        from orchestrator.api_clients import APIResponse
        from orchestrator.models import Model

        (tmp_path / "app.py").write_text("def add(a, b): return a + b\n")

        mock_response = APIResponse(
            text="## Code Review\n\n- Missing type annotations\n- No docstrings",
            input_tokens=100,
            output_tokens=50,
            model=Model.GPT_4O_MINI,
        )
        mock_client = MagicMock()
        mock_client.call = AsyncMock(return_value=mock_response)

        analyzer = CodebaseAnalyzer(client=mock_client)
        report = await analyzer.review(tmp_path)

        assert "Code Review" in report or "review" in report.lower()
        mock_client.call.assert_called_once()

    @pytest.mark.asyncio
    async def test_debug_returns_diagnosis(self, tmp_path):
        from orchestrator.codebase_analyzer import CodebaseAnalyzer
        from orchestrator.api_clients import APIResponse
        from orchestrator.models import Model

        (tmp_path / "buggy.py").write_text("def divide(a, b): return a / b\n")

        mock_response = APIResponse(
            text="## Bug Report\n\nPotential ZeroDivisionError when b=0.",
            input_tokens=100,
            output_tokens=50,
            model=Model.GPT_4O_MINI,
        )
        mock_client = MagicMock()
        mock_client.call = AsyncMock(return_value=mock_response)

        analyzer = CodebaseAnalyzer(client=mock_client)
        report = await analyzer.debug(tmp_path, issue="Division errors")

        assert isinstance(report, str)
        assert len(report) > 0
        mock_client.call.assert_called_once()

    @pytest.mark.asyncio
    async def test_suggest_returns_improvements(self, tmp_path):
        from orchestrator.codebase_analyzer import CodebaseAnalyzer
        from orchestrator.api_clients import APIResponse
        from orchestrator.models import Model

        (tmp_path / "service.py").write_text("class Service:\n    def run(self): pass\n")

        mock_response = APIResponse(
            text="## Suggestions\n\n1. Add logging\n2. Add tests",
            input_tokens=100,
            output_tokens=50,
            model=Model.GPT_4O_MINI,
        )
        mock_client = MagicMock()
        mock_client.call = AsyncMock(return_value=mock_response)

        analyzer = CodebaseAnalyzer(client=mock_client)
        suggestions = await analyzer.suggest(tmp_path)

        assert isinstance(suggestions, str)
        assert len(suggestions) > 0


# ─────────────────────────────────────────────
# OpenRouter sync
# ─────────────────────────────────────────────


class TestOpenRouterSync:
    """OpenRouterSync fetches model list + pricing from OpenRouter API."""

    def test_sync_can_be_instantiated(self):
        from orchestrator.openrouter_sync import OpenRouterSync

        sync = OpenRouterSync(api_key="test-key")
        assert sync is not None

    @pytest.mark.asyncio
    async def test_fetch_models_returns_list(self):
        from orchestrator.openrouter_sync import OpenRouterSync

        mock_data = {
            "data": [
                {
                    "id": "anthropic/claude-3-5-sonnet",
                    "name": "Claude 3.5 Sonnet",
                    "pricing": {"prompt": "0.000003", "completion": "0.000015"},
                    "context_length": 200000,
                    "description": "Latest Claude model",
                },
            ]
        }

        sync = OpenRouterSync(api_key="test-key")
        with patch("aiohttp.ClientSession") as mock_session_cls:
            mock_resp = AsyncMock()
            mock_resp.json = AsyncMock(return_value=mock_data)
            mock_resp.status = 200
            mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_resp.__aexit__ = AsyncMock(return_value=False)
            mock_session = MagicMock()
            mock_session.get = MagicMock(return_value=mock_resp)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)
            mock_session_cls.return_value = mock_session

            models = await sync.fetch_models()

        assert isinstance(models, list)
        assert len(models) >= 1
        assert models[0]["id"] == "anthropic/claude-3-5-sonnet"

    def test_to_cost_table_entry_converts_pricing(self):
        from orchestrator.openrouter_sync import OpenRouterSync

        sync = OpenRouterSync(api_key="test-key")
        entry = sync.to_cost_table_entry(
            {
                "id": "anthropic/claude-3-5-sonnet",
                "pricing": {"prompt": "0.000003", "completion": "0.000015"},
            }
        )
        # pricing is per-token; cost table is per million tokens
        assert entry["input"] == pytest.approx(3.0, rel=1e-3)
        assert entry["output"] == pytest.approx(15.0, rel=1e-3)

    def test_to_cost_table_entry_handles_missing_pricing(self):
        from orchestrator.openrouter_sync import OpenRouterSync

        sync = OpenRouterSync(api_key="test-key")
        entry = sync.to_cost_table_entry({"id": "some/model", "pricing": {}})
        assert "input" in entry
        assert "output" in entry
