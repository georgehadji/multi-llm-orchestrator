"""
Tests for the Decomposer module (extracted from engine.py Phase 2).
==========================================================================
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

pytestmark = pytest.mark.unit

from orchestrator.engine_core.decomposer import Decomposer


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.call = AsyncMock()
    return client


@pytest.fixture
def mock_selector():
    selector = MagicMock()
    selector.decomposition_model = MagicMock(return_value="deepseek/deepseek-chat")
    return selector


@pytest.fixture
def decomposer(mock_client, mock_selector):
    return Decomposer(client=mock_client, selector=mock_selector)


class TestParseDecomposition:
    """Test _parse_decomposition — JSON parsing."""

    VALID_TASKS_JSON = json.dumps(
        [
            {
                "id": "task_001",
                "type": "code_generation",
                "prompt": "Build a CLI",
                "dependencies": [],
            },
            {
                "id": "task_002",
                "type": "code_review",
                "prompt": "Review CLI code",
                "dependencies": ["task_001"],
            },
        ]
    )

    def test_parse_valid_json(self, decomposer):
        tasks = decomposer._parse_decomposition(self.VALID_TASKS_JSON)
        assert len(tasks) == 2
        assert "task_001" in tasks
        assert tasks["task_001"].type.value == "code_generation"

    def test_parse_empty_text(self, decomposer):
        tasks = decomposer._parse_decomposition("")
        assert tasks == {}

    def test_parse_with_fences(self, decomposer):
        fenced = f"```json\n{self.VALID_TASKS_JSON}\n```"
        tasks = decomposer._parse_decomposition(fenced)
        assert len(tasks) == 2

    def test_parse_nested_tasks_key(self, decomposer):
        nested = json.dumps({"tasks": json.loads(self.VALID_TASKS_JSON)})
        tasks = decomposer._parse_decomposition(nested)
        assert len(tasks) == 2

    def test_parse_partial_recovery(self, decomposer):
        """Test _try_parse_partial_json_array with truncated content."""
        partial = (
            '[{"id": "task_001", "type": "code_generation", "prompt": "test"}\n, {"id": "task_002"'
        )
        result = decomposer._try_parse_partial_json_array(partial)
        assert result is not None
        # Should at least recover the first object
        assert len(result) >= 1


class TestTryParsePartial:
    """Test _try_parse_partial_json_array — recovery strategies."""

    def test_closing_brackets(self, decomposer):
        partial = '[{"id": "test", "type": "code_generation", "prompt": "hello"}]'
        result = decomposer._try_parse_partial_json_array(partial)
        assert result is not None
        assert len(result) == 1

    def test_pattern_extraction(self, decomposer):
        text = 'Some text {"id": "t1", "type": "code_generation", "prompt": "p1"} more text'
        result = decomposer._try_parse_partial_json_array(text)
        assert result is not None
        assert len(result) >= 1

    def test_empty_text(self, decomposer):
        assert decomposer._try_parse_partial_json_array("") is None

    def test_not_json(self, decomposer):
        assert decomposer._try_parse_partial_json_array("just random text") is None


class TestGetDecompositionModels:
    """Test model selection for decomposition."""

    def test_returns_models(self, decomposer):
        models = decomposer._get_decomposition_models("Build a web app")
        assert len(models) > 0
        assert all(hasattr(m, "value") or isinstance(m, str) for m in models)

    def test_with_api_health(self, decomposer):
        models = decomposer._get_decomposition_models("Build a web app", api_health={})
        assert len(models) > 0


class TestDecompose:
    """Test the main decompose method."""

    @pytest.mark.asyncio
    async def test_decompose_success(self, decomposer, mock_client):
        mock_client.call.return_value = MagicMock(
            text=json.dumps(
                [
                    {"id": "t1", "type": "code_generation", "prompt": "Create models"},
                    {
                        "id": "t2",
                        "type": "code_generation",
                        "prompt": "Create views",
                        "dependencies": ["t1"],
                    },
                ]
            )
        )
        tasks = await decomposer.decompose("Build a web app", "Must work")
        assert len(tasks) == 2

    @pytest.mark.asyncio
    async def test_decompose_client_error(self, decomposer, mock_client):
        # Contract: decompose() never raises on client errors — it returns an
        # empty dict, which callers (e.g. project_runner) must map to
        # ProjectStatus.SYSTEM_FAILURE. Locked to prevent silent-failure drift.
        mock_client.call.side_effect = Exception("API error")
        tasks = await decomposer.decompose("Build a web app", "Must work")
        assert tasks == {}

    @pytest.mark.asyncio
    async def test_decompose_with_project_context(self, decomposer, mock_client):
        mock_client.call.return_value = MagicMock(
            text=json.dumps([{"id": "t1", "type": "code_generation", "prompt": "test"}])
        )
        from orchestrator.project_context import ProjectContext

        ctx = ProjectContext()
        tasks = await decomposer.decompose(
            "Build a web app",
            "Must work",
            project_context=ctx,
        )
        assert len(tasks) == 1
