"""Tests for ArchitectureDecision dataclass and pure utility functions."""
from __future__ import annotations

import json
import pytest
from orchestrator.architecture_advisor import (
    ArchitectureDecision,
    _select_model,
    _parse_response,
    _ARCH_DEFAULTS,
)
from orchestrator.models import Model


# ─── _select_model ────────────────────────────────────────────────────────────

def test_select_model_short():
    """≤50 words → DeepSeek Chat."""
    desc = "Build a FastAPI auth service with JWT tokens"
    assert _select_model(desc) == Model.DEEPSEEK_CHAT


def test_select_model_long():
    """>50 words → DeepSeek Reasoner."""
    desc = " ".join(["word"] * 51)
    assert _select_model(desc) == Model.DEEPSEEK_REASONER


def test_select_model_exactly_50():
    """Exactly 50 words → DeepSeek Chat (boundary: >50 triggers Reasoner)."""
    desc = " ".join(["word"] * 50)
    assert _select_model(desc) == Model.DEEPSEEK_CHAT


# ─── _parse_response ──────────────────────────────────────────────────────────

FULL_VALID_JSON = json.dumps({
    "app_type": "fastapi",
    "tech_stack": ["python", "fastapi", "postgresql"],
    "entry_point": "src/main.py",
    "test_command": "pytest",
    "run_command": "uvicorn src.main:app --reload",
    "requires_docker": False,
    "structural_pattern": "layered",
    "topology": "monolith",
    "data_paradigm": "relational",
    "api_paradigm": "rest",
    "rationale": "FastAPI is well-suited for REST APIs. Layered architecture keeps things clean.",
})


def test_parse_valid_response():
    """Valid JSON → correct ArchitectureDecision fields."""
    result = _parse_response(FULL_VALID_JSON)
    assert isinstance(result, ArchitectureDecision)
    assert result.app_type == "fastapi"
    assert result.tech_stack == ["python", "fastapi", "postgresql"]
    assert result.structural_pattern == "layered"
    assert result.topology == "monolith"
    assert result.data_paradigm == "relational"
    assert result.api_paradigm == "rest"
    assert "REST" in result.rationale or "rest" in result.rationale.lower()
    assert result.detected_from == "advisor"


def test_parse_invalid_json():
    """Invalid JSON → fallback defaults, no exception."""
    result = _parse_response("this is not json {{{{")
    assert isinstance(result, ArchitectureDecision)
    assert result.app_type == "script"   # fallback default
    assert result.detected_from == "advisor"


def test_parse_missing_arch_fields():
    """JSON with only AppProfile fields → arch fields get sensible defaults."""
    minimal = json.dumps({
        "app_type": "fastapi",
        "tech_stack": ["python", "fastapi"],
        "entry_point": "src/main.py",
        "test_command": "pytest",
        "run_command": "uvicorn src.main:app",
        "requires_docker": False,
    })
    result = _parse_response(minimal)
    assert result.app_type == "fastapi"
    # arch fields should default to known fastapi defaults
    assert result.structural_pattern == _ARCH_DEFAULTS["fastapi"]["structural_pattern"]
    assert result.topology == _ARCH_DEFAULTS["fastapi"]["topology"]


def test_parse_json_in_markdown_fences():
    """LLM sometimes wraps JSON in markdown fences — strip and parse correctly."""
    fenced = f"```json\n{FULL_VALID_JSON}\n```"
    result = _parse_response(fenced)
    assert result.app_type == "fastapi"
    assert result.structural_pattern == "layered"


def test_parse_empty_string():
    """Empty string → fallback defaults, no exception."""
    result = _parse_response("")
    assert isinstance(result, ArchitectureDecision)
    assert result.app_type == "script"

def test_parse_unknown_app_type():
    """Unknown app_type value → normalised to 'generic', no exception."""
    import json
    result = _parse_response(json.dumps({"app_type": "blockchain-dao"}))
    assert result.app_type == "generic"
    assert result.detected_from == "advisor"

# ─── ArchitectureAdvisor ──────────────────────────────────────────────────────

from unittest.mock import AsyncMock, MagicMock
from orchestrator.architecture_advisor import ArchitectureAdvisor


class FakeAPIResponse:
    def __init__(self, text: str, cost_usd: float = 0.002):
        self.text = text
        self.cost_usd = cost_usd


@pytest.mark.asyncio
async def test_analyze_calls_llm_once():
    """analyze() makes exactly one LLM call."""
    mock_client = MagicMock()
    mock_client.call = AsyncMock(return_value=FakeAPIResponse(text=FULL_VALID_JSON))

    advisor = ArchitectureAdvisor(client=mock_client)
    result = await advisor.analyze("Build a FastAPI service", "tests pass")

    assert mock_client.call.call_count == 1
    assert isinstance(result, ArchitectureDecision)
    assert result.app_type == "fastapi"


@pytest.mark.asyncio
async def test_analyze_handles_exception():
    """LLM call raises -> returns fallback defaults, no crash."""
    mock_client = MagicMock()
    mock_client.call = AsyncMock(side_effect=Exception("network error"))

    advisor = ArchitectureAdvisor(client=mock_client)
    result = await advisor.analyze("Build something", "tests pass")

    assert isinstance(result, ArchitectureDecision)
    assert result.app_type == "script"


@pytest.mark.asyncio
async def test_analyze_prints_summary(capsys):
    """Terminal output contains pattern, topology, api, storage."""
    mock_client = MagicMock()
    mock_client.call = AsyncMock(return_value=FakeAPIResponse(text=FULL_VALID_JSON))

    advisor = ArchitectureAdvisor(client=mock_client)
    await advisor.analyze("Build a FastAPI service", "tests pass")

    captured = capsys.readouterr()
    assert "layered" in captured.out.lower() or "Layered" in captured.out
    assert "monolith" in captured.out.lower() or "Monolith" in captured.out
    assert "rest" in captured.out.lower() or "REST" in captured.out


@pytest.mark.asyncio
async def test_analyze_selects_reasoner_for_long_desc():
    """Descriptions >50 words use DeepSeek Reasoner."""
    long_desc = " ".join(["word"] * 51)
    mock_client = MagicMock()
    mock_client.call = AsyncMock(return_value=FakeAPIResponse(text=FULL_VALID_JSON))

    advisor = ArchitectureAdvisor(client=mock_client)
    await advisor.analyze(long_desc, "criteria")

    call_args = mock_client.call.call_args
    used_model = call_args[1].get("model") or call_args[0][0]
    assert used_model == Model.DEEPSEEK_REASONER


@pytest.mark.asyncio
async def test_analyze_selects_chat_for_short_desc():
    """Descriptions <=50 words use DeepSeek Chat."""
    mock_client = MagicMock()
    mock_client.call = AsyncMock(return_value=FakeAPIResponse(text=FULL_VALID_JSON))

    advisor = ArchitectureAdvisor(client=mock_client)
    await advisor.analyze("Build a FastAPI auth service", "tests pass")

    call_args = mock_client.call.call_args
    used_model = call_args[1].get("model") or call_args[0][0]
    assert used_model == Model.DEEPSEEK_CHAT


def test_detect_from_yaml_skips_llm():
    """detect_from_yaml() returns known defaults without LLM call."""
    advisor = ArchitectureAdvisor(client=None)
    result = advisor.detect_from_yaml("fastapi")

    assert result.app_type == "fastapi"
    assert result.structural_pattern == "layered"
    assert result.topology == "monolith"
    assert result.api_paradigm == "rest"
    assert result.detected_from == "yaml_override"


def test_detect_from_yaml_unknown_type():
    """Unknown app_type_override falls back to script defaults."""
    advisor = ArchitectureAdvisor(client=None)
    result = advisor.detect_from_yaml("unknown-framework")

    assert result.app_type == "script"
    assert result.detected_from == "yaml_override"


# ─── backward compatibility ───────────────────────────────────────────────────

def test_app_profile_alias():
    """AppProfile is a type alias for ArchitectureDecision — same object."""
    from orchestrator.app_detector import AppProfile
    assert AppProfile is ArchitectureDecision


# ─── AppBuilder integration ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_app_builder_uses_architecture_advisor():
    """AppBuilder.__init__() creates an ArchitectureAdvisor, not an AppDetector."""
    from orchestrator.app_builder import AppBuilder

    builder = AppBuilder()

    # Must have _advisor attribute
    assert hasattr(builder, "_advisor"), "AppBuilder missing _advisor attribute"
    # _advisor must be an ArchitectureAdvisor instance
    assert isinstance(builder._advisor, ArchitectureAdvisor)
