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
