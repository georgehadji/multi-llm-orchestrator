"""
Tests for AppDetector and AppProfile (Task 2).
All LLM calls are mocked — no real API calls.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from orchestrator.app_detector import AppDetector, AppProfile


# ─────────────────────────────────────────────────────────────────────────────
# AppProfile — dataclass structure
# ─────────────────────────────────────────────────────────────────────────────

def test_app_profile_defaults():
    """AppProfile must have sensible defaults for all fields."""
    p = AppProfile(app_type="fastapi")
    assert p.app_type == "fastapi"
    assert isinstance(p.tech_stack, list)
    assert isinstance(p.entry_point, str)
    assert isinstance(p.test_command, str)
    assert isinstance(p.run_command, str)
    assert isinstance(p.requires_docker, bool)
    assert isinstance(p.detected_from, str)


def test_app_profile_fields_set():
    """All fields can be explicitly set."""
    p = AppProfile(
        app_type="cli",
        tech_stack=["python", "click"],
        entry_point="cli.py",
        test_command="pytest",
        run_command="python cli.py",
        requires_docker=False,
        detected_from="yaml_override",
    )
    assert p.tech_stack == ["python", "click"]
    assert p.entry_point == "cli.py"
    assert p.detected_from == "yaml_override"


# ─────────────────────────────────────────────────────────────────────────────
# AppDetector.detect_from_yaml — YAML override (no LLM call)
# ─────────────────────────────────────────────────────────────────────────────

def test_detect_from_yaml_sets_detected_from():
    detector = AppDetector()
    profile = detector.detect_from_yaml("fastapi")
    assert profile.detected_from == "yaml_override"
    assert profile.app_type == "fastapi"


def test_detect_from_yaml_cli():
    detector = AppDetector()
    profile = detector.detect_from_yaml("cli")
    assert profile.app_type == "cli"
    assert "cli.py" in profile.entry_point or profile.entry_point != ""


def test_detect_from_yaml_unknown_falls_back_to_script():
    detector = AppDetector()
    profile = detector.detect_from_yaml("unknown_type_xyz")
    assert profile.app_type == "script"
    assert profile.detected_from == "yaml_override"


# ─────────────────────────────────────────────────────────────────────────────
# AppDetector.detect — async LLM-based detection
# ─────────────────────────────────────────────────────────────────────────────

def test_detect_uses_llm_response():
    """detect() must parse a valid LLM JSON response into AppProfile."""
    import asyncio

    llm_json = json.dumps({
        "app_type": "fastapi",
        "tech_stack": ["python", "fastapi", "sqlalchemy"],
        "entry_point": "src/main.py",
        "test_command": "pytest",
        "run_command": "uvicorn src.main:app",
        "requires_docker": False,
    })

    detector = AppDetector()
    with patch.object(detector, "_call_llm", new=AsyncMock(return_value=llm_json)):
        profile = asyncio.run(detector.detect("Build a FastAPI REST API"))

    assert profile.app_type == "fastapi"
    assert "fastapi" in profile.tech_stack
    assert profile.detected_from == "auto"


def test_detect_falls_back_on_llm_error():
    """If the LLM call raises, detect() must return a 'script' fallback profile."""
    import asyncio

    detector = AppDetector()
    with patch.object(detector, "_call_llm", new=AsyncMock(side_effect=RuntimeError("LLM failed"))):
        profile = asyncio.run(detector.detect("Build something"))

    assert profile.app_type == "script"
    assert profile.detected_from == "auto"


def test_detect_falls_back_on_invalid_json():
    """If the LLM returns non-JSON, detect() must return a 'script' fallback."""
    import asyncio

    detector = AppDetector()
    with patch.object(detector, "_call_llm", new=AsyncMock(return_value="not json at all")):
        profile = asyncio.run(detector.detect("Build something"))

    assert profile.app_type == "script"


def test_detect_from_yaml_overrides_description():
    """When app_type is provided via YAML, no LLM call should be made."""
    import asyncio

    detector = AppDetector()
    with patch.object(detector, "_call_llm", new=AsyncMock()) as mock_llm:
        profile = asyncio.run(detector.detect("Build a FastAPI app", app_type_override="cli"))

    mock_llm.assert_not_called()
    assert profile.app_type == "cli"
    assert profile.detected_from == "yaml_override"
