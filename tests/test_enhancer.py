"""Tests for Enhancement dataclass and pure utility functions."""

import json
import pytest

from orchestrator.enhancer import (
    Enhancement,
    _select_enhance_model,
    _parse_enhancements,
    _apply_enhancements,
)
from orchestrator.models import Model


# ─────────────────────────────────────────────
# Tests for Enhancement dataclass
# ─────────────────────────────────────────────

def test_enhancement_is_immutable():
    """Enhancement should be frozen (immutable)."""
    enh = Enhancement(
        type="completeness",
        title="Add error handling",
        description="The project needs comprehensive error handling.",
        patch_description="Include error handling for all edge cases.",
        patch_criteria="Error handling in place for all critical paths.",
    )
    with pytest.raises(AttributeError):
        enh.type = "criteria"  # type: ignore


def test_enhancement_creation():
    """Enhancement should create valid instances."""
    enh = Enhancement(
        type="criteria",
        title="Performance threshold",
        description="Add performance metrics.",
        patch_description="Response time must be < 200ms.",
        patch_criteria="Endpoints respond in under 200ms.",
    )
    assert enh.type == "criteria"
    assert enh.title == "Performance threshold"
    assert "Response time" in enh.patch_description


# ─────────────────────────────────────────────
# Tests for _select_enhance_model()
# ─────────────────────────────────────────────

def test_select_enhance_model_empty_string():
    """Empty description should return DEEPSEEK_CHAT."""
    result = _select_enhance_model("")
    assert result == Model.DEEPSEEK_CHAT


def test_select_enhance_model_short_description():
    """Description with ≤50 words should return DEEPSEEK_CHAT."""
    description = "Build a simple app." * 2  # ~8 words total
    result = _select_enhance_model(description)
    assert result == Model.DEEPSEEK_CHAT


def test_select_enhance_model_long_description():
    """Description with >50 words should return DEEPSEEK_REASONER."""
    # Create a description with >50 words
    description = " ".join(["word"] * 60)
    result = _select_enhance_model(description)
    assert result == Model.DEEPSEEK_REASONER


# ─────────────────────────────────────────────
# Tests for _parse_enhancements()
# ─────────────────────────────────────────────

def test_parse_enhancements_valid_json():
    """Valid JSON with 2 enhancements should parse correctly."""
    json_str = json.dumps([
        {
            "type": "completeness",
            "title": "Add logging",
            "description": "Ensure logging is in place.",
            "patch_description": "Add logging to all functions.",
            "patch_criteria": "All functions log their inputs.",
        },
        {
            "type": "criteria",
            "title": "Test coverage",
            "description": "Increase test coverage.",
            "patch_description": "Achieve 80% test coverage.",
            "patch_criteria": "Test coverage must be ≥80%.",
        },
    ])
    result = _parse_enhancements(json_str)
    assert len(result) == 2
    assert result[0].type == "completeness"
    assert result[1].type == "criteria"


def test_parse_enhancements_invalid_json():
    """Invalid JSON should return empty list."""
    json_str = "not valid json {]"
    result = _parse_enhancements(json_str)
    assert result == []


def test_parse_enhancements_missing_field():
    """Missing required field should return empty list."""
    json_str = json.dumps([
        {
            "type": "completeness",
            "title": "Add logging",
            # missing "description"
            "patch_description": "Add logging to all functions.",
            "patch_criteria": "All functions log their inputs.",
        }
    ])
    result = _parse_enhancements(json_str)
    assert result == []


def test_parse_enhancements_invalid_type():
    """Invalid type value should return empty list."""
    json_str = json.dumps([
        {
            "type": "invalid_type",  # Not in completeness|criteria|risk
            "title": "Add logging",
            "description": "Ensure logging is in place.",
            "patch_description": "Add logging to all functions.",
            "patch_criteria": "All functions log their inputs.",
        }
    ])
    result = _parse_enhancements(json_str)
    assert result == []


# ─────────────────────────────────────────────
# Tests for _apply_enhancements()
# ─────────────────────────────────────────────

def test_apply_enhancements_empty_list():
    """Empty enhancements list should return unchanged description and criteria."""
    description = "Build a CLI app"
    criteria = "Must be fast"
    result = _apply_enhancements(description, criteria, [])
    assert result == (description, criteria)


def test_apply_enhancements_single():
    """Single enhancement should be appended correctly."""
    description = "Build a web app"
    criteria = "Must handle 100 requests/sec"
    enhancement = Enhancement(
        type="completeness",
        title="Add auth",
        description="Add authentication",
        patch_description="with JWT authentication",
        patch_criteria="supports JWT authentication",
    )
    new_desc, new_crit = _apply_enhancements(description, criteria, [enhancement])
    assert new_desc == "Build a web app with JWT authentication"
    assert new_crit == "Must handle 100 requests/sec; supports JWT authentication"


def test_apply_enhancements_multiple():
    """Multiple enhancements should all be appended in order."""
    description = "Build a CLI"
    criteria = "Fast"
    enhancements = [
        Enhancement(
            type="completeness",
            title="Logging",
            description="Add logging",
            patch_description="with comprehensive logging",
            patch_criteria="includes logging",
        ),
        Enhancement(
            type="criteria",
            title="Error handling",
            description="Handle errors",
            patch_description="and proper error handling",
            patch_criteria="and handles all errors gracefully",
        ),
    ]
    new_desc, new_crit = _apply_enhancements(description, criteria, enhancements)
    assert new_desc == "Build a CLI with comprehensive logging and proper error handling"
    assert new_crit == "Fast; includes logging; and handles all errors gracefully"


def test_apply_enhancements_empty_string_inputs():
    """Should handle empty string inputs gracefully."""
    enhancement = Enhancement(
        type="completeness",
        title="Add feature",
        description="Add something",
        patch_description="with new feature",
        patch_criteria="has new feature",
    )
    new_desc, new_crit = _apply_enhancements("", "", [enhancement])
    assert new_desc == " with new feature"
    assert new_crit == "; has new feature"


def test_apply_enhancements_already_patched():
    """Should append correctly to already-patched strings."""
    description = "Build app with auth"
    criteria = "Fast; supports auth"
    enhancement = Enhancement(
        type="completeness",
        title="Add caching",
        description="Add caching",
        patch_description="and caching",
        patch_criteria="and supports caching",
    )
    new_desc, new_crit = _apply_enhancements(description, criteria, [enhancement])
    assert new_desc == "Build app with auth and caching"
    assert new_crit == "Fast; supports auth; and supports caching"
