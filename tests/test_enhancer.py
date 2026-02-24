"""Tests for Enhancement dataclass and pure utility functions."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from orchestrator.enhancer import (
    Enhancement,
    _select_enhance_model,
    _parse_enhancements,
    _apply_enhancements,
    ProjectEnhancer,
)
from orchestrator.models import Model
from orchestrator.api_clients import APIResponse
from orchestrator.cli import _async_new_project


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


# ─────────────────────────────────────────────
# Tests for ProjectEnhancer class
# ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_project_enhancer_valid_llm_response():
    """Valid LLM response should return list of Enhancement objects."""
    # Create valid JSON response
    json_response = json.dumps([
        {
            "type": "completeness",
            "title": "Add Performance Metrics",
            "description": "The project lacks specific performance requirements.",
            "patch_description": "with response time < 100ms",
            "patch_criteria": "achieve response time < 100ms for all endpoints"
        },
        {
            "type": "criteria",
            "title": "Add Error Handling",
            "description": "Need comprehensive error handling strategy.",
            "patch_description": "with comprehensive error handling",
            "patch_criteria": "handles all error cases gracefully"
        }
    ])

    # Mock UnifiedClient.call()
    mock_client = AsyncMock()
    mock_response = APIResponse(
        text=json_response,
        input_tokens=100,
        output_tokens=150,
        model=Model.DEEPSEEK_CHAT,
    )
    mock_client.call.return_value = mock_response

    # Create enhancer with mocked client
    enhancer = ProjectEnhancer(client=mock_client)

    # Call analyze
    result = await enhancer.analyze("Build a web app", "Handle 100 requests/sec")

    # Verify result
    assert len(result) == 2
    assert result[0].type == "completeness"
    assert result[0].title == "Add Performance Metrics"
    assert result[1].type == "criteria"


@pytest.mark.asyncio
async def test_project_enhancer_invalid_json_response():
    """Invalid JSON from LLM should return empty list gracefully."""
    # Create invalid JSON response
    invalid_json = "not valid json {]"

    # Mock UnifiedClient.call()
    mock_client = AsyncMock()
    mock_response = APIResponse(
        text=invalid_json,
        input_tokens=100,
        output_tokens=50,
        model=Model.DEEPSEEK_CHAT,
    )
    mock_client.call.return_value = mock_response

    # Create enhancer with mocked client
    enhancer = ProjectEnhancer(client=mock_client)

    # Call analyze
    result = await enhancer.analyze("Build a web app", "Handle 100 requests/sec")

    # Should gracefully return empty list
    assert result == []


@pytest.mark.asyncio
async def test_project_enhancer_llm_exception():
    """LLM exception should return empty list gracefully."""
    # Mock UnifiedClient.call() to raise exception
    mock_client = AsyncMock()
    mock_client.call.side_effect = Exception("LLM call failed")

    # Create enhancer with mocked client
    enhancer = ProjectEnhancer(client=mock_client)

    # Call analyze
    result = await enhancer.analyze("Build a web app", "Handle 100 requests/sec")

    # Should gracefully return empty list
    assert result == []


@pytest.mark.asyncio
async def test_project_enhancer_model_selection_long_description():
    """Long combined description should use DEEPSEEK_REASONER model."""
    # Create long description and criteria (combined > 50 words)
    long_description = " ".join(["word"] * 35)  # 35 words
    long_criteria = " ".join(["word"] * 20)      # 20 words
    # Combined = 55 words > 50, should use DEEPSEEK_REASONER

    json_response = json.dumps([
        {
            "type": "completeness",
            "title": "Test",
            "description": "Test enhancement",
            "patch_description": "test patch",
            "patch_criteria": "test criteria"
        }
    ])

    # Mock UnifiedClient.call()
    mock_client = AsyncMock()
    mock_response = APIResponse(
        text=json_response,
        input_tokens=100,
        output_tokens=100,
        model=Model.DEEPSEEK_REASONER,
    )
    mock_client.call.return_value = mock_response

    # Create enhancer with mocked client
    enhancer = ProjectEnhancer(client=mock_client)

    # Call analyze
    await enhancer.analyze(long_description, long_criteria)

    # Verify DEEPSEEK_REASONER was used
    call_args = mock_client.call.call_args
    assert call_args.kwargs["model"] == Model.DEEPSEEK_REASONER


@pytest.mark.asyncio
async def test_project_enhancer_model_selection_short_description():
    """Short combined description should use DEEPSEEK_CHAT model."""
    # Create short description and criteria (combined ≤ 50 words)
    short_description = "Build a web app"  # ~3 words
    short_criteria = "Must be fast"       # ~3 words
    # Combined = 6 words ≤ 50, should use DEEPSEEK_CHAT

    json_response = json.dumps([
        {
            "type": "completeness",
            "title": "Test",
            "description": "Test enhancement",
            "patch_description": "test patch",
            "patch_criteria": "test criteria"
        }
    ])

    # Mock UnifiedClient.call()
    mock_client = AsyncMock()
    mock_response = APIResponse(
        text=json_response,
        input_tokens=100,
        output_tokens=100,
        model=Model.DEEPSEEK_CHAT,
    )
    mock_client.call.return_value = mock_response

    # Create enhancer with mocked client
    enhancer = ProjectEnhancer(client=mock_client)

    # Call analyze
    await enhancer.analyze(short_description, short_criteria)

    # Verify DEEPSEEK_CHAT was used
    call_args = mock_client.call.call_args
    assert call_args.kwargs["model"] == Model.DEEPSEEK_CHAT


# ─── _present_enhancements ────────────────────────────────────────────────────

from orchestrator.enhancer import _present_enhancements


def _make_three_enhancements() -> list[Enhancement]:
    return [
        Enhancement("completeness", "Missing: refresh tokens",
                    "JWT auth needs refresh tokens.", "with refresh tokens (7d)", ""),
        Enhancement("criteria", "Vague success criteria",
                    "Tests pass is unmeasurable.", "", "≥80% test coverage"),
        Enhancement("risk", "Missing: password hashing",
                    "Plain-text passwords are insecure.", "with bcrypt (cost 12)", ""),
    ]


def test_present_user_accepts_all(monkeypatch):
    """All 'y' responses → all enhancements returned."""
    monkeypatch.setattr("builtins.input", lambda _: "y")
    accepted = _present_enhancements(_make_three_enhancements())
    assert len(accepted) == 3


def test_present_user_rejects_all(monkeypatch):
    """All 'n' responses → empty list returned."""
    monkeypatch.setattr("builtins.input", lambda _: "n")
    accepted = _present_enhancements(_make_three_enhancements())
    assert accepted == []


def test_present_user_mixed(monkeypatch):
    """Mixed responses → only 'y' ones returned."""
    responses = iter(["y", "n", "y"])
    monkeypatch.setattr("builtins.input", lambda _: next(responses))
    accepted = _present_enhancements(_make_three_enhancements())
    assert len(accepted) == 2
    assert accepted[0].title == "Missing: refresh tokens"
    assert accepted[1].title == "Missing: password hashing"


def test_present_empty_enhancements():
    """Empty list → prints completion message, returns empty list."""
    result = _present_enhancements([])
    assert result == []


def test_present_default_is_yes(monkeypatch):
    """Empty Enter (no input) → treated as 'y' (accept)."""
    monkeypatch.setattr("builtins.input", lambda _: "")
    accepted = _present_enhancements(_make_three_enhancements())
    assert len(accepted) == 3


def test_present_ctrl_c_treated_as_no(monkeypatch):
    """KeyboardInterrupt on any prompt → reject all remaining, return what was accepted so far."""
    call_count = 0
    def mock_input(_):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise KeyboardInterrupt
        return "y"
    monkeypatch.setattr("builtins.input", mock_input)
    accepted = _present_enhancements(_make_three_enhancements())
    assert len(accepted) == 1  # first was accepted, then Ctrl-C on second


# ─── CLI integration ──────────────────────────────────────────────────────────

import types


def test_cli_no_enhance_flag():
    """--no-enhance flag causes ProjectEnhancer to be skipped entirely."""
    import asyncio

    with patch("orchestrator.enhancer.ProjectEnhancer.analyze", new_callable=AsyncMock) as mock_analyze:
        args = types.SimpleNamespace(
            project="Build a FastAPI auth service",
            criteria="tests pass",
            budget=8.0,
            time=5400,
            project_id="",
            output_dir="",
            concurrency=3,
            verbose=False,
            raw_tasks=False,
            no_enhance=True,   # ← the flag under test
            tracing=False,
            otlp_endpoint=None,
            dependency_report=False,
            new_project=True,  # skip resume detection too
        )

        with patch("orchestrator.app_builder.AppBuilder.build", new_callable=AsyncMock) as mock_build:
            mock_build.return_value = MagicMock(success=True, output_dir="/tmp/test", errors=[])
            asyncio.run(_async_new_project(args))

        # ProjectEnhancer.analyze must NOT have been called
        mock_analyze.assert_not_called()
