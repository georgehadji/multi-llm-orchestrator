"""Unit tests for the OpenRouter response-healing plugin injection.

Verifies _maybe_add_response_healing only adds the plugin for non-streaming
structured-output requests when the USE_RESPONSE_HEALING flag is enabled, and
that it is routed through ``extra_body`` (the OpenAI SDK does not accept a
top-level ``plugins`` kwarg).
"""

from dataclasses import dataclass

import pytest

from orchestrator.infrastructure.llm_client import _maybe_add_response_healing


@dataclass
class _Opts:
    USE_RESPONSE_HEALING: bool = True


pytestmark = pytest.mark.unit


def test_added_via_extra_body_for_structured_nonstreaming_request():
    params = {"model": "m", "response_format": {"type": "json_schema"}}
    _maybe_add_response_healing(params, _Opts(USE_RESPONSE_HEALING=True))
    # Routed through extra_body, NOT as a top-level kwarg (would TypeError in the SDK).
    assert "plugins" not in params
    assert params["extra_body"]["plugins"] == [{"id": "response-healing"}]


def test_not_added_when_flag_disabled():
    params = {"model": "m", "response_format": {"type": "json_schema"}}
    _maybe_add_response_healing(params, _Opts(USE_RESPONSE_HEALING=False))
    assert "extra_body" not in params


def test_not_added_without_response_format():
    params = {"model": "m"}
    _maybe_add_response_healing(params, _Opts(USE_RESPONSE_HEALING=True))
    assert "extra_body" not in params


def test_not_added_for_streaming():
    params = {"model": "m", "response_format": {"type": "json_schema"}, "stream": True}
    _maybe_add_response_healing(params, _Opts(USE_RESPONSE_HEALING=True))
    assert "extra_body" not in params


def test_safe_when_opts_none():
    params = {"model": "m", "response_format": {"type": "json_schema"}}
    _maybe_add_response_healing(params, None)
    assert "extra_body" not in params


def test_idempotent_and_preserves_existing_plugins():
    params = {
        "model": "m",
        "response_format": {"type": "json_schema"},
        "extra_body": {"plugins": [{"id": "web"}]},
    }
    _maybe_add_response_healing(params, _Opts(USE_RESPONSE_HEALING=True))
    _maybe_add_response_healing(params, _Opts(USE_RESPONSE_HEALING=True))
    assert params["extra_body"]["plugins"] == [{"id": "web"}, {"id": "response-healing"}]
