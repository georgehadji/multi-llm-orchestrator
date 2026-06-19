"""Unit tests for the OpenRouter response-healing plugin injection.

Verifies _maybe_add_response_healing only adds the plugin for non-streaming
structured-output requests when the USE_RESPONSE_HEALING flag is enabled.
"""

from dataclasses import dataclass

import pytest

from orchestrator.infrastructure.llm_client import _maybe_add_response_healing


@dataclass
class _Opts:
    USE_RESPONSE_HEALING: bool = True


pytestmark = pytest.mark.unit


def test_added_for_structured_nonstreaming_request():
    params = {"model": "m", "response_format": {"type": "json_schema"}}
    _maybe_add_response_healing(params, _Opts(USE_RESPONSE_HEALING=True))
    assert params["plugins"] == [{"id": "response-healing"}]


def test_not_added_when_flag_disabled():
    params = {"model": "m", "response_format": {"type": "json_schema"}}
    _maybe_add_response_healing(params, _Opts(USE_RESPONSE_HEALING=False))
    assert "plugins" not in params


def test_not_added_without_response_format():
    params = {"model": "m"}
    _maybe_add_response_healing(params, _Opts(USE_RESPONSE_HEALING=True))
    assert "plugins" not in params


def test_not_added_for_streaming():
    params = {"model": "m", "response_format": {"type": "json_schema"}, "stream": True}
    _maybe_add_response_healing(params, _Opts(USE_RESPONSE_HEALING=True))
    assert "plugins" not in params


def test_safe_when_opts_none():
    params = {"model": "m", "response_format": {"type": "json_schema"}}
    _maybe_add_response_healing(params, None)
    assert "plugins" not in params


def test_idempotent_and_preserves_existing_plugins():
    params = {
        "model": "m",
        "response_format": {"type": "json_schema"},
        "plugins": [{"id": "web"}],
    }
    _maybe_add_response_healing(params, _Opts(USE_RESPONSE_HEALING=True))
    _maybe_add_response_healing(params, _Opts(USE_RESPONSE_HEALING=True))
    assert params["plugins"] == [{"id": "web"}, {"id": "response-healing"}]
