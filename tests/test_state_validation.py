"""Unit tests for StateManager._deserialize_state (corrupt state handling)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.state import StateManager


# ─────────────────────────────────────────────────────────────────────────────
# _deserialize_state — pure unit tests (no DB)
# ─────────────────────────────────────────────────────────────────────────────


def _make_manager() -> StateManager:
    """Create a StateManager without opening a DB connection."""
    return StateManager.__new__(StateManager)


def test_deserialize_state_invalid_json_returns_none(caplog):
    mgr = _make_manager()
    result = mgr._deserialize_state("NOT_VALID_JSON{{{", context="test")
    assert result is None
    assert "Corrupt JSON" in caplog.text


def test_deserialize_state_empty_blob_returns_none(caplog):
    mgr = _make_manager()
    result = mgr._deserialize_state("", context="test")
    assert result is None
    assert "Corrupt JSON" in caplog.text


def test_deserialize_state_missing_keys_returns_none(caplog):
    mgr = _make_manager()
    # Valid JSON but wrong schema (missing required ProjectState fields)
    blob = json.dumps({"foo": "bar"})
    result = mgr._deserialize_state(blob, context="test")
    assert result is None
    assert "Schema mismatch" in caplog.text


def test_deserialize_state_null_returns_none(caplog):
    mgr = _make_manager()
    result = mgr._deserialize_state("null", context="test")
    assert result is None


def test_deserialize_state_context_included_in_log(caplog):
    mgr = _make_manager()
    mgr._deserialize_state("bad json", context="project=abc-123")
    assert "project=abc-123" in caplog.text
