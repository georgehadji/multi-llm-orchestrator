"""Tests for Content Preflight Gate integration."""
import pytest
from orchestrator.hooks import EventType


def test_preflight_check_event_exists():
    assert hasattr(EventType, "PREFLIGHT_CHECK")
    assert EventType.PREFLIGHT_CHECK.value == "preflight_check"
