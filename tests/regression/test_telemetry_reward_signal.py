"""
Regression test: TelemetryCollector.record_call must propagate REAL evaluator
output into ModelProfile.quality_score, not a hardcoded/stub value.

Closes ACR WI-0 risk R-1 ("reward signal might be a stub") as an enforced
regression gate, not just a point-in-time observation.
"""

import pytest

from orchestrator.models import Model
from orchestrator.policy import ModelProfile
from orchestrator.telemetry import TelemetryCollector


@pytest.mark.unit
def test_record_call_moves_quality_score_toward_supplied_evaluator_score():
    # Arrange
    model = next(iter(Model))
    profile = ModelProfile(
        model=model,
        provider="test",
        cost_per_1m_input=1.0,
        cost_per_1m_output=1.0,
        quality_score=0.5,
    )
    collector = TelemetryCollector({model: profile})

    # Act — simulate what evaluator.py does: pass a distinct, non-default
    # evaluator score through the real call path (no monkeypatching quality_score).
    collector.record_call(
        model=model, latency_ms=1200.0, cost_usd=0.01, success=True, quality_score=0.95
    )

    # Assert — EMA moved strictly toward 0.95, away from the 0.5 default.
    assert profile.quality_score > 0.5
    assert profile.quality_score != 0.5


@pytest.mark.unit
def test_record_call_leaves_quality_score_unchanged_when_no_score_supplied():
    """Guards against a future regression where a stub always overwrites
    quality_score regardless of caller intent."""
    # Arrange
    model = next(iter(Model))
    profile = ModelProfile(
        model=model,
        provider="test",
        cost_per_1m_input=1.0,
        cost_per_1m_output=1.0,
        quality_score=0.5,
    )
    collector = TelemetryCollector({model: profile})

    # Act
    collector.record_call(
        model=model, latency_ms=1200.0, cost_usd=0.01, success=True, quality_score=None
    )

    # Assert — untouched, proving the update is conditional on real input
    assert profile.quality_score == 0.5
