"""
Tests for cost_tracker.py — Per-call cost visibility.
"""

from __future__ import annotations


import pytest

from orchestrator.cost_tracker import CostTracker, CallCost


class TestCallCost:
    """Tests for CallCost dataclass."""

    def test_defaults(self):
        """CallCost must have sensible defaults."""
        cc = CallCost(model="gpt-4o")
        assert cc.model == "gpt-4o"
        assert cc.input_tokens == 0
        assert cc.output_tokens == 0
        assert cc.cost_usd == 0.0

    def test_to_dict(self):
        """CallCost must serialize to dict."""
        cc = CallCost(
            model="gpt-4o", input_tokens=100, output_tokens=50, cost_usd=0.005, latency_ms=245.0
        )
        d = cc.to_dict()
        assert d["model"] == "gpt-4o"
        assert d["input_tokens"] == 100
        assert d["cost_usd"] == 0.005


class TestCostTracker:
    """Tests for CostTracker."""

    @pytest.fixture
    def tracker(self):
        """Create isolated CostTracker."""
        import tempfile as _tf
        from pathlib import Path as _Path

        d = str(_Path(_tf.mkdtemp()))
        return CostTracker(storage_dir=d)

    def test_record_single_call(self, tracker):
        """Recording a single call must update totals."""
        tracker.record("gpt-4o", 100, 50, 0.005, 245.0)
        assert tracker.total_cost_usd == 0.005
        assert tracker.total_tokens == 150

    def test_record_multiple_models(self, tracker):
        """Multiple models must be tracked independently."""
        tracker.record("gpt-4o", 100, 50, 0.005)
        tracker.record("claude", 80, 40, 0.003)
        tracker.record("gpt-4o", 200, 100, 0.010)
        assert tracker.total_cost_usd == 0.018
        models = tracker.per_model()
        assert len(models) >= 2

    def test_last_call(self, tracker):
        """last_call must return most recent CallCost."""
        tracker.record("gpt-4o", 50, 25, 0.002)
        tracker.record("claude", 30, 15, 0.001)
        last = tracker.last_call()
        assert last is not None
        assert last.model == "claude"

    def test_status_line(self, tracker):
        """status_line must include model and cost."""
        tracker.record("gpt-4o", 100, 50, 0.005, 245.0)
        status = tracker.status_line("gpt-4o")
        assert "gpt-4o" in status
        assert "100+50" in status

    def test_empty_tracker(self, tracker):
        """Empty tracker must return zero values."""
        assert tracker.total_cost_usd == 0.0
        assert tracker.total_tokens == 0
        assert tracker.last_call() is None
        assert tracker.status_line() == "Total: $0.0000 | 0 tokens | 0 models"

    # ── Edge Cases ──
    @pytest.mark.parametrize(
        "model",
        [
            "",
            "a" * 100,
            "openai/gpt-4o-mini",
        ],
    )
    def test_various_model_names(self, tracker, model):
        """Must accept various model name formats."""
        tracker.record(model, 1, 1, 0.001)
        assert tracker.total_cost_usd == 0.001

    def test_zero_tokens(self, tracker):
        """Zero tokens must not cause errors."""
        tracker.record("test", 0, 0, 0.0)
        assert tracker.total_tokens == 0
