"""
Tests for ENH-3: AutonomyCostCollector — four silent costs from Loop Engineering.

Loop Engineering §IX: "Four Silent Costs":
  1. verification_debt   — skipped deterministic checks per task
  2. comprehension_rot   — context window % consumed by stale/repeated content
  3. cognitive_surrender — % of loop iterations with no judge (auto-approved)
  4. token_blowout       — tokens spent above the generation baseline

Gauges are accumulative; snapshot() returns an immutable dict.
"""
from __future__ import annotations

import pytest

from orchestrator.services.autonomy_costs import AutonomyCostCollector, CostSnapshot


# ── CostSnapshot ──────────────────────────────────────────────────────────────

class TestCostSnapshot:
    def test_is_dataclass_like(self):
        s = CostSnapshot(
            verification_debt=2,
            comprehension_rot_pct=15.0,
            cognitive_surrender_pct=33.3,
            token_blowout=1500,
        )
        assert s.verification_debt == 2
        assert s.comprehension_rot_pct == 15.0
        assert s.cognitive_surrender_pct == 33.3
        assert s.token_blowout == 1500

    def test_snapshot_is_immutable(self):
        s = CostSnapshot(0, 0.0, 0.0, 0)
        with pytest.raises((AttributeError, TypeError)):
            s.verification_debt = 99  # type: ignore[misc]


# ── AutonomyCostCollector ─────────────────────────────────────────────────────

class TestAutonomyCostCollector:
    def test_initial_snapshot_all_zero(self):
        c = AutonomyCostCollector()
        snap = c.snapshot()
        assert snap.verification_debt == 0
        assert snap.comprehension_rot_pct == 0.0
        assert snap.cognitive_surrender_pct == 0.0
        assert snap.token_blowout == 0

    def test_record_skipped_check_increments_debt(self):
        c = AutonomyCostCollector()
        c.record_skipped_check()
        c.record_skipped_check()
        assert c.snapshot().verification_debt == 2

    def test_record_context_window_updates_rot(self):
        c = AutonomyCostCollector()
        c.record_context_window(used=800, capacity=1000)
        assert c.snapshot().comprehension_rot_pct == pytest.approx(80.0)

    def test_record_context_window_zero_capacity_does_not_raise(self):
        c = AutonomyCostCollector()
        c.record_context_window(used=0, capacity=0)
        assert c.snapshot().comprehension_rot_pct == 0.0

    def test_record_judge_skipped_increments_surrender(self):
        c = AutonomyCostCollector()
        c.record_iteration(judge_ran=False)
        c.record_iteration(judge_ran=True)
        c.record_iteration(judge_ran=False)
        snap = c.snapshot()
        # 2 out of 3 iterations had no judge
        assert snap.cognitive_surrender_pct == pytest.approx(200 / 3)

    def test_record_token_blowout(self):
        c = AutonomyCostCollector()
        c.record_tokens(baseline=100, actual=350)
        assert c.snapshot().token_blowout == 250

    def test_record_tokens_below_baseline_clamps_to_zero(self):
        """Blowout cannot be negative."""
        c = AutonomyCostCollector()
        c.record_tokens(baseline=500, actual=200)
        assert c.snapshot().token_blowout == 0

    def test_snapshot_is_independent_copy(self):
        """Mutating collector after snapshot does not change old snapshot."""
        c = AutonomyCostCollector()
        snap1 = c.snapshot()
        c.record_skipped_check()
        snap2 = c.snapshot()
        assert snap1.verification_debt == 0
        assert snap2.verification_debt == 1

    def test_token_blowout_accumulates(self):
        c = AutonomyCostCollector()
        c.record_tokens(baseline=100, actual=200)  # +100
        c.record_tokens(baseline=50, actual=150)   # +100
        assert c.snapshot().token_blowout == 200

    def test_multiple_context_windows_uses_latest(self):
        """comprehension_rot_pct reflects most recent window reading."""
        c = AutonomyCostCollector()
        c.record_context_window(used=200, capacity=1000)  # 20%
        c.record_context_window(used=600, capacity=1000)  # 60%
        assert c.snapshot().comprehension_rot_pct == pytest.approx(60.0)

    def test_reset_clears_all_counters(self):
        c = AutonomyCostCollector()
        c.record_skipped_check()
        c.record_tokens(baseline=10, actual=100)
        c.record_iteration(judge_ran=False)
        c.reset()
        snap = c.snapshot()
        assert snap.verification_debt == 0
        assert snap.token_blowout == 0
        assert snap.cognitive_surrender_pct == 0.0
