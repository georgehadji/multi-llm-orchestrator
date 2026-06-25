"""
Tests for ENH-4: UnattendedGuard — pre-flight check before any unattended run.

RED first.
"""
from __future__ import annotations

import math
import pytest

from orchestrator.application.unattended_guard import (
    UnattendedGuard,
    UnattendedGuardError,
    RunContext,
)
from orchestrator.budget import Budget


def _budget(max_usd: float = 5.0) -> Budget:
    return Budget(max_usd=max_usd)


class TestUnattendedGuard:
    # ── Happy path ───────────────────────────────────────────────────────────

    def test_passes_when_all_caps_and_checkpoint_configured(self):
        ctx = RunContext(
            budget=_budget(5.0),
            daily_cap_usd=20.0,
            max_retries=5,
            has_checkpoint=True,
            is_unattended=True,
        )
        UnattendedGuard.validate(ctx)  # must not raise

    def test_passes_when_attended(self):
        """Attended runs bypass all checks — human is present."""
        ctx = RunContext(
            budget=_budget(0.0),  # no cap
            daily_cap_usd=None,   # no daily
            max_retries=None,     # no retry cap
            has_checkpoint=False,
            is_unattended=False,
        )
        UnattendedGuard.validate(ctx)  # must not raise

    def test_passes_no_checkpoint_with_explicit_ack(self, monkeypatch):
        monkeypatch.setenv("ORCH_NO_CHECKPOINT_ACK", "true")
        ctx = RunContext(
            budget=_budget(5.0),
            daily_cap_usd=20.0,
            max_retries=5,
            has_checkpoint=False,
            is_unattended=True,
        )
        UnattendedGuard.validate(ctx)  # must not raise

    # ── Per-run cap ──────────────────────────────────────────────────────────

    def test_fails_when_per_run_cap_is_infinite(self):
        ctx = RunContext(
            budget=_budget(math.inf),
            daily_cap_usd=20.0,
            max_retries=5,
            has_checkpoint=True,
            is_unattended=True,
        )
        with pytest.raises(UnattendedGuardError) as exc:
            UnattendedGuard.validate(ctx)
        assert "per-run" in str(exc.value).lower()

    def test_fails_when_per_run_cap_is_zero(self):
        ctx = RunContext(
            budget=_budget(0.0),
            daily_cap_usd=20.0,
            max_retries=5,
            has_checkpoint=True,
            is_unattended=True,
        )
        with pytest.raises(UnattendedGuardError) as exc:
            UnattendedGuard.validate(ctx)
        assert "per-run" in str(exc.value).lower()

    # ── Daily cap ────────────────────────────────────────────────────────────

    def test_fails_when_daily_cap_not_set(self):
        ctx = RunContext(
            budget=_budget(5.0),
            daily_cap_usd=None,
            max_retries=5,
            has_checkpoint=True,
            is_unattended=True,
        )
        with pytest.raises(UnattendedGuardError) as exc:
            UnattendedGuard.validate(ctx)
        assert "daily" in str(exc.value).lower()

    def test_fails_when_daily_cap_is_infinite(self):
        ctx = RunContext(
            budget=_budget(5.0),
            daily_cap_usd=math.inf,
            max_retries=5,
            has_checkpoint=True,
            is_unattended=True,
        )
        with pytest.raises(UnattendedGuardError) as exc:
            UnattendedGuard.validate(ctx)
        assert "daily" in str(exc.value).lower()

    # ── Retry cap ────────────────────────────────────────────────────────────

    def test_fails_when_max_retries_not_set(self):
        ctx = RunContext(
            budget=_budget(5.0),
            daily_cap_usd=20.0,
            max_retries=None,
            has_checkpoint=True,
            is_unattended=True,
        )
        with pytest.raises(UnattendedGuardError) as exc:
            UnattendedGuard.validate(ctx)
        assert "retry" in str(exc.value).lower()

    def test_fails_when_max_retries_is_zero(self):
        ctx = RunContext(
            budget=_budget(5.0),
            daily_cap_usd=20.0,
            max_retries=0,
            has_checkpoint=True,
            is_unattended=True,
        )
        with pytest.raises(UnattendedGuardError) as exc:
            UnattendedGuard.validate(ctx)
        assert "retry" in str(exc.value).lower()

    # ── Checkpoint ───────────────────────────────────────────────────────────

    def test_fails_when_no_checkpoint_and_no_ack(self, monkeypatch):
        monkeypatch.delenv("ORCH_NO_CHECKPOINT_ACK", raising=False)
        ctx = RunContext(
            budget=_budget(5.0),
            daily_cap_usd=20.0,
            max_retries=5,
            has_checkpoint=False,
            is_unattended=True,
        )
        with pytest.raises(UnattendedGuardError) as exc:
            UnattendedGuard.validate(ctx)
        assert "checkpoint" in str(exc.value).lower()

    # ── Error lists all missing requirements at once ──────────────────────────

    def test_error_message_lists_all_missing(self, monkeypatch):
        monkeypatch.delenv("ORCH_NO_CHECKPOINT_ACK", raising=False)
        ctx = RunContext(
            budget=_budget(0.0),      # bad per-run
            daily_cap_usd=None,       # bad daily
            max_retries=None,         # bad retry
            has_checkpoint=False,     # bad checkpoint
            is_unattended=True,
        )
        with pytest.raises(UnattendedGuardError) as exc:
            UnattendedGuard.validate(ctx)
        msg = str(exc.value).lower()
        assert "per-run" in msg
        assert "daily" in msg
        assert "retry" in msg
        assert "checkpoint" in msg

    # ── Guard disabled via env ────────────────────────────────────────────────

    def test_guard_disabled_skips_all_checks(self, monkeypatch):
        monkeypatch.setenv("ORCH_UNATTENDED_GUARD", "false")
        ctx = RunContext(
            budget=_budget(0.0),
            daily_cap_usd=None,
            max_retries=None,
            has_checkpoint=False,
            is_unattended=True,
        )
        UnattendedGuard.validate(ctx)  # must not raise when guard disabled
