"""
Tests for HITL gate — FIX-1: replace silent auto-approval.

RED tests first; implementation must make all pass.
"""

from __future__ import annotations

import os
import pytest

pytestmark = pytest.mark.unit

from orchestrator.hitl.gate import Decision, DecisionResult, HumanInTheLoop
from orchestrator.hitl.channel import (
    AutoApproveChannel,
    FailClosedChannel,
    CLIDecisionChannel,
)

SECURITY_DECISION = Decision(
    category="security",
    title="Rotate API key",
    description="Rotate the production API key",
    requires_approval=True,
)

INFO_DECISION = Decision(
    category="info",
    title="Log entry",
    description="Non-critical log",
    requires_approval=False,
)


# ── FailClosedChannel (default when no channel configured) ────────────────────


class TestFailClosedChannel:
    @pytest.mark.asyncio
    async def test_requires_approval_returns_rejected(self):
        gate = HumanInTheLoop()  # no channel → fail-closed default
        result = await gate.request_decision(SECURITY_DECISION)
        assert result == DecisionResult.REJECTED

    @pytest.mark.asyncio
    async def test_no_approval_required_returns_approved(self):
        gate = HumanInTheLoop()
        result = await gate.request_decision(INFO_DECISION)
        assert result == DecisionResult.APPROVED

    @pytest.mark.asyncio
    async def test_decision_logged_as_pending(self):
        gate = HumanInTheLoop()
        await gate.request_decision(SECURITY_DECISION)
        assert len(gate.get_pending()) == 1
        assert gate.get_pending()[0].title == SECURITY_DECISION.title


# ── AutoApproveChannel (explicit dev-only) ────────────────────────────────────


class TestAutoApproveChannel:
    @pytest.mark.asyncio
    async def test_auto_approve_channel_approves_all(self):
        gate = HumanInTheLoop(channel=AutoApproveChannel())
        result = await gate.request_decision(SECURITY_DECISION)
        assert result == DecisionResult.APPROVED

    @pytest.mark.asyncio
    async def test_auto_approve_channel_approves_non_required(self):
        gate = HumanInTheLoop(channel=AutoApproveChannel())
        result = await gate.request_decision(INFO_DECISION)
        assert result == DecisionResult.APPROVED


# ── Env flag: ORCH_HITL_AUTOAPPROVE=true ─────────────────────────────────────


class TestEnvFlagAutoApprove:
    @pytest.mark.asyncio
    async def test_env_flag_true_approves_when_no_channel(self, monkeypatch):
        monkeypatch.setenv("ORCH_HITL_AUTOAPPROVE", "true")
        gate = HumanInTheLoop()  # no channel, but env flag set
        result = await gate.request_decision(SECURITY_DECISION)
        assert result == DecisionResult.APPROVED

    @pytest.mark.asyncio
    async def test_env_flag_false_fails_closed(self, monkeypatch):
        monkeypatch.setenv("ORCH_HITL_AUTOAPPROVE", "false")
        gate = HumanInTheLoop()
        result = await gate.request_decision(SECURITY_DECISION)
        assert result == DecisionResult.REJECTED

    @pytest.mark.asyncio
    async def test_env_flag_absent_fails_closed(self, monkeypatch):
        monkeypatch.delenv("ORCH_HITL_AUTOAPPROVE", raising=False)
        gate = HumanInTheLoop()
        result = await gate.request_decision(SECURITY_DECISION)
        assert result == DecisionResult.REJECTED


# ── FailClosedChannel direct ──────────────────────────────────────────────────


class TestFailClosedChannelDirect:
    @pytest.mark.asyncio
    async def test_always_rejects(self):
        channel = FailClosedChannel()
        result = await channel.ask(SECURITY_DECISION, timeout=5)
        assert result == DecisionResult.REJECTED


# ── Security category invariant ───────────────────────────────────────────────


class TestSecurityCategoryInvariant:
    @pytest.mark.asyncio
    async def test_security_decision_fails_closed_without_channel(self, monkeypatch):
        monkeypatch.delenv("ORCH_HITL_AUTOAPPROVE", raising=False)
        gate = HumanInTheLoop()
        result = await gate.request_decision(
            Decision(
                category="security",
                title="Deploy to production",
                description="Breaking change deployment",
                requires_approval=True,
            )
        )
        assert result == DecisionResult.REJECTED

    @pytest.mark.asyncio
    async def test_security_decision_requires_explicit_autoapprove_opt_in(self):
        # AutoApproveChannel must be wired explicitly — not a magic default
        gate_no_channel = HumanInTheLoop()
        gate_explicit = HumanInTheLoop(channel=AutoApproveChannel())

        result_closed = await gate_no_channel.request_decision(SECURITY_DECISION)
        result_explicit = await gate_explicit.request_decision(SECURITY_DECISION)

        assert result_closed == DecisionResult.REJECTED
        assert result_explicit == DecisionResult.APPROVED


# ── has_real_channel guard ────────────────────────────────────────────────────


class TestHasRealChannel:
    def test_none_channel_is_not_real(self):
        gate = HumanInTheLoop(channel=None)
        assert gate.has_real_channel() is False

    def test_fail_closed_channel_is_not_real(self):
        gate = HumanInTheLoop(channel=FailClosedChannel())
        assert gate.has_real_channel() is False

    def test_auto_approve_channel_is_not_real(self):
        gate = HumanInTheLoop(channel=AutoApproveChannel())
        assert gate.has_real_channel() is False

    def test_cli_channel_is_real(self):
        gate = HumanInTheLoop(channel=CLIDecisionChannel())
        assert gate.has_real_channel() is True
