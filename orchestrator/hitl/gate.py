"""
HumanInTheLoop — Pause for human approval on critical decisions.

FIX-1 (Loop Engineering): replaced silent auto-approval with fail-closed default.

Policy:
  - requires_approval=False  → always APPROVED (no human needed)
  - requires_approval=True, channel wired → delegates to channel
  - requires_approval=True, no channel, ORCH_HITL_AUTOAPPROVE=true → AutoApproveChannel
  - requires_approval=True, no channel, no env flag → FailClosedChannel (REJECTED)

Wire a real channel (CLIDecisionChannel, WebSocketDecisionChannel) for attended
runs. Set ORCH_HITL_AUTOAPPROVE=true only in dev/test — never in production.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .channel import DecisionChannel

logger = logging.getLogger("orchestrator.hitl.gate")


class DecisionResult(str, Enum):
    """Result of a human decision request."""

    APPROVED = "approved"
    REJECTED = "rejected"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class Decision:
    """A decision that may require human approval."""

    category: str
    title: str
    description: str
    context: str = ""
    requires_approval: bool = True


class HumanInTheLoop:
    """Pause execution and request human approval for critical decisions."""

    DEFAULT_TIMEOUT = 300  # 5 minutes

    def __init__(self, channel: DecisionChannel | None = None) -> None:
        self._channel = channel
        self._pending_decisions: list[Decision] = []

    def _resolve_channel(self) -> DecisionChannel:
        """Return the active channel, applying env-flag fallback."""
        if self._channel is not None:
            return self._channel

        # Explicit dev opt-out via env flag
        from .channel import AutoApproveChannel, FailClosedChannel

        if os.getenv("ORCH_HITL_AUTOAPPROVE", "").lower() == "true":
            return AutoApproveChannel()

        return FailClosedChannel()

    async def request_decision(
        self, decision: Decision, timeout: int = DEFAULT_TIMEOUT
    ) -> DecisionResult:
        """Request a human decision.

        Decisions with requires_approval=False are auto-approved without channel
        involvement (informational events, not gates).

        Decisions with requires_approval=True are routed to the configured
        channel. With no channel and no env opt-out, fails closed (REJECTED).
        """
        self._pending_decisions.append(decision)
        logger.info(
            "HITL: decision required — %s [%s] requires_approval=%s",
            decision.title,
            decision.category,
            decision.requires_approval,
        )

        if not decision.requires_approval:
            return DecisionResult.APPROVED

        channel = self._resolve_channel()
        return await channel.ask(decision, timeout)

    def get_pending(self) -> list[Decision]:
        """Return all decisions submitted this session (approved, rejected, pending)."""
        return list(self._pending_decisions)
