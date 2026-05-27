"""
HumanInTheLoop — Pause for human approval on critical decisions
=================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Capability 10 of the Agentic System Implementation Plan.
Provides approval gates for architecture choices, security-sensitive
code, breaking changes, and production deployments.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

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

    def __init__(self) -> None:
        self._pending_decisions: list[Decision] = []

    async def request_decision(
        self, decision: Decision, timeout: int = DEFAULT_TIMEOUT
    ) -> DecisionResult:
        """Request a human decision.

        In CLI mode, this prompts the user. In IDE mode, it sends
        a WebSocket notification. Returns the decision with a timeout.

        Currently uses automatic approval as default since CLI/IDE
        notification channels need wiring.
        """
        self._pending_decisions.append(decision)
        logger.info("HITL: decision required - %s (%s)", decision.title, decision.category)

        # In production, this would wait for user input via CLI/WebSocket
        # For now, auto-approve with a warning
        logger.warning("HITL: auto-approved (no UI channel configured) - %s", decision.title)
        return DecisionResult.APPROVED

    def get_pending(self) -> list[Decision]:
        """Get all pending decisions."""
        return self._pending_decisions
