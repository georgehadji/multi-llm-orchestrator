"""
HITL Decision Channels — transport layer for human approval requests.

Ports & Adapters: DecisionChannel is the port; the concrete classes are adapters.

Default when no channel configured: FailClosedChannel (rejects requires_approval
decisions). Override with AutoApproveChannel ONLY in dev/test via explicit wiring
or ORCH_HITL_AUTOAPPROVE=true env flag.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Protocol, runtime_checkable

from .gate import Decision, DecisionResult

logger = logging.getLogger("orchestrator.hitl.channel")


@runtime_checkable
class DecisionChannel(Protocol):
    """Port: transport that delivers a decision to a human and returns their verdict."""

    async def ask(self, decision: Decision, timeout: int) -> DecisionResult: ...


class FailClosedChannel:
    """Default adapter — rejects every requires_approval decision.

    Used when no real channel is configured and ORCH_HITL_AUTOAPPROVE is not set.
    Prevents silent auto-approval from becoming a security vulnerability.
    """

    async def ask(self, decision: Decision, timeout: int) -> DecisionResult:
        logger.error(
            "HITL: no channel configured — REJECTING '%s' (%s). "
            "Configure a DecisionChannel or set ORCH_HITL_AUTOAPPROVE=true "
            "explicitly to opt out of fail-closed behaviour.",
            decision.title,
            decision.category,
        )
        return DecisionResult.REJECTED


class AutoApproveChannel:
    """Dev/test adapter — approves every decision with a prominent warning.

    Must be wired explicitly (or via ORCH_HITL_AUTOAPPROVE=true).
    Never the silent default.
    """

    async def ask(self, decision: Decision, timeout: int) -> DecisionResult:
        logger.warning(
            "HITL: AUTO-APPROVE active (dev mode) — approved '%s' (%s) "
            "without human review. DO NOT use in production.",
            decision.title,
            decision.category,
        )
        return DecisionResult.APPROVED


class CLIDecisionChannel:
    """Production adapter — prompts the operator via stdin.

    Blocks the event loop via run_in_executor so asyncio tasks remain live.
    Times out after `timeout` seconds and returns TIMEOUT.
    """

    async def ask(self, decision: Decision, timeout: int) -> DecisionResult:
        prompt = (
            f"\n[HITL] Decision required\n"
            f"  Category : {decision.category}\n"
            f"  Title    : {decision.title}\n"
            f"  Details  : {decision.description}\n"
            f"{('  Context  : ' + decision.context + chr(10)) if decision.context else ''}"
            f"Approve? [y/N]: "
        )
        loop = asyncio.get_running_loop()
        try:
            answer = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: input(prompt).strip().lower()),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("HITL: timed out waiting for decision on '%s'", decision.title)
            return DecisionResult.TIMEOUT
        except EOFError:
            logger.warning("HITL: stdin closed — treating as rejection for '%s'", decision.title)
            return DecisionResult.REJECTED

        if answer in ("y", "yes"):
            logger.info("HITL: approved '%s' by operator", decision.title)
            return DecisionResult.APPROVED

        logger.info("HITL: rejected '%s' by operator (input: %r)", decision.title, answer)
        return DecisionResult.REJECTED
