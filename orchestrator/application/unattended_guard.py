"""
UnattendedGuard — pre-flight check for autonomous (unattended) runs.

ENH-4 (Loop Engineering §XI disciplines):
  "Cap before you ship" — all three budget ceilings must be finite.
  "Keep one door open" — at least one human checkpoint reachable.

Validates at run_project / automation trigger time; fails closed with an
actionable error listing every missing requirement so the caller can fix
all issues in one pass rather than discovering them one by one.

Disable for legacy environments: ORCH_UNATTENDED_GUARD=false
Acknowledge no checkpoint (expert opt-out): ORCH_NO_CHECKPOINT_ACK=true
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..budget import Budget


class UnattendedGuardError(RuntimeError):
    """Raised when an unattended run is missing required safety constraints.

    The message lists every missing requirement so callers can fix them all
    in one pass.
    """


@dataclass
class RunContext:
    """Snapshot of safety-relevant configuration for a single run.

    Attributes:
        budget:        Per-run Budget object (checked for finite max_usd > 0).
        daily_cap_usd: Cross-run daily ceiling from BudgetHierarchy (None = not set).
        max_retries:   Per-run retry ceiling (None = not set; 0 = invalid).
        has_checkpoint: True if at least one HITL channel is configured that
                        can reach a human (not FailClosedChannel).
        is_unattended: False for interactive sessions — guard becomes a no-op.
    """

    budget: Budget
    daily_cap_usd: float | None
    max_retries: int | None
    has_checkpoint: bool
    is_unattended: bool


class UnattendedGuard:
    """Stateless validator; call UnattendedGuard.validate(ctx) before any run."""

    @staticmethod
    def validate(ctx: RunContext) -> None:
        """Assert all safety constraints are satisfied for an unattended run.

        No-op when:
          - ctx.is_unattended is False (human present)
          - ORCH_UNATTENDED_GUARD=false (legacy escape hatch)

        Raises UnattendedGuardError listing ALL missing requirements.
        """
        if not ctx.is_unattended:
            return

        if os.getenv("ORCH_UNATTENDED_GUARD", "true").lower() == "false":
            return

        missing: list[str] = []

        # ── Per-run cap ──────────────────────────────────────────────────────
        per_run_ok = (
            ctx.budget is not None
            and ctx.budget.max_usd > 0
            and not math.isinf(ctx.budget.max_usd)
        )
        if not per_run_ok:
            missing.append(
                "per-run budget cap: set Budget(max_usd=<finite positive>) "
                "to bound spend within a single run"
            )

        # ── Daily / cross-run cap ────────────────────────────────────────────
        daily_ok = (
            ctx.daily_cap_usd is not None
            and ctx.daily_cap_usd > 0
            and not math.isinf(ctx.daily_cap_usd)
        )
        if not daily_ok:
            missing.append(
                "daily budget cap: set BudgetHierarchy(org_max_usd=...) or pass "
                "daily_cap_usd to bound cross-run spend"
            )

        # ── Retry cap ────────────────────────────────────────────────────────
        retry_ok = ctx.max_retries is not None and ctx.max_retries > 0
        if not retry_ok:
            missing.append(
                "retry cap: set max_retries > 0 to prevent idle bugs "
                "from burning unbounded tokens overnight"
            )

        # ── Human checkpoint ─────────────────────────────────────────────────
        no_checkpoint_ack = os.getenv("ORCH_NO_CHECKPOINT_ACK", "").lower() == "true"
        checkpoint_ok = ctx.has_checkpoint or no_checkpoint_ack
        if not checkpoint_ok:
            missing.append(
                "human checkpoint: configure a CLIDecisionChannel or WebSocketDecisionChannel "
                "so the loop can pause for a human. To bypass (expert only): "
                "ORCH_NO_CHECKPOINT_ACK=true"
            )

        if missing:
            lines = "\n  - ".join(missing)
            raise UnattendedGuardError(
                f"Unattended run blocked — {len(missing)} required constraint(s) not set:\n"
                f"  - {lines}\n\n"
                "Fix all items above before running unattended, or set "
                "ORCH_UNATTENDED_GUARD=false to bypass (not recommended)."
            )
