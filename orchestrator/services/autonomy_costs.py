"""
AutonomyCostCollector — four silent-cost gauges from Loop Engineering.

ENH-3 (Loop Engineering §IX): "Four Silent Costs" that compound invisibly
until the system becomes unmaintainable:

  1. verification_debt   — skipped deterministic checks (VerificationGate bypasses)
  2. comprehension_rot   — context window fill % (stale/repeated content crowding out signal)
  3. cognitive_surrender — % of iterations that ran without an independent judge
  4. token_blowout       — cumulative tokens spent above generation baseline

These are observability gauges, not enforcement gates. They are emitted to the
ObservabilityService / dashboard so operators can detect drift early.

Usage:
    costs = AutonomyCostCollector()
    costs.record_skipped_check()                  # gate bypassed
    costs.record_context_window(used=6000, capacity=8000)
    costs.record_iteration(judge_ran=False)
    costs.record_tokens(baseline=500, actual=1200)

    snap = costs.snapshot()
    # CostSnapshot(verification_debt=1, comprehension_rot_pct=75.0,
    #              cognitive_surrender_pct=100.0, token_blowout=700)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger("orchestrator.services.autonomy_costs")


@dataclass(frozen=True)
class CostSnapshot:
    """Immutable point-in-time snapshot of the four silent costs.

    Attributes:
        verification_debt:       Cumulative count of skipped deterministic checks.
        comprehension_rot_pct:   Most-recent context window fill percentage (0-100).
        cognitive_surrender_pct: % of recorded iterations that had no judge (0-100).
        token_blowout:           Cumulative tokens above baseline across all calls.
    """

    verification_debt: int
    comprehension_rot_pct: float
    cognitive_surrender_pct: float
    token_blowout: int


class AutonomyCostCollector:
    """Mutable accumulator for the four silent autonomy costs.

    All methods are synchronous; the class is NOT thread-safe by design
    (callers run single-threaded per task; use one instance per project run).
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Clear all counters (e.g. at the start of a new project run)."""
        self._verification_debt: int = 0
        self._comprehension_rot_pct: float = 0.0
        self._iterations_total: int = 0
        self._iterations_no_judge: int = 0
        self._token_blowout: int = 0

    # ── Recording ──────────────────────────────────────────────────────────────

    def record_skipped_check(self) -> None:
        """A deterministic gate check was skipped (VerificationGate bypassed)."""
        self._verification_debt += 1
        logger.debug("autonomy_costs: verification_debt=%d", self._verification_debt)

    def record_context_window(self, used: int, capacity: int) -> None:
        """Update comprehension_rot_pct from current context window usage.

        Args:
            used:     Tokens consumed in the current context.
            capacity: Maximum context window size for the model.
        """
        if capacity <= 0:
            return
        self._comprehension_rot_pct = (used / capacity) * 100.0
        logger.debug("autonomy_costs: comprehension_rot_pct=%.1f%%", self._comprehension_rot_pct)

    def record_iteration(self, judge_ran: bool) -> None:
        """Record one generate-evaluate iteration.

        Args:
            judge_ran: True if CompletionJudge ran; False if auto-approved.
        """
        self._iterations_total += 1
        if not judge_ran:
            self._iterations_no_judge += 1

    def record_tokens(self, baseline: int, actual: int) -> None:
        """Record token usage for one generation call.

        Args:
            baseline: Expected/budgeted token count for this call type.
            actual:   Actual tokens consumed.
        """
        blowout = max(0, actual - baseline)
        self._token_blowout += blowout
        if blowout > 0:
            logger.debug(
                "autonomy_costs: token_blowout +%d (total=%d)", blowout, self._token_blowout
            )

    # ── Snapshot ───────────────────────────────────────────────────────────────

    def snapshot(self) -> CostSnapshot:
        """Return an immutable copy of current cost gauges."""
        surrender_pct = (
            (self._iterations_no_judge / self._iterations_total) * 100.0
            if self._iterations_total > 0
            else 0.0
        )
        return CostSnapshot(
            verification_debt=self._verification_debt,
            comprehension_rot_pct=self._comprehension_rot_pct,
            cognitive_surrender_pct=surrender_pct,
            token_blowout=self._token_blowout,
        )
