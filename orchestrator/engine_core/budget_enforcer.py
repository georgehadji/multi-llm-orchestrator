"""
Budget Enforcer — Budget Monitoring & Enforcement
==================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Handles budget tracking, enforcement, phase partitions, and cost prediction.

Part of Engine Decomposition (Phase 1) - Extracted from engine.py
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..budget import Budget
    from ..cost import BudgetHierarchy, CostPredictor
    from ..models import Task, TaskResult

logger = logging.getLogger(__name__)


class BudgetEnforcer:
    """
    Enforces budget constraints during project execution.

    Responsibilities:
    1. Track spent budget (USD and time)
    2. Enforce phase partitions (decomposition, generation, etc.)
    3. Predict task costs before execution
    4. Warn at budget thresholds
    5. Halt execution when budget exhausted

    Invariants:
    - Budget ceiling is never exceeded (checked mid-task per iteration)
    - Phase partitions are enforced with soft/hard limits
    """

    # Budget warning thresholds (percentage of max)
    WARNING_THRESHOLD = 0.75  # Warn at 75%
    CRITICAL_THRESHOLD = 0.90  # Critical at 90%

    # Phase partition enforcement
    PHASE_WARN_THRESHOLD = 0.80  # Warn at 80% of phase budget
    PHASE_HALT_THRESHOLD = 2.00  # Hard halt at 2× phase budget

    def __init__(
        self,
        budget: Budget,
        budget_hierarchy: BudgetHierarchy | None = None,
        cost_predictor: CostPredictor | None = None,
    ):
        """
        Initialize budget enforcer.

        Args:
            budget: Budget object to enforce
            budget_hierarchy: Optional hierarchy for cross-run tracking
            cost_predictor: Optional cost prediction engine
        """
        self.budget = budget
        self.budget_hierarchy = budget_hierarchy
        self.cost_predictor = cost_predictor

        # Track phase spending
        self.phase_spent: dict[str, float] = dict.fromkeys(budget.phase_limits.keys(), 0.0)

        # Session tracking
        self.session_start_time = time.time()
        self.last_check_time = time.time()

        # Warning state (to avoid repeated warnings)
        self._warned_thresholds: set[str] = set()

        logger.info(
            f"Budget enforcer initialized: max=${budget.max_usd:.2f}, "
            f"time={budget.max_time_seconds}s"
        )

    def check_budget(self, task: Task | None = None) -> tuple[bool, bool]:
        """
        Check if budget allows continued execution.

        FIX #5: Budget checked within iteration loop (mid-task), not just pre-task.

        Args:
            task: Optional task being executed (for cost prediction)

        Returns:
            Tuple of (budget_ok, time_ok)
        """
        # Update elapsed time
        current_time = time.time()
        elapsed = current_time - self.session_start_time

        # Check total budget
        budget_ok = self.budget.remaining_usd > 0
        time_ok = elapsed < self.budget.max_time_seconds

        # Log warnings at thresholds
        self._check_budget_thresholds()

        # Check time threshold
        time_remaining = self.budget.max_time_seconds - elapsed
        if time_remaining < 60:  # Less than 1 minute
            logger.warning(f"Time critical: {time_remaining:.0f}s remaining")

        # Predict task cost if provided
        if task and self.cost_predictor:
            predicted_cost = self.cost_predictor.predict_task_cost(task)
            if predicted_cost > self.budget.remaining_usd:
                logger.warning(
                    f"Predicted task cost ${predicted_cost:.4f} exceeds "
                    f"remaining budget ${self.budget.remaining_usd:.4f}"
                )

        return budget_ok, time_ok

    def check_budget_mid_task(
        self,
        iteration: int,
        cost_so_far: float,
    ) -> bool:
        """
        Check budget during task execution (mid-task).

        Args:
            iteration: Current iteration number
            cost_so_far: Cost accumulated so far in this task

        Returns:
            True if execution can continue
        """
        # Check total remaining
        if self.budget.remaining_usd <= 0:
            logger.warning("Budget exhausted mid-task")
            return False

        # Check if this task is exceeding expectations
        if iteration > 5 and cost_so_far > self.budget.max_usd * 0.1:
            logger.warning(
                f"Task iteration {iteration} has cost ${cost_so_far:.4f}, "
                "consider early termination"
            )

        return True

    def enforce_phase_partition(
        self,
        phase: str,
        cost: float,
    ) -> bool:
        """
        Enforce budget phase partition limits.

        FEAT: Budget phase partition enforcement (warn + soft-halt at 2× soft cap).

        Args:
            phase: Phase name (e.g., "decomposition", "generation")
            cost: Cost to add to phase

        Returns:
            True if allowed, False if would exceed hard limit
        """
        if phase not in self.budget.phase_limits:
            # No limit for this phase
            return True

        phase_limit = self.budget.phase_limits[phase]
        current_phase_spent = self.phase_spent.get(phase, 0.0)
        new_phase_spent = current_phase_spent + cost

        # Check soft cap (warn)
        if new_phase_spent > phase_limit * self.PHASE_WARN_THRESHOLD:
            warning_key = f"{phase}_warn"
            if warning_key not in self._warned_thresholds:
                logger.warning(
                    f"Phase {phase} at {new_phase_spent/phase_limit*100:.1f}% "
                    f"(${new_phase_spent:.4f}/${phase_limit:.4f})"
                )
                self._warned_thresholds.add(warning_key)

        # Check hard cap (halt)
        if new_phase_spent > phase_limit * self.PHASE_HALT_THRESHOLD:
            logger.error(
                f"Phase {phase} would exceed 2× limit "
                f"(${new_phase_spent:.4f} > ${phase_limit * 2:.4f})"
            )
            return False

        # Update phase spending
        self.phase_spent[phase] = new_phase_spent
        return True

    def record_cost(
        self,
        task_id: str,
        cost_usd: float,
        phase: str | None = None,
    ) -> None:
        """
        Record task execution cost.

        Args:
            task_id: Task identifier
            cost_usd: Cost in USD
            phase: Optional phase name
        """
        # Update budget
        self.budget.spent_usd += cost_usd

        # Update phase spending
        if phase and phase in self.phase_spent:
            self.phase_spent[phase] += cost_usd

        # Update hierarchy if available
        if self.budget_hierarchy:
            self.budget_hierarchy.record_cost(task_id, cost_usd)

        logger.debug(f"Recorded cost for {task_id}: ${cost_usd:.6f}")

    def record_time(self, seconds: float) -> None:
        """
        Record elapsed time for a task.

        Args:
            seconds: Elapsed time in seconds
        """
        self.budget.elapsed_time += seconds

    def _check_budget_thresholds(self) -> None:
        """Check and log budget threshold warnings."""
        spent_percentage = self.budget.spent_usd / self.budget.max_usd

        # Check warning threshold (75%)
        if spent_percentage >= self.WARNING_THRESHOLD:
            warning_key = "budget_75"
            if warning_key not in self._warned_thresholds:
                logger.warning(
                    f"Budget warning: {spent_percentage*100:.1f}% spent "
                    f"(${self.budget.spent_usd:.4f}/${self.budget.max_usd:.4f})"
                )
                self._warned_thresholds.add(warning_key)

        # Check critical threshold (90%)
        if spent_percentage >= self.CRITICAL_THRESHOLD:
            warning_key = "budget_90"
            if warning_key not in self._warned_thresholds:
                logger.critical(
                    f"Budget critical: {spent_percentage*100:.1f}% spent "
                    f"(${self.budget.remaining_usd:.4f} remaining)"
                )
                self._warned_thresholds.add(warning_key)

    def get_budget_status(self) -> dict[str, Any]:
        """
        Get current budget status.

        Returns:
            Dictionary with budget status information
        """
        elapsed = time.time() - self.session_start_time

        return {
            "spent_usd": self.budget.spent_usd,
            "remaining_usd": self.budget.remaining_usd,
            "max_usd": self.budget.max_usd,
            "spent_percentage": self.budget.spent_usd / self.budget.max_usd,
            "elapsed_seconds": elapsed,
            "remaining_seconds": self.budget.max_time_seconds - elapsed,
            "phase_spending": dict(self.phase_spent),
        }

    def predict_remaining_tasks(
        self,
        tasks: list[Task],
        completed_results: list[TaskResult],
    ) -> float:
        """
        Predict cost for remaining tasks.

        Args:
            tasks: Remaining tasks to execute
            completed_results: Results from completed tasks (for estimation)

        Returns:
            Predicted cost for remaining tasks
        """
        if not tasks:
            return 0.0

        # Use historical data if available
        if completed_results and self.cost_predictor:
            return self.cost_predictor.predict_remaining_tasks(tasks, completed_results)

        # Simple average-based prediction
        if completed_results:
            avg_cost = sum(r.cost_usd for r in completed_results) / len(completed_results)
            return avg_cost * len(tasks)

        # No data - use budget remaining as upper bound
        return self.budget.remaining_usd

    def reset_session(self) -> None:
        """Reset session tracking (for resumed projects)."""
        self.session_start_time = time.time()
        self.last_check_time = time.time()
        self._warned_thresholds.clear()

        logger.info("Budget session reset")

    def should_halt(self) -> bool:
        """
        Check if execution should halt due to budget constraints.

        Returns:
            True if execution should halt
        """
        budget_ok, time_ok = self.check_budget()
        return not budget_ok or not time_ok
