"""
Canary Deployment System for OpenRouter Optimizations
======================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Gradual rollout system for OpenRouter optimizations.
Supports automatic promotion/rollback based on health metrics.

USAGE:
    from orchestrator.canary_deployment import CanaryDeployment, RolloutStage

    canary = CanaryDeployment()

    # Start rollout
    await canary.start_rollout(
        optimization="json_schema",
        stages=[0.01, 0.05, 0.10, 0.25, 0.50, 1.0],  # 1%, 5%, 10%, etc.
        health_thresholds={
            "max_error_rate": 0.05,
            "max_latency_p95": 5000,
            "max_cost_increase": 0.20
        }
    )

    # Check if enabled for a project
    if canary.is_enabled_for("proj_123", "json_schema"):
        # Use optimization
        pass
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

# FIXED: from ..openrouter_ab_testing import OpenRouterABTester, OptimizationType
from ..openrouter_ab_testing import OpenRouterABTester, OptimizationType

logger = logging.getLogger("orchestrator.canary_deployment")


class RolloutStage(str, Enum):
    """Stages of canary rollout."""

    PAUSED = "paused"  # Not started
    CANARY_1 = "canary_1"  # 1% traffic
    CANARY_5 = "canary_5"  # 5% traffic
    CANARY_10 = "canary_10"  # 10% traffic
    CANARY_25 = "canary_25"  # 25% traffic
    CANARY_50 = "canary_50"  # 50% traffic
    FULL_ROLLOUT = "full"  # 100% traffic
    ROLLED_BACK = "rolled_back"  # Reverted to 0%


class RolloutStatus(str, Enum):
    """Status of a rollout."""

    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ROLLED_BACK = "rolled_back"
    PAUSED = "paused"
    FAILED = "failed"


@dataclass
class HealthThresholds:
    """Health thresholds for automatic rollback."""

    max_error_rate: float = 0.05  # 5% error rate
    max_latency_p95_ms: float = 10000  # 10 seconds
    max_cost_increase: float = 0.20  # 20% cost increase
    min_success_rate: float = 0.95  # 95% success rate

    def to_dict(self) -> dict[str, float]:
        return {
            "max_error_rate": self.max_error_rate,
            "max_latency_p95_ms": self.max_latency_p95_ms,
            "max_cost_increase": self.max_cost_increase,
            "min_success_rate": self.min_success_rate,
        }

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> HealthThresholds:
        return cls(
            max_error_rate=data.get("max_error_rate", 0.05),
            max_latency_p95_ms=data.get("max_latency_p95_ms", 10000),
            max_cost_increase=data.get("max_cost_increase", 0.20),
            min_success_rate=data.get("min_success_rate", 0.95),
        )


@dataclass
class RolloutState:
    """Current state of a rollout."""

    optimization: str
    current_stage: RolloutStage
    status: RolloutStatus
    traffic_percentage: float = 0.0

    # Timing
    started_at: float = field(default_factory=time.time)
    stage_entered_at: float = field(default_factory=time.time)
    completed_at: float | None = None

    # Configuration
    stages: list[float] = field(default_factory=lambda: [0.01, 0.05, 0.10, 0.25, 0.50, 1.0])
    stage_duration_minutes: float = 30.0
    thresholds: HealthThresholds = field(default_factory=HealthThresholds)

    # Metrics
    total_requests: int = 0
    error_count: int = 0
    avg_latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "optimization": self.optimization,
            "current_stage": self.current_stage.value,
            "status": self.status.value,
            "traffic_percentage": self.traffic_percentage,
            "started_at": self.started_at,
            "stage_entered_at": self.stage_entered_at,
            "completed_at": self.completed_at,
            "stages": self.stages,
            "stage_duration_minutes": self.stage_duration_minutes,
            "thresholds": self.thresholds.to_dict(),
            "total_requests": self.total_requests,
            "error_count": self.error_count,
            "avg_latency_ms": self.avg_latency_ms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RolloutState:
        return cls(
            optimization=data["optimization"],
            current_stage=RolloutStage(data["current_stage"]),
            status=RolloutStatus(data["status"]),
            traffic_percentage=data["traffic_percentage"],
            started_at=data["started_at"],
            stage_entered_at=data["stage_entered_at"],
            completed_at=data.get("completed_at"),
            stages=data.get("stages", [0.01, 0.05, 0.10, 0.25, 0.50, 1.0]),
            stage_duration_minutes=data.get("stage_duration_minutes", 30.0),
            thresholds=HealthThresholds.from_dict(data.get("thresholds", {})),
            total_requests=data.get("total_requests", 0),
            error_count=data.get("error_count", 0),
            avg_latency_ms=data.get("avg_latency_ms", 0.0),
        )


class CanaryDeployment:
    """
    Canary deployment system for gradual optimization rollout.

    Automatically progresses through stages (1% → 5% → 10% → ... → 100%)
    based on health metrics. Rolls back if thresholds are breached.
    """

    DEFAULT_STAGES = [0.01, 0.05, 0.10, 0.25, 0.50, 1.0]
    DEFAULT_STAGE_DURATION = 30  # minutes

    def __init__(
        self,
        db_path: str | Path | None = None,
        ab_tester: OpenRouterABTester | None = None,
    ):
        """
        Initialize canary deployment system.

        Args:
            db_path: Path to SQLite database for state storage
            ab_tester: A/B tester instance for metrics collection
        """
        if db_path is None:
            db_path = Path(".orchestrator/canary_deployment.db")

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.ab_tester = ab_tester or OpenRouterABTester()
        self._states: dict[str, RolloutState] = {}

        self._init_db()
        self._load_states()

        logger.info("Canary deployment system initialized")

    def _init_db(self) -> None:
        """Initialize SQLite database."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS canary_rollouts (
                    optimization TEXT PRIMARY KEY,
                    state_json TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
            """)
            conn.commit()

    def _load_states(self) -> None:
        """Load rollout states from database."""
        with sqlite3.connect(str(self.db_path)) as conn:
            rows = conn.execute("SELECT optimization, state_json FROM canary_rollouts").fetchall()

        for opt, state_json in rows:
            try:
                self._states[opt] = RolloutState.from_dict(json.loads(state_json))
            except Exception as e:
                logger.warning(f"Failed to load state for {opt}: {e}")

    def _save_state(self, state: RolloutState) -> None:
        """Save rollout state to database."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO canary_rollouts (optimization, state_json, updated_at)
                VALUES (?, ?, ?)
                """,
                (state.optimization, json.dumps(state.to_dict()), time.time()),
            )
            conn.commit()

    async def start_rollout(
        self,
        optimization: OptimizationType | str,
        stages: list[float] | None = None,
        stage_duration_minutes: float = DEFAULT_STAGE_DURATION,
        thresholds: HealthThresholds | None = None,
    ) -> RolloutState:
        """
        Start a new canary rollout.

        Args:
            optimization: Type of optimization to roll out
            stages: List of traffic percentages for each stage
            stage_duration_minutes: Minimum time to spend in each stage
            thresholds: Health thresholds for automatic rollback

        Returns:
            Initial rollout state
        """
        opt_str = optimization.value if isinstance(optimization, OptimizationType) else optimization

        if opt_str in self._states and self._states[opt_str].status == RolloutStatus.IN_PROGRESS:
            raise ValueError(f"Rollout already in progress for {opt_str}")

        state = RolloutState(
            optimization=opt_str,
            current_stage=RolloutStage.PAUSED,
            status=RolloutStatus.IN_PROGRESS,
            traffic_percentage=0.0,
            stages=stages or self.DEFAULT_STAGES,
            stage_duration_minutes=stage_duration_minutes,
            thresholds=thresholds or HealthThresholds(),
        )

        self._states[opt_str] = state
        self._save_state(state)

        logger.info(f"Started canary rollout for {opt_str}")

        # Start with first stage
        await self._advance_stage(opt_str)

        return state

    async def _advance_stage(self, optimization: str) -> None:
        """Advance rollout to next stage."""
        state = self._states.get(optimization)
        if not state or state.status != RolloutStatus.IN_PROGRESS:
            return

        # Find current stage index
        stage_order = [
            RolloutStage.PAUSED,
            RolloutStage.CANARY_1,
            RolloutStage.CANARY_5,
            RolloutStage.CANARY_10,
            RolloutStage.CANARY_25,
            RolloutStage.CANARY_50,
            RolloutStage.FULL_ROLLOUT,
        ]

        current_idx = stage_order.index(state.current_stage)
        next_idx = current_idx + 1

        if next_idx >= len(stage_order):
            # Complete rollout
            state.status = RolloutStatus.COMPLETED
            state.completed_at = time.time()
            logger.info(f"Canary rollout completed for {optimization}")
        else:
            # Advance to next stage
            next_stage = stage_order[next_idx]
            state.current_stage = next_stage
            state.stage_entered_at = time.time()

            # Map stage to traffic percentage
            stage_idx = next_idx - 1  # Skip PAUSED
            if stage_idx < len(state.stages):
                state.traffic_percentage = state.stages[stage_idx]
            else:
                state.traffic_percentage = 1.0

            logger.info(
                f"Advanced {optimization} to {next_stage.value} "
                f"({state.traffic_percentage:.0%} traffic)"
            )

        self._save_state(state)

    async def check_health(self, optimization: str) -> bool:
        """
        Check health metrics and decide whether to continue or rollback.

        Args:
            optimization: Optimization to check

        Returns:
            True if healthy, False if should rollback
        """
        state = self._states.get(optimization)
        if not state or state.status != RolloutStatus.IN_PROGRESS:
            return True

        # Get A/B test results
        opt_type = OptimizationType(optimization)
        ab_result = self.ab_tester.analyze_experiment(opt_type)

        if ab_result.treatment_samples < 10:
            # Not enough data yet
            return True

        # Check thresholds
        error_rate = ab_result.treatment_error_rate
        if error_rate > state.thresholds.max_error_rate:
            logger.error(
                f"{optimization}: Error rate {error_rate:.2%} exceeds threshold "
                f"{state.thresholds.max_error_rate:.2%}"
            )
            return False

        # Check latency increase
        if ab_result.latency_improvement > 0.5:  # >50% latency increase
            logger.error(
                f"{optimization}: Latency increased by {ab_result.latency_improvement:.1%}"
            )
            return False

        # Check cost increase
        if ab_result.cost_improvement > state.thresholds.max_cost_increase:
            logger.error(
                f"{optimization}: Cost increased by {ab_result.cost_improvement:.1%} "
                f"(threshold: {state.thresholds.max_cost_increase:.1%})"
            )
            return False

        return True

    async def evaluate_and_progress(self, optimization: str) -> RolloutState:
        """
        Evaluate health and progress to next stage if healthy.

        Args:
            optimization: Optimization to evaluate

        Returns:
            Current rollout state
        """
        state = self._states.get(optimization)
        if not state or state.status != RolloutStatus.IN_PROGRESS:
            return state

        # Check if enough time has passed in current stage
        time_in_stage = (time.time() - state.stage_entered_at) / 60  # minutes
        if time_in_stage < state.stage_duration_minutes:
            return state

        # Check health
        is_healthy = await self.check_health(optimization)

        if not is_healthy:
            await self.rollback(optimization)
        else:
            await self._advance_stage(optimization)

        return self._states.get(optimization)

    async def rollback(self, optimization: str) -> RolloutState:
        """
        Rollback a rollout to 0% traffic.

        Args:
            optimization: Optimization to rollback

        Returns:
            Updated rollout state
        """
        state = self._states.get(optimization)
        if not state:
            raise ValueError(f"No rollout found for {optimization}")

        state.status = RolloutStatus.ROLLED_BACK
        state.current_stage = RolloutStage.ROLLED_BACK
        state.traffic_percentage = 0.0

        self._save_state(state)
        logger.warning(f"Rolled back canary deployment for {optimization}")

        return state

    def is_enabled_for(self, project_id: str, optimization: str) -> bool:
        """
        Check if optimization should be enabled for a project.

        Uses consistent hashing for deterministic assignment within the
        current traffic percentage.

        Args:
            project_id: Unique project identifier
            optimization: Optimization type

        Returns:
            True if enabled for this project
        """
        state = self._states.get(optimization)
        if not state:
            return False

        if state.status == RolloutStatus.COMPLETED:
            return True

        if state.status != RolloutStatus.IN_PROGRESS:
            return False

        # Use consistent hashing for deterministic assignment
        import hashlib

        hash_input = f"{project_id}:{optimization}:canary"
        hash_value = int(hashlib.md5(hash_input.encode(), usedforsecurity=False).hexdigest(), 16)
        normalized = (hash_value % 10000) / 10000.0

        return normalized < state.traffic_percentage

    def get_state(self, optimization: str) -> RolloutState | None:
        """Get current state of a rollout."""
        return self._states.get(optimization)

    def list_rollouts(self) -> dict[str, RolloutState]:
        """List all rollout states."""
        return dict(self._states)

    async def run_periodic_check(self) -> None:
        """Run health checks and progress all active rollouts."""
        for optimization, state in self._states.items():
            if state.status == RolloutStatus.IN_PROGRESS:
                await self.evaluate_and_progress(optimization)


# Global instance
_default_canary: CanaryDeployment | None = None


def get_canary_deployment() -> CanaryDeployment:
    """Get or create default canary deployment instance."""
    global _default_canary
    if _default_canary is None:
        _default_canary = CanaryDeployment()
    return _default_canary
