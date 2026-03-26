"""
Gradual Rollout Engine for Meta-Optimization
=============================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Safe deployment of approved strategy proposals through staged rollout.
Automatically progresses through stages based on success metrics and
triggers rollback on failure thresholds.

Features:
- Configurable rollout stages (5% → 25% → 50% → 100%)
- Success/failure tracking per stage
- Auto-progression when thresholds met
- Auto-rollback on failure detection
- Timeout-based stage progression

USAGE:
    from orchestrator.gradual_rollout import GradualRolloutManager, RolloutConfig

    config = RolloutConfig(
        stages=[
            RolloutStage(percentage=5, min_successes=10, max_failures=3),
            RolloutStage(percentage=25, min_successes=25, max_failures=5),
            RolloutStage(percentage=50, min_successes=50, max_failures=10),
            RolloutStage(percentage=100, min_successes=0, max_failures=0),
        ]
    )
    
    rollout_mgr = GradualRolloutManager(archive, config)
    
    # Start rollout for approved proposal
    rollout = await rollout_mgr.start_rollout(proposal)
    
    # Record execution outcomes
    await rollout_mgr.record_execution(rollout.rollout_id, success=True, metrics={...})
    
    # Check progress (called periodically)
    decision = await rollout_mgr.check_stage_progress(rollout.rollout_id)
    if decision.decision == "advance":
        await rollout_mgr.advance_stage(rollout.rollout_id)
    elif decision.decision == "rollback":
        await rollout_mgr.trigger_rollback(rollout.rollout_id, reason="Too many failures")
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from collections import defaultdict

from .models import Model, TaskType
from .meta_orchestrator import ExecutionArchive, StrategyProposal, ProposalStatus

logger = logging.getLogger("orchestrator.rollout")


# ─────────────────────────────────────────────
# Enums & Constants
# ─────────────────────────────────────────────

class RolloutStatus(str, Enum):
    """Status of a rollout."""
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ROLLED_BACK = "rolled_back"
    PAUSED = "paused"
    FAILED = "failed"


@dataclass
class StageDecision:
    """Decision for stage progression."""
    decision: str  # "advance", "rollback", "continue", "timeout"
    reason: str


class RolloutEvent(str, Enum):
    """Events during rollout."""
    STARTED = "started"
    STAGE_ADVANCED = "stage_advanced"
    OUTCOME_RECORDED = "outcome_recorded"
    ROLLED_BACK = "rolled_back"
    COMPLETED = "completed"
    FAILED = "failed"


# Default rollout stages
DEFAULT_STAGES = [
    {"percentage": 5, "min_successes": 10, "max_failures": 3, "timeout_hours": 24},
    {"percentage": 25, "min_successes": 25, "max_failures": 5, "timeout_hours": 48},
    {"percentage": 50, "min_successes": 50, "max_failures": 10, "timeout_hours": 72},
    {"percentage": 100, "min_successes": 0, "max_failures": 0, "timeout_hours": 0},
]


# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────

@dataclass
class RolloutStage:
    """A stage in the gradual rollout."""
    stage_index: int
    percentage: int  # Traffic percentage (5, 25, 50, 100)
    min_successes: int  # Successes required to advance
    max_failures: int  # Failures before rollback
    timeout_hours: float  # Max hours in stage before timeout
    
    def to_dict(self) -> dict:
        return {
            "stage_index": self.stage_index,
            "percentage": self.percentage,
            "min_successes": self.min_successes,
            "max_failures": self.max_failures,
            "timeout_hours": self.timeout_hours,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "RolloutStage":
        return cls(**data)


@dataclass
class StageResult:
    """Results for a rollout stage."""
    stage_index: int
    successes: int = 0
    failures: int = 0
    total_outcomes: int = 0
    avg_score: float = 0.0
    avg_cost: float = 0.0
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    
    @property
    def success_rate(self) -> float:
        if self.total_outcomes == 0:
            return 0.0
        return self.successes / self.total_outcomes
    
    def to_dict(self) -> dict:
        return {
            "stage_index": self.stage_index,
            "successes": self.successes,
            "failures": self.failures,
            "total_outcomes": self.total_outcomes,
            "avg_score": self.avg_score,
            "avg_cost": self.avg_cost,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "StageResult":
        result = cls(
            stage_index=data["stage_index"],
            successes=data["successes"],
            failures=data["failures"],
            total_outcomes=data["total_outcomes"],
            avg_score=data["avg_score"],
            avg_cost=data["avg_cost"],
            started_at=data["started_at"],
        )
        result.completed_at = data.get("completed_at")
        return result


@dataclass
class RolloutOutcome:
    """Single outcome recorded during rollout."""
    outcome_id: str
    rollout_id: str
    stage_index: int
    project_id: str
    success: bool
    score: float
    cost_usd: float
    latency_ms: float
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> dict:
        return {
            "outcome_id": self.outcome_id,
            "rollout_id": self.rollout_id,
            "stage_index": self.stage_index,
            "project_id": self.project_id,
            "success": self.success,
            "score": self.score,
            "cost_usd": self.cost_usd,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "RolloutOutcome":
        return cls(**data)


@dataclass
class Rollout:
    """A gradual rollout for a proposal."""
    rollout_id: str
    proposal: StrategyProposal
    stages: List[RolloutStage]
    current_stage_index: int = 0
    status: RolloutStatus = RolloutStatus.IN_PROGRESS
    
    # Stage results
    stage_results: List[StageResult] = field(default_factory=list)
    
    # All outcomes
    outcomes: List[RolloutOutcome] = field(default_factory=list)
    
    # Timeline
    started_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    
    # Rollback info
    rollback_reason: Optional[str] = None
    rolled_back_at: Optional[float] = None
    
    # Events log
    events: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "rollout_id": self.rollout_id,
            "proposal": self.proposal.to_dict(),
            "stages": [s.to_dict() for s in self.stages],
            "current_stage_index": self.current_stage_index,
            "status": self.status.value,
            "stage_results": [sr.to_dict() for sr in self.stage_results],
            "outcomes": [o.to_dict() for o in self.outcomes[-100:]],  # Last 100
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "rollback_reason": self.rollback_reason,
            "rolled_back_at": self.rolled_back_at,
            "events": self.events[-50:],  # Last 50 events
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Rollout":
        rollout = cls(
            rollout_id=data["rollout_id"],
            proposal=StrategyProposal(**data["proposal"]),
            stages=[RolloutStage.from_dict(s) for s in data["stages"]],
            current_stage_index=data["current_stage_index"],
            status=RolloutStatus(data["status"]),
            started_at=data.get("started_at", time.time()),
            completed_at=data.get("completed_at"),
            rollback_reason=data.get("rollback_reason"),
            rolled_back_at=data.get("rolled_back_at"),
        )
        
        # Load stage results
        rollout.stage_results = [
            StageResult.from_dict(sr) for sr in data.get("stage_results", [])
        ]
        
        # Load outcomes
        rollout.outcomes = [
            RolloutOutcome.from_dict(o) for o in data.get("outcomes", [])
        ]
        
        # Load events
        rollout.events = data.get("events", [])
        
        return rollout
    
    @property
    def current_stage(self) -> Optional[RolloutStage]:
        """Get current stage."""
        if self.current_stage_index >= len(self.stages):
            return None
        return self.stages[self.current_stage_index]
    
    @property
    def current_stage_result(self) -> Optional[StageResult]:
        """Get current stage result."""
        if self.current_stage_index >= len(self.stage_results):
            return None
        return self.stage_results[self.current_stage_index]
    
    def _log_event(self, event_type: RolloutEvent, details: Dict[str, Any]):
        """Log an event."""
        self.events.append({
            "timestamp": time.time(),
            "event_type": event_type.value,
            "details": details,
        })


@dataclass
class RolloutConfig:
    """Configuration for gradual rollout."""
    stages: List[RolloutStage] = field(default_factory=list)
    auto_rollback_enabled: bool = True
    storage_path: Optional[Path] = None
    
    def __post_init__(self):
        if not self.stages:
            # Create default stages
            for i, stage_data in enumerate(DEFAULT_STAGES):
                self.stages.append(RolloutStage(stage_index=i, **stage_data))


# ─────────────────────────────────────────────
# Gradual Rollout Manager
# ─────────────────────────────────────────────

class GradualRolloutManager:
    """
    Manages gradual rollouts for approved proposals.
    
    Handles stage progression, outcome tracking, and auto-rollback.
    """
    
    def __init__(
        self,
        archive: ExecutionArchive,
        config: Optional[RolloutConfig] = None,
    ):
        self.archive = archive
        self.config = config or RolloutConfig()
        
        self._storage_path = self.config.storage_path or (
            Path.home() / ".orchestrator_cache" / "rollouts"
        )
        self._storage_path.mkdir(parents=True, exist_ok=True)
        
        self._rollouts: Dict[str, Rollout] = {}
        self._lock = asyncio.Lock()
        
        self._load_rollouts()
    
    def _load_rollouts(self):
        """Load rollouts from disk."""
        rollouts_file = self._storage_path / "rollouts.jsonl"
        if not rollouts_file.exists():
            return
        
        try:
            with open(rollouts_file, "r") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        rollout = Rollout.from_dict(data)
                        self._rollouts[rollout.rollout_id] = rollout
            
            logger.info(f"Loaded {len(self._rollouts)} rollouts from disk")
        except Exception as e:
            logger.warning(f"Failed to load rollouts: {e}")
    
    def _persist_rollout(self, rollout: Rollout):
        """Persist rollout to disk."""
        rollouts_file = self._storage_path / "rollouts.jsonl"
        
        # Read existing, update/add, write back
        rollouts = []
        if rollouts_file.exists():
            with open(rollouts_file, "r") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        if data["rollout_id"] != rollout.rollout_id:
                            rollouts.append(data)
        
        rollouts.append(rollout.to_dict())
        
        with open(rollouts_file, "w") as f:
            for r in rollouts:
                f.write(json.dumps(r) + "\n")
    
    async def start_rollout(
        self,
        proposal: StrategyProposal,
        stages: Optional[List[RolloutStage]] = None,
    ) -> Rollout:
        """
        Start a gradual rollout for an approved proposal.
        
        Args:
            proposal: The approved proposal to rollout
            stages: Custom stages (uses default if not provided)
        
        Returns:
            Created rollout
        """
        async with self._lock:
            rollout_id = f"rollout_{proposal.proposal_id}_{int(time.time())}"
            
            rollout_stages = stages or self.config.stages
            if not rollout_stages:
                rollout_stages = [
                    RolloutStage(stage_index=i, **stage_data)
                    for i, stage_data in enumerate(DEFAULT_STAGES)
                ]
            
            rollout = Rollout(
                rollout_id=rollout_id,
                proposal=proposal,
                stages=rollout_stages,
            )
            
            # Initialize stage results
            for stage in rollout.stages:
                rollout.stage_results.append(StageResult(stage_index=stage.stage_index))
            
            # Log start event
            rollout._log_event(RolloutEvent.STARTED, {
                "proposal_id": proposal.proposal_id,
                "stages": len(rollout.stages),
            })
            
            self._rollouts[rollout_id] = rollout
            self._persist_rollout(rollout)
            
            logger.info(f"Started rollout {rollout_id} for proposal {proposal.proposal_id}")
            return rollout
    
    async def record_execution(
        self,
        rollout_id: str,
        success: bool,
        score: float,
        cost_usd: float,
        latency_ms: float,
        project_id: str,
    ) -> RolloutOutcome:
        """
        Record an execution outcome for a rollout.
        
        Args:
            rollout_id: Rollout identifier
            success: Whether execution succeeded
            score: Quality score (0-1)
            cost_usd: Actual cost
            latency_ms: Execution latency
            project_id: Project identifier
        
        Returns:
            Recorded outcome
        """
        async with self._lock:
            if rollout_id not in self._rollouts:
                raise ValueError(f"Rollout {rollout_id} not found")
            
            rollout = self._rollouts[rollout_id]
            if rollout.status != RolloutStatus.IN_PROGRESS:
                raise ValueError(f"Rollout {rollout_id} is not in progress")
            
            current_stage_idx = rollout.current_stage_index
            outcome = RolloutOutcome(
                outcome_id=f"out_{int(time.time() * 1000)}",
                rollout_id=rollout_id,
                stage_index=current_stage_idx,
                project_id=project_id,
                success=success,
                score=score,
                cost_usd=cost_usd,
                latency_ms=latency_ms,
            )
            
            rollout.outcomes.append(outcome)
            
            # Update stage result
            stage_result = rollout.stage_results[current_stage_idx]
            stage_result.total_outcomes += 1
            
            if success:
                stage_result.successes += 1
            else:
                stage_result.failures += 1
            
            # Update averages
            stage_result.avg_score = (
                (stage_result.avg_score * (stage_result.total_outcomes - 1) + score)
                / stage_result.total_outcomes
            )
            stage_result.avg_cost = (
                (stage_result.avg_cost * (stage_result.total_outcomes - 1) + cost_usd)
                / stage_result.total_outcomes
            )
            
            # Log event
            rollout._log_event(RolloutEvent.OUTCOME_RECORDED, {
                "outcome_id": outcome.outcome_id,
                "success": success,
                "score": score,
            })
            
            self._persist_rollout(rollout)
            
            return outcome
    
    async def check_stage_progress(self, rollout_id: str) -> StageDecision:
        """
        Check progress of current stage and determine next action.
        
        Args:
            rollout_id: Rollout identifier
        
        Returns:
            Decision for stage progression
        """
        async with self._lock:
            if rollout_id not in self._rollouts:
                raise ValueError(f"Rollout {rollout_id} not found")
            
            rollout = self._rollouts[rollout_id]
            current_stage = rollout.current_stage
            
            if not current_stage:
                # All stages completed
                return StageDecision(
                    decision="advance",
                    reason="All stages completed",
                )
            
            stage_result = rollout.current_stage_result
            
            # Check failure threshold
            if stage_result.failures >= current_stage.max_failures:
                if self.config.auto_rollback_enabled:
                    return StageDecision(
                        decision="rollback",
                        reason=f"Max failures reached ({stage_result.failures} >= {current_stage.max_failures})",
                    )
            
            # Check success threshold
            if stage_result.successes >= current_stage.min_successes:
                return StageDecision(
                    decision="advance",
                    reason=f"Min successes reached ({stage_result.successes} >= {current_stage.min_successes})",
                )
            
            # Check timeout
            if current_stage.timeout_hours > 0:
                elapsed_hours = (time.time() - stage_result.started_at) / 3600
                if elapsed_hours >= current_stage.timeout_hours:
                    return StageDecision(
                        decision="timeout",
                        reason=f"Stage timeout ({elapsed_hours:.1f}h >= {current_stage.timeout_hours}h)",
                    )
            
            # Continue in current stage
            return StageDecision(
                decision="continue",
                reason=f"Progress: {stage_result.successes}/{current_stage.min_successes} successes, {stage_result.failures}/{current_stage.max_failures} failures",
            )
    
    async def advance_stage(self, rollout_id: str) -> bool:
        """
        Advance to next stage.
        
        Args:
            rollout_id: Rollout identifier
        
        Returns:
            True if advanced, False if no more stages
        """
        async with self._lock:
            if rollout_id not in self._rollouts:
                return False
            
            rollout = self._rollouts[rollout_id]
            
            # Complete current stage
            if rollout.current_stage_result:
                rollout.current_stage_result.completed_at = time.time()
            
            # Move to next stage
            rollout.current_stage_index += 1
            
            if rollout.current_stage_index >= len(rollout.stages):
                # All stages completed
                rollout.status = RolloutStatus.COMPLETED
                rollout.completed_at = time.time()
                rollout._log_event(RolloutEvent.COMPLETED, {
                    "total_outcomes": len(rollout.outcomes),
                    "total_stages": len(rollout.stages),
                })
                logger.info(f"Rollout {rollout_id} completed successfully")
            else:
                # Start next stage
                rollout._log_event(RolloutEvent.STAGE_ADVANCED, {
                    "from_stage": rollout.current_stage_index - 1,
                    "to_stage": rollout.current_stage_index,
                    "traffic_percentage": rollout.current_stage.percentage,
                })
                logger.info(
                    f"Rollout {rollout_id} advanced to stage {rollout.current_stage_index} "
                    f"({rollout.current_stage.percentage}% traffic)"
                )
            
            self._persist_rollout(rollout)
            return True
    
    async def trigger_rollback(
        self,
        rollout_id: str,
        reason: str,
    ) -> bool:
        """
        Trigger rollback of a rollout.
        
        Args:
            rollout_id: Rollout identifier
            reason: Reason for rollback
        
        Returns:
            True if rolled back
        """
        async with self._lock:
            if rollout_id not in self._rollouts:
                return False
            
            rollout = self._rollouts[rollout_id]
            rollout.status = RolloutStatus.ROLLED_BACK
            rollout.rollback_reason = reason
            rollout.rolled_back_at = time.time()
            
            rollout._log_event(RolloutEvent.ROLLED_BACK, {
                "reason": reason,
                "stage_at_rollback": rollout.current_stage_index,
            })
            
            self._persist_rollout(rollout)
            
            logger.warning(f"Rollout {rollout_id} rolled back: {reason}")
            return True
    
    async def get_active_rollouts(self) -> List[Rollout]:
        """Get all in-progress rollouts."""
        return [
            r for r in self._rollouts.values()
            if r.status == RolloutStatus.IN_PROGRESS
        ]
    
    async def get_rollout(self, rollout_id: str) -> Optional[Rollout]:
        """Get rollout by ID."""
        return self._rollouts.get(rollout_id)
    
    def get_rollout_stats(self) -> Dict[str, Any]:
        """Get rollout statistics."""
        by_status = defaultdict(int)
        for r in self._rollouts.values():
            by_status[r.status.value] += 1
        
        return {
            "total_rollouts": len(self._rollouts),
            "by_status": dict(by_status),
            "active_rollouts": by_status.get("in_progress", 0),
            "completed_rollouts": by_status.get("completed", 0),
            "rolled_back_rollouts": by_status.get("rolled_back", 0),
        }
    
    async def pause_rollout(self, rollout_id: str) -> bool:
        """Pause a rollout."""
        async with self._lock:
            if rollout_id not in self._rollouts:
                return False
            
            rollout = self._rollouts[rollout_id]
            if rollout.status != RolloutStatus.IN_PROGRESS:
                return False
            
            rollout.status = RolloutStatus.PAUSED
            self._persist_rollout(rollout)
            
            logger.info(f"Paused rollout {rollout_id}")
            return True
    
    async def resume_rollout(self, rollout_id: str) -> bool:
        """Resume a paused rollout."""
        async with self._lock:
            if rollout_id not in self._rollouts:
                return False
            
            rollout = self._rollouts[rollout_id]
            if rollout.status != RolloutStatus.PAUSED:
                return False
            
            rollout.status = RolloutStatus.IN_PROGRESS
            self._persist_rollout(rollout)
            
            logger.info(f"Resumed rollout {rollout_id}")
            return True
