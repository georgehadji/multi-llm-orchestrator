"""
Meta-Orchestrator — Self-optimizing strategy layer for AI Orchestrator
=======================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Inspired by Hyperagents (arXiv:2603.19461), this module adds a meta-optimization
layer that analyzes execution history and proposes improvements to:
- Model routing strategies
- Budget allocation
- Task decomposition patterns
- Template configurations

KEY CONCEPTS:
- ExecutionArchive: Stores successful execution trajectories
- StrategyProposal: Typed proposals for system improvements
- MetaOptimizer: Analyzes patterns and generates proposals
- StagedEvaluation: Fast filtering before full strategy adoption

USAGE:
    from orchestrator.meta_orchestrator import MetaOptimizer, ExecutionArchive
    
    archive = ExecutionArchive()
    optimizer = MetaOptimizer(archive)
    
    # After each project execution
    archive.store_execution(trajectory)
    
    # Periodically optimize
    proposals = await optimizer.analyze_and_propose()
    for proposal in proposals:
        if await optimizer.evaluate_proposal(proposal):
            await optimizer.apply_proposal(proposal)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
from collections import defaultdict

from .models import Model, TaskType, Budget
from .adaptive_router import AdaptiveRouter

logger = logging.getLogger("orchestrator.meta")


# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────

class StrategyType(str, Enum):
    """Types of optimization strategies."""
    MODEL_ROUTING = "model_routing"
    BUDGET_ALLOCATION = "budget_allocation"
    TASK_DECOMPOSITION = "task_decomposition"
    TEMPLATE_CONFIG = "template_config"
    CONCURRENT_EXECUTION = "concurrent_execution"


class ProposalStatus(str, Enum):
    """Status of a strategy proposal."""
    PENDING = "pending"
    EVALUATING = "evaluating"
    APPROVED = "approved"
    REJECTED = "rejected"
    APPLIED = "applied"
    ROLLED_BACK = "rolled_back"


@dataclass
class ExecutionRecord:
    """Record of a single task execution."""
    task_id: str
    task_type: str
    model_used: str
    success: bool
    cost_usd: float
    latency_ms: float
    input_tokens: int
    output_tokens: int
    score: float  # Quality score 0-1
    timestamp: float = field(default_factory=time.time)
    error_message: Optional[str] = None
    project_id: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "model_used": self.model_used,
            "success": self.success,
            "cost_usd": self.cost_usd,
            "latency_ms": self.latency_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "score": self.score,
            "timestamp": self.timestamp,
            "error_message": self.error_message,
            "project_id": self.project_id,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> ExecutionRecord:
        return cls(**data)


@dataclass
class ProjectTrajectory:
    """Complete execution trajectory for a project."""
    project_id: str
    project_description: str
    total_cost: float
    total_time: float
    success: bool
    task_records: List[ExecutionRecord]
    model_sequence: List[str]
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> dict:
        return {
            "project_id": self.project_id,
            "project_description": self.project_description,
            "total_cost": self.total_cost,
            "total_time": self.total_time,
            "success": self.success,
            "task_records": [r.to_dict() for r in self.task_records],
            "model_sequence": self.model_sequence,
            "timestamp": self.timestamp,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> ProjectTrajectory:
        return cls(
            project_id=data["project_id"],
            project_description=data["project_description"],
            total_cost=data["total_cost"],
            total_time=data["total_time"],
            success=data["success"],
            task_records=[ExecutionRecord.from_dict(r) for r in data["task_records"]],
            model_sequence=data["model_sequence"],
            timestamp=data.get("timestamp", time.time()),
        )


@dataclass
class StrategyProposal:
    """A proposal for system improvement."""
    proposal_id: str
    strategy_type: StrategyType
    description: str
    current_config: Dict[str, Any]
    proposed_config: Dict[str, Any]
    expected_improvement: float  # Expected % improvement
    confidence: float  # 0-1 confidence in proposal
    evidence: List[str]  # Supporting evidence from archive
    status: ProposalStatus = ProposalStatus.PENDING
    created_at: float = field(default_factory=time.time)
    evaluated_at: Optional[float] = None
    applied_at: Optional[float] = None
    
    def to_dict(self) -> dict:
        return {
            "proposal_id": self.proposal_id,
            "strategy_type": self.strategy_type.value,
            "description": self.description,
            "current_config": self.current_config,
            "proposed_config": self.proposed_config,
            "expected_improvement": self.expected_improvement,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "status": self.status.value,
            "created_at": self.created_at,
            "evaluated_at": self.evaluated_at,
            "applied_at": self.applied_at,
        }


# ─────────────────────────────────────────────
# Execution Archive
# ─────────────────────────────────────────────

class ExecutionArchive:
    """
    Archive of execution trajectories for pattern mining.
    
    Stores successful and failed executions for analysis.
    Supports similarity-based retrieval and pattern extraction.
    """
    
    def __init__(self, archive_path: Optional[Path] = None):
        if archive_path is None:
            archive_path = Path.home() / ".orchestrator_cache" / "archive"
        archive_path.mkdir(parents=True, exist_ok=True)
        
        self._archive_path = archive_path
        self._trajectories: Dict[str, ProjectTrajectory] = {}
        self._records: List[ExecutionRecord] = []
        self._model_stats: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"success": 0, "total": 0, "cost": 0, "score": 0}
        )
        self._task_type_stats: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"success": 0, "total": 0, "cost": 0, "score": 0}
        )
        self._load_archive()
    
    def _load_archive(self):
        """Load archive from disk."""
        archive_file = self._archive_path / "archive.jsonl"
        if not archive_file.exists():
            return
        
        try:
            with open(archive_file, "r") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        trajectory = ProjectTrajectory.from_dict(data)
                        self._trajectories[trajectory.project_id] = trajectory
                        self._records.extend(trajectory.task_records)
            
            self._rebuild_stats()
            logger.info(f"Loaded {len(self._trajectories)} projects from archive")
        except Exception as e:
            logger.warning(f"Failed to load archive: {e}")
    
    def _rebuild_stats(self):
        """Rebuild statistics from loaded records."""
        self._model_stats.clear()
        self._task_type_stats.clear()
        
        for record in self._records:
            # Model stats
            model = record.model_used
            self._model_stats[model]["total"] += 1
            if record.success:
                self._model_stats[model]["success"] += 1
            self._model_stats[model]["cost"] += record.cost_usd
            self._model_stats[model]["score"] += record.score
            
            # Task type stats
            task_type = record.task_type
            self._task_type_stats[task_type]["total"] += 1
            if record.success:
                self._task_type_stats[task_type]["success"] += 1
            self._task_type_stats[task_type]["cost"] += record.cost_usd
            self._task_type_stats[task_type]["score"] += record.score
    
    def store(self, trajectory: ProjectTrajectory):
        """Store a project trajectory in the archive."""
        self._trajectories[trajectory.project_id] = trajectory
        self._records.extend(trajectory.task_records)
        self._update_stats(trajectory)
        self._persist(trajectory)
        logger.debug(f"Stored trajectory for project {trajectory.project_id}")
    
    def _update_stats(self, trajectory: ProjectTrajectory):
        """Update statistics with new trajectory."""
        for record in trajectory.task_records:
            # Model stats
            model = record.model_used
            self._model_stats[model]["total"] += 1
            if record.success:
                self._model_stats[model]["success"] += 1
            self._model_stats[model]["cost"] += record.cost_usd
            self._model_stats[model]["score"] += record.score
            
            # Task type stats
            task_type = record.task_type
            self._task_type_stats[task_type]["total"] += 1
            if record.success:
                self._task_type_stats[task_type]["success"] += 1
            self._task_type_stats[task_type]["cost"] += record.cost_usd
            self._task_type_stats[task_type]["score"] += record.score
    
    def _persist(self, trajectory: ProjectTrajectory):
        """Persist trajectory to disk."""
        archive_file = self._archive_path / "archive.jsonl"
        try:
            with open(archive_file, "a") as f:
                f.write(json.dumps(trajectory.to_dict()) + "\n")
        except Exception as e:
            logger.warning(f"Failed to persist trajectory: {e}")
    
    def get_model_performance(self, model: str) -> Dict[str, float]:
        """Get performance statistics for a model."""
        stats = self._model_stats.get(model, {"success": 0, "total": 0, "cost": 0, "score": 0})
        total = max(stats["total"], 1)
        return {
            "success_rate": stats["success"] / total,
            "avg_cost": stats["cost"] / total,
            "avg_score": stats["score"] / total,
            "total_executions": stats["total"],
        }
    
    def get_task_type_performance(self, task_type: str) -> Dict[str, float]:
        """Get performance statistics for a task type."""
        stats = self._task_type_stats.get(task_type, {"success": 0, "total": 0, "cost": 0, "score": 0})
        total = max(stats["total"], 1)
        return {
            "success_rate": stats["success"] / total,
            "avg_cost": stats["cost"] / total,
            "avg_score": stats["score"] / total,
            "total_executions": stats["total"],
        }
    
    def get_best_model_for_task_type(self, task_type: str, min_samples: int = 5) -> Optional[str]:
        """Find the best performing model for a specific task type."""
        task_records = [r for r in self._records if r.task_type == task_type]
        if len(task_records) < min_samples:
            return None
        
        model_scores: Dict[str, List[float]] = defaultdict(list)
        for record in task_records:
            if record.success:
                # Score combines success, quality, and cost efficiency
                score = record.score * (1.0 / (1.0 + record.cost_usd))
                model_scores[record.model_used].append(score)
        
        if not model_scores:
            return None
        
        # Return model with highest average score
        return max(model_scores.keys(), key=lambda m: sum(model_scores[m]) / len(model_scores[m]))
    
    def find_similar_projects(self, description: str, limit: int = 5) -> List[ProjectTrajectory]:
        """Find projects with similar descriptions (simple keyword matching)."""
        keywords = set(description.lower().split())
        
        scored = []
        for proj in self._trajectories.values():
            proj_keywords = set(proj.project_description.lower().split())
            overlap = len(keywords & proj_keywords)
            if overlap > 0:
                scored.append((overlap, proj))
        
        scored.sort(reverse=True, key=lambda x: x[0])
        return [proj for _, proj in scored[:limit]]
    
    def get_patterns(self) -> Dict[str, Any]:
        """Extract patterns from archive for meta-optimization."""
        patterns = {
            "model_task_affinity": {},  # Which models work best for which tasks
            "cost_anomalies": [],  # Unusually high/low cost executions
            "failure_patterns": [],  # Common failure modes
            "success_patterns": [],  # Common success factors
        }
        
        # Model-task affinity
        for task_type in self._task_type_stats.keys():
            best_model = self.get_best_model_for_task_type(task_type)
            if best_model:
                patterns["model_task_affinity"][task_type] = best_model
        
        # Cost anomalies (executions > 2x average cost)
        avg_cost = sum(r.cost_usd for r in self._records) / max(len(self._records), 1)
        for record in self._records:
            if record.cost_usd > avg_cost * 2:
                patterns["cost_anomalies"].append({
                    "task_id": record.task_id,
                    "model": record.model_used,
                    "cost": record.cost_usd,
                    "avg": avg_cost,
                })
        
        # Failure patterns
        failures_by_model: Dict[str, int] = defaultdict(int)
        for record in self._records:
            if not record.success:
                failures_by_model[record.model_used] += 1
        
        for model, count in failures_by_model.items():
            total = self._model_stats[model]["total"]
            if total >= 5 and count / total > 0.3:  # >30% failure rate
                patterns["failure_patterns"].append({
                    "model": model,
                    "failure_rate": count / total,
                    "total": total,
                })
        
        return patterns
    
    @property
    def total_projects(self) -> int:
        return len(self._trajectories)
    
    @property
    def total_executions(self) -> int:
        return len(self._records)


# ─────────────────────────────────────────────
# Meta-Optimizer
# ─────────────────────────────────────────────

class MetaOptimizer:
    """
    Meta-optimization engine for AI Orchestrator.
    
    Analyzes execution archive to propose and evaluate strategy improvements.
    """
    
    def __init__(
        self,
        archive: ExecutionArchive,
        min_samples: int = 10,
        improvement_threshold: float = 0.05,  # 5% improvement required
    ):
        self.archive = archive
        self.min_samples = min_samples
        self.improvement_threshold = improvement_threshold
        self._pending_proposals: Dict[str, StrategyProposal] = {}
        self._applied_proposals: List[StrategyProposal] = []
    
    async def analyze_and_propose(self) -> List[StrategyProposal]:
        """Analyze archive and generate improvement proposals."""
        proposals = []
        
        if self.archive.total_executions < self.min_samples:
            logger.info(f"Insufficient data for optimization ({self.archive.total_executions} < {self.min_samples})")
            return proposals
        
        # Generate proposals for each strategy type
        proposals.extend(self._propose_routing_optimizations())
        proposals.extend(self._propose_budget_optimizations())
        proposals.extend(self._propose_template_optimizations())
        
        # Store pending proposals
        for proposal in proposals:
            self._pending_proposals[proposal.proposal_id] = proposal
        
        logger.info(f"Generated {len(proposals)} optimization proposals")
        return proposals
    
    def _propose_routing_optimizations(self) -> List[StrategyProposal]:
        """Propose model routing improvements."""
        proposals = []
        patterns = self.archive.get_patterns()
        
        # Propose routing changes based on model-task affinity
        for task_type, best_model in patterns.get("model_task_affinity", {}).items():
            proposal = StrategyProposal(
                proposal_id=f"route_{task_type}_{int(time.time())}",
                strategy_type=StrategyType.MODEL_ROUTING,
                description=f"Route {task_type} tasks to {best_model} (best performing)",
                current_config={"task_type": task_type, "routing": "adaptive"},
                proposed_config={"task_type": task_type, "routing": "fixed", "model": best_model},
                expected_improvement=0.10,  # 10% expected improvement
                confidence=0.8,
                evidence=[f"Best model for {task_type} based on {self.archive.total_executions} executions"],
            )
            proposals.append(proposal)
        
        # Propose removing underperforming models
        for failure in patterns.get("failure_patterns", []):
            if failure["failure_rate"] > 0.5:  # >50% failure rate
                proposal = StrategyProposal(
                    proposal_id=f"disable_{failure['model']}_{int(time.time())}",
                    strategy_type=StrategyType.MODEL_ROUTING,
                    description=f"Temporarily disable {failure['model']} (high failure rate)",
                    current_config={"model": failure["model"], "enabled": True},
                    proposed_config={"model": failure["model"], "enabled": False},
                    expected_improvement=0.15,
                    confidence=0.9,
                    evidence=[f"Failure rate: {failure['failure_rate']:.2%} over {failure['total']} executions"],
                )
                proposals.append(proposal)
        
        return proposals
    
    def _propose_budget_optimizations(self) -> List[StrategyProposal]:
        """Propose budget allocation improvements."""
        proposals = []
        
        # Analyze cost patterns by task type
        task_costs: Dict[str, List[float]] = defaultdict(list)
        for record in self.archive._records:
            task_costs[record.task_type].append(record.cost_usd)
        
        # Propose budget adjustments for over/under-budgeted task types
        for task_type, costs in task_costs.items():
            if len(costs) < 5:
                continue
            
            avg_cost = sum(costs) / len(costs)
            p90_cost = sorted(costs)[int(len(costs) * 0.9)]
            
            # If p90 is 2x average, suggest higher budget allocation
            if p90_cost > avg_cost * 2:
                proposal = StrategyProposal(
                    proposal_id=f"budget_{task_type}_{int(time.time())}",
                    strategy_type=StrategyType.BUDGET_ALLOCATION,
                    description=f"Increase budget allocation for {task_type} (high variance)",
                    current_config={"task_type": task_type, "budget_factor": 1.0},
                    proposed_config={"task_type": task_type, "budget_factor": 1.5},
                    expected_improvement=0.05,
                    confidence=0.7,
                    evidence=[f"P90 cost ({p90_cost:.3f}) is {p90_cost/avg_cost:.1f}x average ({avg_cost:.3f})"],
                )
                proposals.append(proposal)
        
        return proposals
    
    def _propose_template_optimizations(self) -> List[StrategyProposal]:
        """Propose task template improvements."""
        proposals = []
        patterns = self.archive.get_patterns()
        
        # Analyze success patterns for template hints
        success_by_type: Dict[str, List[ExecutionRecord]] = defaultdict(list)
        for record in self.archive._records:
            if record.success and record.score > 0.9:
                success_by_type[record.task_type].append(record)
        
        # Propose template adjustments for task types with consistent high performers
        for task_type, records in success_by_type.items():
            if len(records) >= 10:
                # Analyze common characteristics of successful executions
                models_used = [r.model_used for r in records]
                most_common_model = max(set(models_used), key=models_used.count)
                
                proposal = StrategyProposal(
                    proposal_id=f"template_{task_type}_{int(time.time())}",
                    strategy_type=StrategyType.TEMPLATE_CONFIG,
                    description=f"Update {task_type} template defaults based on successful patterns",
                    current_config={"task_type": task_type, "default_model": "auto"},
                    proposed_config={"task_type": task_type, "default_model": most_common_model},
                    expected_improvement=0.08,
                    confidence=0.75,
                    evidence=[f"{len(records)} high-quality executions favor {most_common_model}"],
                )
                proposals.append(proposal)
        
        return proposals
    
    async def evaluate_proposal(self, proposal: StrategyProposal) -> bool:
        """
        Evaluate a proposal using staged evaluation.
        
        Stage 1: Fast simulation based on historical data
        Stage 2: A/B test on live executions (future enhancement)
        """
        proposal.status = ProposalStatus.EVALUATING
        
        # Stage 1: Fast simulation
        simulated_improvement = self._simulate_proposal(proposal)
        
        if simulated_improvement >= self.improvement_threshold:
            proposal.status = ProposalStatus.APPROVED
            proposal.evaluated_at = time.time()
            logger.info(f"Proposal {proposal.proposal_id} approved (simulated improvement: {simulated_improvement:.2%})")
            return True
        else:
            proposal.status = ProposalStatus.REJECTED
            proposal.evaluated_at = time.time()
            logger.info(f"Proposal {proposal.proposal_id} rejected (simulated improvement: {simulated_improvement:.2%} < threshold)")
            return False
    
    def _simulate_proposal(self, proposal: StrategyProposal) -> float:
        """Simulate proposal impact using historical data."""
        if proposal.strategy_type == StrategyType.MODEL_ROUTING:
            return self._simulate_routing_change(proposal)
        elif proposal.strategy_type == StrategyType.BUDGET_ALLOCATION:
            return self._simulate_budget_change(proposal)
        else:
            return 0.0  # No simulation for other types yet
    
    def _simulate_routing_change(self, proposal: StrategyProposal) -> float:
        """Simulate impact of routing change."""
        task_type = proposal.proposed_config.get("task_type")
        if not task_type:
            return 0.0
        
        # Get historical performance with current vs proposed routing
        records = [r for r in self.archive._records if r.task_type == task_type]
        if len(records) < 5:
            return 0.0
        
        # Calculate what would have happened with proposed routing
        proposed_model = proposal.proposed_config.get("model")
        if proposed_model:
            proposed_records = [r for r in records if r.model_used == proposed_model]
            if len(proposed_records) < 3:
                return 0.0
            
            current_avg_score = sum(r.score for r in records) / len(records)
            proposed_avg_score = sum(r.score for r in proposed_records) / len(proposed_records)
            
            return (proposed_avg_score - current_avg_score) / max(current_avg_score, 0.01)
        
        return 0.0
    
    def _simulate_budget_change(self, proposal: StrategyProposal) -> float:
        """Simulate impact of budget change."""
        # Simplified: assume higher budget reduces failures proportionally
        budget_factor = proposal.proposed_config.get("budget_factor", 1.0)
        current_factor = proposal.current_config.get("budget_factor", 1.0)
        
        # Estimate 5% improvement per 50% budget increase (diminishing returns)
        increase = (budget_factor - current_factor) / current_factor
        return min(0.05 * increase * 2, 0.15)  # Cap at 15%
    
    async def apply_proposal(self, proposal: StrategyProposal) -> bool:
        """Apply an approved proposal."""
        if proposal.status != ProposalStatus.APPROVED:
            logger.warning(f"Cannot apply proposal {proposal.proposal_id}: status={proposal.status.value}")
            return False
        
        # Apply based on strategy type
        success = await self._apply_proposal_impl(proposal)
        
        if success:
            proposal.status = ProposalStatus.APPLIED
            proposal.applied_at = time.time()
            self._applied_proposals.append(proposal)
            logger.info(f"Applied proposal {proposal.proposal_id}")
        else:
            proposal.status = ProposalStatus.ROLLED_BACK
            logger.warning(f"Failed to apply proposal {proposal.proposal_id}")
        
        return success
    
    async def _apply_proposal_impl(self, proposal: StrategyProposal) -> bool:
        """Implementation of proposal application."""
        # For now, proposals are logged and tracked
        # Actual application would modify orchestrator configuration
        
        if proposal.strategy_type == StrategyType.MODEL_ROUTING:
            # Would update routing tables
            logger.info(f"Would update routing: {proposal.description}")
            return True
        
        elif proposal.strategy_type == StrategyType.BUDGET_ALLOCATION:
            # Would update budget partitions
            logger.info(f"Would update budget: {proposal.description}")
            return True
        
        elif proposal.strategy_type == StrategyType.TEMPLATE_CONFIG:
            # Would update template defaults
            logger.info(f"Would update template: {proposal.description}")
            return True
        
        return False
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate optimization report."""
        return {
            "archive_stats": {
                "total_projects": self.archive.total_projects,
                "total_executions": self.archive.total_executions,
            },
            "patterns": self.archive.get_patterns(),
            "pending_proposals": len(self._pending_proposals),
            "applied_proposals": len(self._applied_proposals),
            "model_performance": {
                model: self.archive.get_model_performance(model)
                for model in list(self.archive._model_stats.keys())[:10]  # Top 10
            },
        }


# ─────────────────────────────────────────────
# Integration Hooks
# ─────────────────────────────────────────────

class MetaOptimizationIntegration:
    """
    Integration layer for meta-optimization in the orchestrator.
    
    Use this class to integrate meta-optimization with the main Orchestrator.
    """
    
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.archive = ExecutionArchive()
        self.optimizer = MetaOptimizer(self.archive)
        self._optimization_lock = asyncio.Lock()
    
    async def record_execution(self, trajectory: ProjectTrajectory):
        """Record a project execution for later optimization."""
        self.archive.store(trajectory)
    
    async def maybe_optimize(self) -> List[StrategyProposal]:
        """
        Periodically analyze and propose optimizations.
        
        Call this after every N project completions.
        """
        async with self._optimization_lock:
            # Only optimize if we have enough data
            if self.archive.total_executions < 50:
                return []
            
            # Generate and evaluate proposals
            proposals = await self.optimizer.analyze_and_propose()
            
            approved = []
            for proposal in proposals[:3]:  # Limit to top 3 proposals
                if await self.optimizer.evaluate_proposal(proposal):
                    if await self.optimizer.apply_proposal(proposal):
                        approved.append(proposal)
            
            return approved
    
    def get_status(self) -> Dict[str, Any]:
        """Get meta-optimization status."""
        return self.optimizer.get_optimization_report()
