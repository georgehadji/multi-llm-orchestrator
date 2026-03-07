"""
CQRS Projections (Read Models)
==============================

Pre-computed, read-optimized views of data for fast queries.
These subscribe to domain events and update their state accordingly.

Usage:
    from orchestrator.projections import ModelPerformanceProjection
    from orchestrator.events import get_event_bus
    
    projection = ModelPerformanceProjection(get_event_bus())
    
    # Query the read model
    score = projection.get_model_score("gpt-4o", "code_gen")
    leaderboard = projection.get_leaderboard()
"""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from collections import defaultdict
import sqlite3

from .events import (
    EventBus, DomainEvent,
    TaskCompletedEvent, TaskFailedEvent, ModelSelectedEvent,
    ProductionOutcomeRecordedEvent, BudgetWarningEvent,
    CircuitBreakerTrippedEvent
)
from .models import Model, TaskType

logger = logging.getLogger("orchestrator.projections")


# ═══════════════════════════════════════════════════════════════════════════════
# Projection Base Classes
# ═══════════════════════════════════════════════════════════════════════════════

class Projection(ABC):
    """
    Base class for CQRS projections (read models).
    
    Projections subscribe to events and maintain read-optimized state.
    They can be rebuilt from the event store at any time.
    """
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self._subscriptions: List[Callable[[], None]] = []
        self._is_building = False
    
    def subscribe_to(self, event_type: str) -> None:
        """Subscribe to an event type."""
        handler_name = f'on_{event_type.replace(".", "_")}'
        handler = getattr(self, handler_name, None)
        
        if handler:
            unsub = self.event_bus.subscribe(event_type, handler)
            self._subscriptions.append(unsub)
            logger.debug(f"{self.__class__.__name__} subscribed to {event_type}")
        else:
            logger.warning(f"{self.__class__.__name__} has no handler for {event_type}")
    
    def unsubscribe_all(self) -> None:
        """Unsubscribe from all events."""
        for unsub in self._subscriptions:
            unsub()
        self._subscriptions.clear()
    
    @abstractmethod
    async def rebuild(self) -> None:
        """Rebuild the projection from event history."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all projection state."""
        pass


class PersistentProjection(Projection):
    """Projection that persists state to disk."""
    
    def __init__(self, event_bus: EventBus, storage_path: Optional[Path] = None):
        super().__init__(event_bus)
        self.storage_path = storage_path or Path(".projections")
        self.storage_path.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def save(self) -> None:
        """Save projection state to disk."""
        pass
    
    @abstractmethod
    def load(self) -> None:
        """Load projection state from disk."""
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# Model Performance Projection
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ModelPerformanceStats:
    """Statistics for a model on a specific task type."""
    model: str
    task_type: str
    
    # Counters
    total_calls: int = 0
    success_count: int = 0
    failure_count: int = 0
    
    # EMA scores (0.0 - 1.0)
    quality_score_ema: float = 0.5
    success_rate_ema: float = 0.5
    production_score_ema: float = 0.5
    
    # Cost tracking
    total_cost_usd: float = 0.0
    avg_latency_ms: float = 0.0
    
    # Metadata
    last_used: Optional[datetime] = None
    circuit_breaker_tripped: int = 0
    
    def update_success(self, score: float, cost: float, latency_ms: float) -> None:
        """Update stats with a successful task."""
        self.total_calls += 1
        self.success_count += 1
        self.total_cost_usd += cost
        
        # Update EMAs
        alpha = 0.1
        self.quality_score_ema = (1 - alpha) * self.quality_score_ema + alpha * score
        self.success_rate_ema = (1 - alpha) * self.success_rate_ema + alpha * 1.0
        
        # Update latency EMA
        if self.avg_latency_ms == 0:
            self.avg_latency_ms = latency_ms
        else:
            self.avg_latency_ms = (1 - alpha) * self.avg_latency_ms + alpha * latency_ms
        
        self.last_used = datetime.utcnow()
    
    def update_failure(self) -> None:
        """Update stats with a failed task."""
        self.total_calls += 1
        self.failure_count += 1
        
        alpha = 0.1
        self.success_rate_ema = (1 - alpha) * self.success_rate_ema + alpha * 0.0
        self.last_used = datetime.utcnow()
    
    def update_production_score(self, score: float) -> None:
        """Update with production feedback score."""
        alpha = 0.2  # Weight production data more heavily
        self.production_score_ema = (1 - alpha) * self.production_score_ema + alpha * score
    
    @property
    def composite_score(self) -> float:
        """Calculate composite score for ranking."""
        # Weight: Quality 30%, Success Rate 30%, Production 40%
        return (
            0.3 * self.quality_score_ema +
            0.3 * self.success_rate_ema +
            0.4 * self.production_score_ema
        )
    
    @property
    def efficiency_score(self) -> float:
        """Calculate cost efficiency (quality per dollar)."""
        if self.total_cost_usd <= 0:
            return self.quality_score_ema * 100
        return (self.quality_score_ema * 100) / self.total_cost_usd
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "task_type": self.task_type,
            "total_calls": self.total_calls,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "quality_score": round(self.quality_score_ema, 3),
            "success_rate": round(self.success_rate_ema, 3),
            "production_score": round(self.production_score_ema, 3),
            "composite_score": round(self.composite_score, 3),
            "total_cost": round(self.total_cost_usd, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
        }


class ModelPerformanceProjection(PersistentProjection):
    """
    Projection that maintains model performance statistics.
    
    This is a CQRS read model optimized for fast queries about model performance.
    """
    
    def __init__(
        self,
        event_bus: EventBus,
        storage_path: Optional[Path] = None,
    ):
        super().__init__(event_bus, storage_path)
        
        # In-memory state: (model, task_type) -> stats
        self._stats: Dict[Tuple[str, str], ModelPerformanceStats] = {}
        
        # Subscribe to events
        self._subscribe_to_events()
        
        # Try to load persisted state
        self.load()
    
    def _subscribe_to_events(self) -> None:
        """Subscribe to relevant events."""
        self.subscribe_to("task.completed")
        self.subscribe_to("task.failed")
        self.subscribe_to("production.outcome_recorded")
        self.subscribe_to("circuit_breaker.tripped")
    
    def _get_or_create_stats(
        self,
        model: str,
        task_type: str,
    ) -> ModelPerformanceStats:
        """Get or create stats for a model/task_type pair."""
        key = (model, task_type)
        if key not in self._stats:
            self._stats[key] = ModelPerformanceStats(
                model=model,
                task_type=task_type,
            )
        return self._stats[key]
    
    # Event Handlers
    
    async def on_task_completed(self, event: TaskCompletedEvent) -> None:
        """Handle task completion."""
        # Determine task type from event metadata if available
        task_type = event.metadata.get("task_type", "code_gen")
        
        stats = self._get_or_create_stats(event.model, task_type)
        stats.update_success(
            score=event.score,
            cost=event.cost_usd,
            latency_ms=event.latency_ms,
        )
        
        # Persist periodically
        if stats.total_calls % 10 == 0:
            self.save()
    
    async def on_task_failed(self, event: TaskFailedEvent) -> None:
        """Handle task failure."""
        task_type = event.metadata.get("task_type", "code_gen")
        
        stats = self._get_or_create_stats(event.model, task_type)
        stats.update_failure()
    
    async def on_production_outcome_recorded(self, event: ProductionOutcomeRecordedEvent) -> None:
        """Handle production feedback."""
        # Map status to score
        status_scores = {
            "success": 1.0,
            "partial": 0.6,
            "failure": 0.0,
        }
        score = status_scores.get(event.status, 0.5)
        
        # Try to find matching stats
        for key, stats in self._stats.items():
            if stats.model == event.model:
                stats.update_production_score(score)
    
    async def on_circuit_breaker_tripped(self, event: CircuitBreakerTrippedEvent) -> None:
        """Handle circuit breaker trip."""
        for key, stats in self._stats.items():
            if stats.model == event.model:
                stats.circuit_breaker_tripped += 1
    
    # Query Methods
    
    def get_model_score(
        self,
        model: str,
        task_type: str,
        include_production: bool = True,
    ) -> float:
        """
        Get performance score for a model on a task type.
        
        Returns 0.0-1.0 score. Higher is better.
        """
        key = (model, task_type)
        stats = self._stats.get(key)
        
        if not stats or stats.total_calls == 0:
            return 0.5  # Unknown - neutral
        
        if include_production:
            return stats.composite_score
        else:
            return (stats.quality_score_ema + stats.success_rate_ema) / 2
    
    def get_model_stats(
        self,
        model: str,
        task_type: Optional[str] = None,
    ) -> Optional[ModelPerformanceStats]:
        """Get full stats for a model."""
        if task_type:
            return self._stats.get((model, task_type))
        
        # Aggregate across all task types
        model_stats = [s for k, s in self._stats.items() if k[0] == model]
        if not model_stats:
            return None
        
        # Return aggregate
        return ModelPerformanceStats(
            model=model,
            task_type="aggregate",
            total_calls=sum(s.total_calls for s in model_stats),
            success_count=sum(s.success_count for s in model_stats),
            failure_count=sum(s.failure_count for s in model_stats),
            quality_score_ema=sum(s.quality_score_ema for s in model_stats) / len(model_stats),
            success_rate_ema=sum(s.success_rate_ema for s in model_stats) / len(model_stats),
            production_score_ema=sum(s.production_score_ema for s in model_stats) / len(model_stats),
            total_cost_usd=sum(s.total_cost_usd for s in model_stats),
            avg_latency_ms=sum(s.avg_latency_ms for s in model_stats) / len(model_stats),
        )
    
    def get_leaderboard(
        self,
        task_type: Optional[str] = None,
        min_calls: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Get model leaderboard sorted by composite score.
        
        Args:
            task_type: Filter by task type, or None for all
            min_calls: Minimum number of calls to be ranked
        """
        # Filter stats
        if task_type:
            stats_list = [
                s for (m, tt), s in self._stats.items()
                if tt == task_type and s.total_calls >= min_calls
            ]
        else:
            # Aggregate by model
            model_scores: Dict[str, List[ModelPerformanceStats]] = defaultdict(list)
            for (model, tt), stats in self._stats.items():
                if stats.total_calls >= min_calls:
                    model_scores[model].append(stats)
            
            stats_list = []
            for model, model_stats in model_scores.items():
                aggregated = ModelPerformanceStats(
                    model=model,
                    task_type="aggregate",
                    total_calls=sum(s.total_calls for s in model_stats),
                    success_count=sum(s.success_count for s in model_stats),
                    quality_score_ema=sum(s.quality_score_ema for s in model_stats) / len(model_stats),
                    success_rate_ema=sum(s.success_rate_ema for s in model_stats) / len(model_stats),
                    production_score_ema=sum(s.production_score_ema for s in model_stats) / len(model_stats),
                )
                stats_list.append(aggregated)
        
        # Sort by composite score
        stats_list.sort(key=lambda s: s.composite_score, reverse=True)
        
        # Add rank and convert to dict
        result = []
        for rank, stats in enumerate(stats_list, 1):
            data = stats.to_dict()
            data["rank"] = rank
            result.append(data)
        
        return result
    
    def get_best_model_for_task(
        self,
        task_type: str,
        strategy: str = "quality",
    ) -> Optional[str]:
        """
        Get the best model for a specific task type.
        
        Args:
            task_type: The task type
            strategy: "quality", "cost", "balanced", or "production"
        """
        candidates = [
            s for (m, tt), s in self._stats.items()
            if tt == task_type and s.total_calls >= 3
        ]
        
        if not candidates:
            return None
        
        if strategy == "quality":
            candidates.sort(key=lambda s: s.quality_score_ema, reverse=True)
        elif strategy == "cost":
            candidates.sort(key=lambda s: s.efficiency_score, reverse=True)
        elif strategy == "production":
            candidates.sort(key=lambda s: s.production_score_ema, reverse=True)
        else:  # balanced
            candidates.sort(key=lambda s: s.composite_score, reverse=True)
        
        return candidates[0].model
    
    # Persistence
    
    def save(self) -> None:
        """Save projection state to SQLite."""
        db_path = self.storage_path / "model_performance.db"
        
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_stats (
                    model TEXT,
                    task_type TEXT,
                    data TEXT,
                    updated_at TEXT,
                    PRIMARY KEY (model, task_type)
                )
            """)
            
            now = datetime.utcnow().isoformat()
            for key, stats in self._stats.items():
                conn.execute(
                    """
                    INSERT OR REPLACE INTO model_stats (model, task_type, data, updated_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (stats.model, stats.task_type, json.dumps(stats.to_dict()), now)
                )
            
            conn.commit()
    
    def load(self) -> None:
        """Load projection state from SQLite."""
        db_path = self.storage_path / "model_performance.db"
        
        if not db_path.exists():
            return
        
        try:
            with sqlite3.connect(str(db_path)) as conn:
                cursor = conn.execute("SELECT model, task_type, data FROM model_stats")
                
                for row in cursor.fetchall():
                    model, task_type, data_json = row
                    data = json.loads(data_json)
                    
                    stats = ModelPerformanceStats(
                        model=model,
                        task_type=task_type,
                        total_calls=data.get("total_calls", 0),
                        success_count=data.get("success_count", 0),
                        failure_count=data.get("failure_count", 0),
                        quality_score_ema=data.get("quality_score", 0.5),
                        success_rate_ema=data.get("success_rate", 0.5),
                        production_score_ema=data.get("production_score", 0.5),
                        total_cost_usd=data.get("total_cost", 0.0),
                        avg_latency_ms=data.get("avg_latency_ms", 0.0),
                    )
                    
                    self._stats[(model, task_type)] = stats
            
            logger.info(f"Loaded {len(self._stats)} model stats from disk")
            
        except Exception as e:
            logger.error(f"Failed to load projection state: {e}")
    
    def clear(self) -> None:
        """Clear all projection state."""
        self._stats.clear()
    
    async def rebuild(self) -> None:
        """Rebuild projection from event history."""
        logger.info("Rebuilding ModelPerformanceProjection...")
        
        self.clear()
        self.unsubscribe_all()
        
        await self.event_bus.replay(
            event_types=[
                "task.completed",
                "task.failed",
                "production.outcome_recorded",
            ],
            handler_filter=lambda name: name.startswith("on_"),
        )
        
        self._subscribe_to_events()
        self.save()
        
        logger.info(f"Rebuild complete. {len(self._stats)} stats loaded.")


# ═══════════════════════════════════════════════════════════════════════════════
# Budget Projection
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BudgetSnapshot:
    """Budget state at a point in time."""
    project_id: str
    total_budget: float
    spent: float
    remaining: float
    phase_breakdown: Dict[str, float]
    timestamp: datetime


class BudgetProjection(Projection):
    """Projection for real-time budget tracking."""
    
    def __init__(self, event_bus: EventBus):
        super().__init__(event_bus)
        self._budgets: Dict[str, BudgetSnapshot] = {}
        self._alerts: List[Dict] = []
        
        self.subscribe_to("budget.warning")
        self.subscribe_to("project.started")
        self.subscribe_to("project.completed")
    
    async def on_budget_warning(self, event: BudgetWarningEvent) -> None:
        """Track budget warnings."""
        self._alerts.append({
            "project_id": event.project_id,
            "phase": event.phase,
            "ratio": event.ratio,
            "timestamp": datetime.utcnow(),
        })
    
    def get_budget_status(self, project_id: str) -> Optional[BudgetSnapshot]:
        """Get current budget status for a project."""
        return self._budgets.get(project_id)
    
    def get_alert_history(
        self,
        project_id: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> List[Dict]:
        """Get budget alert history."""
        alerts = self._alerts
        
        if project_id:
            alerts = [a for a in alerts if a["project_id"] == project_id]
        
        if since:
            alerts = [a for a in alerts if a["timestamp"] >= since]
        
        return alerts
    
    def clear(self) -> None:
        self._budgets.clear()
        self._alerts.clear()
    
    async def rebuild(self) -> None:
        # Budget projection is ephemeral, no need to persist
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# Global Projections
# ═══════════════════════════════════════════════════════════════════════════════

_projections: Dict[str, Projection] = {}


def get_model_performance_projection(
    event_bus: Optional[EventBus] = None,
) -> ModelPerformanceProjection:
    """Get or create global model performance projection."""
    global _projections
    
    if "model_performance" not in _projections:
        from .events import get_event_bus
        bus = event_bus or get_event_bus()
        _projections["model_performance"] = ModelPerformanceProjection(bus)
    
    return _projections["model_performance"]


def get_budget_projection(
    event_bus: Optional[EventBus] = None,
) -> BudgetProjection:
    """Get or create global budget projection."""
    global _projections
    
    if "budget" not in _projections:
        from .events import get_event_bus
        bus = event_bus or get_event_bus()
        _projections["budget"] = BudgetProjection(bus)
    
    return _projections["budget"]


def reset_projections() -> None:
    """Reset all projections (for testing)."""
    global _projections
    for projection in _projections.values():
        projection.unsubscribe_all()
    _projections.clear()
