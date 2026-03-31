"""
Nash Stability Event System
============================

Event-driven architecture για cross-component communication.
Επιτρέπει στα 4 Nash stability features να επικοινωνούν μεταξύ τους
και να ενημερώνονται σε real-time.

Key Features:
- Async event bus with pub/sub pattern
- Typed events with validation
- Event persistence for replay
- Cross-component subscriptions
- Performance monitoring

Usage:
    from orchestrator.nash_events import NashEventBus, event_bus

    # Subscribe to events
    @event_bus.on(EventType.KNOWLEDGE_GRAPH_UPDATED)
    async def handle_update(event):
        print(f"Graph updated: {event.data}")

    # Publish event
    await event_bus.publish(KnowledgeGraphUpdatedEvent(
        nodes_added=5,
        edges_added=10,
    ))
"""

from __future__ import annotations

import asyncio
import inspect
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .log_config import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger(__name__)


class EventType(Enum):
    """Event types for Nash stability system."""

    # Knowledge Graph Events
    KNOWLEDGE_GRAPH_UPDATED = "kg:updated"
    KNOWLEDGE_NODE_ADDED = "kg:node_added"
    KNOWLEDGE_EDGE_ADDED = "kg:edge_added"
    KNOWLEDGE_SIMILARITY_MATCHED = "kg:similarity_matched"

    # Template Events
    TEMPLATE_SELECTED = "template:selected"
    TEMPLATE_RESULT_REPORTED = "template:result_reported"
    TEMPLATE_CONVERGED = "template:converged"

    # Pareto Frontier Events
    FRONTIER_COMPUTED = "frontier:computed"
    PREDICTION_MADE = "frontier:prediction_made"
    DRIFT_DETECTED = "frontier:drift_detected"

    # Federated Learning Events
    INSIGHT_CONTRIBUTED = "federated:insight_contributed"
    BASELINE_UPDATED = "federated:baseline_updated"
    AGGREGATION_COMPLETED = "federated:aggregation_completed"

    # System Events
    AUTO_TUNING_TRIGGERED = "system:auto_tuning"
    BACKUP_CREATED = "system:backup_created"
    BACKUP_RESTORED = "system:backup_restored"

    # Nash Stability Events
    STABILITY_SCORE_UPDATED = "nash:score_updated"
    SWITCHING_COST_CHANGED = "nash:switching_cost_changed"


@dataclass
class NashEvent:
    """Base class for all Nash events."""

    event_type: EventType
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: str = "unknown"
    correlation_id: str | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "correlation_id": self.correlation_id,
            "data": self.data,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NashEvent:
        return cls(
            event_type=EventType(data["event_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            source=data["source"],
            correlation_id=data.get("correlation_id"),
            data=data.get("data", {}),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Specific Event Types
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class KnowledgeGraphUpdatedEvent(NashEvent):
    """Emitted when knowledge graph is updated."""

    nodes_added: int = 0
    edges_added: int = 0
    nodes_total: int = 0
    edges_total: int = 0

    def __post_init__(self):
        if not self.event_type:
            self.event_type = EventType.KNOWLEDGE_GRAPH_UPDATED
        self.data = {
            "nodes_added": self.nodes_added,
            "edges_added": self.edges_added,
            "nodes_total": self.nodes_total,
            "edges_total": self.edges_total,
        }


@dataclass
class TemplateSelectedEvent(NashEvent):
    """Emitted when a template is selected."""

    task_type: str = ""
    model: str = ""
    variant_name: str = ""
    strategy: str = ""
    confidence: float = 0.0

    def __post_init__(self):
        if not self.event_type:
            self.event_type = EventType.TEMPLATE_SELECTED
        self.data = {
            "task_type": self.task_type,
            "model": self.model,
            "variant_name": self.variant_name,
            "strategy": self.strategy,
            "confidence": self.confidence,
        }


@dataclass
class TemplateResultReportedEvent(NashEvent):
    """Emitted when template result is reported."""

    task_type: str = ""
    model: str = ""
    variant_name: str = ""
    score: float = 0.0
    success: bool = True
    ema_score: float = 0.0

    def __post_init__(self):
        if not self.event_type:
            self.event_type = EventType.TEMPLATE_RESULT_REPORTED
        self.data = {
            "task_type": self.task_type,
            "model": self.model,
            "variant_name": self.variant_name,
            "score": self.score,
            "success": self.success,
            "ema_score": self.ema_score,
        }


@dataclass
class FrontierComputedEvent(NashEvent):
    """Emitted when Pareto frontier is computed."""

    task_type: str = ""
    models_considered: int = 0
    pareto_optimal_count: int = 0
    computation_time_ms: float = 0.0

    def __post_init__(self):
        if not self.event_type:
            self.event_type = EventType.FRONTIER_COMPUTED
        self.data = {
            "task_type": self.task_type,
            "models_considered": self.models_considered,
            "pareto_optimal_count": self.pareto_optimal_count,
            "computation_time_ms": self.computation_time_ms,
        }


@dataclass
class DriftDetectedEvent(NashEvent):
    """Emitted when model drift is detected."""

    model: str = ""
    metric: str = ""
    expected_value: float = 0.0
    observed_value: float = 0.0
    p_value: float = 0.0
    severity: str = "warning"  # warning, critical

    def __post_init__(self):
        if not self.event_type:
            self.event_type = EventType.DRIFT_DETECTED
        self.data = {
            "model": self.model,
            "metric": self.metric,
            "expected_value": self.expected_value,
            "observed_value": self.observed_value,
            "p_value": self.p_value,
            "severity": self.severity,
        }


@dataclass
class InsightContributedEvent(NashEvent):
    """Emitted when insight is contributed to federation."""

    model: str = ""
    task_type: str = ""
    success_rate: float = 0.0
    privacy_epsilon_spent: float = 0.0
    global_insight_count: int = 0

    def __post_init__(self):
        if not self.event_type:
            self.event_type = EventType.INSIGHT_CONTRIBUTED
        self.data = {
            "model": self.model,
            "task_type": self.task_type,
            "success_rate": self.success_rate,
            "privacy_epsilon_spent": self.privacy_epsilon_spent,
            "global_insight_count": self.global_insight_count,
        }


@dataclass
class StabilityScoreUpdatedEvent(NashEvent):
    """Emitted when Nash stability score changes."""

    previous_score: float = 0.0
    new_score: float = 0.0
    score_change: float = 0.0
    interpretation: str = ""

    def __post_init__(self):
        if not self.event_type:
            self.event_type = EventType.STABILITY_SCORE_UPDATED
        self.data = {
            "previous_score": self.previous_score,
            "new_score": self.new_score,
            "score_change": self.score_change,
            "interpretation": self.interpretation,
        }


@dataclass
class AutoTuningTriggeredEvent(NashEvent):
    """Emitted when auto-tuning runs."""

    parameter_name: str = ""
    old_value: float = 0.0
    new_value: float = 0.0
    reason: str = ""
    expected_improvement: float = 0.0

    def __post_init__(self):
        if not self.event_type:
            self.event_type = EventType.AUTO_TUNING_TRIGGERED
        self.data = {
            "parameter_name": self.parameter_name,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "reason": self.reason,
            "expected_improvement": self.expected_improvement,
        }


@dataclass
class BackupCreatedEvent(NashEvent):
    """Emitted when backup is created."""

    backup_path: str = ""
    backup_size_bytes: int = 0
    components_backed_up: list[str] = field(default_factory=list)
    checksum: str = ""
    estimated_value_usd: float = 0.0

    def __post_init__(self):
        if not self.event_type:
            self.event_type = EventType.BACKUP_CREATED
        self.data = {
            "backup_path": self.backup_path,
            "backup_size_bytes": self.backup_size_bytes,
            "components_backed_up": self.components_backed_up,
            "checksum": self.checksum,
            "estimated_value_usd": self.estimated_value_usd,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Event Bus
# ═══════════════════════════════════════════════════════════════════════════════


class NashEventBus:
    """
    Async event bus for Nash stability system.

    Features:
    - Pub/sub with type-safe handlers
    - Event persistence for replay
    - Handler error isolation
    - Performance metrics
    """

    def __init__(self, storage_path: Path | None = None, persist_events: bool = True):
        self.storage_path = storage_path or Path(".nash_events")
        self.persist_events = persist_events

        if self.persist_events:
            self.storage_path.mkdir(exist_ok=True)

        # Subscribers: event_type -> list of handlers
        self._subscribers: dict[EventType, list[Callable]] = defaultdict(list)

        # Event history for replay
        self._event_history: list[NashEvent] = []
        self._max_history = 10000

        # Metrics
        self._events_published = 0
        self._events_processed = 0
        self._handler_errors = 0

        # Load persisted events
        if self.persist_events:
            self._load_history()

    def _load_history(self) -> None:
        """Load event history from disk."""
        history_file = self.storage_path / "event_history.jsonl"
        if history_file.exists():
            try:
                with history_file.open("r") as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            event = NashEvent.from_dict(data)
                            self._event_history.append(event)
                logger.info(f"Loaded {len(self._event_history)} events from history")
            except Exception as e:
                logger.error(f"Failed to load event history: {e}")

    def _persist_event(self, event: NashEvent) -> None:
        """Persist event to disk."""
        if not self.persist_events:
            return

        try:
            history_file = self.storage_path / "event_history.jsonl"
            with history_file.open("a") as f:
                f.write(json.dumps(event.to_dict()) + "\n")
        except Exception as e:
            logger.error(f"Failed to persist event: {e}")

    def on(self, event_type: EventType):
        """
        Decorator to subscribe to an event type.

        Usage:
            @event_bus.on(EventType.KNOWLEDGE_GRAPH_UPDATED)
            async def handle_update(event):
                ...
        """

        def decorator(func: Callable):
            self.subscribe(event_type, func)
            return func

        return decorator

    def subscribe(self, event_type: EventType, handler: Callable) -> None:
        """
        Subscribe a handler to an event type.

        Args:
            event_type: Type of event to subscribe to
            handler: Async or sync callable that receives the event
        """
        self._subscribers[event_type].append(handler)
        logger.debug(f"Subscribed {handler.__name__} to {event_type.value}")

    def unsubscribe(self, event_type: EventType, handler: Callable) -> bool:
        """Unsubscribe a handler from an event type."""
        if event_type in self._subscribers:
            try:
                self._subscribers[event_type].remove(handler)
                return True
            except ValueError:
                pass
        return False

    async def publish(self, event: NashEvent) -> None:
        """
        Publish an event to all subscribers.

        Args:
            event: Event to publish
        """
        self._events_published += 1

        # Add to history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            self._event_history.pop(0)

        # Persist
        self._persist_event(event)

        # Notify subscribers
        handlers = self._subscribers.get(event.event_type, [])

        if handlers:
            # Run handlers concurrently with error isolation
            tasks = [self._run_handler(handler, event) for handler in handlers]
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _run_handler(self, handler: Callable, event: NashEvent) -> None:
        """Run a handler with error isolation."""
        try:
            if inspect.iscoroutinefunction(handler):
                await handler(event)
            else:
                handler(event)
            self._events_processed += 1
        except Exception as e:
            self._handler_errors += 1
            logger.error(f"Handler error for {event.event_type.value}: {e}")

    def get_event_history(
        self,
        event_type: EventType | None = None,
        limit: int = 100,
    ) -> list[NashEvent]:
        """Get event history with optional filtering."""
        events = self._event_history

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        return events[-limit:]

    def get_stats(self) -> dict[str, Any]:
        """Get event bus statistics."""
        return {
            "events_published": self._events_published,
            "events_processed": self._events_processed,
            "handler_errors": self._handler_errors,
            "history_size": len(self._event_history),
            "subscriber_count": sum(len(h) for h in self._subscribers.values()),
            "subscribers_by_type": {
                et.value: len(handlers) for et, handlers in self._subscribers.items()
            },
        }

    async def replay_events(
        self,
        event_type: EventType | None = None,
        from_timestamp: datetime | None = None,
    ) -> int:
        """
        Replay events from history.

        Returns:
            Number of events replayed
        """
        events = self._event_history

        if event_type:
            events = [e for e in events if e.event_type == event_type]

        if from_timestamp:
            events = [e for e in events if e.timestamp >= from_timestamp]

        for event in events:
            await self.publish(event)

        return len(events)


# ═══════════════════════════════════════════════════════════════════════════════
# Cross-Component Event Handlers
# ═══════════════════════════════════════════════════════════════════════════════


class NashEventHandlers:
    """
    Built-in event handlers that wire components together.

    These handlers implement cross-component reactions:
    - KG update → Trigger frontier recomputation
    - Template convergence → Update KG
    - Drift detection → Alert + Auto-tuning
    - Insight contribution → Update baseline
    """

    def __init__(
        self,
        event_bus: NashEventBus,
        knowledge_graph=None,
        adaptive_templates=None,
        pareto_frontier=None,
        federated_learning=None,
    ):
        self.event_bus = event_bus
        self.kg = knowledge_graph
        self.templates = adaptive_templates
        self.frontier = pareto_frontier
        self.federated = federated_learning

        self._setup_handlers()

    def _setup_handlers(self):
        """Set up cross-component event handlers."""
        # Knowledge Graph → Pareto Frontier
        self.event_bus.subscribe(
            EventType.KNOWLEDGE_GRAPH_UPDATED, self._on_knowledge_graph_updated
        )

        # Template Results → Knowledge Graph
        self.event_bus.subscribe(EventType.TEMPLATE_RESULT_REPORTED, self._on_template_result)

        # Drift Detection → Auto-tuning
        self.event_bus.subscribe(EventType.DRIFT_DETECTED, self._on_drift_detected)

        # Federated Insights → Baseline Update
        self.event_bus.subscribe(EventType.INSIGHT_CONTRIBUTED, self._on_insight_contributed)

    async def _on_knowledge_graph_updated(self, event: KnowledgeGraphUpdatedEvent):
        """Handle knowledge graph update - trigger frontier refresh."""
        logger.debug(f"KG updated: {event.nodes_added} nodes, {event.edges_added} edges")

        # Invalidate frontier cache if significant change
        if (event.nodes_added > 5 or event.edges_added > 10) and self.frontier:
            # Clear cache to force recomputation
            self.frontier._prediction_cache.clear()
            logger.info("Frontier cache invalidated due to KG update")

    async def _on_template_result(self, event: TemplateResultReportedEvent):
        """Handle template result - update KG with performance data."""
        logger.debug(f"Template result: {event.variant_name} = {event.score:.2f}")

        # Could trigger KG update here
        # if self.kg and event.score < 0.5:
        #     # Low score - might indicate model struggles with pattern
        #     pass

    async def _on_drift_detected(self, event: DriftDetectedEvent):
        """Handle drift detection - trigger auto-tuning."""
        logger.warning(
            f"Drift detected in {event.model}: "
            f"{event.metric} changed from {event.expected_value:.2f} "
            f"to {event.observed_value:.2f} (p={event.p_value:.4f})"
        )

        # Trigger auto-tuning
        if event.severity == "critical":
            await self.event_bus.publish(
                AutoTuningTriggeredEvent(
                    parameter_name=f"model_weight_{event.model}",
                    old_value=1.0,
                    new_value=0.5,
                    reason=f"Drift detected: {event.metric}",
                    expected_improvement=0.1,
                )
            )

    async def _on_insight_contributed(self, event: InsightContributedEvent):
        """Handle federated insight - trigger baseline update."""
        logger.debug(
            f"Insight contributed: {event.model} " f"(ε spent: {event.privacy_epsilon_spent:.3f})"
        )

        # Trigger baseline update if enough insights
        if event.global_insight_count % 10 == 0:
            logger.info(f"Baseline update triggered ({event.global_insight_count} insights)")


# ═══════════════════════════════════════════════════════════════════════════════
# Global Event Bus
# ═══════════════════════════════════════════════════════════════════════════════

_event_bus: NashEventBus | None = None


def get_event_bus() -> NashEventBus:
    """Get global Nash event bus."""
    global _event_bus
    if _event_bus is None:
        _event_bus = NashEventBus()
    return _event_bus


def reset_event_bus() -> None:
    """Reset global event bus (for testing)."""
    global _event_bus
    _event_bus = None


# Convenience decorator
def on_event(event_type: EventType):
    """Decorator to subscribe to events on global bus."""
    return get_event_bus().on(event_type)
