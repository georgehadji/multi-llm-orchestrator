"""
Command Center Integration
==========================
Connects existing orchestrator events to the command center dashboard.

This module bridges:
- UnifiedEventBus → CommandCenterServer
- TelemetryCollector → Real-time metrics
- Budget → Cost burn rate tracking
- AdaptiveRouter → Model health status
"""

from __future__ import annotations

import asyncio
import logging

from .command_center_server import (
    Severity,
    SystemMetrics,
    get_command_center_server,
)
from .models import Model

logger = logging.getLogger("orchestrator.command_center")


class CommandCenterIntegration:
    """
    Integration layer between orchestrator and command center dashboard.

    Automatically converts orchestrator events to dashboard alerts and metrics.
    """

    def __init__(self, orchestrator: Orchestrator):
        self._orchestrator = orchestrator
        self._server = get_command_center_server()
        self._running = False
        self._monitor_task: asyncio.Task | None = None

        # Track state for change detection
        self._last_model_health: dict = {}
        self._last_budget: float = 0.0
        self._alerted_budget_overrun = False

    def start(self):
        """Start the integration."""
        if self._running:
            return

        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Command Center integration started")

    def stop(self):
        """Stop the integration."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
        logger.info("Command Center integration stopped")

    async def _monitor_loop(self):
        """Main monitoring loop - 1 second intervals."""
        while self._running:
            try:
                await self._update_metrics()
                await self._check_alerts()
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                await asyncio.sleep(5)

    async def _update_metrics(self):
        """Update dashboard metrics from orchestrator state."""
        # Model health from adaptive router
        models = {}
        for model in Model:
            if hasattr(self._orchestrator, "_adaptive_router"):
                state = self._orchestrator._adaptive_router.get_state(model)
                models[model.value] = state.value
            else:
                # Fallback to api_health
                is_healthy = self._orchestrator.api_health.get(model, False)
                models[model.value] = "healthy" if is_healthy else "disabled"

        # Task queue from results
        task_queue = {
            "pending": len(
                [
                    t
                    for t in self._orchestrator.results.values()
                    if hasattr(t, "status") and t.status == "pending"
                ]
            ),
            "active": len(
                [
                    t
                    for t in self._orchestrator.results.values()
                    if hasattr(t, "status") and t.status == "running"
                ]
            ),
            "failed": len(
                [
                    t
                    for t in self._orchestrator.results.values()
                    if hasattr(t, "status") and t.status == "failed"
                ]
            ),
        }

        # Cost burn rate (hourly)
        budget = self._orchestrator.budget
        elapsed_hours = budget.elapsed_seconds / 3600
        cost_burn_rate = budget.spent_usd / elapsed_hours if elapsed_hours > 0 else 0

        # Quality score from telemetry
        quality_score = 0.85  # default
        if hasattr(self._orchestrator, "_telemetry"):
            profiles = self._orchestrator._telemetry._profiles
            if profiles:
                qualities = [
                    p.quality_score for p in profiles.values() if hasattr(p, "quality_score")
                ]
                if qualities:
                    quality_score = sum(qualities) / len(qualities)

        # Cache hit rate from semantic cache
        cache_hit_rate = 0.0
        if hasattr(self._orchestrator, "_semantic_cache"):
            stats = self._orchestrator._semantic_cache.get_stats()
            total_uses = stats.get("total_uses", 0)
            if total_uses > 0:
                cache_hit_rate = stats.get("hot_entries", 0) / total_uses

        metrics = SystemMetrics(
            timestamp=__import__("time").time(),
            models=models,
            task_queue=task_queue,
            cost_burn_rate=cost_burn_rate,
            quality_score=quality_score,
            cache_hit_rate=cache_hit_rate,
            latency_ms=45,  # estimated
        )

        self._server.update_metrics(metrics)

    async def _check_alerts(self):
        """Check for alert conditions."""
        # Check model health changes
        if hasattr(self._orchestrator, "_adaptive_router"):
            from .adaptive_router import ModelState

            for model in Model:
                state = self._orchestrator._adaptive_router.get_state(model)
                last_state = self._last_model_health.get(model.value)

                if state == ModelState.DEGRADED and last_state != "degraded":
                    self._server.raise_alert(
                        severity=Severity.WARNING,
                        title=f"Model {model.value} Degraded",
                        message=f"Model {model.value} has been marked as degraded due to timeouts",
                        source="adaptive_router",
                    )
                elif state == ModelState.DISABLED and last_state != "disabled":
                    self._server.raise_alert(
                        severity=Severity.CRITICAL,
                        title=f"Model {model.value} Disabled",
                        message=f"Model {model.value} has been permanently disabled (auth failure)",
                        source="adaptive_router",
                    )

                self._last_model_health[model.value] = state.value

        # Check budget overrun
        budget = self._orchestrator.budget
        if budget.max_usd > 0:
            spent_ratio = budget.spent_usd / budget.max_usd
            if spent_ratio > 2.0 and not self._alerted_budget_overrun:
                self._server.raise_alert(
                    severity=Severity.CRITICAL,
                    title="Budget Overrun",
                    message=f"Spent ${budget.spent_usd:.2f} of ${budget.max_usd:.2f} budget ({spent_ratio:.1f}×)",
                    source="budget_monitor",
                )
                self._alerted_budget_overrun = True
            elif spent_ratio < 1.5:
                self._alerted_budget_overrun = False

    # Event handlers for unified event bus

    def on_task_completed(self, event: dict):
        """Handle task completion events."""
        if event.get("status") == "failed":
            self._server.raise_alert(
                severity=Severity.WARNING,
                title=f"Task Failed: {event.get('task_id', 'unknown')}",
                message=f"Task failed after {event.get('iterations', 0)} iterations",
                source="task_executor",
            )

    def on_fallback_triggered(self, event: dict):
        """Handle fallback events."""
        self._server.raise_alert(
            severity=Severity.INFO,
            title="Model Fallback",
            message=f"Fell back from {event.get('from_model')} to {event.get('to_model')}",
            source="router",
        )

    def on_budget_warning(self, event: dict):
        """Handle budget warning events."""
        self._server.raise_alert(
            severity=Severity.WARNING,
            title="Budget Warning",
            message=f"Phase '{event.get('phase')}' at {event.get('ratio', 0):.0%} of budget",
            source="budget_monitor",
        )


# Convenience function
def enable_command_center(orchestrator: Orchestrator) -> CommandCenterIntegration:
    """
    Enable command center dashboard for an orchestrator instance.

    Usage:
        orch = Orchestrator()
        cc = enable_command_center(orch)
        cc.start()
    """
    integration = CommandCenterIntegration(orchestrator)
    integration.start()
    return integration
