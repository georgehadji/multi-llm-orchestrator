"""
Orchestrator Engine with Event-Driven Architecture
==================================================

This is an enhanced version of engine.py that uses the new event system
while maintaining backward compatibility with hooks.

Key changes:
- Events are now first-class citizens
- Hooks are wrapped as event handlers for backward compatibility
- All major lifecycle points emit domain events
- Projections can subscribe to events for CQRS

Usage:
    from orchestrator.engine_with_events import EventDrivenOrchestrator
    
    orch = EventDrivenOrchestrator()
    
    # Subscribe to events
    from orchestrator.events import get_event_bus
    bus = get_event_bus()
    
    @bus.subscribe("task.completed")
    async def on_task_complete(event):
        print(f"Task {event.task_id} done!")
    
    # Run project
    result = await orch.run_project("Build API", "Works", 5.0)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

# Import original engine components
from .engine import Orchestrator as BaseOrchestrator
from .models import (
    Task, TaskResult, TaskType, TaskStatus,
    ProjectState, ProjectStatus, Model, Budget,
    AttemptRecord
)
from .events import (
    EventBus, get_event_bus,
    TaskStartedEvent, TaskCompletedEvent, TaskFailedEvent,
    ModelSelectedEvent, ValidationFailedEvent, BudgetWarningEvent,
    ProjectStartedEvent, ProjectCompletedEvent, CircuitBreakerTrippedEvent,
    DomainEvent
)
from .hooks import EventType, HookRegistry

logger = logging.getLogger("orchestrator.engine_events")


class EventDrivenOrchestrator(BaseOrchestrator):
    """
    Extended Orchestrator that emits domain events throughout the lifecycle.
    
    Maintains full backward compatibility with the base Orchestrator while
    adding event-driven capabilities.
    """
    
    def __init__(self, *args, event_bus: Optional[EventBus] = None, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Event bus (singleton by default)
        self._event_bus = event_bus or get_event_bus()
        
        # Track project context for events
        self._current_project_id: str = ""
        self._project_start_time: Optional[datetime] = None
        
        logger.info("EventDrivenOrchestrator initialized")
    
    async def run_project(
        self,
        project_description: str,
        success_criteria: str,
        budget: Optional[float] = None,
    ) -> ProjectState:
        """
        Run a project with event emission.
        
        Emits:
        - project.started
        - task.started (for each task)
        - task.completed / task.failed
        - project.completed
        """
        # Set budget if provided
        if budget is not None:
            self.budget = Budget(max_usd=budget)
        
        # Generate project ID and track start
        import hashlib
        self._current_project_id = hashlib.sha256(
            f"{project_description}:{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:12]
        self._project_start_time = datetime.utcnow()
        
        # Emit project started event
        await self._emit_event(ProjectStartedEvent(
            aggregate_id=self._current_project_id,
            project_id=self._current_project_id,
            description=project_description[:100],  # Truncate for event size
            budget=self.budget.max_usd if self.budget else 0.0,
        ))
        
        try:
            # Call parent implementation
            result = await super().run_project(project_description, success_criteria)
            
            # Calculate duration
            duration = 0.0
            if self._project_start_time:
                duration = (datetime.utcnow() - self._project_start_time).total_seconds()
            
            # Emit project completed event
            await self._emit_event(ProjectCompletedEvent(
                aggregate_id=self._current_project_id,
                project_id=self._current_project_id,
                status=self._map_project_status(result.status),
                total_cost=result.budget.spent_usd if result.budget else 0.0,
                duration_seconds=duration,
            ))
            
            return result
            
        except Exception as e:
            # Emit project failed event
            await self._emit_event(ProjectCompletedEvent(
                aggregate_id=self._current_project_id,
                project_id=self._current_project_id,
                status="failed",
                total_cost=self.budget.spent_usd if self.budget else 0.0,
                duration_seconds=0.0,
            ))
            raise
    
    async def _execute_task(self, task: Task, attempt: int = 0) -> TaskResult:
        """
        Execute a single task with event emission.
        
        Wraps the parent _execute_task with event publishing.
        """
        # Emit task started event
        await self._emit_event(TaskStartedEvent(
            aggregate_id=task.id,
            task_id=task.id,
            task_type=task.task_type.value,
            model="",  # Will be filled after model selection
            project_id=self._current_project_id,
        ))
        
        start_time = datetime.utcnow()
        
        try:
            # Track model selection for event
            original_select_model = self._planner.select_model
            
            async def wrapped_select_model(task_obj: Task, *args, **kwargs):
                model = await original_select_model(task_obj, *args, **kwargs)
                
                # Emit model selected event
                await self._emit_event(ModelSelectedEvent(
                    aggregate_id=task_obj.id,
                    task_id=task_obj.id,
                    model=model.value,
                    strategy="constraint_planner",
                    reason="default_routing",
                    confidence=0.8,
                ))
                
                return model
            
            # Temporarily replace model selector
            self._planner.select_model = wrapped_select_model
            
            try:
                # Call parent implementation
                result = await super()._execute_task(task, attempt)
            finally:
                # Restore original selector
                self._planner.select_model = original_select_model
            
            # Calculate latency
            latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Emit appropriate event based on result
            if result.success:
                await self._emit_event(TaskCompletedEvent(
                    aggregate_id=task.id,
                    task_id=task.id,
                    model=result.model.value if result.model else "",
                    score=result.score or 0.0,
                    cost_usd=result.cost_usd or 0.0,
                    latency_ms=latency_ms,
                    tokens_input=result.tokens_used or 0,
                    tokens_output=0,  # Not tracked in current model
                ))
            else:
                await self._emit_event(TaskFailedEvent(
                    aggregate_id=task.id,
                    task_id=task.id,
                    model=result.model.value if result.model else "",
                    error=result.error or "Unknown error",
                    attempt=attempt,
                    will_retry=attempt < 2 and result.score < 0.5,
                ))
            
            return result
            
        except Exception as e:
            # Emit task failed event for exceptions
            await self._emit_event(TaskFailedEvent(
                aggregate_id=task.id,
                task_id=task.id,
                model="",
                error=str(e),
                attempt=attempt,
                will_retry=False,
            ))
            raise
    
    async def _emit_event(self, event: DomainEvent) -> None:
        """Emit an event to the event bus."""
        try:
            await self._event_bus.publish(event)
        except Exception as e:
            # Events should never fail the main operation
            logger.error(f"Failed to emit event {event.event_type}: {e}")
    
    def _map_project_status(self, status: ProjectStatus) -> str:
        """Map internal status to event status."""
        status_map = {
            ProjectStatus.SUCCESS: "success",
            ProjectStatus.PARTIAL_SUCCESS: "partial",
            ProjectStatus.COMPLETED_DEGRADED: "degraded",
            ProjectStatus.BUDGET_EXHAUSTED: "budget_exhausted",
            ProjectStatus.TIMEOUT: "timeout",
            ProjectStatus.SYSTEM_FAILURE: "failed",
        }
        return status_map.get(status, "unknown")
    
    # Backward compatibility: wrap hooks as event handlers
    def _setup_hook_bridge(self) -> None:
        """
        Bridge between old hooks and new event system.
        
        This allows existing code using hooks to continue working
        while also emitting events.
        """
        # Bridge TASK_STARTED hook
        self._hook_registry.add(EventType.TASK_STARTED, self._on_task_started_hook)
        
        # Bridge TASK_COMPLETED hook
        self._hook_registry.add(EventType.TASK_COMPLETED, self._on_task_completed_hook)
        
        # Bridge VALIDATION_FAILED hook
        self._hook_registry.add(EventType.VALIDATION_FAILED, self._on_validation_failed_hook)
        
        # Bridge BUDGET_WARNING hook
        self._hook_registry.add(EventType.BUDGET_WARNING, self._on_budget_warning_hook)
    
    def _on_task_started_hook(self, task_id: str, task: Task, **kwargs) -> None:
        """Bridge hook to event."""
        asyncio.create_task(self._emit_event(TaskStartedEvent(
            aggregate_id=task_id,
            task_id=task_id,
            task_type=task.task_type.value,
            project_id=self._current_project_id,
        )))
    
    def _on_task_completed_hook(self, task_id: str, result: TaskResult, **kwargs) -> None:
        """Bridge hook to event."""
        if result.success:
            asyncio.create_task(self._emit_event(TaskCompletedEvent(
                aggregate_id=task_id,
                task_id=task_id,
                model=result.model.value if result.model else "",
                score=result.score or 0.0,
                cost_usd=result.cost_usd or 0.0,
                latency_ms=0.0,
            )))
        else:
            asyncio.create_task(self._emit_event(TaskFailedEvent(
                aggregate_id=task_id,
                task_id=task_id,
                model=result.model.value if result.model else "",
                error=result.error or "Failed",
            )))
    
    def _on_validation_failed_hook(self, task_id: str, model: str, validators: List[str], **kwargs) -> None:
        """Bridge hook to event."""
        asyncio.create_task(self._emit_event(ValidationFailedEvent(
            aggregate_id=task_id,
            task_id=task_id,
            model=model,
            validators=validators,
        )))
    
    def _on_budget_warning_hook(self, phase: str, spent: float, cap: float, ratio: float, **kwargs) -> None:
        """Bridge hook to event."""
        asyncio.create_task(self._emit_event(BudgetWarningEvent(
            phase=phase,
            spent=spent,
            budget=cap,
            ratio=ratio,
            project_id=self._current_project_id,
        )))


# Convenience function for migration
def create_orchestrator(
    use_events: bool = True,
    event_backend: str = "sqlite",
    **kwargs
) -> BaseOrchestrator:
    """
    Factory function to create orchestrator with optional event support.
    
    Args:
        use_events: If True, returns EventDrivenOrchestrator
        event_backend: "memory" or "sqlite"
        **kwargs: Passed to orchestrator constructor
    
    Returns:
        Orchestrator instance
    """
    if use_events:
        # Initialize event bus
        bus = get_event_bus(backend=event_backend)
        return EventDrivenOrchestrator(event_bus=bus, **kwargs)
    else:
        # Return legacy orchestrator
        return BaseOrchestrator(**kwargs)
