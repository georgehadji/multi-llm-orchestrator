"""
Saga Pattern for Distributed Transactions
=========================================

Implements the saga pattern for managing long-running transactions
with support for compensation (rollback) on failure.

Usage:
    from orchestrator.sagas import Saga, SagaStep
    
    saga = Saga(
        steps=[
            SagaStep(
                name="decompose",
                action=DecomposeAction(),
                compensation=DeleteTasksAction(),
            ),
            SagaStep(
                name="execute",
                action=ExecuteAction(),
                compensation=MarkFailedAction(),
            ),
        ]
    )
    
    result = await saga.execute(context)
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, TypeVar, Generic, Awaitable
from uuid import uuid4

from .events import (
    EventBus, get_event_bus, DomainEvent,
    TaskStartedEvent, TaskCompletedEvent, TaskFailedEvent
)
from .log_config import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


# ═══════════════════════════════════════════════════════════════════════════════
# Saga State and Types
# ═══════════════════════════════════════════════════════════════════════════════

class SagaState(Enum):
    """States of a saga execution."""
    PENDING = auto()
    RUNNING = auto()
    COMPENSATING = auto()
    COMPLETED = auto()
    FAILED = auto()
    COMPENSATED = auto()


class SagaStatus(Enum):
    """Final status of a saga."""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"  # Some steps succeeded, then failed and compensated


# ═══════════════════════════════════════════════════════════════════════════════
# Saga Events
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SagaStartedEvent(DomainEvent):
    """Emitted when a saga starts."""
    saga_id: str = ""
    saga_type: str = ""
    step_count: int = 0
    
    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, 'event_type', 'saga.started')


@dataclass
class SagaStepStartedEvent(DomainEvent):
    """Emitted when a saga step starts."""
    saga_id: str = ""
    step_name: str = ""
    step_index: int = 0
    
    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, 'event_type', 'saga.step_started')


@dataclass
class SagaStepCompletedEvent(DomainEvent):
    """Emitted when a saga step completes."""
    saga_id: str = ""
    step_name: str = ""
    step_index: int = 0
    result: Any = None
    
    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, 'event_type', 'saga.step_completed')


@dataclass
class SagaStepFailedEvent(DomainEvent):
    """Emitted when a saga step fails."""
    saga_id: str = ""
    step_name: str = ""
    step_index: int = 0
    error: str = ""
    
    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, 'event_type', 'saga.step_failed')


@dataclass
class SagaCompensatingEvent(DomainEvent):
    """Emitted when saga starts compensating."""
    saga_id: str = ""
    failed_step: str = ""
    steps_to_compensate: int = 0
    
    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, 'event_type', 'saga.compensating')


@dataclass
class SagaCompensationStepEvent(DomainEvent):
    r"""Emitted when a compensation step runs."""
    saga_id: str = ""
    step_name: str = ""
    success: bool = True
    
    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, 'event_type', 'saga.compensation_step')


@dataclass
class SagaCompletedEvent(DomainEvent):
    """Emitted when a saga completes."""
    saga_id: str = ""
    status: str = ""  # success, failure, partial
    steps_succeeded: int = 0
    steps_failed: int = 0
    steps_compensated: int = 0
    duration_seconds: float = 0.0
    
    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(self, 'event_type', 'saga.completed')


# ═══════════════════════════════════════════════════════════════════════════════
# Saga Action Interface
# ═══════════════════════════════════════════════════════════════════════════════

class SagaAction(ABC):
    """Abstract base class for saga actions."""
    
    @abstractmethod
    async def execute(self, context: SagaContext) -> SagaActionResult:
        """Execute the action."""
        pass
    
    async def on_success(self, result: Any, context: SagaContext) -> None:
        """Called when action succeeds. Override to add side effects."""
        pass
    
    async def on_failure(self, error: Exception, context: SagaContext) -> None:
        """Called when action fails. Override to add side effects."""
        pass


class SagaCompensation(ABC):
    """Abstract base class for saga compensation actions."""
    
    @abstractmethod
    async def compensate(self, context: SagaContext, original_result: Any) -> bool:
        """
        Compensate (undo) the original action.
        
        Returns True if compensation succeeded.
        """
        pass


@dataclass
class SagaActionResult:
    """Result of a saga action."""
    success: bool
    result: Any = None
    error: Optional[str] = None
    should_compensate: bool = True


@dataclass
class SagaContext:
    """Context passed through saga execution."""
    saga_id: str
    data: Dict[str, Any] = field(default_factory=dict)
    step_results: List[SagaStepResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def set(self, key: str, value: Any) -> None:
        """Store data in context."""
        self.data[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve data from context."""
        return self.data.get(key, default)


@dataclass
class SagaStepResult:
    """Result of executing a saga step."""
    step_name: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    compensated: bool = False
    compensation_error: Optional[str] = None
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    
    @property
    def duration_seconds(self) -> float:
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time).total_seconds()


# ═══════════════════════════════════════════════════════════════════════════════
# Saga Step
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SagaStep:
    """A single step in a saga."""
    name: str
    action: SagaAction
    compensation: Optional[SagaCompensation] = None
    retry_count: int = 0
    retry_delay_seconds: float = 1.0
    
    async def execute(self, context: SagaContext) -> SagaStepResult:
        """Execute the step with optional retries."""
        start_time = datetime.utcnow()
        
        last_error = None
        for attempt in range(self.retry_count + 1):
            try:
                result = await self.action.execute(context)
                
                if result.success:
                    await self.action.on_success(result.result, context)
                    
                    return SagaStepResult(
                        step_name=self.name,
                        success=True,
                        result=result.result,
                        end_time=datetime.utcnow(),
                    )
                else:
                    # Action returned failure
                    if attempt < self.retry_count:
                        await asyncio.sleep(self.retry_delay_seconds)
                        continue
                    
                    return SagaStepResult(
                        step_name=self.name,
                        success=False,
                        error=result.error or "Action returned failure",
                        end_time=datetime.utcnow(),
                    )
                    
            except Exception as e:
                last_error = e
                await self.action.on_failure(e, context)
                
                if attempt < self.retry_count:
                    logger.warning(f"Step {self.name} failed (attempt {attempt + 1}), retrying...")
                    await asyncio.sleep(self.retry_delay_seconds)
                    continue
                
                return SagaStepResult(
                    step_name=self.name,
                    success=False,
                    error=str(e),
                    end_time=datetime.utcnow(),
                )
        
        # Should not reach here, but just in case
        return SagaStepResult(
            step_name=self.name,
            success=False,
            error=str(last_error) if last_error else "Unknown error",
            end_time=datetime.utcnow(),
        )
    
    async def compensate(self, context: SagaContext, original_result: Any) -> bool:
        """Execute compensation if available."""
        if self.compensation is None:
            logger.warning(f"Step {self.name} has no compensation action")
            return False
        
        try:
            return await self.compensation.compensate(context, original_result)
        except Exception as e:
            logger.error(f"Compensation failed for step {self.name}: {e}")
            return False


# ═══════════════════════════════════════════════════════════════════════════════
# Saga Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SagaResult:
    """Final result of saga execution."""
    success: bool
    status: SagaStatus
    context: SagaContext
    step_results: List[SagaStepResult]
    duration_seconds: float
    error: Optional[str] = None
    
    @property
    def steps_succeeded(self) -> int:
        return sum(1 for r in self.step_results if r.success)
    
    @property
    def steps_failed(self) -> int:
        return sum(1 for r in self.step_results if not r.success)
    
    @property
    def steps_compensated(self) -> int:
        return sum(1 for r in self.step_results if r.compensated)


class Saga:
    """
    Saga orchestrator for managing distributed transactions.
    
    A saga is a sequence of transactions where each step has a compensation
    action that can undo it if subsequent steps fail.
    """
    
    def __init__(
        self,
        steps: List[SagaStep],
        saga_type: str = "generic",
        event_bus: Optional[EventBus] = None,
    ):
        self.steps = steps
        self.saga_type = saga_type
        self.saga_id = str(uuid4())[:8]
        self.state = SagaState.PENDING
        self.event_bus = event_bus or get_event_bus()
    
    async def execute(self, initial_context: Optional[Dict[str, Any]] = None) -> SagaResult:
        """
        Execute the saga.
        
        Flow:
        1. Execute steps in order
        2. If a step fails, run compensations for completed steps (in reverse order)
        3. Return result with final state
        """
        start_time = datetime.utcnow()
        self.state = SagaState.RUNNING
        
        # Create context
        context = SagaContext(
            saga_id=self.saga_id,
            data=initial_context or {},
        )
        
        # Emit saga started event
        await self._emit_event(SagaStartedEvent(
            aggregate_id=self.saga_id,
            saga_id=self.saga_id,
            saga_type=self.saga_type,
            step_count=len(self.steps),
        ))
        
        step_results: List[SagaStepResult] = []
        
        try:
            # Execute each step
            for i, step in enumerate(self.steps):
                # Emit step started
                await self._emit_event(SagaStepStartedEvent(
                    aggregate_id=self.saga_id,
                    saga_id=self.saga_id,
                    step_name=step.name,
                    step_index=i,
                ))
                
                # Execute step
                result = await step.execute(context)
                step_results.append(result)
                context.step_results.append(result)
                
                if result.success:
                    # Emit step completed
                    await self._emit_event(SagaStepCompletedEvent(
                        aggregate_id=self.saga_id,
                        saga_id=self.saga_id,
                        step_name=step.name,
                        step_index=i,
                        result=result.result,
                    ))
                else:
                    # Step failed - emit event and compensate
                    await self._emit_event(SagaStepFailedEvent(
                        aggregate_id=self.saga_id,
                        saga_id=self.saga_id,
                        step_name=step.name,
                        step_index=i,
                        error=result.error or "Unknown error",
                    ))
                    
                    # Compensate completed steps
                    await self._compensate(context, step_results)
                    
                    duration = (datetime.utcnow() - start_time).total_seconds()
                    
                    await self._emit_event(SagaCompletedEvent(
                        aggregate_id=self.saga_id,
                        saga_id=self.saga_id,
                        status="partial" if self.steps_compensated(step_results) > 0 else "failure",
                        steps_succeeded=self.steps_succeeded(step_results),
                        steps_failed=1,
                        steps_compensated=self.steps_compensated(step_results),
                        duration_seconds=duration,
                    ))
                    
                    return SagaResult(
                        success=False,
                        status=SagaStatus.PARTIAL if self.steps_compensated(step_results) > 0 else SagaStatus.FAILURE,
                        context=context,
                        step_results=step_results,
                        duration_seconds=duration,
                        error=f"Step {step.name} failed: {result.error}",
                    )
            
            # All steps completed successfully
            self.state = SagaState.COMPLETED
            duration = (datetime.utcnow() - start_time).total_seconds()
            
            await self._emit_event(SagaCompletedEvent(
                aggregate_id=self.saga_id,
                saga_id=self.saga_id,
                status="success",
                steps_succeeded=len(self.steps),
                steps_failed=0,
                steps_compensated=0,
                duration_seconds=duration,
            ))
            
            return SagaResult(
                success=True,
                status=SagaStatus.SUCCESS,
                context=context,
                step_results=step_results,
                duration_seconds=duration,
            )
            
        except Exception as e:
            # Unexpected error during execution
            logger.exception("Unexpected error during saga execution")
            
            # Try to compensate if we have any results
            if step_results:
                await self._compensate(context, step_results)
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            
            await self._emit_event(SagaCompletedEvent(
                aggregate_id=self.saga_id,
                saga_id=self.saga_id,
                status="failure",
                steps_succeeded=self.steps_succeeded(step_results),
                steps_failed=len(self.steps) - self.steps_succeeded(step_results),
                steps_compensated=self.steps_compensated(step_results),
                duration_seconds=duration,
            ))
            
            return SagaResult(
                success=False,
                status=SagaStatus.FAILURE,
                context=context,
                step_results=step_results,
                duration_seconds=duration,
                error=str(e),
            )
    
    async def _compensate(
        self,
        context: SagaContext,
        step_results: List[SagaStepResult],
    ) -> None:
        """
        Run compensation actions for completed steps in reverse order.
        """
        self.state = SagaState.COMPENSATING
        
        completed_steps = [
            (self.steps[i], step_results[i])
            for i in range(len(step_results))
            if step_results[i].success
        ]
        
        if not completed_steps:
            return
        
        # Emit compensating event
        await self._emit_event(SagaCompensatingEvent(
            aggregate_id=self.saga_id,
            saga_id=self.saga_id,
            failed_step=step_results[-1].step_name if step_results else "",
            steps_to_compensate=len(completed_steps),
        ))
        
        # Compensate in reverse order
        for step, result in reversed(completed_steps):
            if step.compensation is None:
                continue
            
            try:
                success = await step.compensate(context, result.result)
                result.compensated = success
                
                await self._emit_event(SagaCompensationStepEvent(
                    aggregate_id=self.saga_id,
                    saga_id=self.saga_id,
                    step_name=step.name,
                    success=success,
                ))
                
            except Exception as e:
                logger.error(f"Compensation error for step {step.name}: {e}")
                result.compensated = False
                result.compensation_error = str(e)
                
                await self._emit_event(SagaCompensationStepEvent(
                    aggregate_id=self.saga_id,
                    saga_id=self.saga_id,
                    step_name=step.name,
                    success=False,
                ))
    
    def steps_succeeded(self, step_results: List[SagaStepResult]) -> int:
        """Count succeeded steps."""
        return sum(1 for r in step_results if r.success)
    
    def steps_compensated(self, step_results: List[SagaStepResult]) -> int:
        """Count compensated steps."""
        return sum(1 for r in step_results if r.compensated)
    
    async def _emit_event(self, event: DomainEvent) -> None:
        """Emit event if event bus is available."""
        try:
            await self.event_bus.publish(event)
        except Exception as e:
            logger.error(f"Failed to emit saga event: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# Pre-built Actions for Common Use Cases
# ═══════════════════════════════════════════════════════════════════════════════

class FunctionAction(SagaAction):
    """Action that wraps a function."""
    
    def __init__(
        self,
        func: Callable[[SagaContext], Awaitable[SagaActionResult]],
        name: str = "function",
    ):
        self.func = func
        self.name = name
    
    async def execute(self, context: SagaContext) -> SagaActionResult:
        return await self.func(context)


class FunctionCompensation(SagaCompensation):
    """Compensation that wraps a function."""
    
    def __init__(
        self,
        func: Callable[[SagaContext, Any], Awaitable[bool]],
        name: str = "compensation",
    ):
        self.func = func
        self.name = name
    
    async def compensate(self, context: SagaContext, original_result: Any) -> bool:
        return await self.func(context, original_result)


# ═══════════════════════════════════════════════════════════════════════════════
# Project Execution Saga
# ═══════════════════════════════════════════════════════════════════════════════

class ProjectExecutionSaga:
    """
    Pre-built saga for project execution.
    
    This saga orchestrates:
    1. Project enhancement
    2. Task decomposition
    3. Task execution
    4. Validation
    5. Output generation
    
    With compensation at each step.
    """
    
    @classmethod
    def create(
        cls,
        orchestrator: Any,  # The orchestrator instance
        project_description: str,
        success_criteria: str,
    ) -> Saga:
        """Create a project execution saga."""
        
        steps = [
            SagaStep(
                name="enhance_project",
                action=EnhanceProjectAction(orchestrator, project_description),
                compensation=DeleteEnhancementAction(),
            ),
            SagaStep(
                name="decompose",
                action=DecomposeAction(orchestrator, success_criteria),
                compensation=DeleteTasksAction(),
            ),
            SagaStep(
                name="execute_tasks",
                action=ExecuteTasksAction(orchestrator),
                compensation=MarkTasksFailedAction(),
            ),
            SagaStep(
                name="validate",
                action=ValidateAction(orchestrator),
                compensation=NoOpCompensation(),  # Validation is read-only
            ),
        ]
        
        return Saga(steps, saga_type="project_execution")


# Placeholder actions for project execution saga

class EnhanceProjectAction(SagaAction):
    def __init__(self, orchestrator: Any, description: str):
        self.orchestrator = orchestrator
        self.description = description
    
    async def execute(self, context: SagaContext) -> SagaActionResult:
        # Would call actual enhancement
        context.set("enhanced_description", f"Enhanced: {self.description}")
        return SagaActionResult(success=True, result={"enhanced": True})


class DeleteEnhancementAction(SagaCompensation):
    async def compensate(self, context: SagaContext, original_result: Any) -> bool:
        context.set("enhanced_description", None)
        return True


class DecomposeAction(SagaAction):
    def __init__(self, orchestrator: Any, criteria: str):
        self.orchestrator = orchestrator
        self.criteria = criteria
    
    async def execute(self, context: SagaContext) -> SagaActionResult:
        # Would call actual decomposition
        tasks = ["task1", "task2", "task3"]
        context.set("tasks", tasks)
        return SagaActionResult(success=True, result={"tasks": tasks})


class DeleteTasksAction(SagaCompensation):
    async def compensate(self, context: SagaContext, original_result: Any) -> bool:
        context.set("tasks", [])
        return True


class ExecuteTasksAction(SagaAction):
    def __init__(self, orchestrator: Any):
        self.orchestrator = orchestrator
    
    async def execute(self, context: SagaContext) -> SagaActionResult:
        # Would call actual task execution
        return SagaActionResult(success=True, result={"executed": True})


class MarkTasksFailedAction(SagaCompensation):
    async def compensate(self, context: SagaContext, original_result: Any) -> bool:
        # Mark tasks as failed
        return True


class ValidateAction(SagaAction):
    def __init__(self, orchestrator: Any):
        self.orchestrator = orchestrator
    
    async def execute(self, context: SagaContext) -> SagaActionResult:
        # Would call actual validation
        return SagaActionResult(success=True, result={"valid": True})


class NoOpCompensation(SagaCompensation):
    """Compensation that does nothing (for read-only steps)."""
    
    async def compensate(self, context: SagaContext, original_result: Any) -> bool:
        return True


# ═══════════════════════════════════════════════════════════════════════════════
# Example Usage
# ═══════════════════════════════════════════════════════════════════════════════

async def example():
    """Example saga usage."""
    
    # Define custom actions
    class ReserveInventoryAction(SagaAction):
        async def execute(self, context: SagaContext) -> SagaActionResult:
            # Simulate reservation
            context.set("reservation_id", "RES-123")
            return SagaActionResult(success=True, result={"id": "RES-123"})
    
    class ReleaseInventoryCompensation(SagaCompensation):
        async def compensate(self, context: SagaContext, original_result: Any) -> bool:
            reservation_id = context.get("reservation_id")
            print(f"Releasing reservation {reservation_id}")
            return True
    
    class ProcessPaymentAction(SagaAction):
        async def execute(self, context: SagaContext) -> SagaActionResult:
            # Simulate payment (sometimes fails)
            import random
            if random.random() < 0.5:
                return SagaActionResult(success=False, error="Payment declined")
            return SagaActionResult(success=True, result={"payment_id": "PAY-456"})
    
    class RefundPaymentCompensation(SagaCompensation):
        async def compensate(self, context: SagaContext, original_result: Any) -> bool:
            print("Refunding payment")
            return True
    
    # Create saga
    saga = Saga(
        steps=[
            SagaStep(
                name="reserve_inventory",
                action=ReserveInventoryAction(),
                compensation=ReleaseInventoryCompensation(),
            ),
            SagaStep(
                name="process_payment",
                action=ProcessPaymentAction(),
                compensation=RefundPaymentCompensation(),
                retry_count=2,
                retry_delay_seconds=1.0,
            ),
        ],
        saga_type="order_processing",
    )
    
    # Execute
    result = await saga.execute(initial_context={"order_id": "ORD-789"})
    
    print(f"Saga completed: {result.status.value}")
    print(f"Duration: {result.duration_seconds:.2f}s")
    print(f"Steps: {len(result.step_results)}")


if __name__ == "__main__":
    asyncio.run(example())
