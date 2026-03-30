"""
Streaming Pipeline for Real-Time Project Execution
==================================================

Enables real-time streaming of project execution events for WebSocket/HTTP
streaming to clients. Essential for large projects (50+ tasks) where blocking
until completion is not acceptable.

Usage:
    from orchestrator.streaming import StreamingPipeline, ProjectEvent

    pipeline = StreamingPipeline()

    async for event in pipeline.execute_streaming(project_spec):
        await websocket.send_json({
            "type": event.type,
            "data": event.to_dict()
        })
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

from .events import (
    BudgetWarningEvent,
    DomainEvent,
    EventBus,
    ProjectCompletedEvent,
    ProjectStartedEvent,
    TaskCompletedEvent,
    TaskFailedEvent,
    TaskProgressEvent,
    TaskStartedEvent,
    get_event_bus,
)
from .models import Budget, Task, TaskResult

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

logger = logging.getLogger("orchestrator.streaming")


# Re-export common event classes for legacy consumers.
ProjectStarted = ProjectStartedEvent
ProjectCompleted = ProjectCompletedEvent
TaskStarted = TaskStartedEvent
TaskCompleted = TaskCompletedEvent
TaskFailed = TaskFailedEvent
TaskProgressUpdate = TaskProgressEvent
BudgetWarning = BudgetWarningEvent


# ═══════════════════════════════════════════════════════════════════════════════
# Streaming Events
# ═══════════════════════════════════════════════════════════════════════════════

class PipelineEventType(Enum):
    """Types of pipeline events for streaming."""
    PROJECT_START = auto()
    STAGE_START = auto()
    STAGE_PROGRESS = auto()
    STAGE_COMPLETE = auto()
    TASK_START = auto()
    TASK_PROGRESS = auto()
    TASK_COMPLETE = auto()
    TASK_FAILED = auto()
    MODEL_SELECTED = auto()
    VALIDATION_RESULT = auto()
    BUDGET_UPDATE = auto()
    PROJECT_COMPLETE = auto()
    ERROR = auto()
    INFO = auto()


@dataclass
class PipelineEvent:
    """Event emitted during streaming pipeline execution."""
    type: PipelineEventType
    project_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type.name,
            "project_id": self.project_id,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
        }


@dataclass
class StreamingTask:
    """Task wrapper for streaming execution."""
    task: Task
    status: str = "pending"  # pending, running, completed, failed
    result: TaskResult | None = None
    start_time: float | None = None
    end_time: float | None = None
    retry_count: int = 0

    @property
    def duration_ms(self) -> float:
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.time()
        return (end - self.start_time) * 1000

    @property
    def is_complete(self) -> bool:
        return self.status in ("completed", "failed")


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline Stages
# ═══════════════════════════════════════════════════════════════════════════════

class PipelineStage(ABC):
    """Abstract base class for pipeline stages."""

    def __init__(self, name: str):
        self.name = name
        self._progress_callback: Callable[[float, str], Awaitable[None]] | None = None

    def on_progress(self, callback: Callable[[float, str], Awaitable[None]]) -> None:
        """Set progress callback."""
        self._progress_callback = callback

    async def _report_progress(self, percent: float, message: str = "") -> None:
        """Report progress if callback is set."""
        if self._progress_callback:
            await self._progress_callback(percent, message)

    @abstractmethod
    async def execute(
        self,
        context: StreamingContext,
        event_queue: asyncio.Queue[PipelineEvent],
    ) -> None:
        """Execute the stage and emit events."""
        pass


StreamingStage = PipelineStage


@dataclass
class StreamingContext:
    """Context object passed through pipeline stages."""
    project_id: str
    description: str
    budget: Budget
    tasks: dict[str, StreamingTask] = field(default_factory=dict)
    completed_tasks: set[str] = field(default_factory=set)
    failed_tasks: set[str] = field(default_factory=set)
    current_stage: str = ""
    stage_progress: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_ready_tasks(self) -> list[StreamingTask]:
        """Get tasks whose dependencies are all complete."""
        ready = []
        for _task_id, st in self.tasks.items():
            if st.status != "pending":
                continue

            # Check all dependencies are complete
            deps_complete = all(
                dep_id in self.completed_tasks
                for dep_id in st.task.dependencies
            )

            if deps_complete:
                ready.append(st)

        return ready


class DecomposeStage(PipelineStage):
    """Stage for project decomposition into tasks."""

    def __init__(self):
        super().__init__("decompose")

    async def execute(
        self,
        context: StreamingContext,
        event_queue: asyncio.Queue[PipelineEvent],
    ) -> None:
        await event_queue.put(PipelineEvent(
            type=PipelineEventType.STAGE_START,
            project_id=context.project_id,
            data={"stage": self.name, "message": "Decomposing project into tasks"},
        ))

        await self._report_progress(0.0, "Analyzing project description...")

        # Simulate decomposition (would call actual decomposer)
        await asyncio.sleep(0.5)  # Placeholder

        await self._report_progress(100.0, f"Created {len(context.tasks)} tasks")

        await event_queue.put(PipelineEvent(
            type=PipelineEventType.STAGE_COMPLETE,
            project_id=context.project_id,
            data={"stage": self.name, "tasks_created": len(context.tasks)},
        ))


class ExecuteStage(PipelineStage):
    """Stage for parallel task execution with dependency resolution."""

    def __init__(self, max_parallel: int = 3):
        super().__init__("execute")
        self.max_parallel = max_parallel

    async def execute(
        self,
        context: StreamingContext,
        event_queue: asyncio.Queue[PipelineEvent],
    ) -> None:
        await event_queue.put(PipelineEvent(
            type=PipelineEventType.STAGE_START,
            project_id=context.project_id,
            data={"stage": self.name, "max_parallel": self.max_parallel},
        ))

        semaphore = asyncio.Semaphore(self.max_parallel)
        running_tasks: set[asyncio.Task] = set()

        while True:
            # Start ready tasks up to max_parallel
            ready = context.get_ready_tasks()

            for st in ready[:self.max_parallel - len(running_tasks)]:
                task_coro = self._execute_single_task(st, context, event_queue, semaphore)
                running_tasks.add(asyncio.create_task(task_coro))
                st.status = "running"
                st.start_time = time.time()

            if not running_tasks and not ready:
                # No more tasks to run
                break

            # Wait for at least one task to complete
            done, running_tasks = await asyncio.wait(
                running_tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Process completed tasks
            for task in done:
                try:
                    await task
                except Exception as e:
                    logger.error(f"Task execution failed: {e}")

            # Report progress
            total = len(context.tasks)
            completed = len(context.completed_tasks)
            progress = (completed / total * 100) if total > 0 else 100

            await self._report_progress(progress, f"{completed}/{total} tasks complete")

            await event_queue.put(PipelineEvent(
                type=PipelineEventType.STAGE_PROGRESS,
                project_id=context.project_id,
                data={
                    "stage": self.name,
                    "progress": progress,
                    "completed": completed,
                    "total": total,
                    "failed": len(context.failed_tasks),
                },
            ))

        await event_queue.put(PipelineEvent(
            type=PipelineEventType.STAGE_COMPLETE,
            project_id=context.project_id,
            data={
                "stage": self.name,
                "completed": len(context.completed_tasks),
                "failed": len(context.failed_tasks),
            },
        ))

    async def _execute_single_task(
        self,
        st: StreamingTask,
        context: StreamingContext,
        event_queue: asyncio.Queue[PipelineEvent],
        semaphore: asyncio.Semaphore,
    ) -> None:
        """Execute a single task with semaphore-controlled concurrency."""
        async with semaphore:
            # Emit task start
            await event_queue.put(PipelineEvent(
                type=PipelineEventType.TASK_START,
                project_id=context.project_id,
                data={
                    "task_id": st.task.id,
                    "task_type": st.task.task_type.value,
                    "attempt": st.retry_count + 1,
                },
            ))

            try:
                # Simulate task execution (would call actual executor)
                # This is where the real orchestrator integration happens
                await asyncio.sleep(0.1)  # Placeholder

                # Mark as complete
                st.status = "completed"
                st.end_time = time.time()
                context.completed_tasks.add(st.task.id)

                # Emit task complete
                await event_queue.put(PipelineEvent(
                    type=PipelineEventType.TASK_COMPLETE,
                    project_id=context.project_id,
                    data={
                        "task_id": st.task.id,
                        "duration_ms": st.duration_ms,
                        "status": "success",
                    },
                ))

            except Exception as e:
                st.status = "failed"
                st.end_time = time.time()
                context.failed_tasks.add(st.task.id)

                await event_queue.put(PipelineEvent(
                    type=PipelineEventType.TASK_FAILED,
                    project_id=context.project_id,
                    data={
                        "task_id": st.task.id,
                        "error": str(e),
                        "duration_ms": st.duration_ms,
                    },
                ))


class ValidateStage(PipelineStage):
    """Stage for validation of outputs."""

    def __init__(self):
        super().__init__("validate")

    async def execute(
        self,
        context: StreamingContext,
        event_queue: asyncio.Queue[PipelineEvent],
    ) -> None:
        await event_queue.put(PipelineEvent(
            type=PipelineEventType.STAGE_START,
            project_id=context.project_id,
            data={"stage": self.name},
        ))

        # Validation logic here
        await asyncio.sleep(0.2)  # Placeholder

        await event_queue.put(PipelineEvent(
            type=PipelineEventType.STAGE_COMPLETE,
            project_id=context.project_id,
            data={"stage": self.name, "passed": True},
        ))


# ═══════════════════════════════════════════════════════════════════════════════
# Streaming Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

class StreamingPipeline:
    """
    Real-time streaming pipeline for project execution.

    Emits events as they happen, allowing clients to track progress
    without blocking until completion.
    """

    def __init__(
        self,
        max_parallel: int = 3,
        event_bus: EventBus | None = None,
    ):
        self.max_parallel = max_parallel
        self.event_bus = event_bus or get_event_bus()
        self.stages: list[PipelineStage] = [
            DecomposeStage(),
            ExecuteStage(max_parallel=max_parallel),
            ValidateStage(),
        ]

    async def execute_streaming(
        self,
        project_description: str,
        success_criteria: str,
        budget: float = 5.0,
        project_id: str | None = None,
    ) -> AsyncIterator[PipelineEvent]:
        """
        Execute a project with real-time streaming.

        Yields PipelineEvent objects as execution progresses.
        """
        # Generate project ID
        if project_id is None:
            import hashlib
            project_id = hashlib.sha256(
                f"{project_description}:{time.time()}".encode()
            ).hexdigest()[:12]

        # Create context
        context = StreamingContext(
            project_id=project_id,
            description=project_description,
            budget=Budget(max_usd=budget),
        )

        # Event queue for internal communication
        event_queue: asyncio.Queue[PipelineEvent] = asyncio.Queue()

        # Start pipeline execution in background
        pipeline_task = asyncio.create_task(
            self._run_pipeline(context, event_queue)
        )

        # Yield events as they arrive
        try:
            while True:
                # Wait for event with timeout to check if pipeline is done
                try:
                    event = await asyncio.wait_for(
                        event_queue.get(),
                        timeout=0.1
                    )
                    yield event

                    # Also emit to global event bus
                    await self._emit_to_bus(event)

                except asyncio.TimeoutError:
                    # Check if pipeline is done
                    if pipeline_task.done():
                        # Drain remaining events
                        while not event_queue.empty():
                            yield event_queue.get_nowait()
                        break
        finally:
            # Ensure pipeline task is cleaned up
            if not pipeline_task.done():
                pipeline_task.cancel()
                try:
                    await pipeline_task
                except asyncio.CancelledError:
                    pass

    async def _run_pipeline(
        self,
        context: StreamingContext,
        event_queue: asyncio.Queue[PipelineEvent],
    ) -> None:
        """Run all pipeline stages."""
        try:
            # Emit project start
            await event_queue.put(PipelineEvent(
                type=PipelineEventType.PROJECT_START,
                project_id=context.project_id,
                data={
                    "description": project_description[:100],
                    "budget": context.budget.max_usd,
                },
            ))

            # Run each stage
            for stage in self.stages:
                stage.on_progress(
                    lambda p, m: self._on_stage_progress(
                        stage.name, p, m, context, event_queue
                    )
                )
                await stage.execute(context, event_queue)

            # Emit project complete
            await event_queue.put(PipelineEvent(
                type=PipelineEventType.PROJECT_COMPLETE,
                project_id=context.project_id,
                data={
                    "status": "success" if not context.failed_tasks else "partial",
                    "completed_tasks": len(context.completed_tasks),
                    "failed_tasks": len(context.failed_tasks),
                    "total_cost": context.budget.spent_usd,
                },
            ))

        except Exception as e:
            logger.exception("Pipeline execution failed")
            await event_queue.put(PipelineEvent(
                type=PipelineEventType.ERROR,
                project_id=context.project_id,
                data={"error": str(e), "type": type(e).__name__},
            ))

    async def _on_stage_progress(
        self,
        stage_name: str,
        percent: float,
        message: str,
        context: StreamingContext,
        event_queue: asyncio.Queue[PipelineEvent],
    ) -> None:
        """Handle stage progress updates."""
        context.current_stage = stage_name
        context.stage_progress = percent

    async def _emit_to_bus(self, event: PipelineEvent) -> None:
        """Convert pipeline event to domain event and emit to bus."""
        try:
            domain_event = self._to_domain_event(event)
            if domain_event:
                await self.event_bus.publish(domain_event)
        except Exception as e:
            logger.error(f"Failed to emit to event bus: {e}")

    def _to_domain_event(self, event: PipelineEvent) -> DomainEvent | None:
        """Convert pipeline event to domain event."""
        if event.type == PipelineEventType.PROJECT_START:
            return ProjectStartedEvent(
                aggregate_id=event.project_id,
                project_id=event.project_id,
                description=event.data.get("description", ""),
                budget=event.data.get("budget", 0.0),
            )
        elif event.type == PipelineEventType.TASK_START:
            return TaskStartedEvent(
                aggregate_id=event.data.get("task_id", ""),
                task_id=event.data.get("task_id", ""),
                task_type=event.data.get("task_type", ""),
            )
        elif event.type == PipelineEventType.TASK_COMPLETE:
            return TaskCompletedEvent(
                aggregate_id=event.data.get("task_id", ""),
                task_id=event.data.get("task_id", ""),
                score=1.0,  # Would come from actual result
            )
        elif event.type == PipelineEventType.TASK_FAILED:
            return TaskFailedEvent(
                aggregate_id=event.data.get("task_id", ""),
                task_id=event.data.get("task_id", ""),
                error=event.data.get("error", ""),
            )
        elif event.type == PipelineEventType.PROJECT_COMPLETE:
            return ProjectCompletedEvent(
                aggregate_id=event.project_id,
                project_id=event.project_id,
                status=event.data.get("status", "unknown"),
                total_cost=event.data.get("total_cost", 0.0),
            )
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# WebSocket Handler Helper
# ═══════════════════════════════════════════════════════════════════════════════

class WebSocketStreamingHandler:
    """Helper for streaming pipeline events over WebSocket."""

    def __init__(self, pipeline: StreamingPipeline):
        self.pipeline = pipeline

    async def handle(
        self,
        websocket,
        project_description: str,
        success_criteria: str,
        budget: float = 5.0,
    ) -> None:
        """
        Handle WebSocket connection with streaming execution.

        Usage (FastAPI):
            @app.websocket("/ws/execute")
            async def ws_endpoint(websocket: WebSocket):
                await websocket.accept()
                handler = WebSocketStreamingHandler(pipeline)
                await handler.handle(websocket, "Build API", "Works", 5.0)
        """
        try:
            async for event in self.pipeline.execute_streaming(
                project_description,
                success_criteria,
                budget,
            ):
                await websocket.send_json(event.to_dict())

                # Allow client to cancel
                try:
                    message = await asyncio.wait_for(
                        websocket.receive_text(),
                        timeout=0.01
                    )
                    if message == "cancel":
                        await websocket.send_json({
                            "type": "CANCELLED",
                            "message": "Execution cancelled by client"
                        })
                        break
                except asyncio.TimeoutError:
                    pass  # No message, continue

        except Exception as e:
            await websocket.send_json({
                "type": "ERROR",
                "error": str(e),
            })
        finally:
            await websocket.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Project Event Bus (Legacy Compatibility)
# ═══════════════════════════════════════════════════════════════════════════════

class ProjectEventBus:
    """
    Legacy event bus for streaming project execution.
    Wraps the standard EventBus for backward compatibility.
    """

    def __init__(self):
        self._event_bus = get_event_bus()
        self._queue: asyncio.Queue = asyncio.Queue()
        self._subscribers: list[asyncio.Queue] = []
        self._running = False
        self._task: asyncio.Task | None = None

    async def subscribe(self) -> AsyncIterator[Any]:
        """Subscribe to events. Returns an async iterator."""
        while True:
            event = await self._queue.get()
            if event is None:  # Sentinel to signal end
                break
            yield event

    async def publish(self, event: Any) -> None:
        """Publish an event."""
        await self._queue.put(event)

    async def close(self) -> None:
        """Close the event bus. Sends sentinel to unblock subscribers."""
        self._running = False
        await self._queue.put(None)  # Sentinel to unblock subscribers
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass


# ═══════════════════════════════════════════════════════════════════════════════
# Example Usage
# ═══════════════════════════════════════════════════════════════════════════════

async def example():
    """Example of streaming pipeline usage."""
    pipeline = StreamingPipeline(max_parallel=3)

    print("Starting streaming execution...")

    async for event in pipeline.execute_streaming(
        project_description="Build a FastAPI REST API",
        success_criteria="All endpoints tested",
        budget=5.0,
    ):
        print(f"[{event.type.name}] {event.data}")


if __name__ == "__main__":
    asyncio.run(example())
