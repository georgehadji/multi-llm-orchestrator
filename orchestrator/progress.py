"""
Terminal progress renderer for run_project_streaming().

Prints compact task-by-task progress to stderr, leaving stdout clean for
piped output. Use quiet=True in tests or when --quiet CLI flag is set.
"""

from __future__ import annotations

import sys
from typing import Any

try:
    from .unified_events import (
        BudgetWarningEvent as BudgetWarning,
    )
    from .unified_events import (
        ProjectCompletedEvent as ProjectCompleted,
    )
    from .unified_events import (
        ProjectStartedEvent as ProjectStarted,
    )
    from .unified_events import (
        TaskCompletedEvent as TaskCompleted,
    )
    from .unified_events import (
        TaskFailedEvent as TaskFailed,
    )
    from .unified_events import (
        TaskProgressEvent as TaskProgressUpdate,
    )
    from .unified_events import (
        TaskStartedEvent as TaskStarted,
    )
except ImportError:
    # Fallback to standard events
    from .events import (
        BudgetWarningEvent as BudgetWarning,
    )
    from .events import (
        ProjectCompletedEvent as ProjectCompleted,
    )
    from .events import (
        ProjectStartedEvent as ProjectStarted,
    )
    from .events import (
        TaskCompletedEvent as TaskCompleted,
    )
    from .events import (
        TaskFailedEvent as TaskFailed,
    )
    from .events import (
        TaskProgressEvent as TaskProgressUpdate,
    )
    from .events import (
        TaskStartedEvent as TaskStarted,
    )
from .models import TaskStatus

_STATUS_ICONS: dict[TaskStatus, str] = {
    TaskStatus.COMPLETED: "✓",
    TaskStatus.DEGRADED: "~",
    TaskStatus.FAILED: "✗",
}


class ProgressRenderer:
    """
    Stateful event handler that prints live progress to stderr.
    Maintains counters so callers can inspect final state.
    """

    def __init__(self, quiet: bool = False) -> None:
        self.quiet = quiet
        self.total: int = 0
        self.completed: int = 0
        self.failed: int = 0
        self._active: dict[str, str] = {}  # task_id → model

    def handle(self, event: Any) -> None:
        if isinstance(event, ProjectStarted):
            # Handle both unified_events (metadata) and events.py (direct attribute)
            total_tasks = getattr(event, "total_tasks", None) or event.metadata.get(
                "total_tasks", 0
            )
            budget_usd = (
                getattr(event, "budget_usd", None)
                or getattr(event, "budget", None)
                or event.metadata.get("budget", 0.0)
            )
            self.total = total_tasks
            if not self.quiet:
                print(
                    f"\n▶  Project started — {total_tasks} tasks  " f"budget=${budget_usd:.2f}",
                    file=sys.stderr,
                )
        elif isinstance(event, TaskStarted):
            # Handle both unified_events (metadata) and events.py (direct attribute)
            task_id = getattr(event, "task_id", None) or event.metadata.get("task_id", "unknown")
            model = getattr(event, "model", None) or event.metadata.get("model", "unknown")
            task_type = getattr(event, "task_type", None) or event.metadata.get(
                "task_type", "unknown"
            )
            self._active[task_id] = model
            if not self.quiet:
                print(
                    f"   → {task_id}  [{task_type}]  model={model}",
                    file=sys.stderr,
                )
        elif TaskProgressUpdate and isinstance(event, TaskProgressUpdate):
            if not self.quiet:
                # Handle both unified_events (metadata) and events.py (direct attribute)
                task_id = getattr(event, "task_id", None) or event.metadata.get(
                    "task_id", "unknown"
                )
                iteration = getattr(event, "iteration", None) or event.metadata.get("iteration", 0)
                score = getattr(event, "score", None) or event.metadata.get("score", 0.0)
                best_score = getattr(event, "best_score", None)
                if best_score is None:
                    best_score = (
                        event.metadata.get("score", score) if hasattr(event, "metadata") else score
                    )
                print(
                    f"     {task_id}  iter={iteration}  "
                    f"score={score:.3f}  best={best_score:.3f}",
                    file=sys.stderr,
                )
        elif isinstance(event, TaskCompleted):
            # Handle both unified_events (metadata) and events.py (direct attribute)
            task_id = getattr(event, "task_id", None) or event.metadata.get("task_id", "unknown")
            score = getattr(event, "score", None) or event.metadata.get("score", 0.0)
            cost_usd = getattr(event, "cost_usd", None) or event.metadata.get("cost_usd", 0.0)
            iterations = getattr(event, "iterations", None) or event.metadata.get("iterations", 1)
            status = getattr(event, "status", None)
            if status is None:
                status_str = event.metadata.get("status", "completed")
                from .models import TaskStatus

                status = (
                    TaskStatus(status_str)
                    if status_str in ["completed", "failed", "degraded"]
                    else TaskStatus.COMPLETED
                )
            self.completed += 1
            self._active.pop(task_id, None)
            icon = _STATUS_ICONS.get(status, "?")
            if not self.quiet:
                print(
                    f"   {icon} {task_id}  score={score:.3f}  "
                    f"${cost_usd:.4f}  iters={iterations}  "
                    f"[{self.completed}/{self.total}]",
                    file=sys.stderr,
                )
        elif isinstance(event, TaskFailed):
            # Handle both unified_events (metadata) and events.py (direct attribute)
            task_id = getattr(event, "task_id", None) or event.metadata.get("task_id", "unknown")
            reason = (
                getattr(event, "reason", None)
                or getattr(event, "error", None)
                or event.metadata.get("reason")
                or event.metadata.get("error", "Unknown error")
            )
            self.failed += 1
            self._active.pop(task_id, None)
            if not self.quiet:
                print(
                    f"   ✗ {task_id}  FAILED: {reason}",
                    file=sys.stderr,
                )
        elif isinstance(event, BudgetWarning):
            # Handle both unified_events (metadata) and events.py (direct attribute)
            phase = getattr(event, "phase", None) or event.metadata.get("phase", "unknown")
            spent_usd = (
                getattr(event, "spent_usd", None)
                or getattr(event, "spent", None)
                or event.metadata.get("spent", 0.0)
            )
            cap_usd = (
                getattr(event, "cap_usd", None)
                or getattr(event, "budget", None)
                or event.metadata.get("budget", 0.0)
            )
            ratio = getattr(event, "ratio", None) or event.metadata.get("ratio", 0.0)
            if not self.quiet:
                print(
                    f"   ⚠  Budget {phase}: "
                    f"${spent_usd:.4f} / ${cap_usd:.4f} "
                    f"({ratio:.0%})",
                    file=sys.stderr,
                )
        elif isinstance(event, ProjectCompleted):
            # Handle both unified_events (metadata) and events.py (direct attribute)
            status = getattr(event, "status", None) or event.metadata.get("status", "unknown")
            total_cost_usd = (
                getattr(event, "total_cost_usd", None)
                or getattr(event, "total_cost", None)
                or event.metadata.get("total_cost", 0.0)
            )
            elapsed_seconds = (
                getattr(event, "elapsed_seconds", None)
                or getattr(event, "duration_seconds", None)
                or event.metadata.get("duration_seconds", 0.0)
            )
            tasks_completed = getattr(event, "tasks_completed", None) or event.metadata.get(
                "tasks_completed", 0
            )
            tasks_failed = getattr(event, "tasks_failed", None) or event.metadata.get(
                "tasks_failed", 0
            )
            if not self.quiet:
                marker = "✓" if "SUCCESS" in status else "~"
                print(
                    f"\n{marker} Project {status}  "
                    f"${total_cost_usd:.4f}  "
                    f"{elapsed_seconds:.0f}s  "
                    f"{tasks_completed} completed  "
                    f"{tasks_failed} failed",
                    file=sys.stderr,
                )

    def summary(self) -> str:
        return f"{self.completed} completed / {self.total} total, " f"{self.failed} failed"
