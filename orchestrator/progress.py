"""
Terminal progress renderer for run_project_streaming().

Prints compact task-by-task progress to stderr, leaving stdout clean for
piped output. Use quiet=True in tests or when --quiet CLI flag is set.
"""
from __future__ import annotations
import sys
from typing import Any

from .streaming import (
    ProjectStarted, TaskStarted, TaskProgressUpdate,
    TaskCompleted, TaskFailed, BudgetWarning, ProjectCompleted,
)
from .models import TaskStatus

_STATUS_ICONS: dict[TaskStatus, str] = {
    TaskStatus.COMPLETED: "✓",
    TaskStatus.DEGRADED:  "~",
    TaskStatus.FAILED:    "✗",
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
            self.total = event.total_tasks
            if not self.quiet:
                print(
                    f"\n▶  Project started — {event.total_tasks} tasks  "
                    f"budget=${event.budget_usd:.2f}",
                    file=sys.stderr,
                )
        elif isinstance(event, TaskStarted):
            self._active[event.task_id] = event.model
            if not self.quiet:
                print(
                    f"   → {event.task_id}  [{event.task_type}]  model={event.model}",
                    file=sys.stderr,
                )
        elif isinstance(event, TaskProgressUpdate):
            if not self.quiet:
                print(
                    f"     {event.task_id}  iter={event.iteration}  "
                    f"score={event.score:.3f}  best={event.best_score:.3f}",
                    file=sys.stderr,
                )
        elif isinstance(event, TaskCompleted):
            self.completed += 1
            self._active.pop(event.task_id, None)
            icon = _STATUS_ICONS.get(event.status, "?")
            if not self.quiet:
                print(
                    f"   {icon} {event.task_id}  score={event.score:.3f}  "
                    f"${event.cost_usd:.4f}  iters={event.iterations}  "
                    f"[{self.completed}/{self.total}]",
                    file=sys.stderr,
                )
        elif isinstance(event, TaskFailed):
            self.failed += 1
            self._active.pop(event.task_id, None)
            if not self.quiet:
                print(
                    f"   ✗ {event.task_id}  FAILED: {event.reason}",
                    file=sys.stderr,
                )
        elif isinstance(event, BudgetWarning):
            if not self.quiet:
                print(
                    f"   ⚠  Budget {event.phase}: "
                    f"${event.spent_usd:.4f} / ${event.cap_usd:.4f} "
                    f"({event.ratio:.0%})",
                    file=sys.stderr,
                )
        elif isinstance(event, ProjectCompleted):
            if not self.quiet:
                marker = "✓" if "SUCCESS" in event.status else "~"
                print(
                    f"\n{marker} Project {event.status}  "
                    f"${event.total_cost_usd:.4f}  "
                    f"{event.elapsed_seconds:.0f}s  "
                    f"{event.tasks_completed} completed  "
                    f"{event.tasks_failed} failed",
                    file=sys.stderr,
                )

    def summary(self) -> str:
        return (
            f"{self.completed} completed / {self.total} total, "
            f"{self.failed} failed"
        )
