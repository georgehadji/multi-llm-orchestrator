"""
HookRegistry — lightweight event hook system for the orchestration lifecycle.
=============================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis
Provides a simple pub-sub mechanism for observing engine events without
modifying core orchestration logic. Callbacks are synchronous (fire-and-forget);
async callers can wrap with asyncio.create_task() if needed.

Events fired by Orchestrator (see EventType):
  TASK_STARTED      — before _execute_task() is called
  TASK_COMPLETED    — after _execute_task() returns
  VALIDATION_FAILED — when deterministic validators fail (det_passed=False)
  BUDGET_WARNING    — when a phase budget soft-cap is exceeded
  MODEL_SELECTED    — optional: when ConstraintPlanner selects a model
"""
from __future__ import annotations

import logging
from collections import defaultdict
from enum import Enum
from typing import Callable, Optional

logger = logging.getLogger("orchestrator.hooks")


# ─────────────────────────────────────────────────────────────────────────────
# EventType
# ─────────────────────────────────────────────────────────────────────────────

class EventType(str, Enum):
    """
    Standard lifecycle events fired by Orchestrator.

    Callback signatures (all kwargs):
      TASK_STARTED            — task_id: str, task: Task
      TASK_COMPLETED          — task_id: str, result: TaskResult
      VALIDATION_FAILED       — task_id: str, model: str, validators: list[str]
      BUDGET_WARNING          — phase: str, spent: float, cap: float, ratio: float
      MODEL_SELECTED          — task_id: str, model: str, backend: str
      TASK_RETRY_WITH_HISTORY — task_id: str, attempt_num: int, record: AttemptRecord
    """
    TASK_STARTED            = "task_started"
    TASK_COMPLETED          = "task_completed"
    VALIDATION_FAILED       = "validation_failed"
    BUDGET_WARNING          = "budget_warning"
    MODEL_SELECTED          = "model_selected"
    TASK_RETRY_WITH_HISTORY = "task_retry_with_history"


# ─────────────────────────────────────────────────────────────────────────────
# HookRegistry
# ─────────────────────────────────────────────────────────────────────────────

class HookRegistry:
    """
    Maps event names to lists of callback functions.

    Usage:
        registry = HookRegistry()
        registry.add(EventType.TASK_COMPLETED, lambda task_id, result, **_: print(task_id))
        registry.fire(EventType.TASK_COMPLETED, task_id="t_001", result=result)

    Exceptions thrown by individual callbacks are caught and logged, so one bad
    hook never prevents subsequent hooks or engine logic from running.
    """

    def __init__(self) -> None:
        self._hooks: dict[str, list[Callable]] = defaultdict(list)

    def add(self, event: str | EventType, callback: Callable) -> None:
        """Register a callback for the given event."""
        key = event.value if isinstance(event, EventType) else str(event)
        self._hooks[key].append(callback)

    def fire(self, event: str | EventType, **kwargs) -> None:
        """
        Invoke all callbacks registered for the event, passing kwargs.

        Exceptions from individual callbacks are caught and logged at WARNING
        level so one bad hook never crashes the engine loop.
        """
        key = event.value if isinstance(event, EventType) else str(event)
        for cb in self._hooks.get(key, []):
            try:
                cb(**kwargs)
            except Exception as exc:  # noqa: BLE001 — intentional broad catch
                logger.warning(
                    "Hook callback %r raised for event %r: %s",
                    cb, key, exc,
                )

    def clear(self, event: Optional[str | EventType] = None) -> None:
        """
        Remove callbacks.

        If event is specified, only that event's callbacks are cleared.
        If event is None, all callbacks for all events are cleared.
        """
        if event is None:
            self._hooks.clear()
        else:
            key = event.value if isinstance(event, EventType) else str(event)
            self._hooks.pop(key, None)

    def registered_events(self) -> list[str]:
        """Return the list of events that have at least one callback registered."""
        return [k for k, v in self._hooks.items() if v]

    def __len__(self) -> int:
        """Total number of registered callbacks across all events."""
        return sum(len(v) for v in self._hooks.values())
