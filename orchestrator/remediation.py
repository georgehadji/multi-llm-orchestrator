"""
Auto-Remediation Engine.

When a task fails or scores below threshold, RemediationEngine selects
the next recovery strategy from an ordered RemediationPlan.

Strategies (tried in the order specified):
  AUTO_RETRY       — retry with same model and prompt
  FALLBACK_MODEL   — try the next model in the fallback chain
  ADJUST_PROMPT    — rephrase the task prompt and retry
  SKIP_VALIDATOR   — bypass hard validators, accept any output
  DEGRADE_QUALITY  — lower acceptance threshold by 15%
  ABORT_TASK       — give up, mark FAILED, continue with next task
"""
from __future__ import annotations
from enum import Enum
from typing import Optional
from .models import TaskResult, TaskStatus


class RemediationStrategy(str, Enum):
    AUTO_RETRY      = "auto_retry"
    FALLBACK_MODEL  = "fallback_model"
    ADJUST_PROMPT   = "adjust_prompt"
    SKIP_VALIDATOR  = "skip_validator"
    DEGRADE_QUALITY = "degrade_quality"
    ABORT_TASK      = "abort_task"


_DEFAULT_PLAN = [
    RemediationStrategy.AUTO_RETRY,
    RemediationStrategy.FALLBACK_MODEL,
    RemediationStrategy.DEGRADE_QUALITY,
    RemediationStrategy.ABORT_TASK,
]

_DEGRADE_FACTOR = 0.85   # reduce threshold by 15%


class RemediationPlan:
    """Ordered list of strategies; call advance() after each attempt."""

    def __init__(self, strategies: list[RemediationStrategy]) -> None:
        self._strategies = list(strategies)
        self._index = 0

    def next_strategy(self) -> Optional[RemediationStrategy]:
        if self._index >= len(self._strategies):
            return None
        return self._strategies[self._index]

    def advance(self) -> None:
        self._index += 1

    def exhausted(self) -> bool:
        return self._index >= len(self._strategies)

    def reset(self) -> None:
        self._index = 0


class RemediationEngine:
    """
    Decides whether a TaskResult warrants remediation and
    what the next strategy should be.
    """

    def should_remediate(self, result: TaskResult, threshold: float) -> bool:
        return (
            result.status == TaskStatus.FAILED
            or result.score < threshold
        )

    def adjusted_threshold(
        self,
        strategy: RemediationStrategy,
        original_threshold: float,
    ) -> float:
        if strategy == RemediationStrategy.DEGRADE_QUALITY:
            return max(0.0, original_threshold * _DEGRADE_FACTOR)
        return original_threshold

    def rephrase_prompt(self, original_prompt: str) -> str:
        return (
            "Please provide a complete, detailed, and correct response "
            "to the following task. Be thorough and precise.\n\n"
            + original_prompt
        )

    def default_plan(self) -> RemediationPlan:
        return RemediationPlan(list(_DEFAULT_PLAN))

    def plan_from_list(self, strategies: list[str]) -> RemediationPlan:
        return RemediationPlan([RemediationStrategy(s) for s in strategies])
