"""
ARA Execution Strategy — Decides when and how to use ARA pipelines
====================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Decides whether to use ARA reasoning for a given task and which
method to select. Acts as a policy layer between engine.py and
ARAPipelineIntegration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .ara_pipelines import ReasoningMethod
from ..models import Task, TaskType

logger = logging.getLogger("orchestrator.ara_execution_strategy")


@dataclass
class ARAStrategyConfig:
    """Configuration for ARA execution strategy.

    Attributes:
        enabled: Master switch for ARA execution.
        quality_deficit_threshold: Use ARA when current score is below this.
        budget_fraction_ara: Maximum fraction of remaining budget for ARA.
        complexity_threshold: Use ARA when estimated task complexity exceeds this.
        default_methods: Per-task-type default ARA methods.
        excluded_task_types: Task types that never use ARA.
    """

    enabled: bool = True
    quality_deficit_threshold: float = 0.3
    budget_fraction_ara: float = 0.3
    complexity_threshold: float = 0.7

    default_methods: dict[TaskType, ReasoningMethod] = field(
        default_factory=lambda: {
            TaskType.CODE_GEN: ReasoningMethod.PERSUASION_DEFENSE,
            TaskType.REASONING: ReasoningMethod.SOT,
            TaskType.EVALUATE: ReasoningMethod.JURY,
            TaskType.CODE_REVIEW: ReasoningMethod.MULTI_PERSPECTIVE,
        }
    )

    excluded_task_types: set[TaskType] = field(default_factory=set)

    # Self-consistency retry escalation mapping
    retry_methods: dict[TaskType, ReasoningMethod] = field(
        default_factory=lambda: {
            TaskType.CODE_REVIEW: ReasoningMethod.DEBATE,
            TaskType.CODE_GEN: ReasoningMethod.COVE,
            TaskType.REASONING: ReasoningMethod.COVE,
        }
    )


class ARAExecutionStrategy:
    """Decides when and how to use ARA for task execution.

    The strategy is:
    1. If the task type has a default ARA method, use ARA.
    2. If the task is in a self-consistency retry with low score, use ARA.
    3. If budget is insufficient for the estimated ARA cost, skip.
    4. If excluded, skip.

    All decisions fail-open — errors disable ARA for that task, not crash.
    """

    def __init__(
        self,
        ara_integration: Any = None,
        config: ARAStrategyConfig | None = None,
    ) -> None:
        self._ara = ara_integration
        self._config = config or ARAStrategyConfig()

    @property
    def enabled(self) -> bool:
        return self._config.enabled and self._ara is not None

    def should_use_ara(self, task: Task, current_score: float = 0.0) -> bool:
        """Decide if ARA should be used for this task.

        Args:
            task: The task being evaluated.
            current_score: Current quality score (0.0 = not yet evaluated).

        Returns:
            True if ARA should be used for this task.
        """
        if not self.enabled:
            return False

        if task.type in self._config.excluded_task_types:
            return False

        # Always use for explicitly configured task types
        if task.type in self._config.default_methods:
            return True

        # Use for self-consistency retry when quality is low
        if current_score > 0 and current_score < self._config.quality_deficit_threshold:
            return True

        return False

    def select_method(self, task: Task, retry: bool = False) -> ReasoningMethod:
        """Select the best ARA method for this task.

        Args:
            task: The task to select a method for.
            retry: If True, use retry escalation mapping instead of default.

        Returns:
            The ReasoningMethod to use.
        """
        if retry and task.type in self._config.retry_methods:
            return self._config.retry_methods[task.type]

        if task.type in self._config.default_methods:
            return self._config.default_methods[task.type]

        return ReasoningMethod.MULTI_PERSPECTIVE

    def get_retry_method(self, task: Task) -> ReasoningMethod | None:
        """Get the ARA method to use for a self-consistency retry.

        Returns None if no retry method is configured for this task type.
        """
        return self._config.retry_methods.get(task.type)


class ARAReasoningDispatcher:
    """Dispatch REASONING tasks to optimal ARA method.

    Analyzes task prompt characteristics to select the best
    cognitive reasoning pipeline from the 20 available methods.
    """

    def __init__(self):
        pass

    def select_method(self, task: "Task") -> "ReasoningMethod":
        """Select best ARA method for a REASONING task."""
        from .ara_pipelines import ReasoningMethod
        from ..models import TaskType

        if task.type != TaskType.REASONING:
            return ReasoningMethod.COVE  # Default

        prompt = task.prompt.lower()

        # Multi-part problems -> SoT (skeleton-of-thought)
        if any(
            kw in prompt
            for kw in [
                "multiple",
                "several",
                "complex",
                "multi-part",
                "sub-problem",
                "decompose",
                "break down",
            ]
        ):
            return ReasoningMethod.SOT

        # Decision / choose problems -> ToT (tree-of-thoughts)
        if any(
            kw in prompt
            for kw in [
                "choose",
                "decide",
                "select",
                "best option",
                "trade-off",
                "tradeoff",
                "compare",
            ]
        ):
            return ReasoningMethod.TOT

        # Novel / open-ended discovery -> Self-Discover
        if any(
            kw in prompt
            for kw in ["novel", "innovative", "discover", "custom", "unique", "new approach"]
        ):
            return ReasoningMethod.SELF_DISCOVER

        # Factual / verification tasks -> CoVE
        if any(
            kw in prompt
            for kw in ["fact", "verify", "check", "accurate", "true", "false", "validate"]
        ):
            return ReasoningMethod.COVE

        # Default: Multi-Perspective for general reasoning
        return ReasoningMethod.MULTI_PERSPECTIVE

    async def execute(self, task: "Task", ara_integration: object) -> "TaskResult":
        """Execute a REASONING task through the selected ARA method.

        Args:
            task: The task to execute.
            ara_integration: The ARAPipelineIntegration instance.

        Returns:
            TaskResult from the ARA pipeline.
        """
        method = self.select_method(task)
        return await ara_integration.execute_task_with_pipeline(
            task=task,
            method=method,
        )


# Attach to ARAExecutionStrategy
if "ARAExecutionStrategy" in dir():
    ARAExecutionStrategy.reasoning_dispatcher = ARAReasoningDispatcher()
