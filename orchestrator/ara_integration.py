"""
ARA Pipeline Integration — Engine Extension
============================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Integrates ARA reasoning pipelines into the AI Orchestrator engine.
Provides method dispatch, configuration, and execution wrappers.
"""

from __future__ import annotations

import logging
from typing import Any

from .api_clients import UnifiedClient
from .ara_pipelines import (
    PipelineFactory,
    ReasoningMethod,
)
from .cache import DiskCache
from .method_selector import (
    ComplexityLevel,
    MethodSelection,
    MethodSelector,
    RiskLevel,
    select_method_for_task,
)
from .models import Model, Task, TaskResult, TaskStatus, TaskType
from .telemetry import TelemetryCollector

logger = logging.getLogger("orchestrator")


class ARAPipelineIntegration:
    """
    Integration layer for ARA reasoning pipelines.

    Provides:
    - Method selection for tasks
    - Pipeline dispatch and execution
    - Result conversion and telemetry
    - Configuration management
    """

    def __init__(
        self,
        client: UnifiedClient,
        cache: DiskCache | None = None,
        telemetry: TelemetryCollector | None = None,
        default_method: ReasoningMethod = ReasoningMethod.MULTI_PERSPECTIVE,
        auto_select_method: bool = True,
        use_llm_for_selection: bool = True,
    ):
        """
        Initialize ARA pipeline integration.

        Args:
            client: API client for LLM calls
            cache: Optional disk cache
            telemetry: Optional telemetry collector
            default_method: Default reasoning method if auto-select disabled
            auto_select_method: Whether to automatically select method per task
            use_llm_for_selection: Whether to use LLM for method selection optimization
        """
        self.client = client
        self.cache = cache or DiskCache()
        self.telemetry = telemetry or TelemetryCollector({})
        self.default_method = default_method
        self.auto_select_method = auto_select_method
        self.use_llm_for_selection = use_llm_for_selection

        # Method selector
        self.selector = MethodSelector(client=client if use_llm_for_selection else None)

        # Configuration
        self.config = {
            "enabled": True,
            "max_cost_multiplier": 5.0,
            "max_time_multiplier": 2.0,
            "require_approval_for_method": [],  # Methods requiring approval
            "method_overrides": {},  # task_id → method override
        }

        # Statistics
        self.stats = {
            "tasks_executed": 0,
            "methods_used": {},
            "avg_cost_multiplier": 0.0,
            "avg_time_multiplier": 0.0,
        }

    def configure(
        self,
        enabled: bool = True,
        max_cost_multiplier: float | None = None,
        max_time_multiplier: float | None = None,
        method_overrides: dict[str, str] | None = None,
    ):
        """
        Configure ARA pipeline integration.

        Args:
            enabled: Enable/disable ARA pipelines
            max_cost_multiplier: Maximum allowed cost multiplier
            max_time_multiplier: Maximum allowed time multiplier
            method_overrides: Dict of task_id → method_name overrides
        """
        self.config["enabled"] = enabled

        if max_cost_multiplier is not None:
            self.config["max_cost_multiplier"] = max_cost_multiplier

        if max_time_multiplier is not None:
            self.config["max_time_multiplier"] = max_time_multiplier

        if method_overrides:
            self.config["method_overrides"].update(method_overrides)

    def select_method_for_task(
        self,
        task: Task,
        complexity: str | None = None,
        risk: str | None = None,
        override_check: bool = False,
    ) -> MethodSelection:
        """
        Select reasoning method for a task.

        Args:
            task: Task to select method for
            complexity: Optional complexity override
            risk: Optional risk level override
            override_check: If True, check for manual overrides first

        Returns:
            MethodSelection with recommended method
        """
        # Check for manual override
        if override_check and task.id in self.config["method_overrides"]:
            override_method = self.config["method_overrides"][task.id]
            try:
                method = ReasoningMethod(override_method)
                return MethodSelection(
                    method=method,
                    confidence=1.0,
                    rationale="Manual override",
                    alternative_methods=[],
                    estimated_cost_multiplier=1.0,
                    estimated_time_multiplier=1.0,
                )
            except ValueError:
                logger.warning(f"Invalid method override: {override_method}")

        # Auto-select method
        if self.auto_select_method:
            # Determine complexity and risk from task
            if complexity is None:
                complexity = self._estimate_complexity(task)

            if risk is None:
                risk = self._estimate_risk(task)

            return select_method_for_task(
                task=task,
                complexity=complexity,
                risk=risk,
                use_llm=self.use_llm_for_selection,
                client=self.client,
            )

        # Use default method
        return MethodSelection(
            method=self.default_method,
            confidence=0.5,
            rationale="Default method",
            alternative_methods=[],
            estimated_cost_multiplier=1.0,
            estimated_time_multiplier=1.0,
        )

    async def execute_task_with_pipeline(
        self,
        task: Task,
        context: str = "",
        method: ReasoningMethod | None = None,
    ) -> TaskResult:
        """
        Execute a task using an ARA reasoning pipeline.

        Args:
            task: Task to execute
            context: Optional context from dependencies
            method: Optional method override (otherwise auto-selected)

        Returns:
            TaskResult from pipeline execution
        """
        if not self.config["enabled"]:
            # Fallback to standard execution
            logger.info("ARA pipelines disabled, using standard execution")
            return await self._fallback_execute(task, context)

        # Select method
        if method:
            method_selection = MethodSelection(
                method=method,
                confidence=1.0,
                rationale="Manual selection",
                alternative_methods=[],
                estimated_cost_multiplier=1.0,
                estimated_time_multiplier=1.0,
            )
        else:
            method_selection = self.select_method_for_task(task, override_check=True)

        selected_method = method_selection.method

        # Check cost/time constraints
        cost_mult = method_selection.estimated_cost_multiplier
        time_mult = method_selection.estimated_time_multiplier

        if cost_mult > self.config["max_cost_multiplier"]:
            logger.warning(
                f"Method {selected_method.value} exceeds cost limit "
                f"({cost_mult} > {self.config['max_cost_multiplier']})"
            )
            # Fallback to cheaper method
            selected_method = ReasoningMethod.MULTI_PERSPECTIVE

        if time_mult > self.config["max_time_multiplier"]:
            logger.warning(
                f"Method {selected_method.value} exceeds time limit "
                f"({time_mult} > {self.config['max_time_multiplier']})"
            )
            # Fallback to faster method
            selected_method = ReasoningMethod.ITERATIVE

        # Create pipeline
        try:
            pipeline = PipelineFactory.create(
                method=selected_method,
                client=self.client,
                cache=self.cache,
                telemetry=self.telemetry,
            )
        except ValueError as e:
            logger.error(f"Failed to create pipeline: {e}")
            return await self._fallback_execute(task, context)

        # Execute pipeline
        logger.info(
            f"Executing task {task.id} with {selected_method.value} pipeline "
            f"(cost×{cost_mult:.1f}, time×{time_mult:.1f})"
        )

        try:
            result = await pipeline.execute(task=task, context=context)

            # Update statistics
            self._update_stats(selected_method, cost_mult, time_mult)

            # Add method metadata
            result.metadata["ara_method"] = selected_method.value
            result.metadata["method_confidence"] = method_selection.confidence
            result.metadata["method_rationale"] = method_selection.rationale

            return result

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            # Fallback to standard execution
            return await self._fallback_execute(task, context)

    async def _fallback_execute(self, task: Task, context: str) -> TaskResult:
        """Fallback to standard single-pass execution."""
        logger.info(f"Falling back to standard execution for task {task.id}")

        models = self._get_available_models(task.type)
        if not models:
            return TaskResult(
                task_id=task.id,
                output="",
                score=0.0,
                model_used=Model.GPT_4O_MINI,
                status=TaskStatus.FAILED,
            )

        primary = models[0]

        prompt = f"Task: {task.prompt}\n\nContext: {context}" if context else task.prompt

        response, _ = await self.client.call(
            model=primary,
            system_prompt="You are an expert AI assistant. Provide a comprehensive solution.",
            user_prompt=prompt,
            max_tokens=task.max_output_tokens,
            temperature=0.3,
        )

        return TaskResult(
            task_id=task.id,
            output=response.text,
            score=0.7,
            model_used=primary,
            status=TaskStatus.COMPLETED,
            metadata={"execution_mode": "fallback_standard"},
        )

    def _estimate_complexity(self, task: Task) -> ComplexityLevel:
        """Estimate task complexity from prompt."""
        prompt = task.prompt.lower()

        # Simple heuristics
        complexity_indicators = {
            ComplexityLevel.CRITICAL: [
                "mission-critical", "production", "enterprise", "scalable",
                "high-performance", "distributed", "real-time",
            ],
            ComplexityLevel.HIGH: [
                "complex", "advanced", "sophisticated", "multi-step",
                "integrate", "orchestrate", "architecture",
            ],
            ComplexityLevel.MEDIUM: [
                "implement", "build", "create", "develop",
                "feature", "module", "component",
            ],
            ComplexityLevel.LOW: [
                "simple", "basic", "hello world", "example",
                "utility", "helper", "snippet",
            ],
        }

        for level, indicators in complexity_indicators.items():
            if any(indicator in prompt for indicator in indicators):
                return level

        return ComplexityLevel.MEDIUM

    def _estimate_risk(self, task: Task) -> RiskLevel:
        """Estimate task risk from type and prompt."""
        # High-risk task types
        if task.type in [TaskType.CODE_REVIEW, TaskType.EVALUATE]:
            return RiskLevel.MEDIUM

        prompt = task.prompt.lower()

        # Risk indicators
        high_risk_keywords = [
            "security", "authentication", "payment", "financial",
            "database", "migration", "delete", "remove",
            "production", "live", "user data", "pii",
        ]

        medium_risk_keywords = [
            "api", "endpoint", "integration", "external",
            "cache", "performance", "optimization",
        ]

        if any(keyword in prompt for keyword in high_risk_keywords):
            return RiskLevel.HIGH

        if any(keyword in prompt for keyword in medium_risk_keywords):
            return RiskLevel.MEDIUM

        return RiskLevel.LOW

    def _get_available_models(self, task_type: TaskType) -> list:
        """Get available models for task type."""
        from .models import ROUTING_TABLE

        routing = ROUTING_TABLE.get(task_type, [])
        return [m for m in routing if True]  # Could check api_health

    def _update_stats(self, method: ReasoningMethod, cost_mult: float, time_mult: float):
        """Update execution statistics."""
        self.stats["tasks_executed"] += 1

        # Update method counts
        method_name = method.value
        self.stats["methods_used"][method_name] = (
            self.stats["methods_used"].get(method_name, 0) + 1
        )

        # Update averages
        n = self.stats["tasks_executed"]
        self.stats["avg_cost_multiplier"] = (
            (self.stats["avg_cost_multiplier"] * (n - 1) + cost_mult) / n
        )
        self.stats["avg_time_multiplier"] = (
            (self.stats["avg_time_multiplier"] * (n - 1) + time_mult) / n
        )

    def get_stats(self) -> dict[str, Any]:
        """Get execution statistics."""
        return self.stats.copy()

    def get_method_distribution(self) -> dict[str, float]:
        """Get method usage distribution as percentages."""
        total = self.stats["tasks_executed"]
        if total == 0:
            return {}

        return {
            method: (count / total) * 100
            for method, count in self.stats["methods_used"].items()
        }


# ─────────────────────────────────────────────
# Engine Integration Helper
# ─────────────────────────────────────────────

def create_ara_integration(
    client: UnifiedClient | None = None,
    cache: DiskCache | None = None,
    telemetry: TelemetryCollector | None = None,
    enabled: bool = True,
    auto_select: bool = True,
) -> ARAPipelineIntegration:
    """
    Create ARA pipeline integration with standard configuration.

    Args:
        client: API client (uses default if None)
        cache: Disk cache (uses default if None)
        telemetry: Telemetry collector (uses default if None)
        enabled: Enable ARA pipelines
        auto_select: Auto-select method per task

    Returns:
        Configured ARAPipelineIntegration instance
    """
    if client is None:
        client = UnifiedClient()

    integration = ARAPipelineIntegration(
        client=client,
        cache=cache,
        telemetry=telemetry,
        default_method=ReasoningMethod.MULTI_PERSPECTIVE,
        auto_select_method=auto_select,
        use_llm_for_selection=auto_select,
    )

    integration.configure(enabled=enabled)

    return integration


# ─────────────────────────────────────────────
# Exports
# ─────────────────────────────────────────────

__all__ = [
    "ARAPipelineIntegration",
    "create_ara_integration",
]
