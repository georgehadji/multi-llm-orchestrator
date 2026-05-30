"""
ConfigServices — Domain services for architectural logic
========================================================
Provides high-level access to routing, costs, and thresholds.
Abstracts away the underlying configuration storage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...models import Model, TaskType
    from ..ports import ConfigPort


class RoutingService:
    """Handles model selection and fallback logic."""

    def __init__(self, config: ConfigPort):
        self._config = config

    def get_models_for_task(self, task_type: TaskType) -> list[Model]:
        """Returns priority-ordered list of models for a task type."""
        from ...models import Model

        routing = self._config.get_routing()
        model_names = routing.get(task_type.value, [])
        return [Model(name) for name in model_names if name in Model._value2member_map_]

    def get_fallback_for_model(self, model: Model) -> Model | None:
        """Returns the configured fallback model for a given model."""
        from ...models import Model

        fallbacks = self._config.get_fallbacks()
        fallback_name = fallbacks.get(model.value)
        if fallback_name and fallback_name in Model._value2member_map_:
            return Model(fallback_name)
        return None


class CostService:
    """Handles pricing and budget calculations."""

    def __init__(self, config: ConfigPort):
        self._config = config

    def get_cost(self, model: Model) -> dict[str, float]:
        """Returns input/output costs per 1M tokens."""
        costs = self._config.get_costs()
        return costs.get(model.value, {"input": 0.0, "output": 0.0})


class ConfigurationService:
    """Handles task-specific thresholds and limits."""

    def __init__(self, config: ConfigPort):
        self._config = config

    def get_threshold(self, task_type: TaskType) -> float:
        """Returns the quality threshold for a task type."""
        thresholds = self._config.get_thresholds()
        return thresholds.get(task_type.value, 0.8)

    def get_max_tokens(self, task_type: TaskType) -> int:
        """Returns the max output tokens for a task type."""
        limits = self._config.get_limits()
        return limits.get(task_type.value, 4096)
