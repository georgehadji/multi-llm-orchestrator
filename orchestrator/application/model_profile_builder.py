"""
ModelProfile Builder — factory for default ModelProfile instances.
====================================================================

Moved from orchestrator/models.py to break a circular import:
policy.py → models.py, models.py → policy.py.

Rule 2: models.py = pure data. This factory belongs in the application
layer where it can freely import from both models.py and policy.py.
"""

from __future__ import annotations

from ..models import COST_TABLE, Model, ROUTING_TABLE, TaskType, get_provider
from ..policy import ModelProfile


def build_default_profiles() -> dict[Model, ModelProfile]:
    """
    Build a ModelProfile for every Model enum value using the static
    COST_TABLE and ROUTING_TABLE as the source of truth.

    Called once at Orchestrator construction time. Telemetry fields
    (quality_score, trust_factor, avg_latency_ms, …) start at their
    defaults and are updated at runtime by TelemetryCollector.
    """
    # Build capability map: {TaskType → priority_rank} for each model
    capability_map: dict[Model, dict[TaskType, int]] = {m: {} for m in Model}

    for task_type, model_list in ROUTING_TABLE.items():
        for rank, model in enumerate(model_list):
            capability_map[model][task_type] = rank

    profiles: dict[Model, ModelProfile] = {}

    for model in Model:
        costs = COST_TABLE.get(model, {"input": 5.0, "output": 20.0})

        profiles[model] = ModelProfile(
            model=model,
            provider=get_provider(model),
            cost_per_1m_input=costs["input"],
            cost_per_1m_output=costs["output"],
            capable_task_types=capability_map.get(model, {}),
        )

    return profiles
