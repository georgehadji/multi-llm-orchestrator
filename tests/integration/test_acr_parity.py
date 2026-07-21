"""
Integration test: ORCH_ACR_BACKEND=off must produce byte-identical routing
to the pre-ACR ConstraintPlanner. This is the Phase 0 acceptance criterion,
not just unit coverage.
"""

import pytest

pytestmark = pytest.mark.integration

from orchestrator.models import Model, TaskType
from orchestrator.planner import ConstraintPlanner
from orchestrator.policy import ModelProfile
from orchestrator.policy_engine import PolicyEngine


def _profiles_fixture() -> dict[Model, ModelProfile]:
    models = list(Model)[:5]
    profiles = {}
    for i, m in enumerate(models):
        profiles[m] = ModelProfile(
            model=m,
            provider="test",
            cost_per_1m_input=1.0 + i,
            cost_per_1m_output=2.0 + i,
            quality_score=0.5 + i * 0.05,
            trust_factor=1.0,
            capable_task_types={
                TaskType.CODE_GEN: i,
                TaskType.REASONING: i,
                TaskType.EVALUATE: i,
            },
        )
    return profiles


@pytest.mark.integration
@pytest.mark.parametrize("task_type", [TaskType.CODE_GEN, TaskType.REASONING, TaskType.EVALUATE])
def test_flag_off_routing_identical_to_pre_acr_baseline(task_type):
    # Arrange — "old way": no backend kwarg, exactly as container.py did
    # before ACR wiring existed.
    profiles_a = _profiles_fixture()
    api_health_a = dict.fromkeys(profiles_a, True)
    baseline_planner = ConstraintPlanner(
        profiles=profiles_a, policy_engine=PolicyEngine(), api_health=api_health_a
    )

    # Arrange — planner as container.py constructs it today, then run through
    # _wire_acr_backend with flag "off" (the documented default no-op path).
    from orchestrator.engine_core.container import _wire_acr_backend
    from types import SimpleNamespace

    profiles_b = _profiles_fixture()
    api_health_b = dict.fromkeys(profiles_b, True)
    wired_planner = ConstraintPlanner(
        profiles=profiles_b, policy_engine=PolicyEngine(), api_health=api_health_b
    )
    _wire_acr_backend(wired_planner, SimpleNamespace(acr_backend="off"))

    # Act
    baseline_pick = baseline_planner.select_model(task_type, policies=[], budget_remaining=1000.0)
    wired_pick = wired_planner.select_model(task_type, policies=[], budget_remaining=1000.0)

    # Assert
    assert wired_pick == baseline_pick
