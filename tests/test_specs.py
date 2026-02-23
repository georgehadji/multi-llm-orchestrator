"""
Unit tests for specs.py (B1).
"""
from __future__ import annotations

import pytest

from orchestrator.specs import (
    Constraints,
    EscalationRule,
    InputSpec,
    JobSpecV2,
    PolicySpecV2,
    RoutingHint,
    SLAs,
    ValidationRule,
)


def test_job_spec_v2_defaults():
    job = JobSpecV2(goal="test")
    assert job.goal == "test"
    assert job.inputs.data_locality == "any"
    assert job.inputs.contains_pii is False
    assert job.slas.min_quality_tier == 0.85
    assert job.constraints.hard == []
    assert job.constraints.soft == {}


def test_input_spec_data_locality():
    spec = InputSpec(data_locality="eu", contains_pii=True)
    assert spec.data_locality == "eu"
    assert spec.contains_pii is True


def test_slas_max_cost():
    slas = SLAs(max_cost_usd=2.5, min_quality_tier=0.9)
    assert slas.max_cost_usd == 2.5
    assert slas.min_quality_tier == 0.9


def test_constraints_hard_and_soft():
    c = Constraints(
        hard=["no_training", "eu_only"],
        soft={"prefer_low_cost": 0.8},
    )
    assert "no_training" in c.hard
    assert c.soft["prefer_low_cost"] == 0.8


def test_routing_hint():
    hint = RoutingHint(condition="eu_only AND contains_pii", target="self_hosted_only")
    assert "eu_only" in hint.condition


def test_validation_rule():
    rule = ValidationRule(node_pattern="code_generation", mandatory_validators=["ruff"])
    assert rule.node_pattern == "code_generation"
    assert "ruff" in rule.mandatory_validators


def test_escalation_rule():
    rule = EscalationRule(trigger="validator_failed AND iterations >= 3", action="human_review")
    assert rule.action == "human_review"


def test_policy_spec_v2_defaults():
    pol = PolicySpecV2()
    assert pol.allow_deny_rules == []
    assert pol.routing_hints == []
    assert pol.validation_rules == []
    assert pol.escalation_rules == []


def test_job_spec_v2_budget_default():
    job = JobSpecV2(goal="hello")
    assert job.budget.max_usd == 8.0
