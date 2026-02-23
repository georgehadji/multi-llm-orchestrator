"""
Constraint Control Plane — Typed Specs
=======================================
JobSpecV2 and PolicySpecV2 extend the existing JobSpec / PolicySet with
structured SLAs, input descriptions, hard/soft constraints, routing hints,
validation rules and escalation rules.

These are used by ControlPlane.submit() and OrchestrationAgent.draft().
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .policy import Budget, PolicySet


# ─────────────────────────────────────────────
# SLA + Input description
# ─────────────────────────────────────────────

@dataclass
class SLAs:
    """Service-level agreements for a job run."""

    max_latency_ms: Optional[int] = None
    max_cost_usd: Optional[float] = None
    min_quality_tier: float = 0.85
    reliability_target: float = 0.95


@dataclass
class InputSpec:
    """Describes the shape and locality of input data."""

    schema: dict = field(default_factory=dict)
    data_locality: str = "any"   # "eu" | "us" | "any"
    contains_pii: bool = False


@dataclass
class Constraints:
    """Hard (boolean) and soft (weighted) constraints for a job."""

    hard: list[str] = field(default_factory=list)
    # e.g. ["no_training", "eu_only", "no_pii_logging"]
    soft: dict[str, float] = field(default_factory=dict)
    # e.g. {"prefer_low_cost": 0.8, "prefer_low_latency": 0.6}


# ─────────────────────────────────────────────
# JobSpecV2
# ─────────────────────────────────────────────

@dataclass
class JobSpecV2:
    """
    Extended job specification that includes structured SLAs, input
    descriptions, and constraint groups.

    Backward-compatible: ``budget`` and ``policy_set`` are kept so that
    ControlPlane can delegate to Orchestrator.run_job() via a thin adapter.
    """

    goal: str = ""
    inputs: InputSpec = field(default_factory=InputSpec)
    slas: SLAs = field(default_factory=SLAs)
    constraints: Constraints = field(default_factory=Constraints)
    metrics: list[str] = field(default_factory=list)
    task_tree: list[dict] = field(default_factory=list)
    # Backward-compatibility
    budget: Budget = field(default_factory=lambda: Budget(max_usd=8.0))
    policy_set: PolicySet = field(default_factory=PolicySet)


# ─────────────────────────────────────────────
# PolicySpecV2 components
# ─────────────────────────────────────────────

@dataclass
class RoutingHint:
    """Route tasks to specific model sets when a condition matches."""

    condition: str    # "eu_only AND contains_pii"
    target: str       # "self_hosted_only" | "eu_models_only"


@dataclass
class ValidationRule:
    """Require specific validators on tasks matching a node pattern."""

    node_pattern: str                    # TaskType value or "*"
    mandatory_validators: list[str] = field(default_factory=list)


@dataclass
class EscalationRule:
    """Trigger an action when a condition is met during execution."""

    trigger: str    # "validator_failed AND iterations >= 3"
    action: str     # "human_review" | "abort" | "fallback_model"


@dataclass
class PolicySpecV2:
    """
    Extended policy specification with routing hints, per-node validation
    rules, and escalation rules.
    """

    allow_deny_rules: list[dict] = field(default_factory=list)
    # [{"effect": "deny", "when": "risk_level == high AND jurisdiction != eu"}]
    routing_hints: list[RoutingHint] = field(default_factory=list)
    validation_rules: list[ValidationRule] = field(default_factory=list)
    escalation_rules: list[EscalationRule] = field(default_factory=list)


__all__ = [
    "SLAs", "InputSpec", "Constraints",
    "JobSpecV2",
    "RoutingHint", "ValidationRule", "EscalationRule",
    "PolicySpecV2",
]
