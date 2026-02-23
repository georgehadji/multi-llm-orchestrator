"""
Policy data model — compliance rules, job specifications, and model profiles.
=============================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis
All types here are pure dataclasses with no side effects. They are seeded from
the existing ROUTING_TABLE / COST_TABLE / FALLBACK_CHAIN via
models.build_default_profiles() at Orchestrator construction time.

Novelty angle: This module makes the orchestrator's routing concern a
*first-class policy artifact* rather than implicit code. Every routing
decision can now be explained in terms of compliance tags, region constraints,
cost caps, and latency SLAs — not just "which model string is in a list".

Design: Placing policy types in their own module prevents circular imports.
policy.py imports from models.py (Budget, Model, TaskType).
models.py does NOT import from policy.py.
engine.py imports from both.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .models import Model, TaskType, Budget


# ─────────────────────────────────────────────────────────────────────────────
# EnforcementMode — controls how violations are handled
# ─────────────────────────────────────────────────────────────────────────────

class EnforcementMode(str, Enum):
    """
    Controls how PolicyEngine handles violations.

    MONITOR — log the violation but allow the model through (observability only)
    SOFT    — block hard violations (provider/model/region/compliance) but allow
              soft violations (latency SLA, cost cap) through
    HARD    — block on any violation (default; existing behaviour)
    """
    MONITOR = "monitor"
    SOFT    = "soft"
    HARD    = "hard"


# ─────────────────────────────────────────────────────────────────────────────
# RateLimit — per-policy rate cap (enforcement counted externally)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RateLimit:
    """Per-scope rate cap attached to a Policy. Actual enforcement is external."""
    calls_per_minute:  Optional[int]   = None
    cost_usd_per_hour: Optional[float] = None
    tokens_per_day:    Optional[int]   = None


# ─────────────────────────────────────────────────────────────────────────────
# ModelProfile — replaces static COST_TABLE + ROUTING_TABLE entries per model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelProfile:
    """
    Rich descriptor for a single LLM model.

    Static fields (cost, capability, region) are populated once from the
    existing lookup tables via build_default_profiles(). Telemetry fields
    (latency, quality, trust) are mutable and updated at runtime by
    TelemetryCollector after each real API call.

    The scoring function used by ConstraintPlanner is:
        score = quality_score * trust_factor / (estimated_cost + ε)

    This encodes the joint optimization: prefer high quality, high trust,
    low cost — subject to policy and budget constraints.
    """
    model: Model
    provider: str

    # ── Cost metadata (static, from COST_TABLE) ──────────────────────────────
    cost_per_1m_input: float
    cost_per_1m_output: float

    # ── Capability mapping (static, from ROUTING_TABLE) ───────────────────────
    # dict[TaskType → priority_rank]: lower rank = higher priority (0 = first choice)
    # A model is "capable" for a task type if it appears in ROUTING_TABLE[task_type].
    # Models not in ROUTING_TABLE for a given type are excluded from selection.
    capable_task_types: dict[TaskType, int] = field(default_factory=dict)

    # ── Adaptive telemetry (mutable, updated by TelemetryCollector) ───────────
    avg_latency_ms: float = 2000.0       # EMA of observed call latency
    latency_p95_ms: float = 5000.0       # approximated as 2× avg
    success_rate: float = 1.0            # rolling window: last 10 calls
    quality_score: float = 0.8           # EMA of LLM evaluator scores
    trust_factor: float = 1.0            # degrades on failures/violations, recovers on success

    # ── Compliance metadata (static, configurable per deployment) ─────────────
    region: str = "global"               # e.g. "eu", "us", "global"
    compliance_tags: list[str] = field(default_factory=list)
    # Known compliance_tags:
    #   "no_train"    — provider guarantees no training on API outputs
    #   "pii_allowed" — provider accepts PII in requests (GDPR compliant config)
    #   "soc2"        — SOC 2 Type II certified
    #   "hipaa"       — HIPAA compliant configuration available

    # ── Call counters (for telemetry windows) ─────────────────────────────────
    call_count: int = 0
    failure_count: int = 0

    # ── Extended telemetry (Improvement 3) ───────────────────────────────────
    avg_cost_usd:         float       = 0.0                           # EMA of actual per-call USD cost
    validator_fail_count: int         = 0                             # cumulative deterministic-check failures
    latency_samples:      list[float] = field(default_factory=list)   # sorted buffer, last 50 samples

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Compute estimated USD cost for a hypothetical call."""
        return (
            input_tokens * self.cost_per_1m_input
            + output_tokens * self.cost_per_1m_output
        ) / 1_000_000


# ─────────────────────────────────────────────────────────────────────────────
# Policy — a single named compliance rule set
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Policy:
    """
    Encodes a compliance rule that constrains model selection.
    Multiple Policy objects compose into a PolicySet.
    Any dimension set to None means "no restriction on this dimension".

    Examples:
        Policy(name="eu_only", allowed_regions=["eu", "global"])
        Policy(name="no_anthropic", blocked_providers=["anthropic"])
        Policy(name="low_latency_sla", max_latency_ms=3000.0)
        Policy(name="gdpr", allow_training_on_output=False, pii_allowed=False)
    """
    name: str

    # ── Provider constraints ──────────────────────────────────────────────────
    allowed_providers: Optional[list[str]] = None    # whitelist; None = all allowed
    blocked_providers: Optional[list[str]] = None    # blacklist; None = none blocked

    # ── Region constraints ────────────────────────────────────────────────────
    allowed_regions: Optional[list[str]] = None      # e.g. ["eu", "global"]

    # ── Model-level blocks ────────────────────────────────────────────────────
    blocked_models: Optional[list[Model]] = None

    # ── Compliance requirements ───────────────────────────────────────────────
    allow_training_on_output: bool = True            # False requires "no_train" tag
    pii_allowed: bool = True                         # False requires "pii_allowed" tag

    # ── Performance constraints ───────────────────────────────────────────────
    max_cost_per_task_usd: Optional[float] = None    # hard per-task cap
    max_latency_ms: Optional[float] = None           # reject if avg_latency > this

    # ── Enforcement mode ──────────────────────────────────────────────────────
    enforcement_mode: Optional["EnforcementMode"] = None   # None → HARD (default)
    rate_limit: Optional["RateLimit"] = None


# ─────────────────────────────────────────────────────────────────────────────
# PolicySet — container of policies with per-node overrides
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PolicySet:
    """
    Maps global policies and per-task-id overrides.
    The engine calls policies_for(task_id) to get the merged list.

    Node-level policies extend (not replace) global policies.
    To override a global policy for a specific task, use a node-level policy
    with the same name — the PolicyEngine processes all policies in order.

    Example:
        PolicySet(
            global_policies=[Policy("eu_only", allowed_regions=["eu"])],
            node_policies={
                "task_004": [Policy("allow_openai_for_frontend",
                                    allowed_providers=["openai", "google"])]
            }
        )
    """
    global_policies: list[Policy] = field(default_factory=list)
    node_policies: dict[str, list[Policy]] = field(default_factory=dict)

    def policies_for(self, task_id: str) -> list[Policy]:
        """
        Returns global policies merged with node-level overrides for task_id.
        Order: global first, then node-level. PolicyEngine evaluates all.
        """
        return self.global_policies + self.node_policies.get(task_id, [])


# ─────────────────────────────────────────────────────────────────────────────
# JobSpec — first-class job specification
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class JobSpec:
    """
    First-class replacement for the ad-hoc (project_description, success_criteria,
    budget) triple passed to Orchestrator.run_project().

    Use Orchestrator.run_job(spec) to use the full policy-driven interface.
    The existing run_project() signature is preserved by constructing a default
    JobSpec from its arguments internally.

    Example:
        spec = JobSpec(
            project_description="Build a FastAPI auth service",
            success_criteria="All tests pass, docs complete",
            budget=Budget(max_usd=8.0, max_time_seconds=5400),
            policy_set=PolicySet(global_policies=[
                Policy("gdpr", allow_training_on_output=False),
                Policy("low_latency", max_latency_ms=5000.0),
            ]),
            quality_targets={TaskType.CODE_GEN: 0.90},
        )
        state = await orch.run_job(spec)
    """
    project_description: str
    success_criteria: str
    budget: Budget
    policy_set: PolicySet = field(default_factory=PolicySet)
    quality_targets: dict[TaskType, float] = field(default_factory=dict)
    preferred_regions: list[str] = field(default_factory=list)
    max_parallel_tasks: int = 3


# ─────────────────────────────────────────────────────────────────────────────
# PolicyHierarchy — 4-level additive policy merge (Org → Team → Job → Node)
# ─────────────────────────────────────────────────────────────────────────────

class PolicyHierarchy:
    """
    4-level policy merge: Org → Team → Job → Node.
    Each level's policies extend (not replace) parent-level policies.

    Example:
        hier = PolicyHierarchy(
            org=[Policy("gdpr", allow_training_on_output=False)],
            team={"eng": [Policy("eu_only", allowed_regions=["eu"])]},
        )
        policies = hier.policies_for(team="eng", job_id="job_001", task_id="t_007")
    """

    def __init__(
        self,
        org:  Optional[list["Policy"]]            = None,
        team: Optional[dict[str, list["Policy"]]] = None,
        job:  Optional[dict[str, list["Policy"]]] = None,
        node: Optional[dict[str, list["Policy"]]] = None,
    ):
        self._org  = org  or []
        self._team = team or {}
        self._job  = job  or {}
        self._node = node or {}

    def policies_for(
        self,
        team:    str = "",
        job_id:  str = "",
        task_id: str = "",
    ) -> list["Policy"]:
        """Return merged policy list: org + team[team] + job[job_id] + node[task_id]."""
        return (
            list(self._org)
            + list(self._team.get(team, []))
            + list(self._job.get(job_id, []))
            + list(self._node.get(task_id, []))
        )

    def as_policy_set(
        self,
        team:    str = "",
        job_id:  str = "",
        task_id: str = "",
    ) -> "PolicySet":
        """Convert to a flat PolicySet compatible with the existing engine API."""
        return PolicySet(global_policies=self.policies_for(team, job_id, task_id))
