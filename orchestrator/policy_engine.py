"""
PolicyEngine — validates a (model, profile) pair against a list of policies.
=============================================================================
Purely functional: check() returns a result object; enforce() raises on failure.
No state is mutated here. Trust degradation after violations is handled by
TelemetryCollector.record_policy_violation() in engine.py.

Policy-as-code novelty:
  Compliance rules are first-class Python objects (Policy dataclasses) rather
  than scattered if-statements in routing logic. This makes compliance auditable:
  you can print(policy_set) to see every constraint that was applied to a run,
  and PolicyCheckResult.violations explains exactly why a model was rejected.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .models import Model
from .policy import ModelProfile, Policy


# ─────────────────────────────────────────────────────────────────────────────
# Exceptions
# ─────────────────────────────────────────────────────────────────────────────

class PolicyViolationError(Exception):
    """
    Raised by PolicyEngine.enforce() when a model is non-compliant with
    the active policy set.

    Also raised by engine.py when ConstraintPlanner.select_model() returns
    None and the task cannot be degraded or skipped — i.e., when no compliant
    model exists at all.

    Attributes
    ----------
    task_id : str
    policies : list[Policy]  — the policies that were applied
    reason : str             — human-readable explanation of all violations
    """
    def __init__(self, task_id: str, policies: list[Policy], reason: str):
        self.task_id = task_id
        self.policies = policies
        self.reason = reason
        policy_names = [p.name for p in policies]
        super().__init__(
            f"No compliant model for task '{task_id}': {reason}. "
            f"Policies applied: {policy_names}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# PolicyCheckResult
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PolicyCheckResult:
    """
    Result of checking a single (model, profile) pair against a list of policies.

    passed=True and violations=[] means the model is fully compliant.
    passed=False and violations=[...] lists every specific rule that was broken.
    """
    passed: bool
    violations: list[str] = field(default_factory=list)
    model: Model = None  # type: ignore[assignment]  # populated by check()


# ─────────────────────────────────────────────────────────────────────────────
# PolicyEngine
# ─────────────────────────────────────────────────────────────────────────────

class PolicyEngine:
    """
    Stateless compliance checker. The same instance is reused across all tasks.
    All check methods are deterministic given the same (model, profile, policies).

    Checks performed by check() in order:
      1. blocked_providers      — hard blacklist
      2. allowed_providers      — whitelist (None = all allowed)
      3. allowed_regions        — region constraint (None = all allowed)
      4. blocked_models         — specific model blacklist
      5. max_latency_ms         — latency SLA gate
      6. allow_training_on_output → requires "no_train" compliance tag
      7. pii_allowed            → requires "pii_allowed" compliance tag
    """

    def check(
        self,
        model: Model,
        profile: ModelProfile,
        policies: list[Policy],
    ) -> PolicyCheckResult:
        """
        Evaluate all policies against the (model, profile) pair.

        Returns PolicyCheckResult with:
          passed=True  if all policies are satisfied (violations=[])
          passed=False if any policy is violated (violations lists each one)

        Does NOT raise. Use enforce() to get exception-based enforcement.
        """
        violations: list[str] = []

        for policy in policies:

            # 1. Blocked providers (hard blacklist)
            if (
                policy.blocked_providers is not None
                and profile.provider in policy.blocked_providers
            ):
                violations.append(
                    f"[{policy.name}] provider '{profile.provider}' is in "
                    f"blocked_providers {policy.blocked_providers}"
                )

            # 2. Allowed providers (whitelist — None means all allowed)
            if (
                policy.allowed_providers is not None
                and profile.provider not in policy.allowed_providers
            ):
                violations.append(
                    f"[{policy.name}] provider '{profile.provider}' not in "
                    f"allowed_providers {policy.allowed_providers}"
                )

            # 3. Allowed regions (None means all allowed)
            if (
                policy.allowed_regions is not None
                and profile.region not in policy.allowed_regions
            ):
                violations.append(
                    f"[{policy.name}] region '{profile.region}' not in "
                    f"allowed_regions {policy.allowed_regions}"
                )

            # 4. Blocked models (specific model-level block)
            if (
                policy.blocked_models is not None
                and model in policy.blocked_models
            ):
                violations.append(
                    f"[{policy.name}] model '{model.value}' is in blocked_models"
                )

            # 5. Latency SLA
            if (
                policy.max_latency_ms is not None
                and profile.avg_latency_ms > policy.max_latency_ms
            ):
                violations.append(
                    f"[{policy.name}] avg_latency {profile.avg_latency_ms:.0f}ms "
                    f"> max_latency_ms {policy.max_latency_ms:.0f}ms"
                )

            # 6. Training consent — policy says no training, model must carry
            #    "no_train" in its compliance_tags to prove it
            if not policy.allow_training_on_output:
                if "no_train" not in profile.compliance_tags:
                    violations.append(
                        f"[{policy.name}] allow_training_on_output=False requires "
                        f"'no_train' compliance tag; model has: {profile.compliance_tags}"
                    )

            # 7. PII policy — if PII is disallowed, model must carry "pii_allowed" tag
            if not policy.pii_allowed:
                if "pii_allowed" not in profile.compliance_tags:
                    violations.append(
                        f"[{policy.name}] pii_allowed=False requires 'pii_allowed' "
                        f"compliance tag; model has: {profile.compliance_tags}"
                    )

        return PolicyCheckResult(
            passed=(len(violations) == 0),
            violations=violations,
            model=model,
        )

    def enforce(
        self,
        model: Model,
        profile: ModelProfile,
        policies: list[Policy],
    ) -> None:
        """
        Call check() and raise PolicyViolationError if any violations are found.

        Used as a hard gate before each API call in engine.py to catch
        configuration drift (e.g. region tags changed between selection
        and execution).
        """
        result = self.check(model, profile, policies)
        if not result.passed:
            raise PolicyViolationError(
                task_id="pre-flight",
                policies=policies,
                reason="; ".join(result.violations),
            )
