"""
PolicyEngine — validates a (model, profile) pair against a list of policies.
=============================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis
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
from typing import Optional

from .models import Model
from .policy import EnforcementMode, ModelProfile, Policy
from .tracing import traced_policy_check


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

    raw_passed reflects whether violations were found before EnforcementMode
    override. passed is the effective result after the mode is applied.
    """
    passed: bool
    violations: list[str] = field(default_factory=list)
    model: Model = None  # type: ignore[assignment]  # populated by check()
    raw_passed: bool = True   # before EnforcementMode override


# ─────────────────────────────────────────────────────────────────────────────
# Violation severity classification for SOFT mode
# ─────────────────────────────────────────────────────────────────────────────

# Keywords identifying "hard" violations: these always block in SOFT mode.
# Any violation message not matching these keywords is treated as "soft"
# (latency SLA, cost cap) and allowed through in SOFT mode.
_HARD_VIOLATION_PREFIXES = (
    "blocked_providers",
    "allowed_providers",
    "allowed_regions",
    "blocked_models",
    "allow_training_on_output",
    "pii_allowed",
)


def _is_hard_violation(violation: str) -> bool:
    """Return True if the violation message represents a hard (structural) violation."""
    v_lower = violation.lower()
    return any(kw in v_lower for kw in _HARD_VIOLATION_PREFIXES)


# ─────────────────────────────────────────────────────────────────────────────
# PolicyEngine
# ─────────────────────────────────────────────────────────────────────────────

class PolicyEngine:
    """
    Stateless compliance checker. The same instance is reused across all tasks.
    All check methods are deterministic given the same (model, profile, policies).

    Optionally accepts an AuditLog to record one entry per check() call.

    Checks performed by check() in order:
      1. blocked_providers      — hard blacklist
      2. allowed_providers      — whitelist (None = all allowed)
      3. allowed_regions        — region constraint (None = all allowed)
      4. blocked_models         — specific model blacklist
      5. max_latency_ms         — latency SLA gate
      6. allow_training_on_output → requires "no_train" compliance tag
      7. pii_allowed            → requires "pii_allowed" compliance tag
    """

    def __init__(self, audit_log: Optional[object] = None):
        """
        Parameters
        ----------
        audit_log : AuditLog | None
            Optional AuditLog instance. If provided, every check() call emits
            one AuditRecord. Pass None (default) to disable auditing.
        """
        self._audit_log = audit_log

    def check(
        self,
        model: Model,
        profile: ModelProfile,
        policies: list[Policy],
        task_id: str = "",
    ) -> PolicyCheckResult:
        """
        Evaluate all policies against the (model, profile) pair.

        Returns PolicyCheckResult with:
          passed=True  if all policies are satisfied (violations=[])
          passed=False if any policy is violated (violations lists each one)

        EnforcementMode (per policy, most-restrictive across all policies wins):
          HARD    → any violation blocks (default; wins over SOFT/MONITOR)
          SOFT    → only hard violations (provider/model/region/compliance) block
          MONITOR → always passed=True (violations still logged)

        Does NOT raise. Use enforce() to get exception-based enforcement.
        """
        with traced_policy_check(len(policies)) as span:
            violations: list[str] = []
            # Track the most-restrictive mode across all policies.
            # HARD (0) > SOFT (1) > MONITOR (2) — stricter policies always win.
            # Rationale: if ANY policy is HARD, all violations must block the model.
            # A permissive MONITOR policy from one rule must never override a
            # HARD compliance rule from another (e.g. GDPR region constraint).
            _MODE_RANK = {
                EnforcementMode.HARD:    0,  # most restrictive → wins
                EnforcementMode.SOFT:    1,
                EnforcementMode.MONITOR: 2,  # most permissive
                None:                    0,  # None → HARD
            }
            effective_mode_rank = 2          # start at most-permissive, tighten downward
            effective_mode = EnforcementMode.MONITOR

            for policy in policies:
                mode = policy.enforcement_mode  # type: ignore[attr-defined]
                rank = _MODE_RANK.get(mode, 0)
                if rank < effective_mode_rank:  # lower rank = more restrictive = wins
                    effective_mode_rank = rank
                    effective_mode = mode if mode is not None else EnforcementMode.HARD

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

            raw_passed = (len(violations) == 0)

            # Apply enforcement mode to determine effective result
            if not raw_passed:
                if effective_mode == EnforcementMode.MONITOR:
                    effective_passed = True   # log but allow
                elif effective_mode == EnforcementMode.SOFT:
                    # Block only if any hard violation exists
                    has_hard = any(_is_hard_violation(v) for v in violations)
                    effective_passed = not has_hard
                else:
                    # HARD (default): block on any violation
                    effective_passed = False
            else:
                effective_passed = True

            result = PolicyCheckResult(
                passed=effective_passed,
                violations=violations,
                model=model,
                raw_passed=raw_passed,
            )

            # Emit audit record if audit log is configured
            if self._audit_log is not None:
                self._audit_log.record(
                    task_id=task_id,
                    model=model.value,
                    passed=effective_passed,
                    raw_passed=raw_passed,
                    violations=violations,
                    enforcement_mode=(
                        effective_mode.value
                        if effective_mode is not None
                        else EnforcementMode.HARD.value
                    ),
                    policies_applied=[p.name for p in policies],
                )

            span.set_attribute("policy.passed", result.passed)
            span.set_attribute("policy.violations", len(result.violations))
            return result

    def enforce(
        self,
        model: Model,
        profile: ModelProfile,
        policies: list[Policy],
        task_id: str = "pre-flight",
    ) -> None:
        """
        Call check() and raise PolicyViolationError if any violations are found.

        Used as a hard gate before each API call in engine.py to catch
        configuration drift (e.g. region tags changed between selection
        and execution).
        """
        with traced_policy_check(len(policies)) as span:
            result = self.check(model, profile, policies, task_id=task_id)
            span.set_attribute("policy.passed", result.passed)
            span.set_attribute("policy.violations", len(result.violations))
            if not result.passed:
                raise PolicyViolationError(
                    task_id=task_id,
                    policies=policies,
                    reason="; ".join(result.violations),
                )
