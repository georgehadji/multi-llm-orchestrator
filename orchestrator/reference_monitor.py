"""
Reference Monitor — Hard Constraint Enforcement
===============================================
Called synchronously before EVERY task execution by ControlPlane.
Returns ALLOW | DENY(reason) | ESCALATE(rule).

Hard rules:
  no_training    — deny any task whose prompt mentions model training/fine-tuning
  eu_only        — deny tasks routed to non-EU models when data_locality=eu
  no_pii_logging — deny tasks that would log PII (contains_pii=True + logging validator)

Design principles:
  • Synchronous — no async, no I/O, cannot be bypassed by prompt content
  • Stateless per call — no mutable shared state
  • Fail-closed — unknown hard constraint → DENY
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

from .models import Task
from .specs import JobSpecV2, PolicySpecV2, EscalationRule


class Decision(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    ESCALATE = "escalate"


@dataclass
class MonitorResult:
    decision: Decision
    reason: str = ""
    rule: Optional[EscalationRule] = None


# ─────────────────────────────────────────────
# EU-only model sets
# ─────────────────────────────────────────────

# Models whose infrastructure is EU-hosted or self-hosted (not US cloud)
_EU_SAFE_MODELS: frozenset[str] = frozenset({
    "self_hosted",
    "mistral",
    "mistral-large",
    "mistral-small",
})

# Known US-based model name prefixes to block under eu_only
_US_MODEL_PREFIXES: tuple[str, ...] = (
    "gpt-",
    "claude-",
    "gemini-",
    "deepseek-",
    "kimi-",
)


# ─────────────────────────────────────────────
# Rule implementations
# ─────────────────────────────────────────────

_TRAINING_KEYWORDS_RE = re.compile(
    r"(?:"
    r"fine[- ]?tun(?:e|ing|ed)?"           # fine-tune, fine-tuning, finetune
    r"|train(?:ing)?[ _](?:data|set|loop|model)"  # training data/set/loop/model
    r"|gradient[ _]descent"                # gradient descent
    r"|backprop(?:agation)?"               # backprop, backpropagation
    r"|loss\.backward"                     # PyTorch loss.backward()
    r"|model\.fit"                         # Keras model.fit()
    r")",
    re.IGNORECASE,
)


def _check_no_training(task: Task, job: JobSpecV2, policy: PolicySpecV2) -> MonitorResult:
    """Deny tasks whose prompt mentions model training/fine-tuning activities."""
    if _TRAINING_KEYWORDS_RE.search(task.prompt):
        return MonitorResult(
            decision=Decision.DENY,
            reason=(
                "no_training constraint violated: task prompt references training/fine-tuning. "
                f"Task id={task.id!r}"
            ),
        )
    return MonitorResult(Decision.ALLOW)


def _check_eu_only(task: Task, job: JobSpecV2, policy: PolicySpecV2) -> MonitorResult:
    """
    Deny tasks if data_locality=eu and the task appears bound to a US-based model.

    We can only check the task's tech_context (which may name the model) and
    the routing hints from the policy. A conservative fail-closed heuristic:
    if the task's tech_context or prompt names a US model prefix, deny.
    """
    if job.inputs.data_locality not in ("eu", "EU"):
        return MonitorResult(Decision.ALLOW)

    combined = f"{task.prompt} {task.tech_context}".lower()
    for prefix in _US_MODEL_PREFIXES:
        if prefix in combined:
            return MonitorResult(
                decision=Decision.DENY,
                reason=(
                    f"eu_only constraint violated: task references US-based model '{prefix}*'. "
                    f"Task id={task.id!r}"
                ),
            )
    return MonitorResult(Decision.ALLOW)


def _check_no_pii_logging(task: Task, job: JobSpecV2, policy: PolicySpecV2) -> MonitorResult:
    """Deny tasks that would log PII when no_pii_logging constraint is set."""
    if not job.inputs.contains_pii:
        return MonitorResult(Decision.ALLOW)

    # If the task uses any logging-related validators, flag it
    logging_validators = frozenset({"logging", "audit_log", "pii_scan"})
    if logging_validators.intersection(task.hard_validators):
        return MonitorResult(
            decision=Decision.DENY,
            reason=(
                "no_pii_logging constraint violated: task uses logging validators "
                f"with contains_pii=True. Task id={task.id!r}"
            ),
        )
    return MonitorResult(Decision.ALLOW)


# ─────────────────────────────────────────────
# Simple rule-expression evaluator
# ─────────────────────────────────────────────

def _eval_condition(condition: str, context: dict) -> bool:
    """
    Evaluate a simple boolean condition string against a context dict.

    Supported syntax:
      key == value
      key != value
      key AND key
      key OR key
      NOT key

    Values are compared as strings (case-insensitive).
    """
    condition = condition.strip()

    # OR
    if " OR " in condition:
        parts = condition.split(" OR ", 1)
        return _eval_condition(parts[0], context) or _eval_condition(parts[1], context)

    # AND
    if " AND " in condition:
        parts = condition.split(" AND ", 1)
        return _eval_condition(parts[0], context) and _eval_condition(parts[1], context)

    # NOT
    if condition.startswith("NOT "):
        return not _eval_condition(condition[4:], context)

    # Equality / inequality
    if "==" in condition:
        key, val = condition.split("==", 1)
        return str(context.get(key.strip(), "")).lower() == val.strip().strip("\"'").lower()
    if "!=" in condition:
        key, val = condition.split("!=", 1)
        return str(context.get(key.strip(), "")).lower() != val.strip().strip("\"'").lower()

    # Boolean key lookup
    return bool(context.get(condition, False))


class ReferenceMonitor:
    """
    Synchronous, bypass-proof hard constraint checker.
    Called before every task execution by ControlPlane.

    Usage:
        monitor = ReferenceMonitor()
        result = monitor.check(task, job, policy)
        if result.decision == Decision.DENY:
            raise PolicyViolation(result.reason)
    """

    HARD_RULES: dict[str, Callable[[Task, JobSpecV2, PolicySpecV2], MonitorResult]] = {
        "no_training":    _check_no_training,
        "eu_only":        _check_eu_only,
        "no_pii_logging": _check_no_pii_logging,
    }

    def check(
        self,
        task: Task,
        job: JobSpecV2,
        policy: PolicySpecV2,
    ) -> MonitorResult:
        """
        Check all hard constraints and allow/deny rules.

        Order:
        1. Hard constraint rules (fail-closed for unknown constraints)
        2. Allow/deny rules from PolicySpecV2
        3. Escalation rules
        """
        # 1. Hard constraints
        for constraint in job.constraints.hard:
            checker = self.HARD_RULES.get(constraint)
            if checker is None:
                # Unknown hard constraint → fail-closed
                return MonitorResult(
                    decision=Decision.DENY,
                    reason=f"Unknown hard constraint '{constraint}' — failing closed. "
                           f"Task id={task.id!r}",
                )
            result = checker(task, job, policy)
            if result.decision != Decision.ALLOW:
                return result

        # 2. Allow/deny rules
        context = self._build_context(task, job, policy)
        for rule in policy.allow_deny_rules:
            result = self._eval_rule(rule, context, task)
            if result.decision != Decision.ALLOW:
                return result

        # 3. Escalation rules
        for rule in policy.escalation_rules:
            if _eval_condition(rule.trigger, context):
                return MonitorResult(
                    decision=Decision.ESCALATE,
                    reason=f"Escalation rule triggered: {rule.trigger!r}",
                    rule=rule,
                )

        return MonitorResult(Decision.ALLOW)

    def check_global(
        self,
        job: JobSpecV2,
        policy: PolicySpecV2,
    ) -> MonitorResult:
        """
        Pre-run global check — validate job-level constraints before any task
        is executed.  Does not require a specific Task.
        """
        # For now, check with a sentinel task to run the hard rules
        from .models import Task as _Task, TaskType
        sentinel = _Task(id="__global__", type=TaskType.EVALUATE, prompt="")
        return self.check(sentinel, job, policy)

    def _build_context(
        self,
        task: Task,
        job: JobSpecV2,
        policy: PolicySpecV2,
    ) -> dict:
        return {
            "task_type":     task.type.value,
            "data_locality": job.inputs.data_locality,
            "contains_pii":  str(job.inputs.contains_pii).lower(),
            "jurisdiction":  job.inputs.data_locality,
            "risk_level":    "low",  # placeholder — extend when risk scoring is added
        }

    def _eval_rule(
        self,
        rule: dict,
        context: dict,
        task: Task,
    ) -> MonitorResult:
        """Evaluate a single allow/deny rule dict."""
        effect = rule.get("effect", "allow").lower()
        condition = rule.get("when", "")
        if not condition:
            return MonitorResult(Decision.ALLOW)

        matched = _eval_condition(condition, context)
        if matched and effect == "deny":
            return MonitorResult(
                decision=Decision.DENY,
                reason=f"Policy deny rule matched: {condition!r}. Task id={task.id!r}",
            )
        return MonitorResult(Decision.ALLOW)


__all__ = ["Decision", "MonitorResult", "ReferenceMonitor"]
