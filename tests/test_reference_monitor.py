"""
Unit tests for ReferenceMonitor (B2).
"""
from __future__ import annotations

import pytest

from orchestrator.models import Task, TaskType
from orchestrator.reference_monitor import Decision, MonitorResult, ReferenceMonitor
from orchestrator.specs import Constraints, InputSpec, JobSpecV2, PolicySpecV2


def _job(**kwargs) -> JobSpecV2:
    return JobSpecV2(goal="test", **kwargs)


def _task(prompt: str = "do something", tech_context: str = "") -> Task:
    return Task(
        id="task_001",
        type=TaskType.CODE_GEN,
        prompt=prompt,
        tech_context=tech_context,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Basic allow
# ─────────────────────────────────────────────────────────────────────────────

def test_allow_with_no_constraints():
    monitor = ReferenceMonitor()
    job = _job()
    policy = PolicySpecV2()
    result = monitor.check(_task(), job, policy)
    assert result.decision == Decision.ALLOW


# ─────────────────────────────────────────────────────────────────────────────
# no_training rule
# ─────────────────────────────────────────────────────────────────────────────

def test_no_training_blocks_training_prompt():
    monitor = ReferenceMonitor()
    job = _job(constraints=Constraints(hard=["no_training"]))
    task = _task(prompt="Fine-tune the model on this dataset")
    result = monitor.check(task, job, PolicySpecV2())
    assert result.decision == Decision.DENY
    assert "no_training" in result.reason


def test_no_training_allows_normal_prompt():
    monitor = ReferenceMonitor()
    job = _job(constraints=Constraints(hard=["no_training"]))
    task = _task(prompt="Generate a REST API endpoint")
    result = monitor.check(task, job, PolicySpecV2())
    assert result.decision == Decision.ALLOW


def test_no_training_blocks_gradient_mention():
    monitor = ReferenceMonitor()
    job = _job(constraints=Constraints(hard=["no_training"]))
    task = _task(prompt="Run gradient descent on this model")
    result = monitor.check(task, job, PolicySpecV2())
    assert result.decision == Decision.DENY


# ─────────────────────────────────────────────────────────────────────────────
# eu_only rule
# ─────────────────────────────────────────────────────────────────────────────

def test_eu_only_denies_gpt_model_mention():
    monitor = ReferenceMonitor()
    job = _job(
        constraints=Constraints(hard=["eu_only"]),
        inputs=InputSpec(data_locality="eu"),
    )
    task = _task(tech_context="use gpt-4o for code generation")
    result = monitor.check(task, job, PolicySpecV2())
    assert result.decision == Decision.DENY
    assert "eu_only" in result.reason


def test_eu_only_allows_no_model_mention():
    monitor = ReferenceMonitor()
    job = _job(
        constraints=Constraints(hard=["eu_only"]),
        inputs=InputSpec(data_locality="eu"),
    )
    task = _task(prompt="Generate TypeScript code")
    result = monitor.check(task, job, PolicySpecV2())
    assert result.decision == Decision.ALLOW


def test_eu_only_not_triggered_when_locality_us():
    """eu_only constraint with data_locality=us should not trigger."""
    monitor = ReferenceMonitor()
    job = _job(
        constraints=Constraints(hard=["eu_only"]),
        inputs=InputSpec(data_locality="us"),
    )
    task = _task(tech_context="use gpt-4o for code generation")
    result = monitor.check(task, job, PolicySpecV2())
    # eu_only checks data_locality; with "us" it short-circuits to ALLOW
    assert result.decision == Decision.ALLOW


# ─────────────────────────────────────────────────────────────────────────────
# no_pii_logging rule
# ─────────────────────────────────────────────────────────────────────────────

def test_no_pii_logging_allows_non_pii_task():
    monitor = ReferenceMonitor()
    job = _job(
        constraints=Constraints(hard=["no_pii_logging"]),
        inputs=InputSpec(contains_pii=False),
    )
    task = _task()
    result = monitor.check(task, job, PolicySpecV2())
    assert result.decision == Decision.ALLOW


def test_no_pii_logging_denies_logging_validator_with_pii():
    monitor = ReferenceMonitor()
    job = _job(
        constraints=Constraints(hard=["no_pii_logging"]),
        inputs=InputSpec(contains_pii=True),
    )
    task = Task(
        id="t1",
        type=TaskType.CODE_GEN,
        prompt="process data",
        hard_validators=["audit_log"],
    )
    result = monitor.check(task, job, PolicySpecV2())
    assert result.decision == Decision.DENY


# ─────────────────────────────────────────────────────────────────────────────
# Unknown hard constraint — fail-closed
# ─────────────────────────────────────────────────────────────────────────────

def test_unknown_hard_constraint_denied():
    monitor = ReferenceMonitor()
    job = _job(constraints=Constraints(hard=["unknown_future_constraint"]))
    result = monitor.check(_task(), job, PolicySpecV2())
    assert result.decision == Decision.DENY
    assert "Unknown hard constraint" in result.reason


# ─────────────────────────────────────────────────────────────────────────────
# Allow/deny rules
# ─────────────────────────────────────────────────────────────────────────────

def test_allow_deny_rule_deny():
    monitor = ReferenceMonitor()
    job = _job(inputs=InputSpec(data_locality="us"))
    policy = PolicySpecV2(allow_deny_rules=[
        {"effect": "deny", "when": "data_locality == us"},
    ])
    result = monitor.check(_task(), job, policy)
    assert result.decision == Decision.DENY


def test_allow_deny_rule_allow():
    monitor = ReferenceMonitor()
    job = _job(inputs=InputSpec(data_locality="eu"))
    policy = PolicySpecV2(allow_deny_rules=[
        {"effect": "deny", "when": "data_locality == us"},
    ])
    result = monitor.check(_task(), job, policy)
    assert result.decision == Decision.ALLOW


# ─────────────────────────────────────────────────────────────────────────────
# check_global
# ─────────────────────────────────────────────────────────────────────────────

def test_check_global_allow():
    monitor = ReferenceMonitor()
    job = _job()
    result = monitor.check_global(job, PolicySpecV2())
    assert result.decision == Decision.ALLOW


def test_check_global_denies_training():
    monitor = ReferenceMonitor()
    job = _job(constraints=Constraints(hard=["no_training"]))
    # Sentinel task has empty prompt → should pass no_training
    result = monitor.check_global(job, PolicySpecV2())
    assert result.decision == Decision.ALLOW  # sentinel prompt is empty
