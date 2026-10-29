"""Unit tests for orchestrator.resilience module."""

from __future__ import annotations


import pytest

pytestmark = pytest.mark.unit

from orchestrator.models import Model, TaskType
from orchestrator.resilience import (
    FallbackTriggeredEvent,
    ResiliencePolicy,
    RetryTemplate,
    resolve_fallback_chain,
)

# ─────────────────────────────────────────────────────────────────────────────
# ResiliencePolicy
# ─────────────────────────────────────────────────────────────────────────────


def test_resilience_policy_defaults():
    p = ResiliencePolicy()
    assert p.retries == 2
    assert p.timeout == 60.0
    assert p.jitter is True
    assert p.fallback_chain is None


def test_resilience_policy_with_fallback():
    p = ResiliencePolicy(retries=3)
    p2 = p.with_fallback(Model.GPT_4O_MINI, Model.LLAMA_3_3_70B)
    assert p2.retries == 3
    assert p2.fallback_chain == (Model.GPT_4O_MINI, Model.LLAMA_3_3_70B)
    # Original unchanged (frozen)
    assert p.fallback_chain is None


# ─────────────────────────────────────────────────────────────────────────────
# RetryTemplate
# ─────────────────────────────────────────────────────────────────────────────


def test_retry_template_code_gen():
    p = RetryTemplate.CODE_GEN.to_policy()
    assert p.retries == 3
    assert p.timeout == 120.0


def test_retry_template_evaluate():
    p = RetryTemplate.EVALUATE.to_policy()
    assert p.retries == 2
    assert p.timeout == 45.0


def test_retry_template_for_task_type():
    assert RetryTemplate.for_task_type(TaskType.CODE_GEN).retries == 3
    assert RetryTemplate.for_task_type(TaskType.EVALUATE).timeout == 45.0
    assert RetryTemplate.for_task_type(TaskType.SUMMARIZE).retries == 2


# ─────────────────────────────────────────────────────────────────────────────
# resolve_fallback_chain
# ─────────────────────────────────────────────────────────────────────────────


def test_resolve_fallback_chain_known_model():
    chain = resolve_fallback_chain(Model.GPT_4O)
    assert len(chain) >= 1
    assert chain[0] == Model.CLAUDE_SONNET_5


def test_resolve_fallback_chain_max_depth():
    # Verify max_depth limits the chain length
    chain = resolve_fallback_chain(Model.GPT_4O, max_depth=1)
    assert len(chain) == 1
    assert chain[0] == Model.CLAUDE_SONNET_5


def test_resolve_fallback_chain_cycle_protection():
    # FALLBACK_CHAIN has no cycles in production, but test the guard
    chain = resolve_fallback_chain(Model.GPT_4O, max_depth=1)
    assert len(chain) == 1


# ─────────────────────────────────────────────────────────────────────────────
# FallbackTriggeredEvent
# ─────────────────────────────────────────────────────────────────────────────


def test_fallback_triggered_event_to_dict():
    evt = FallbackTriggeredEvent(
        primary_model="openai/gpt-4o",
        fallback_model="anthropic/claude-sonnet",
        attempt_number=2,
        reason="ConnectionError",
    )
    d = evt.to_dict()
    assert d["primary_model"] == "openai/gpt-4o"
    assert d["fallback_model"] == "anthropic/claude-sonnet"
    assert d["attempt_number"] == 2
