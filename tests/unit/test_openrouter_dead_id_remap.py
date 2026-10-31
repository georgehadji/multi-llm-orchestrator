"""Pins the 2026-10-31 remap of OpenRouter ids that left the live catalogue.

Offline counterpart of ``test_openrouter_model_audit.py``: no network. Each dead
id must redirect to its approved replacement in a single hop, and no
``UNAVAILABLE_MODELS`` value may itself be a key (a chain would resolve to a
dead id after one hop).
"""

from __future__ import annotations

import pytest

from orchestrator.domain.model_registry import ModelRegistry

pytestmark = pytest.mark.unit

APPROVED_REMAP = {
    "anthropic/claude-opus-4": "anthropic/claude-opus-4.1",
    "anthropic/claude-sonnet-4": "anthropic/claude-sonnet-4.5",
    "openai/o1": "openai/o3",
    "openai/o1-pro": "openai/o3-pro",
    "openai/o3-mini": "openai/gpt-5-mini",
    "openai/o4-mini": "openai/gpt-5-mini",
    "openai/o4": "openai/gpt-5-mini",
    "qwen/qwen3-max": "qwen/qwen3.7-max",
    "qwen/qwen3-max-thinking": "qwen/qwen3.7-max",
    "qwen/qwen-3-max-thinking": "qwen/qwen3.7-max",
    "qwen/qwen3-235b-a22b": "qwen/qwen3-235b-a22b-2507",
    "qwen/qwen3-coder-plus": "qwen/qwen3-coder",
    "qwen/qwen3-next-80b-a3b-thinking": "qwen/qwen3-next-80b-a3b-instruct",
    "poolside/laguna-s-2.1": "qwen/qwen3-coder-next",
}


@pytest.mark.parametrize("dead_id,live_id", sorted(APPROVED_REMAP.items()))
def test_dead_id_resolves_to_approved_target(dead_id: str, live_id: str):
    assert ModelRegistry.get_replacement_model(dead_id) == live_id


def test_no_redirect_chains():
    chained = {
        dead: repl
        for dead, repl in ModelRegistry.UNAVAILABLE_MODELS.items()
        if repl in ModelRegistry.UNAVAILABLE_MODELS
    }
    assert chained == {}, f"UNAVAILABLE_MODELS has multi-hop chains: {chained}"
