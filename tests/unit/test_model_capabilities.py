"""Unit tests for OpenRouter structured-output capability gating.

Several OpenRouter models do NOT advertise ``structured_outputs`` in their
``supported_parameters`` (confirmed live: ``z-ai/glm-4.5-air``,
``minimax/minimax-01``, several ``nvidia/nemotron-*:free`` variants). Sending
``response_format`` to those is ignored or rejected, so the dispatch path must
only attach a JSON schema when the resolved model supports it.

These cover the pure predicate; the I/O fetch/cache is exercised separately.
"""

import pytest

pytestmark = pytest.mark.unit

from orchestrator.infrastructure.model_capabilities import (
    build_capability_map,
    supports_structured_output,
)

pytestmark = pytest.mark.unit


# A trimmed-down shape of the OpenRouter /models `data` payload.
_MODELS_PAYLOAD = [
    {
        "id": "openai/gpt-4o",
        "canonical_slug": "openai/gpt-4o",
        "supported_parameters": ["tools", "response_format", "structured_outputs"],
    },
    {
        "id": "z-ai/glm-4.5-air",
        "canonical_slug": "z-ai/glm-4.5-air",
        "supported_parameters": ["tools", "response_format"],  # no structured_outputs
    },
    {
        "id": "minimax/minimax-01",
        "supported_parameters": ["temperature", "max_tokens"],
    },
]


def test_build_capability_map_indexes_id_and_slug():
    cap = build_capability_map(_MODELS_PAYLOAD)
    assert "structured_outputs" in cap["openai/gpt-4o"]
    assert "structured_outputs" not in cap["z-ai/glm-4.5-air"]
    # canonical_slug is indexed too when present.
    assert cap["z-ai/glm-4.5-air"] == cap["z-ai/glm-4.5-air"]


def test_supports_when_structured_outputs_listed():
    cap = build_capability_map(_MODELS_PAYLOAD)
    assert supports_structured_output("openai/gpt-4o", cap) is True


def test_skips_when_structured_outputs_absent():
    cap = build_capability_map(_MODELS_PAYLOAD)
    assert supports_structured_output("z-ai/glm-4.5-air", cap) is False
    assert supports_structured_output("minimax/minimax-01", cap) is False


def test_variant_suffix_resolves_to_base_slug():
    """A ':free'/':nitro' style suffix must fall back to the base model's caps."""
    cap = build_capability_map(_MODELS_PAYLOAD)
    assert supports_structured_output("openai/gpt-4o:nitro", cap) is True
    assert supports_structured_output("z-ai/glm-4.5-air:free", cap) is False


def test_unknown_model_is_permissive():
    """Models we have not catalogued default to True — only KNOWN-unsupported skip."""
    cap = build_capability_map(_MODELS_PAYLOAD)
    assert supports_structured_output("some/brand-new-model", cap) is True


def test_empty_or_none_map_is_permissive():
    """When the capability map could not be loaded, do not suppress schemas."""
    assert supports_structured_output("z-ai/glm-4.5-air", None) is True
    assert supports_structured_output("z-ai/glm-4.5-air", {}) is True
