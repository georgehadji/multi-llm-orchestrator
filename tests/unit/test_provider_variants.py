"""Unit tests for OpenRouter sorting-alias variant resolution.

``:nitro``/``:floor`` are sorting aliases (equivalent to provider.sort
throughput/price) and must be stripped from the slug and translated to a
provider.sort value. ``:exacto`` is a virtual variant with no provider.sort
equivalent, so it stays on the slug and signals that a task-strategy sort must
not override it. Endpoint variants (:free/:thinking/:extended) are left intact.
"""

import pytest

from orchestrator.infrastructure.llm_client import _resolve_provider_variant

pytestmark = pytest.mark.unit


def test_nitro_maps_to_throughput_and_strips_suffix():
    model_id, sort, exacto = _resolve_provider_variant("anthropic/claude-3.5-sonnet:nitro")
    assert model_id == "anthropic/claude-3.5-sonnet"
    assert sort == "throughput"
    assert exacto is False


def test_floor_maps_to_price_and_strips_suffix():
    model_id, sort, exacto = _resolve_provider_variant("openai/gpt-4o:floor")
    assert model_id == "openai/gpt-4o"
    assert sort == "price"
    assert exacto is False


def test_exacto_stays_on_slug_and_flags():
    model_id, sort, exacto = _resolve_provider_variant("moonshotai/kimi-k2-0905:exacto")
    assert model_id == "moonshotai/kimi-k2-0905:exacto"  # kept on slug; OpenRouter resolves it
    assert sort is None  # no provider.sort equivalent
    assert exacto is True


@pytest.mark.parametrize("variant", [":free", ":thinking", ":extended"])
def test_endpoint_variants_untouched(variant):
    raw = f"x-ai/grok-4.3{variant}"
    model_id, sort, exacto = _resolve_provider_variant(raw)
    assert model_id == raw  # endpoint variants must reach OpenRouter intact
    assert sort is None
    assert exacto is False


def test_plain_model_unchanged():
    model_id, sort, exacto = _resolve_provider_variant("z-ai/glm-4.7-flash")
    assert model_id == "z-ai/glm-4.7-flash"
    assert sort is None
    assert exacto is False
