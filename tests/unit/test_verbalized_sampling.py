"""
Tests for VerbalizedSampler primitive.

Covers:
  - _build_system: format definitions, tail injection, k injection
  - _parse: clean JSON, fences, json5, partial recovery, missing probs, edge cases
  - sample(): integration shape with fake LLMClient port
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from orchestrator.application.verbalized_sampling import VSCandidate, VerbalizedSampler
from orchestrator.models import Model, ProbabilityFormat, TaskType, VSConfig


# ── Fake LLMClient for port-level testing ────────────────────────────────────


@dataclass
class FakeAPIResponse:
    text: str
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0


class FakeLLMClient:
    """Implements the LLMClient port shape for testing."""

    def __init__(self, response_text: str = ""):
        self.response_text = response_text
        self.last_call: dict[str, Any] = {}

    async def call(self, **kwargs: Any) -> FakeAPIResponse:
        self.last_call = kwargs
        return FakeAPIResponse(text=self.response_text)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_response() -> str:
    """A clean VS response with 3 candidates."""
    return json.dumps(
        {
            "responses": [
                {"text": "response A", "probability": 0.5},
                {"text": "response B", "probability": 0.3},
                {"text": "response C", "probability": 0.2},
            ]
        }
    )


@pytest.fixture
def fence_response() -> str:
    """A VS response wrapped in ```json fences."""
    return (
        "```json\n"
        + json.dumps(
            {
                "responses": [
                    {"text": "fenced A", "probability": 0.7},
                    {"text": "fenced B", "probability": 0.3},
                ]
            }
        )
        + "\n```"
    )


@pytest.fixture
def sampler() -> VerbalizedSampler:
    return VerbalizedSampler(client=FakeLLMClient())


# ─────────────────────────────────────────────────────────────────────────────
# _build_system tests
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildSystem:
    def test_injects_k(self):
        s = VerbalizedSampler(client=FakeLLMClient())
        system = s._build_system(VSConfig(k=5), "")
        assert "Generate 5 possible responses" in system

    def test_injects_k_varied(self):
        s = VerbalizedSampler(client=FakeLLMClient())
        system = s._build_system(VSConfig(k=1), "")
        assert "Generate 1 possible responses" in system  # not 0/empty

    def test_explicit_format_string(self):
        s = VerbalizedSampler(client=FakeLLMClient())
        system = s._build_system(VSConfig(fmt=ProbabilityFormat.EXPLICIT), "")
        assert "estimated probability" in system
        assert "relative to the full distribution" in system

    def test_confidence_format_string(self):
        s = VerbalizedSampler(client=FakeLLMClient())
        system = s._build_system(VSConfig(fmt=ProbabilityFormat.CONFIDENCE), "")
        assert "likelihood score" in system
        assert "representative or typical" in system

    def test_threshold_injects_tail_instruction(self):
        s = VerbalizedSampler(client=FakeLLMClient())
        system = s._build_system(VSConfig(probability_threshold=0.10), "")
        assert "probability of each response is below 0.1" in system

    def test_no_threshold_omits_tail(self):
        s = VerbalizedSampler(client=FakeLLMClient())
        system = s._build_system(VSConfig(probability_threshold=None), "")
        assert "below" not in system

    def test_system_extra_prepended(self):
        s = VerbalizedSampler(client=FakeLLMClient())
        system = s._build_system(VSConfig(), "CUSTOM_PREFIX")
        assert system.startswith("CUSTOM_PREFIX")
        assert "Generate 5" in system

    def test_json_format_instruction_present(self):
        s = VerbalizedSampler(client=FakeLLMClient())
        system = s._build_system(VSConfig(), "")
        assert '"responses"' in system
        assert '"text"' in system
        assert '"probability"' in system


# ─────────────────────────────────────────────────────────────────────────────
# _parse tests
# ─────────────────────────────────────────────────────────────────────────────


class TestParse:
    def test_clean_json(self, sample_response):
        s = VerbalizedSampler(client=FakeLLMClient())
        candidates = s._parse(sample_response, 3)
        assert len(candidates) == 3
        assert candidates[0].text == "response A"
        assert candidates[0].probability == 0.5
        assert candidates[1].text == "response B"
        assert candidates[1].probability == 0.3

    def test_fenced_json(self, fence_response):
        s = VerbalizedSampler(client=FakeLLMClient())
        candidates = s._parse(fence_response, 2)
        assert len(candidates) == 2
        assert candidates[0].text == "fenced A"
        assert candidates[0].probability == 0.7

    def test_fence_no_language_tag(self):
        s = VerbalizedSampler(client=FakeLLMClient())
        text = "```\n" + json.dumps({"responses": [{"text": "x", "probability": 0.5}]}) + "\n```"
        candidates = s._parse(text, 1)
        assert len(candidates) == 1
        assert candidates[0].text == "x"

    def test_missing_probability_defaults_to_uniform(self):
        s = VerbalizedSampler(client=FakeLLMClient())
        text = json.dumps({"responses": [{"text": "only text"}, {"text": "also text"}]})
        candidates = s._parse(text, 2)
        assert len(candidates) == 2
        assert candidates[0].probability == 0.5  # uniform 1/2
        assert candidates[1].probability == 0.5

    def test_probability_clamped_high(self):
        s = VerbalizedSampler(client=FakeLLMClient())
        text = json.dumps({"responses": [{"text": "overconfident", "probability": 5.0}]})
        candidates = s._parse(text, 1)
        assert candidates[0].probability == 1.0

    def test_probability_clamped_low(self):
        s = VerbalizedSampler(client=FakeLLMClient())
        text = json.dumps({"responses": [{"text": "negative", "probability": -1.0}]})
        candidates = s._parse(text, 1)
        assert candidates[0].probability == 0.0

    def test_probability_as_string(self):
        s = VerbalizedSampler(client=FakeLLMClient())
        text = json.dumps({"responses": [{"text": "stringy", "probability": "0.75"}]})
        candidates = s._parse(text, 1)
        assert candidates[0].probability == 0.75

    def test_probability_as_invalid_string(self):
        s = VerbalizedSampler(client=FakeLLMClient())
        text = json.dumps({"responses": [{"text": "bad", "probability": "abc"}]})
        candidates = s._parse(text, 1)
        assert candidates[0].probability == 1.0  # uniform default

    def test_truncated_array_partial_recovery(self):
        """Partial JSON — recovery by appending missing closing brackets."""
        s = VerbalizedSampler(client=FakeLLMClient())
        # Truncated JSON with first complete object; closing } and ] missing.
        text = '{"responses": [{"text": "first", "probability": 0.6}'
        candidates = s._parse(text, 3)
        # Should recover the first complete item
        assert len(candidates) >= 1
        if candidates:
            assert candidates[0].text == "first"
            assert candidates[0].probability == 0.6

    def test_partial_object_recovery_close(self):
        """JSON with only closing brace missing — recovery appends it."""
        s = VerbalizedSampler(client=FakeLLMClient())
        text = '{"responses": [{"text": "A", "probability": 0.5}'
        candidates = s._parse(text, 1)
        assert len(candidates) >= 1
        assert candidates[0].text == "A"

    def test_bare_array_format(self):
        s = VerbalizedSampler(client=FakeLLMClient())
        text = json.dumps([{"text": "bare A", "probability": 0.4}, {"text": "bare B", "probability": 0.6}])
        candidates = s._parse(text, 2)
        assert len(candidates) == 2

    def test_empty_response(self):
        s = VerbalizedSampler(client=FakeLLMClient())
        candidates = s._parse("", 5)
        assert candidates == []

    def test_garbage_non_json(self):
        s = VerbalizedSampler(client=FakeLLMClient())
        candidates = s._parse("This is not JSON at all. Not even close.", 3)
        assert candidates == []

    def test_respects_k_limit(self):
        s = VerbalizedSampler(client=FakeLLMClient())
        text = json.dumps({"responses": [{"text": f"r{i}", "probability": 0.1} for i in range(10)]})
        candidates = s._parse(text, 3)
        assert len(candidates) == 3  # limited to k

    def test_skips_items_without_text(self):
        s = VerbalizedSampler(client=FakeLLMClient())
        text = json.dumps(
            {
                "responses": [
                    {"text": "", "probability": 0.5},
                    {"text": "real", "probability": 0.5},
                ]
            }
        )
        candidates = s._parse(text, 2)
        assert len(candidates) == 1
        assert candidates[0].text == "real"

    def test_extra_fields_ignored(self):
        s = VerbalizedSampler(client=FakeLLMClient())
        text = json.dumps(
            {
                "responses": [
                    {"text": "real", "probability": 0.5, "extra": "ignored"},
                ]
            }
        )
        candidates = s._parse(text, 1)
        assert len(candidates) == 1
        assert candidates[0].text == "real"


# ─────────────────────────────────────────────────────────────────────────────
# sample() integration tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSample:
    @pytest.mark.asyncio
    async def test_returns_candidates(self, sample_response):
        fake = FakeLLMClient(response_text=sample_response)
        s = VerbalizedSampler(client=fake)
        candidates = await s.sample(
            prompt="test prompt",
            model=Model.GPT_4O_MINI,
            cfg=VSConfig(k=3),
        )
        assert len(candidates) == 3
        assert all(isinstance(c, VSCandidate) for c in candidates)

    @pytest.mark.asyncio
    async def test_passes_correct_model_and_temperature(self, sample_response):
        fake = FakeLLMClient(response_text=sample_response)
        s = VerbalizedSampler(client=fake)
        await s.sample(
            prompt="test",
            model=Model.GPT_4O_MINI,
            cfg=VSConfig(k=2, temperature=0.7),
        )
        assert fake.last_call.get("model") == Model.GPT_4O_MINI
        assert fake.last_call.get("temperature") == 0.7

    @pytest.mark.asyncio
    async def test_passes_system_prompt(self):
        fake = FakeLLMClient(response_text='{"responses": []}')
        s = VerbalizedSampler(client=fake)
        await s.sample(
            prompt="test",
            model=Model.GPT_4O_MINI,
            cfg=VSConfig(k=1),
            system_extra="CUSTOM",
        )
        system = fake.last_call.get("system", "")
        assert "CUSTOM" in system
        assert "Generate 1 possible response" in system

    @pytest.mark.asyncio
    async def test_passes_timeout_and_max_tokens(self):
        fake = FakeLLMClient(response_text='{"responses": []}')
        s = VerbalizedSampler(client=fake)
        await s.sample(
            prompt="test",
            model=Model.GPT_4O_MINI,
            cfg=VSConfig(k=1),
            max_tokens=2048,
            timeout=90,
        )
        assert fake.last_call.get("max_tokens") == 2048
        assert fake.last_call.get("timeout") == 90

    @pytest.mark.asyncio
    async def test_passes_task_type_for_response_schema(self):
        fake = FakeLLMClient(response_text='{"responses": []}')
        s = VerbalizedSampler(client=fake)
        await s.sample(
            prompt="test",
            model=Model.GPT_4O_MINI,
            cfg=VSConfig(k=1),
            task_type=TaskType.CODE_GEN,
        )
        assert fake.last_call.get("task_type") == TaskType.CODE_GEN
        assert fake.last_call.get("response_schema") is True

    @pytest.mark.asyncio
    async def test_empty_response_returns_empty_list(self):
        fake = FakeLLMClient(response_text="")
        s = VerbalizedSampler(client=fake)
        candidates = await s.sample(prompt="test", model=Model.GPT_4O_MINI)
        assert candidates == []

    @pytest.mark.asyncio
    async def test_client_error_returns_empty_list(self):
        class BrokenClient:
            async def call(self, **kwargs):
                raise RuntimeError("API down")

        s = VerbalizedSampler(client=BrokenClient())
        candidates = await s.sample(prompt="test", model=Model.GPT_4O_MINI)
        assert candidates == []

    @pytest.mark.asyncio
    async def test_tail_config(self):
        """Tail threshold makes it into the system prompt."""
        fake = FakeLLMClient(response_text='{"responses": []}')
        s = VerbalizedSampler(client=fake)
        await s.sample(
            prompt="test",
            model=Model.GPT_4O_MINI,
            cfg=VSConfig(k=3, probability_threshold=0.10),
        )
        system = fake.last_call.get("system", "")
        assert "below 0.1" in system
