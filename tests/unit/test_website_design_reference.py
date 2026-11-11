"""WebsiteGenerator pulls an Awwwards design brief (flag-gated) and puts it in section prompts."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

BRIEF = (
    '<design_reference source="awwwards-mcp" trust="untrusted">\nuse big type\n</design_reference>'
)


def _generator(engine=None):
    from orchestrator.generators.website_generator import WebsiteGenerator

    gen = WebsiteGenerator.__new__(WebsiteGenerator)
    gen._engine = engine
    return gen


def _client():
    from orchestrator.generators.website_generator import ClientInfo

    return ClientInfo(name="Gadini", industry="barbershop", description="A barber shop")


def _section_prompt(config):
    from orchestrator.design_system import DesignSystem
    from orchestrator.generators.website_generator import ContentBrief

    return _generator()._build_section_prompt(
        section="hero",
        component="hero",
        design_system=DesignSystem(tone="luxury"),
        content_brief=ContentBrief(),
        config=config,
    )


def test_flag_is_on_by_default():
    from orchestrator.crosscutting.config import FeatureFlags

    assert FeatureFlags.model_fields["awwwards_enabled"].default is True


def test_section_prompt_carries_design_reference():
    from orchestrator.generators.website_generator import WebsiteConfig

    prompt = _section_prompt(WebsiteConfig(design_reference=BRIEF))
    assert BRIEF in prompt


def test_section_prompt_without_reference_has_no_block():
    from orchestrator.generators.website_generator import WebsiteConfig

    assert "<design_reference" not in _section_prompt(WebsiteConfig())


@pytest.fixture
def fetch_calls(monkeypatch):
    calls = []

    async def fake_fetch(goal, query="", **_kw):
        calls.append((goal, query))
        return BRIEF

    monkeypatch.setattr("orchestrator.design.awwwards_client.fetch_design_brief", fake_fetch)
    return calls


@pytest.mark.asyncio
async def test_reference_fetched_with_goal_and_industry(fetch_calls, monkeypatch):
    from orchestrator.crosscutting.config import flags
    from orchestrator.generators.website_generator import WebsiteConfig

    monkeypatch.setattr(flags, "awwwards_enabled", True)
    out = await _generator(engine=object())._design_reference(_client(), WebsiteConfig())

    assert out == BRIEF
    goal, query = fetch_calls[0]
    assert "Gadini" in goal and "barbershop" in goal
    assert query == "barbershop"


@pytest.mark.asyncio
async def test_no_reference_when_flag_off(fetch_calls, monkeypatch):
    from orchestrator.crosscutting.config import flags
    from orchestrator.generators.website_generator import WebsiteConfig

    monkeypatch.setattr(flags, "awwwards_enabled", False)
    out = await _generator(engine=object())._design_reference(_client(), WebsiteConfig())

    assert out == "" and fetch_calls == []


@pytest.mark.asyncio
async def test_no_reference_without_llm_engine(fetch_calls, monkeypatch):
    """Template-only builds make no LLM calls, so they must not spawn npx either."""
    from orchestrator.crosscutting.config import flags
    from orchestrator.generators.website_generator import WebsiteConfig

    monkeypatch.setattr(flags, "awwwards_enabled", True)
    out = await _generator(engine=None)._design_reference(_client(), WebsiteConfig())

    assert out == "" and fetch_calls == []
