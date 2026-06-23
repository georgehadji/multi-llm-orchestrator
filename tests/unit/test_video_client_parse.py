"""Unit tests for VideoGenClient._find_video_url and hero-video wiring.

The video URL is returned by OpenRouter in a provider-specific field, so the
parser must walk the whole response payload for any video URL or data: URI.
"""

import base64

import pytest

from orchestrator.infrastructure.video_client import VideoGenClient

_MP4_BYTES = b"\x00\x00\x00\x18ftypmp42fake-mp4-bytes"
_MP4_B64 = base64.b64encode(_MP4_BYTES).decode()


@pytest.mark.unit
def test_http_mp4_url_found_anywhere_in_payload():
    """A remote .mp4 URL is located regardless of nesting depth."""
    payload = {
        "choices": [
            {"message": {"content": None, "video": {"url": "https://cdn.example/clip.mp4"}}}
        ]
    }
    url, data = VideoGenClient._find_video_url(payload)
    assert url == "https://cdn.example/clip.mp4"
    assert data is None


@pytest.mark.unit
def test_http_url_with_query_string_matches():
    payload = {"v": ["https://cdn.example/clip.webm?token=abc&exp=1"]}
    url, data = VideoGenClient._find_video_url(payload)
    assert url == "https://cdn.example/clip.webm?token=abc&exp=1"


@pytest.mark.unit
def test_data_video_uri_decoded_inline():
    payload = {"choices": [{"message": {"images": [f"data:video/mp4;base64,{_MP4_B64}"]}}]}
    url, data = VideoGenClient._find_video_url(payload)
    assert url is None
    assert data == _MP4_BYTES


@pytest.mark.unit
def test_non_video_url_ignored():
    payload = {"message": {"content": "see https://example.com/page.html and image.png"}}
    url, data = VideoGenClient._find_video_url(payload)
    assert url is None
    assert data is None


@pytest.mark.unit
async def test_generate_without_api_key_fails_gracefully():
    client = VideoGenClient(api_key="placeholder")
    client._api_key = ""  # force the no-key guard regardless of env
    result = await client.generate(prompt="x", model="google/veo-3.1-fast")
    assert result.success is False
    assert "OPENROUTER_API_KEY" in (result.error or "")


@pytest.mark.unit
def test_hero_component_emits_video_when_enabled():
    from orchestrator.generators.website_generator import WebsiteGenerator

    gen = WebsiteGenerator.__new__(WebsiteGenerator)
    html = gen._build_hero_component(
        "Hero", "Headline", "Tagline", ["Start", "Learn"], None, hero_video=True
    )
    assert "<video" in html
    assert "/videos/hero.mp4" in html
    assert 'poster="/images/hero-bg.webp"' in html
    assert "loop" in html and "muted" in html


@pytest.mark.unit
def test_hero_component_no_video_by_default():
    from orchestrator.generators.website_generator import WebsiteGenerator

    gen = WebsiteGenerator.__new__(WebsiteGenerator)
    html = gen._build_hero_component("Hero", "Headline", "Tagline", ["Start"], None)
    assert "<video" not in html


@pytest.mark.unit
def test_hero_video_model_selection_by_tier():
    from orchestrator.generators.website_generator import WebsiteGenerator

    gen = WebsiteGenerator.__new__(WebsiteGenerator)
    gen._quality_tier = "draft"
    assert gen._select_hero_video_model() == "bytedance/seedance-1-5-pro"
    gen._quality_tier = "premium"
    assert gen._select_hero_video_model() == "google/veo-3.1"
    gen._quality_tier = "balanced"
    assert gen._select_hero_video_model() == "google/veo-3.1-fast"
