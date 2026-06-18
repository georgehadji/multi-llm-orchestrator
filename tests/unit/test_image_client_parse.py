"""Unit tests for ImageGenClient._parse_response.

Covers the OpenRouter image-model response formats, with emphasis on the
``message.images[].image_url.url`` shape that returns a base64 ``data:`` URI —
the format that previously fell through to "Unrecognised response format".
"""

import base64

import pytest

from orchestrator.infrastructure.image_client import ImageGenClient

# 1x1 transparent PNG
_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)
_PNG_B64 = base64.b64encode(_PNG_BYTES).decode()


@pytest.fixture
def client():
    return ImageGenClient(api_key="test-key")


def _response(message: dict) -> dict:
    return {"choices": [{"message": message}]}


@pytest.mark.unit
def test_data_uri_in_message_images_with_null_content(client):
    """OpenRouter image models: data: URI in images[].image_url.url, content=None."""
    data_uri = f"data:image/png;base64,{_PNG_B64}"
    resp = _response(
        {
            "role": "assistant",
            "content": None,
            "images": [{"type": "image_url", "image_url": {"url": data_uri}}],
        }
    )

    result = client._parse_response(resp, None)

    assert result.success is True
    assert result.image_data == _PNG_BYTES


@pytest.mark.unit
def test_data_uri_in_message_images_with_empty_string_content(client):
    """Same shape but content is an empty string (not None) — must still parse."""
    data_uri = f"data:image/png;base64,{_PNG_B64}"
    resp = _response(
        {
            "role": "assistant",
            "content": "",
            "images": [{"type": "image_url", "image_url": {"url": data_uri}}],
        }
    )

    result = client._parse_response(resp, None)

    assert result.success is True
    assert result.image_data == _PNG_BYTES


@pytest.mark.unit
def test_http_url_in_message_images_returns_url(client):
    """Recraft-style: http URL in images[].image_url.url should be returned as URL."""
    resp = _response(
        {
            "role": "assistant",
            "content": None,
            "images": [
                {"type": "image_url", "image_url": {"url": "https://cdn.example/x.svg"}}
            ],
        }
    )

    result = client._parse_response(resp, None)

    assert result.success is True
    assert result.image_url == "https://cdn.example/x.svg"


@pytest.mark.unit
def test_b64_json_block_still_works(client):
    """Regression: content as a list of b64_json blocks still parses."""
    resp = _response({"role": "assistant", "content": [{"b64_json": _PNG_B64}]})

    result = client._parse_response(resp, None)

    assert result.success is True
    assert result.image_data == _PNG_BYTES


@pytest.mark.unit
def test_unrecognised_format_reports_error(client):
    """Truly empty message yields a graceful failure, not an exception."""
    resp = _response({"role": "assistant", "content": None})

    result = client._parse_response(resp, None)

    assert result.success is False
    assert result.error
