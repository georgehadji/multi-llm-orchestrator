"""
ImageGenClient — generates images via OpenRouter image models.
=============================================================
Wraps OpenRouter's chat/completions endpoint for image-generation models
(text+image->image). Handles base64 decoding and file output.

Currently supports three models:
  - google/gemini-3.1-flash-image-preview  (Nano Banana 2, #1 Design Arena)
  - recraft/recraft-v4.1-pro-vector         (SVG output)
  - black-forest-labs/flux.2-klein-4b       (fast, cost-effective)

Usage:
    client = ImageGenClient(api_key="...")
    result = await client.generate(
        prompt="A dark premium website hero background...",
        model="google/gemini-3.1-flash-image-preview",
        output_path=Path("public/images/hero-bg.png"),
    )
"""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger("orchestrator.infrastructure.image_client")

DEFAULT_MODEL = "google/gemini-3.1-flash-image-preview"
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
MAX_RETRIES = 2
TIMEOUT = 60


@dataclass
class ImageGenResult:
    """Result of an image generation call."""

    success: bool = False
    image_data: bytes | None = None
    image_url: str | None = None
    mime_type: str = "image/png"
    model: str = DEFAULT_MODEL
    cost_usd: float = 0.0
    error: str | None = None
    output_path: Path | None = None


class ImageGenClient:
    """Generate images via OpenRouter's chat/completions API.

    Uses the standard chat completions endpoint — image models return
    content blocks with ``b64_json`` or ``url`` fields.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = OPENROUTER_BASE,
    ):
        self._api_key = api_key or self._resolve_api_key()
        self._base_url = base_url

    @staticmethod
    def _resolve_api_key() -> str:
        import os

        key = os.environ.get("OPENROUTER_API_KEY", "")
        if not key:
            logger.warning("OPENROUTER_API_KEY not set — image generation disabled")
        return key

    async def generate(
        self,
        prompt: str,
        model: str = DEFAULT_MODEL,
        width: int = 1200,
        height: int = 630,
        output_path: Path | None = None,
    ) -> ImageGenResult:
        """Generate an image from a text prompt.

        Args:
            prompt: Text description of the image.
            model: OpenRouter model ID.
            width: Desired image width (hint — model may adjust).
            height: Desired image height.
            output_path: If set, save the decoded image to this path.

        Returns:
            ImageGenResult with image_data (bytes) or error.
        """
        if not self._api_key:
            return ImageGenResult(success=False, error="OPENROUTER_API_KEY not set")

        system_prompt = (
            "Generate a high-quality image based on the user's description. "
            "Do NOT include any text, typography, or words in the image. "
            "Output only the image content."
        )

        messages = [
            {"role": "user", "content": f"Generate a {width}x{height} image: {prompt}"},
        ]

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": 4096,
        }

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        last_error: str | None = None
        for attempt in range(1 + MAX_RETRIES):
            try:
                async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                    response = await client.post(
                        f"{self._base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                    )

                if response.status_code == 429:
                    logger.warning("Image gen rate limited, retry %d/%d", attempt + 1, MAX_RETRIES)
                    import asyncio

                    await asyncio.sleep(2**attempt)
                    continue

                if response.status_code != 200:
                    last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                    logger.error("Image gen failed: %s", last_error)
                    continue

                data = response.json()
                result = self._parse_response(data, output_path)
                result.model = model
                if result.success and output_path:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_bytes(result.image_data)
                    result.output_path = output_path
                return result

            except httpx.TimeoutException:
                last_error = "Request timed out"
                logger.warning("Image gen timeout, retry %d/%d", attempt + 1, MAX_RETRIES)
                import asyncio

                await asyncio.sleep(1)
                continue
            except Exception as e:
                last_error = str(e)
                logger.error("Image gen exception: %s", e)
                break

        return ImageGenResult(success=False, error=last_error or "Unknown error")

    def _parse_response(
        self,
        data: dict[str, Any],
        output_path: Path | None,
    ) -> ImageGenResult:
        """Extract image data from an OpenRouter chat completions response.

        OpenRouter image models return content blocks in one of these formats:

        .. code-block:: json
            {
              "choices": [{
                "message": {
                  "content": [
                    {"type": "image", "source": {"data": "<b64>", "media_type": "image/png"}},
                    {"type": "text", "text": "..."}
                  ]
                }
              }]
            }

        Or for some models, the content is a single text block containing a URL.
        """
        try:
            choices = data.get("choices", [])
            if not choices:
                return ImageGenResult(success=False, error="No choices in response")

            message = choices[0].get("message", {})
            content = message.get("content", "")

            # Format 1: content is a list of content blocks
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "image":
                        source = block.get("source", {})
                        b64 = source.get("data", "")
                        mime = source.get("media_type", "image/png")
                        if b64:
                            try:
                                img_bytes = base64.b64decode(b64)
                                return ImageGenResult(success=True, image_data=img_bytes, mime_type=mime)
                            except Exception as e:
                                return ImageGenResult(success=False, error=f"Base64 decode failed: {e}")
                    # Recraft vector models sometimes return b64_json in a different format
                    if isinstance(block, dict) and "b64_json" in block:
                        mime = block.get("media_type", "image/svg+xml")
                        return ImageGenResult(
                            success=True,
                            image_data=base64.b64decode(block["b64_json"]),
                            mime_type=mime,
                        )

            # Format 2: Recraft SVG models return content as a URL string
            if isinstance(content, str) and content.startswith("http"):
                return ImageGenResult(success=True, image_url=content, mime_type="image/svg+xml")

            # Format 3: b64_json at the top level of the choice
            if "b64_json" in choices[0]:
                return ImageGenResult(
                    success=True,
                    image_data=base64.b64decode(choices[0]["b64_json"]),
                )

            return ImageGenResult(
                success=False,
                error=f"Unrecognised response format: no image content found in {type(content).__name__}",
            )

        except (KeyError, IndexError, TypeError) as e:
            return ImageGenResult(success=False, error=f"Response parse error: {e}")
