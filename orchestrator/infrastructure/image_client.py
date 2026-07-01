"""
ImageGenClient — generates images via OpenRouter image models.
=============================================================
Wraps OpenRouter's chat/completions endpoint for image-generation models
(text+image->image). Handles base64 decoding and file output.

Currently supports four models:
  - google/gemini-3.1-flash-image-preview  (Nano Banana 2, #1 Design Arena)
  - google/gemini-3.1-flash-lite-image     (Ultra-fast, cost-effective Nano Banana 2 Lite)
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

import asyncio
import base64
import hashlib
import logging
from dataclasses import dataclass
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
    cached: bool = False


class ImageGenClient:
    """Generate images via OpenRouter's chat/completions API.

    Uses the standard chat completions endpoint — image models return
    content blocks with ``b64_json`` or ``url`` fields.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = OPENROUTER_BASE,
        cache=None,
    ):
        self._api_key = api_key or self._resolve_api_key()
        self._base_url = base_url
        self._cache = cache

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
            {"role": "system", "content": system_prompt},
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

        # ── Cache check ────────────────────────────────────────────────────
        cache_key = ""
        if self._cache:
            cache_key = hashlib.sha256(f"{model}:{prompt}:{width}x{height}".encode()).hexdigest()
            cached = await self._cache.get(cache_key, "", 4096, "", 0.0)
            if cached and output_path:
                try:
                    img_bytes = base64.b64decode(cached)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_bytes(img_bytes)
                    return ImageGenResult(
                        success=True,
                        image_data=img_bytes,
                        mime_type="image/png",
                        model=model,
                        cached=True,
                    )
                except Exception:
                    pass

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
                    await asyncio.sleep(2**attempt)
                    continue

                if response.status_code >= 500:
                    logger.warning(
                        "Image gen server error %d, retry %d/%d",
                        response.status_code,
                        attempt + 1,
                        MAX_RETRIES,
                    )
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
                    if result.image_data:
                        output_path.write_bytes(result.image_data)
                    elif result.image_url:
                        # Download SVG URLs (Recraft models return URLs, not base64)
                        try:
                            async with httpx.AsyncClient(timeout=30) as http:
                                resp = await http.get(result.image_url)
                                if resp.status_code == 200:
                                    output_path.write_bytes(resp.content)
                                    result.image_data = resp.content
                                    logger.debug("Downloaded SVG from %s", result.image_url[:60])
                                else:
                                    logger.warning(
                                        "Failed to download image URL: HTTP %d", resp.status_code
                                    )
                        except Exception as e:
                            logger.warning("Failed to download image URL: %s", e)
                    # ── Cache write ──
                    if result.success and result.image_data and self._cache:
                        try:
                            await self._cache.put(
                                cache_key,
                                "",
                                4096,
                                base64.b64encode(result.image_data).decode(),
                                len(result.image_data),
                                0,
                                "",
                                0.0,
                            )
                        except Exception:
                            pass
                    result.output_path = output_path
                return result

            except httpx.TimeoutException:
                last_error = "Request timed out"
                logger.warning("Image gen timeout, retry %d/%d", attempt + 1, MAX_RETRIES)
                await asyncio.sleep(1)
                continue
            except Exception as e:
                last_error = str(e)
                logger.error("Image gen exception: %s", e)
                break

        return ImageGenResult(success=False, error=last_error or "Unknown error")

    @staticmethod
    def _result_from_url(url: str, mime: str = "image/png") -> ImageGenResult | None:
        """Build a result from a URL string.

        Handles base64 ``data:`` URIs (decoded inline) and remote ``http(s)``
        URLs (returned for later download). Returns None if ``url`` is neither.
        """
        if not url or not isinstance(url, str):
            return None
        if url.startswith("data:"):
            # data:[<mime>][;base64],<payload>
            try:
                header, _, payload = url.partition(",")
                if not payload:
                    return None
                if header[5:].split(";")[0]:
                    mime = header[5:].split(";")[0]
                return ImageGenResult(
                    success=True,
                    image_data=base64.b64decode(payload),
                    mime_type=mime,
                )
            except Exception as e:  # noqa: BLE001 - report, do not raise
                return ImageGenResult(success=False, error=f"data URI decode failed: {e}")
        if url.startswith("http"):
            svg = url.lower().endswith(".svg")
            return ImageGenResult(
                success=True,
                image_url=url,
                mime_type="image/svg+xml" if svg else mime,
            )
        return None

    def _extract_message_images(
        self,
        message: dict[str, Any],
        choice: dict[str, Any],
    ) -> ImageGenResult | None:
        """Extract an image from a message's ``images[]`` / ``image_url`` fields.

        Returns an ImageGenResult on success/decode-failure, or None when no
        image payload is present (so the caller can keep trying other formats).
        """
        # Some models put image_url directly on the message or choice.
        direct = message.get("image_url") or choice.get("image_url", "")
        if isinstance(direct, dict):
            direct = direct.get("url", "")
        direct_result = self._result_from_url(direct)
        if direct_result is not None:
            return direct_result

        msg_images = message.get("images", [])
        if not isinstance(msg_images, list) or not msg_images:
            return None

        img = msg_images[0]
        if isinstance(img, str):
            # Bare base64 string or a data/http URL.
            url_result = self._result_from_url(img)
            if url_result is not None:
                return url_result
            try:
                return ImageGenResult(success=True, image_data=base64.b64decode(img))
            except Exception:
                return None

        if isinstance(img, dict):
            mime = img.get("mime_type", img.get("media_type", "image/png"))
            b64 = img.get("b64_json") or img.get("data", "")
            if b64:
                try:
                    return ImageGenResult(
                        success=True,
                        image_data=base64.b64decode(b64),
                        mime_type=mime,
                    )
                except Exception as e:  # noqa: BLE001
                    return ImageGenResult(success=False, error=f"Message images decode: {e}")
            # Nested image_url object or string (commonly a base64 data: URI).
            img_url = img.get("image_url", "")
            if isinstance(img_url, dict):
                img_url = img_url.get("url", "")
            return self._result_from_url(img_url, mime)

        return None

    def _parse_response(
        self,
        data: dict[str, Any],
        output_path: Path | None,
    ) -> ImageGenResult:
        """Extract image data from an OpenRouter chat completions response.

        Handles multiple response formats across different image models:
        - Gemini: content as list of content blocks with {'type': 'image', 'source': {...}}
        - Recraft: b64_json field in content blocks or URL string
        - FLUX: content may be None; image data in choices[0]['images'] or top-level
        """
        try:
            choices = data.get("choices", [])
            if not choices:
                return ImageGenResult(success=False, error="No choices in response")

            message = choices[0].get("message", {})
            content = message.get("content", None)

            # Format 1: content is a list of content blocks
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    # Gemini-style image blocks
                    if block.get("type") == "image":
                        source = block.get("source", {})
                        b64 = source.get("data", "")
                        mime = source.get("media_type", "image/png")
                        if b64:
                            try:
                                img_bytes = base64.b64decode(b64)
                                return ImageGenResult(
                                    success=True, image_data=img_bytes, mime_type=mime
                                )
                            except Exception as e:
                                return ImageGenResult(
                                    success=False, error=f"Base64 decode failed: {e}"
                                )
                    # Recraft/FLUX b64_json format
                    if "b64_json" in block:
                        mime = block.get("media_type", "image/png")
                        try:
                            return ImageGenResult(
                                success=True,
                                image_data=base64.b64decode(block["b64_json"]),
                                mime_type=mime,
                            )
                        except Exception as e:
                            return ImageGenResult(success=False, error=f"b64_json decode: {e}")

            # Format 2: Recraft SVG models return content as a URL string
            if isinstance(content, str):
                if content.startswith("http"):
                    return ImageGenResult(
                        success=True, image_url=content, mime_type="image/svg+xml"
                    )
                # Might be base64 directly
                if len(content) > 100 and not content.startswith("{"):
                    try:
                        return ImageGenResult(success=True, image_data=base64.b64decode(content))
                    except Exception:
                        pass

            # Format 3: Image data in top-level choices[0] fields (not nested in message.content)
            if "b64_json" in choices[0]:
                try:
                    return ImageGenResult(
                        success=True, image_data=base64.b64decode(choices[0]["b64_json"])
                    )
                except Exception as e:
                    return ImageGenResult(success=False, error=f"Top-level b64_json: {e}")

            # Format 4: Images array at top level (some models return this)
            if "images" in choices[0]:
                img = choices[0]["images"]
                if isinstance(img, list) and img:
                    img = img[0]
                if isinstance(img, dict):
                    b64 = img.get("b64_json") or img.get("data", "")
                    if b64:
                        return ImageGenResult(success=True, image_data=base64.b64decode(b64))

            # Format 5: image payload carried on the message object (not in content).
            # OpenRouter image models (Gemini, FLUX, etc.) return images[] on the
            # message with a nested image_url.url that is usually a base64 ``data:``
            # URI. ``content`` may be None OR an empty string, so this branch must
            # NOT be gated on ``content is None``.
            msg_result = self._extract_message_images(message, choices[0])
            if msg_result is not None:
                return msg_result

            # Content is None — check for base64 payload carried at the top level.
            if content is None:
                for key in ("data", "image", "image_data"):
                    val = data.get(key, "")
                    if isinstance(val, str) and len(val) > 100:
                        try:
                            return ImageGenResult(success=True, image_data=base64.b64decode(val))
                        except Exception:
                            pass

            return ImageGenResult(
                success=False,
                error=f"Unrecognised response format: content={type(content).__name__}, "
                f"message_keys={list(message.keys())[:8]}, choice_keys={list(choices[0].keys())[:8]}",
            )

        except (KeyError, IndexError, TypeError) as e:
            return ImageGenResult(success=False, error=f"Response parse error: {e}")
