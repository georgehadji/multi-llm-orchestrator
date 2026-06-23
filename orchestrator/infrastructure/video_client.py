"""
VideoGenClient — generates short videos via OpenRouter video models.
====================================================================
Wraps OpenRouter's chat/completions endpoint for video-generation models
(``text+image->video``, e.g. ``google/veo-3.1-fast``, ``bytedance/seedance-2.0``).

Video generation is slow (tens of seconds) and the resulting MP4/WebM is large,
so providers return a downloadable URL rather than inline base64. The exact
response field varies by provider, so :meth:`_find_video_url` walks the whole
response JSON looking for any video URL (``http(s)`` ending in a video
extension, or a ``data:video/...`` URI) — robust to contract differences.

Usage:
    client = VideoGenClient(api_key="...")
    result = await client.generate(
        prompt="Slow cinematic aerial flythrough of a misty mountain range, looping",
        model="google/veo-3.1-fast",
        output_path=Path("public/videos/hero.mp4"),
    )
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger("orchestrator.infrastructure.video_client")

DEFAULT_MODEL = "google/veo-3.1-fast"
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
MAX_RETRIES = 2
# Video generation is slow — allow well beyond the image-gen timeout.
TIMEOUT = 240

_VIDEO_EXT_RE = re.compile(r"\.(mp4|webm|mov|m4v)(\?|$)", re.IGNORECASE)
_DATA_VIDEO_RE = re.compile(r"^data:video/", re.IGNORECASE)


@dataclass
class VideoGenResult:
    """Result of a video generation call."""

    success: bool = False
    video_data: bytes | None = None
    video_url: str | None = None
    mime_type: str = "video/mp4"
    model: str = DEFAULT_MODEL
    cost_usd: float = 0.0
    error: str | None = None
    output_path: Path | None = None
    cached: bool = False


class VideoGenClient:
    """Generate short videos via OpenRouter's chat/completions API."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = OPENROUTER_BASE,
        cache: Any = None,
    ) -> None:
        self._api_key = api_key or self._resolve_api_key()
        self._base_url = base_url
        self._cache = cache

    @staticmethod
    def _resolve_api_key() -> str:
        import os

        key = os.environ.get("OPENROUTER_API_KEY", "")
        if not key:
            logger.warning("OPENROUTER_API_KEY not set — video generation disabled")
        return key

    async def generate(
        self,
        prompt: str,
        model: str = DEFAULT_MODEL,
        output_path: Path | None = None,
        duration_seconds: int = 6,
    ) -> VideoGenResult:
        """Generate a video from a text prompt and (optionally) save it.

        Args:
            prompt: Text description of the desired video.
            model: OpenRouter video model ID.
            output_path: If set, the decoded/downloaded video is written here.
            duration_seconds: Requested clip length (hint — provider may clamp).
        """
        if not self._api_key:
            return VideoGenResult(success=False, error="OPENROUTER_API_KEY not set", model=model)

        system_prompt = (
            "Generate a high-quality, seamless looping background video based on "
            "the user's description. No text, captions, watermarks, or logos. "
            "Smooth, subtle motion suitable for a website hero background."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Generate a ~{duration_seconds}s looping video: {prompt}",
            },
        ]
        payload = {"model": model, "messages": messages}
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        # ── Cache check ────────────────────────────────────────────────────
        cache_key = ""
        if self._cache:
            cache_key = hashlib.sha256(f"{model}:{prompt}:{duration_seconds}s".encode()).hexdigest()
            cached = await self._cache.get(cache_key, "", 4096, "", 0.0)
            if cached and output_path:
                try:
                    vid_bytes = base64.b64decode(cached)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_bytes(vid_bytes)
                    return VideoGenResult(
                        success=True,
                        video_data=vid_bytes,
                        model=model,
                        output_path=output_path,
                        cached=True,
                    )
                except Exception:  # noqa: BLE001 - corrupt cache entry, regenerate
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
                if response.status_code in (429,) or response.status_code >= 500:
                    logger.warning(
                        "Video gen transient %d, retry %d/%d",
                        response.status_code,
                        attempt + 1,
                        MAX_RETRIES,
                    )
                    await asyncio.sleep(2**attempt)
                    continue
                if response.status_code != 200:
                    last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                    logger.error("Video gen failed: %s", last_error)
                    continue

                url, data_uri_bytes = self._find_video_url(response.json())
                if not url and data_uri_bytes is None:
                    last_error = "No video URL in response"
                    continue

                result = VideoGenResult(success=True, model=model, output_path=output_path)
                if data_uri_bytes is not None:
                    result.video_data = data_uri_bytes
                else:
                    result.video_url = url
                    if output_path:
                        async with httpx.AsyncClient(timeout=TIMEOUT) as http:
                            resp = await http.get(url)
                            if resp.status_code != 200:
                                last_error = f"Video download HTTP {resp.status_code}"
                                continue
                            result.video_data = resp.content

                if output_path and result.video_data:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_bytes(result.video_data)
                    if self._cache and cache_key:
                        try:
                            await self._cache.put(
                                cache_key,
                                "",
                                4096,
                                base64.b64encode(result.video_data).decode(),
                                len(result.video_data),
                                0,
                                "",
                                0.0,
                            )
                        except Exception:  # noqa: BLE001 - cache write is best-effort
                            pass
                return result
            except (httpx.TimeoutException, httpx.HTTPError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning("Video gen attempt %d failed: %s", attempt + 1, last_error)
                await asyncio.sleep(2**attempt)

        return VideoGenResult(success=False, error=last_error or "unknown error", model=model)

    @classmethod
    def _find_video_url(cls, data: Any) -> tuple[str | None, bytes | None]:
        """Walk a response payload for the first video URL or ``data:`` URI.

        Returns ``(url, None)`` for a remote URL, ``(None, bytes)`` for an inline
        base64 ``data:video/...`` URI, or ``(None, None)`` when none is found.
        """
        stack: list[Any] = [data]
        while stack:
            node = stack.pop()
            if isinstance(node, str):
                if _DATA_VIDEO_RE.match(node):
                    try:
                        payload = node.partition(",")[2]
                        if payload:
                            return None, base64.b64decode(payload)
                    except Exception:  # noqa: BLE001 - malformed data URI, keep scanning
                        pass
                elif node.startswith("http") and _VIDEO_EXT_RE.search(node):
                    return node, None
            elif isinstance(node, dict):
                stack.extend(node.values())
            elif isinstance(node, list):
                stack.extend(node)
        return None, None
