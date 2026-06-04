"""
Context Compressor — LLM-Powered Summarization for Dependency Context
======================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Compresses dependency context text via LLM summarization when it exceeds
the truncation limit. Preserves function/class signatures for code text.
Falls back to hard truncation on failure.

Uses the cheapest available model (ZHIPU_GLM_5_1 — canonical GLM model)
for summarization. Summaries are cached by content hash for 48 hours.

Integration: Called from DependencyResolver._format_dependency_context()
when output length exceeds context_truncation_limit. The compressor is
injected via constructor — defaults to disabled for backward compatibility.

Environment variable: ORCH_CONTEXT_COMPRESSION=true enables compression.
ORCH_CONTEXT_COMPRESSION_MODEL overrides the summarization model.
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import TYPE_CHECKING, Any

from ..models import Model

if TYPE_CHECKING:
    from ..domain.ports import LLMClient

logger = logging.getLogger("orchestrator.context_compressor")

# Default cache TTL: 48 hours (matching L2 DiskCache TTL)
_CACHE_TTL_SECONDS: int = 172800

# Default summarization model — cheapest in the routing table
_DEFAULT_MODEL: Model = Model.ZHIPU_GLM_5_1

# Maximum input length to send to the summarizer (chunks to avoid flooding)
_MAX_SUMMARIZER_INPUT: int = 8000


class ContextCompressor:
    """Compresses dependency context via LLM summarization.

    Usage:
        compressor = ContextCompressor(
            client=unified_client,
            enabled=os.getenv("ORCH_CONTEXT_COMPRESSION", "").lower() == "true",
        )
        summary = await compressor.compress(long_text, max_chars=40000)
    """

    def __init__(
        self,
        client: LLMClient | None = None,
        enabled: bool = False,
        cache_ttl: int = _CACHE_TTL_SECONDS,
        compression_model: str = "",
    ) -> None:
        """Initialize compressor.

        Args:
            client: LLMClient instance for LLM calls. Required when enabled.
            enabled: When False, falls back to hard truncation.
            cache_ttl: Seconds before a cached summary is evicted.
            compression_model: Model to use for compression (defaults to
                _DEFAULT_MODEL if empty).  Callers should read this from
                crosscutting/config.py rather than os.environ.
        """
        self._client = client
        self._enabled = enabled
        self._cache_ttl = cache_ttl
        self._cache: dict[str, tuple[float, str]] = {}  # hash → (timestamp, summary)
        self._model = Model(compression_model) if compression_model else _DEFAULT_MODEL

    # ── Public API ──────────────────────────────────────────────────────────

    async def compress(
        self,
        text: str,
        max_chars: int,
        *,
        preserve_signatures: bool = True,
        dep_id: str = "",
    ) -> str:
        """Compress text to fit within ``max_chars``.

        Args:
            text: The text to compress (e.g. dependency output).
            max_chars: Maximum character length for the compressed result.
            preserve_signatures: When True (default), instructs the LLM to
                preserve function/class signatures. Set to False for
                non-code text.
            dep_id: Optional dependency ID for logging.

        Returns:
            Compressed text within ``max_chars`` length.
        """
        if len(text) <= max_chars:
            return text

        if not self._enabled or self._client is None:
            self._log_truncation(dep_id, len(text), max_chars)
            return text[:max_chars] + "\n\n[... truncated ...]"

        # Check cache
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        cached = self._cache.get(text_hash)
        if cached is not None and (time.time() - cached[0]) < self._cache_ttl:
            logger.debug("ContextCompressor cache hit for %s (%d chars)", dep_id, len(text))
            return cached[1]

        # Summarize via LLM
        try:
            summary = await self._summarize(text, max_chars, preserve_signatures)
            self._cache[text_hash] = (time.time(), summary)
            return summary
        except Exception as exc:
            logger.warning(
                "ContextCompressor LLM summarization failed for %s: %s. "
                "Falling back to hard truncation.",
                dep_id or "unknown",
                exc,
            )
            return text[:max_chars] + "\n\n[... truncated ...]"

    # ── Internal ────────────────────────────────────────────────────────────

    async def _summarize(
        self,
        text: str,
        max_chars: int,
        preserve_signatures: bool,
    ) -> str:
        """Call LLM to summarize text."""
        if preserve_signatures:
            instruction = (
                "Summarize the following code while preserving ALL function "
                "signatures, class definitions, and import statements. "
                "Remove implementation bodies (replace with '# ...'). "
                "Keep variable names and type annotations. "
                f"Output must be under {max_chars} characters."
            )
        else:
            instruction = (
                f"Summarize the following text to under {max_chars} characters. "
                "Keep all named entities, key numbers, and structured data."
            )

        # Don't send full text to the summarizer — first chunk is enough
        # for proper structure detection
        input_text = text[:_MAX_SUMMARIZER_INPUT]
        if len(text) > _MAX_SUMMARIZER_INPUT:
            input_text += "\n\n[... more content follows, total "
            input_text += f"{len(text)} chars prior to compression ...]"

        response = await self._client.call(  # type: ignore[no-untyped-call]
            model=self._model,
            system=instruction,
            prompt=input_text,
            max_tokens=min(2048, max_chars // 4),  # ~4 chars per token est.
        )

        result = response.text.strip()
        if len(result) > max_chars:
            result = result[:max_chars] + "\n\n[... truncated ...]"

        return result  # type: ignore[no-any-return]

    def _log_truncation(self, dep_id: str, original_len: int, max_chars: int) -> None:
        """Log a truncation warning."""
        logger.warning(
            "ContextCompressor disabled — truncated %s from %d to %d chars",
            dep_id or "unknown",
            original_len,
            max_chars,
        )

    def clear_cache(self) -> None:
        """Clear the in-memory summary cache."""
        self._cache.clear()
        logger.debug("ContextCompressor cache cleared")

    @property
    def is_enabled(self) -> bool:
        return self._enabled
