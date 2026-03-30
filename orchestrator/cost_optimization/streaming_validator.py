"""
Streaming Validator Module
===========================
Author: Georgios-Chrysovalantis Chatzivantsidis

Implements streaming generation with early validation for 10-15% wasted token reduction.

Strategy: Stream output chunks, validate early, abort if model goes off-track.

Features:
- Streaming output processing
- Early failure detection (first 500 tokens)
- Automatic retry with different model
- Token waste prevention

Usage:
    from orchestrator.cost_optimization import StreamingValidator

    validator = StreamingValidator(client=api_client)

    result = await validator.stream_and_validate(
        model="claude-sonnet-4.6",
        prompt="Generate Python code...",
        task_type="code_generation",
    )
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from orchestrator.log_config import get_logger

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = get_logger(__name__)


@dataclass
class StreamingMetrics:
    """Metrics for streaming validation."""
    total_streams: int = 0
    early_aborts: int = 0
    successful_streams: int = 0
    retries: int = 0
    tokens_saved: int = 0
    estimated_savings: float = 0.0

    @property
    def early_abort_rate(self) -> float:
        """Calculate early abort rate."""
        if self.total_streams == 0:
            return 0.0
        return self.early_aborts / self.total_streams

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_streams": self.total_streams,
            "early_aborts": self.early_aborts,
            "successful_streams": self.successful_streams,
            "retries": self.retries,
            "tokens_saved": self.tokens_saved,
            "estimated_savings": self.estimated_savings,
            "early_abort_rate": self.early_abort_rate,
        }


@dataclass
class StreamingResult:
    """Result of streaming validation."""
    response: str
    chunks: list[str]
    total_tokens: int
    cost: float
    early_aborted: bool
    abort_reason: str | None
    retry_count: int
    latency_seconds: float


class StreamingValidator:
    """
    Streaming generation with early validation.

    Usage:
        validator = StreamingValidator(client=api_client)
        result = await validator.stream_and_validate(model, prompt, task_type)
    """

    # Early abort patterns (indicate model is going off-track)
    EARLY_ABORT_PATTERNS = [
        # Refusals
        r"i cannot",
        r"i'm unable",
        r"i am unable",
        r"as an ai",
        r"as a language model",

        # Off-topic
        r"this is not related",
        r"without more context",
        r"i need more information",

        # Errors in code
        r"# error:",
        r"# todo:",
        r"# fixme:",
        r"raise NotImplementedError",

        # Incomplete code
        r"pass  # TODO",
        r"pass  # FIXME",
        r"\\.\\.\\.",  # Ellipsis in code
    ]

    # Model fallback chain
    FALLBACK_CHAIN = [
        "claude-sonnet-4.6",
        "claude-opus-4.6",
        "gpt-4o",
        "deepseek/deepseek-chat",
    ]

    # Cost per 1M tokens
    MODEL_COSTS = {
        "deepseek/deepseek-chat": {"input": 1.0, "output": 4.0},
        "claude-sonnet-4.6": {"input": 3.0, "output": 15.0},
        "claude-opus-4.6": {"input": 15.0, "output": 75.0},
        "gpt-4o": {"input": 5.0, "output": 15.0},
    }

    def __init__(self, client=None):
        """
        Initialize streaming validator.

        Args:
            client: API client with streaming support
        """
        self.client = client
        self.metrics = StreamingMetrics()
        self._compiled_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.EARLY_ABORT_PATTERNS
        ]

    async def stream_and_validate(
        self,
        model: str,
        prompt: str,
        task_type: str = "code_generation",
        max_tokens: int = 4000,
        early_abort_tokens: int = 500,
        **kwargs,
    ) -> StreamingResult:
        """
        Stream generation with early validation.

        Args:
            model: Model to use
            prompt: Prompt text
            task_type: Task type
            max_tokens: Maximum output tokens
            early_abort_tokens: Check for failures in first N tokens
            **kwargs: Additional API parameters

        Returns:
            StreamingResult with response, chunks, tokens, cost
        """
        start_time = time.time()
        self.metrics.total_streams += 1

        chunks: list[str] = []
        total_tokens = 0
        early_aborted = False
        abort_reason: str | None = None
        retry_count = 0
        current_model = model

        for attempt in range(len(self.FALLBACK_CHAIN)):
            try:
                logger.info(
                    f"Streaming with {current_model} "
                    f"(attempt {attempt + 1}/{len(self.FALLBACK_CHAIN)})"
                )

                chunks.clear()
                total_tokens = 0
                early_aborted = False

                # Stream the response
                async for chunk in self._stream_with_model(
                    current_model, prompt, max_tokens, **kwargs
                ):
                    chunks.append(chunk)
                    total_tokens += len(chunk) / 4  # Estimate

                    # Early abort check (first N tokens)
                    if total_tokens < early_abort_tokens / 4:
                        partial = "".join(chunks)
                        failure = self._detect_early_failure(partial, task_type)

                        if failure:
                            logger.warning(
                                f"Early abort at {total_tokens:.0f} tokens: {failure}"
                            )

                            early_aborted = True
                            abort_reason = failure
                            self.metrics.early_aborts += 1

                            # Calculate tokens saved
                            tokens_saved = max_tokens - total_tokens
                            self.metrics.tokens_saved += tokens_saved

                            # Estimate cost savings
                            savings = self._estimate_cost(current_model, tokens_saved)
                            self.metrics.estimated_savings += savings

                            break

                if early_aborted:
                    # Retry with next model in chain
                    if attempt < len(self.FALLBACK_CHAIN) - 1:
                        current_model = self.FALLBACK_CHAIN[attempt + 1]
                        retry_count += 1
                        self.metrics.retries += 1
                        logger.info(f"Retrying with {current_model}")
                        continue
                    else:
                        # No more models to try
                        logger.warning("All models in fallback chain exhausted")
                        break

                else:
                    # Success
                    self.metrics.successful_streams += 1
                    break

            except Exception as e:
                logger.error(f"Streaming failed with {current_model}: {e}")

                # Try next model
                if attempt < len(self.FALLBACK_CHAIN) - 1:
                    current_model = self.FALLBACK_CHAIN[attempt + 1]
                    retry_count += 1
                    self.metrics.retries += 1
                    continue
                else:
                    raise

        latency = time.time() - start_time
        response = "".join(chunks)
        cost = self._estimate_cost(current_model, len(response))

        return StreamingResult(
            response=response,
            chunks=chunks,
            total_tokens=int(total_tokens),
            cost=cost,
            early_aborted=early_aborted,
            abort_reason=abort_reason,
            retry_count=retry_count,
            latency_seconds=latency,
        )

    async def _stream_with_model(
        self,
        model: str,
        prompt: str,
        max_tokens: int,
        **kwargs,
    ) -> AsyncIterator[str]:
        """
        Stream response from model.

        Args:
            model: Model to use
            prompt: Prompt text
            max_tokens: Maximum output tokens
            **kwargs: Additional parameters

        Yields:
            Text chunks
        """
        if self.client is None:
            raise RuntimeError("No client available")

        # Check if client supports streaming
        if hasattr(self.client, 'stream'):
            async for chunk in self.client.stream(model, prompt, **kwargs):
                yield chunk
        elif hasattr(self.client, 'stream_create'):
            stream = await self.client.stream_create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                **kwargs,
            )
            async for chunk in stream:
                if hasattr(chunk, 'choices') and chunk.choices:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content') and delta.content:
                        yield delta.content
        else:
            # Fallback: simulate streaming with regular call
            logger.warning(f"Client doesn't support streaming, using fallback for {model}")
            response = await self.client.call(
                model=model,
                system_prompt=prompt,
                max_tokens=max_tokens,
                **kwargs,
            )

            text = response.text if hasattr(response, 'text') else str(response)

            # Yield in chunks
            chunk_size = 100
            for i in range(0, len(text), chunk_size):
                yield text[i:i + chunk_size]
                await asyncio.sleep(0.01)  # Simulate streaming delay

    def _detect_early_failure(
        self,
        text: str,
        task_type: str,
    ) -> str | None:
        """
        Detect early failure in partial text.

        Args:
            text: Partial generated text
            task_type: Task type

        Returns:
            Failure reason or None
        """
        text_lower = text.lower()

        # Check abort patterns
        for pattern in self._compiled_patterns:
            if pattern.search(text_lower):
                return f"Matched pattern: {pattern.pattern}"

        # Task-specific checks
        if task_type == "code_generation":
            # Check for code structure
            if len(text) > 200:
                if "```" not in text and "def " not in text and "class " not in text:
                    return "No code structure detected"

        elif task_type == "code_review":
            # Check for review structure
            if len(text) > 200:
                if "score" not in text_lower and "quality" not in text_lower:
                    return "No review content detected"

        elif task_type == "decomposition":
            # Check for JSON/list structure
            if len(text) > 200 and "[" not in text and "{" not in text:
                return "No structured output detected"

        return None

    def _estimate_cost(
        self,
        model: str,
        tokens: int,
    ) -> float:
        """
        Estimate cost for tokens.

        Args:
            model: Model used
            tokens: Number of tokens

        Returns:
            Estimated cost in USD
        """
        model_key = model.lower()
        costs = self.MODEL_COSTS.get(model_key, {"input": 3.0, "output": 15.0})

        output_cost = (tokens / 1_000_000) * costs["output"]
        return output_cost

    def get_metrics(self) -> dict[str, Any]:
        """Get streaming metrics."""
        return self.metrics.to_dict()

    def reset_metrics(self) -> None:
        """Reset metrics."""
        self.metrics = StreamingMetrics()


# ─────────────────────────────────────────────
# Convenience Functions
# ─────────────────────────────────────────────

async def stream_and_validate(
    client,
    model: str,
    prompt: str,
    task_type: str = "code_generation",
) -> StreamingResult:
    """Convenience function for streaming validation."""
    validator = StreamingValidator(client=client)
    return await validator.stream_and_validate(model, prompt, task_type)


__all__ = [
    "StreamingValidator",
    "StreamingMetrics",
    "StreamingResult",
    "stream_and_validate",
]
