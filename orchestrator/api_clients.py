"""
API Clients — OpenRouter-only interface
========================================
Author: Georgios-Chrysovalantis Chatzivantsidis
Uses OpenRouter API exclusively for all LLM access.

OpenRouter: https://openrouter.ai/api/v1
Env var: OPENROUTER_API_KEY

FIX #9: Rate-limit detection for all providers.
Note: Updated 2026-04-01 with runtime model validation.

"""

from __future__ import annotations

import asyncio
import logging
import os
import time

from .cache import DiskCache
from .models import Model, estimate_cost, get_provider
from .model_registry import ModelRegistry
from .tracing import traced_llm_call

logger = logging.getLogger("orchestrator.api")

# FIX #9: Rate-limit error patterns across providers
_RATE_LIMIT_PATTERNS = (
    "rate_limit",
    "rate limit",
    "429",
    "too many requests",
    "resource_exhausted",
    "quota",
    "overloaded",
)


def _is_rate_limit_error(error: Exception) -> bool:
    """Detect rate-limit errors across all providers."""
    err_str = str(error).lower()
    return any(p in err_str for p in _RATE_LIMIT_PATTERNS)


def validate_model_available(model: Model) -> tuple[bool, str | None]:
    """
    Validate that a model is available on OpenRouter.

    Args:
        model: Model to validate

    Returns:
        Tuple of (is_available, replacement_model_if_any)
    """
    model_id = model.value

    # Check if model is in unavailable list
    if model_id in ModelRegistry.UNAVAILABLE_MODELS:
        replacement = ModelRegistry.UNAVAILABLE_MODELS[model_id]
        return False, replacement

    # Check if model is in cost table (indicates it's valid)
    if model_id in ModelRegistry.COST_TABLE:
        return True, None

    # Unknown model - allow it through (might be new)
    logger.warning(f"Unknown model {model_id} - allowing through")
    return True, None


class APIResponse:
    """Normalized response from any provider."""

    __slots__ = (
        "text",
        "input_tokens",
        "output_tokens",
        "model",
        "cost_usd",
        "cached",
        "latency_ms",
    )

    def __init__(
        self,
        text: str,
        input_tokens: int,
        output_tokens: int,
        model: Model,
        cached: bool = False,
        latency_ms: float = 0.0,
    ):
        self.text = text
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.model = model
        self.cost_usd = 0.0 if cached else estimate_cost(model, input_tokens, output_tokens)
        self.cached = cached
        self.latency_ms = latency_ms


class UnifiedClient:
    """
    Async API client for OpenRouter exclusively with:
    - Disk caching
    - Retry with exponential backoff
    - Timeout enforcement
    - Concurrency limiting
    """

    # Default timeouts (seconds)
    DEFAULT_CONNECT_TIMEOUT: float = 10.0
    DEFAULT_READ_TIMEOUT: float = 60.0
    DEFAULT_TOTAL_TIMEOUT: float = 90.0

    def __init__(
        self,
        cache: DiskCache | None = None,
        max_concurrency: int = 3,
        connect_timeout: float | None = None,
        read_timeout: float | None = None,
    ):
        """
        Initialize UnifiedClient.

        Args:
            cache: Disk cache for responses
            max_concurrency: Maximum concurrent requests
            connect_timeout: Connection timeout in seconds
            read_timeout: Read timeout in seconds
        """
        self.cache = cache or DiskCache()
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self._clients: dict[str, object] = {}

        # Timeout configuration
        self._connect_timeout = connect_timeout or self.DEFAULT_CONNECT_TIMEOUT
        self._read_timeout = read_timeout or self.DEFAULT_READ_TIMEOUT
        self._init_clients()

    def _init_clients(self):
        """Initialize OpenRouter client exclusively."""
        import os

        # Configure timeouts
        try:
            import httpx

            timeout = httpx.Timeout(
                connect=self._connect_timeout,
                read=self._read_timeout,
                write=self._connect_timeout,
                pool=self._connect_timeout,
            )
        except ImportError:
            timeout = self._read_timeout

        # ═══════════════════════════════════════════════════════
        # OPENROUTER ONLY - All models via single endpoint
        # ═══════════════════════════════════════════════════════
        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        if openrouter_key:
            from openai import AsyncOpenAI

            self._clients["openrouter"] = AsyncOpenAI(
                api_key=openrouter_key,
                base_url="https://openrouter.ai/api/v1",
                timeout=timeout,
                max_retries=0,
                default_headers={
                    "HTTP-Referer": "https://github.com/georgehadji/multi-llm-orchestrator",
                    "X-Title": "Multi-LLM Orchestrator",
                },
            )
            logger.info("OpenRouter client initialized")
        else:
            logger.warning("OPENROUTER_API_KEY not set - no LLM client available")

    def is_available(self, model: Model) -> bool:
        """Check if model is available (always True for OpenRouter models)."""
        provider = get_provider(model)
        return provider == "openrouter" or "/" in model.value

    async def call(
        self,
        model: Model,
        prompt: str,
        system: str = "",
        max_tokens: int = 1500,
        temperature: float = 0.3,
        timeout: int = 60,
        retries: int = 2,
        bypass_cache: bool = False,
    ) -> APIResponse:
        """
        Unified call with model validation → cache check → semaphore → retry → OpenRouter dispatch.

        Note: Validates model availability before making API call.
        If model is unavailable and a replacement is configured, silently redirects
        with a warning rather than raising, to keep the pipeline running.
        """
        # Validate model availability; auto-redirect to replacement if configured
        is_available, replacement = validate_model_available(model)
        if not is_available:
            if replacement:
                from .models import Model as _Model
                try:
                    original_id = model.value
                    model = _Model(replacement)
                    logger.warning(
                        f"Model {original_id!r} is unavailable on OpenRouter, "
                        f"redirecting to {replacement!r}"
                    )
                except ValueError:
                    # replacement string not in Model enum — can't redirect, must fail
                    raise ValueError(
                        f"Model {model.value} is not available on OpenRouter and "
                        f"replacement {replacement!r} is not in the Model enum"
                    )
            else:
                raise ValueError(f"Model {model.value} is not available on OpenRouter")

        if not bypass_cache:
            cached = await self.cache.get(model.value, prompt, max_tokens, system, temperature)
            if cached:
                logger.debug(f"Cache hit for {model.value}")
                return APIResponse(
                    text=cached["response"],
                    input_tokens=cached["tokens_input"],
                    output_tokens=cached["tokens_output"],
                    model=model,
                    cached=True,
                )

        async with self.semaphore:
            with traced_llm_call(model.value, "api_call") as span:
                response = await self._call_with_retry(
                    model, prompt, system, max_tokens, temperature, timeout, retries
                )
                span.set_attribute("llm.tokens_in", response.input_tokens)
                span.set_attribute("llm.tokens_out", response.output_tokens)
                span.set_attribute("llm.cost_usd", response.cost_usd)
                span.set_attribute("llm.latency_ms", response.latency_ms)
                span.set_attribute("llm.cached", False)
                return response

    async def _call_with_retry(
        self,
        model: Model,
        prompt: str,
        system: str,
        max_tokens: int,
        temperature: float,
        timeout: int,
        retries: int,
    ) -> APIResponse:
        # NEW: Use Tenacity for robust retry logic with exponential backoff
        try:
            from .retry_utils import (
                llm_retry,
                TimeoutError,
                RateLimitError,
                ServiceUnavailableError,
            )

            # Create retry decorator with custom settings
            @llm_retry
            async def _call_with_tenacity() -> APIResponse:
                t0 = time.monotonic()
                response = await asyncio.wait_for(
                    self._dispatch(model, prompt, system, max_tokens, temperature), timeout=timeout
                )
                response.latency_ms = (time.monotonic() - t0) * 1000

                await self.cache.put(
                    model.value,
                    prompt,
                    max_tokens,
                    response.text,
                    response.input_tokens,
                    response.output_tokens,
                    system,
                    temperature,
                )
                return response

            # Execute with Tenacity retry logic
            return await _call_with_tenacity()

        except ImportError:
            logger.warning("Tenacity not available, using manual retry logic")
        except Exception as e:
            logger.warning(f"Tenacity retry failed: {e}, using manual retry logic")

        # FALLBACK: Original manual retry logic
        last_error = None
        for attempt in range(retries + 1):
            try:
                t0 = time.monotonic()
                response = await asyncio.wait_for(
                    self._dispatch(model, prompt, system, max_tokens, temperature), timeout=timeout
                )
                response.latency_ms = (time.monotonic() - t0) * 1000

                await self.cache.put(
                    model.value,
                    prompt,
                    max_tokens,
                    response.text,
                    response.input_tokens,
                    response.output_tokens,
                    system,
                    temperature,
                )
                return response
            except asyncio.TimeoutError:
                logger.warning(f"Timeout calling {model.value} (attempt {attempt + 1})")
                last_error = TimeoutError(f"{model.value} timed out after {timeout}s")
            except asyncio.CancelledError:
                elapsed = time.monotonic() - t0
                logger.warning(
                    f"Timeout calling {model.value} (attempt {attempt + 1}) "
                    f"[CancelledError after {elapsed:.1f}s]"
                )
                last_error = TimeoutError(f"{model.value} timed out after {timeout}s")
            except Exception as e:
                logger.warning(f"Error calling {model.value}: {e} (attempt {attempt + 1})")
                last_error = e
                if _is_rate_limit_error(e):
                    backoff = 2 ** (attempt + 1)
                    logger.info(f"Rate-limited by {model.value}, backing off {backoff}s")
                    await asyncio.sleep(backoff)
                    continue

        raise last_error or RuntimeError(f"Failed to call {model.value}")

    async def _dispatch(
        self, model: Model, prompt: str, system: str, max_tokens: int, temperature: float
    ) -> APIResponse:
        """Dispatch to OpenRouter."""
        client = self._clients.get("openrouter")
        if not client:
            raise RuntimeError("OpenRouter client not initialized")

        # Check for reasoning models
        if self._is_reasoning_model(model):
            return await self._call_reasoning_model(
                client, model, prompt, system, max_tokens, temperature
            )

        # Standard chat completion
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = await client.chat.completions.create(
            model=model.value,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        choice = response.choices[0]
        usage = response.usage

        return APIResponse(
            text=choice.message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            model=model,
        )

    def _is_reasoning_model(self, model: Model) -> bool:
        """Check if model is a reasoning model requiring special handling."""
        reasoning_models = {
            "o1",
            "o1-preview",
            "o3",
            "o3-mini",
            "o4-mini",
            "deepseek-r1",
            "deepseek/deepseek-r1",
            "grok-4",
            "grok-4-reasoning",
            "grok-4.20-reasoning",
        }
        return model.value in reasoning_models

    async def _call_reasoning_model(
        self, client, model: Model, prompt: str, system: str, max_tokens: int, temperature: float
    ) -> APIResponse:
        """Call reasoning model with special handling."""
        reasoning_timeout = 3600  # 1 hour for complex reasoning

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = await client.chat.completions.create(
            model=model.value,
            messages=messages,
            max_tokens=max_tokens,
            timeout=reasoning_timeout,
        )

        choice = response.choices[0]
        usage = response.usage

        return APIResponse(
            text=choice.message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            model=model,
        )
