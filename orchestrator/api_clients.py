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
import time

from typing import Any, Awaitable, Callable

from openai import AsyncOpenAI

from .cache import DiskCache
from .circuit_breaker import CircuitBreaker
from .config import OPENROUTER_OPTS
from .models import (
    Model, 
    TaskType,
    TASK_PROVIDER_STRATEGIES,
    estimate_cost, 
    get_provider
)
from .model_registry import ModelRegistry
from .resilience import ResiliencePolicy, resolve_fallback_chain, run_with_resilience
from .task_schemas import generate_openrouter_schema
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


def _is_auth_error(error: Exception) -> bool:
    """Detect authentication errors (401)."""
    err_str = str(error).lower()
    return any(p in err_str for p in ["401", "unauthorized", "user not found", "invalid api key"])


class AuthenticationError(RuntimeError):
    """Raised when API authentication fails (invalid/missing API key)."""
    pass


def validate_model_available(model: Model | str) -> tuple[bool, str | None]:
    """
    Validate that a model is available on OpenRouter.

    Args:
        model: Model enum or string model ID (supports variants like "openai/gpt-4o:nitro")

    Returns:
        Tuple of (is_available, replacement_model_if_any)
    """
    # Handle both Model enum and string model IDs
    if isinstance(model, Model):
        model_id = model.value
    else:
        model_id = model
        # Strip variant suffix for validation (e.g., "openai/gpt-4o:nitro" -> "openai/gpt-4o")
        if ":" in model_id:
            model_id = model_id.split(":")[0]

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
        circuit_breaker: CircuitBreaker | None = None,
    ):
        """
        Initialize UnifiedClient.

        Args:
            cache: Disk cache for responses
            max_concurrency: Maximum concurrent requests
            connect_timeout: Connection timeout in seconds
            read_timeout: Read timeout in seconds
            circuit_breaker: Optional pre-configured CircuitBreaker; a default one is created if None
        """
        self.cache = cache or DiskCache()
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self._clients: dict[str, object] = {}
        self.circuit_breaker = circuit_breaker or CircuitBreaker(
            name="openrouter",
            failure_threshold=5,
            reset_timeout=60.0,
            success_threshold=2,
        )

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
        model: Model | str,
        prompt: str,
        system: str = "",
        max_tokens: int = 1500,
        temperature: float = 0.3,
        timeout: int = 60,
        retries: int = 2,
        bypass_cache: bool = False,
        task_type: TaskType | None = None,
        response_schema: bool = False,
        fallback_models: list[str] | None = None,
        policy: ResiliencePolicy | None = None,
    ) -> APIResponse:
        """
        Unified call with model validation → cache check → semaphore → retry → OpenRouter dispatch.

        Args:
            model: Model enum or string model ID (supports variants like "openai/gpt-4o:nitro")
            prompt: User prompt
            system: System message
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            timeout: Request timeout in seconds
            retries: Number of retries on failure
            bypass_cache: Skip cache read/write
            task_type: Task type for variant/strategy selection
            response_schema: Use JSON schema structured output (requires task_type)
            fallback_models: List of fallback model IDs for OpenRouter native fallbacks
            policy: Optional ResiliencePolicy. When provided, retries and model-level
                    fallback cascade are handled by the resilience layer instead of
                    the legacy manual loop.

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

        # Get string model ID for cache key
        model_id = model.value if isinstance(model, Model) else model
        
        if not bypass_cache:
            cached = await self.cache.get(model_id, prompt, max_tokens, system, temperature)
            if cached:
                logger.debug(f"Cache hit for {model_id}")
                # Create model instance for response (handle string IDs)
                model_for_response = model if isinstance(model, Model) else Model(model_id.split(":")[0])
                return APIResponse(
                    text=cached["response"],
                    input_tokens=cached["tokens_input"],
                    output_tokens=cached["tokens_output"],
                    model=model_for_response,
                    cached=True,
                )

        # Fail-fast if circuit is open (provider is down)
        async with self.circuit_breaker.context():
            async with self.semaphore:
                with traced_llm_call(model_id, "api_call") as span:
                    if policy is not None:
                        response = await self._call_with_policy(
                            model, prompt, system, max_tokens, temperature,
                            policy=policy,
                            task_type=task_type,
                            response_schema=response_schema,
                            fallback_models=fallback_models,
                        )
                    else:
                        # LEGACY: remove after full migration to ResiliencePolicy
                        response = await self._call_with_retry(
                            model, prompt, system, max_tokens, temperature, timeout, retries,
                            task_type=task_type,
                            response_schema=response_schema,
                            fallback_models=fallback_models,
                        )
                    span.set_attribute("llm.tokens_in", response.input_tokens)
                    span.set_attribute("llm.tokens_out", response.output_tokens)
                    span.set_attribute("llm.cost_usd", response.cost_usd)
                    span.set_attribute("llm.latency_ms", response.latency_ms)
                    span.set_attribute("llm.cached", False)
                    return response

    async def _call_with_retry(
        self,
        model: Model | str,
        prompt: str,
        system: str,
        max_tokens: int,
        temperature: float,
        timeout: int,
        retries: int,
        task_type: TaskType | None = None,
        response_schema: bool = False,
        fallback_models: list[str] | None = None,
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
                    self._dispatch(
                        model, prompt, system, max_tokens, temperature,
                        task_type=task_type,
                        response_schema=response_schema,
                        fallback_models=fallback_models,
                    ), 
                    timeout=timeout
                )
                response.latency_ms = (time.monotonic() - t0) * 1000

                model_id = model.value if isinstance(model, Model) else model
                await self.cache.put(
                    model_id,
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
        model_id = model.value if isinstance(model, Model) else model
        for attempt in range(retries + 1):
            try:
                t0 = time.monotonic()
                response = await asyncio.wait_for(
                    self._dispatch(
                        model, prompt, system, max_tokens, temperature,
                        task_type=task_type,
                        response_schema=response_schema,
                        fallback_models=fallback_models,
                    ), 
                    timeout=timeout
                )
                response.latency_ms = (time.monotonic() - t0) * 1000

                await self.cache.put(
                    model_id,
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
                logger.warning(f"Timeout calling {model_id} (attempt {attempt + 1})")
                last_error = TimeoutError(f"{model_id} timed out after {timeout}s")
            except asyncio.CancelledError:
                # Must re-raise — swallowing CancelledError breaks asyncio.wait_for
                # callers and can hang the entire orchestration loop.
                raise
            except Exception as e:
                logger.warning(f"Error calling {model_id}: {e} (attempt {attempt + 1})")
                last_error = e
                # Check for authentication errors (401)
                if _is_auth_error(e):
                    raise AuthenticationError(
                        "OpenRouter API authentication failed. "
                        "Your OPENROUTER_API_KEY may be invalid or expired. "
                        "Get a new key at https://openrouter.ai/keys"
                    ) from e
                if _is_rate_limit_error(e):
                    backoff = 2 ** (attempt + 1)
                    logger.info(f"Rate-limited by {model_id}, backing off {backoff}s")
                    await asyncio.sleep(backoff)
                    continue

        raise last_error or RuntimeError(f"Failed to call {model_id}")

    async def _call_with_policy(
        self,
        model: Model | str,
        prompt: str,
        system: str,
        max_tokens: int,
        temperature: float,
        policy: ResiliencePolicy,
        task_type: TaskType | None = None,
        response_schema: bool = False,
        fallback_models: list[str] | None = None,
    ) -> APIResponse:
        """Dispatch with ResiliencePolicy: tenacity retries + model cascade fallback."""

        def _make_callable(m: Model | str) -> Callable[[], Awaitable[APIResponse]]:
            async def _callable() -> APIResponse:
                t0 = time.monotonic()
                response = await asyncio.wait_for(
                    self._dispatch(
                        m, prompt, system, max_tokens, temperature,
                        task_type=task_type,
                        response_schema=response_schema,
                        fallback_models=fallback_models,
                    ),
                    timeout=policy.timeout,
                )
                response.latency_ms = (time.monotonic() - t0) * 1000

                mid = m.value if isinstance(m, Model) else m
                await self.cache.put(
                    mid,
                    prompt,
                    max_tokens,
                    response.text,
                    response.input_tokens,
                    response.output_tokens,
                    system,
                    temperature,
                )
                return response

            return _callable

        # Primary model (already validated in call())
        callables = [_make_callable(model)]

        # Append fallback chain from policy or static FALLBACK_CHAIN
        chain = policy.fallback_chain
        if chain is None:
            try:
                from_model = model if isinstance(model, Model) else Model(model)
                chain = tuple(resolve_fallback_chain(from_model))
            except ValueError:
                chain = ()

        for fallback_model in chain:
            callables.append(_make_callable(fallback_model))

        return await run_with_resilience(callables, policy)

    async def _dispatch(
        self, 
        model: Model | str, 
        prompt: str, 
        system: str, 
        max_tokens: int, 
        temperature: float,
        task_type: TaskType | None = None,
        response_schema: bool = False,
        fallback_models: list[str] | None = None,
    ) -> APIResponse:
        """Dispatch to OpenRouter with optimization features.
        
        Args:
            model: Model enum or string model ID (supports variants like "openai/gpt-4o:nitro")
            prompt: User prompt
            system: System message
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            task_type: Task type for variant/strategy selection
            response_schema: Use JSON schema structured output
            fallback_models: List of fallback model IDs for OpenRouter native fallbacks
        """
        client: AsyncOpenAI = self._clients.get("openrouter")
        if not client:
            raise RuntimeError("OpenRouter client not initialized")

        # Get string model ID
        model_id = model.value if isinstance(model, Model) else model
        
        # Validate and sanitize variant suffixes
        # NOTE: As of 2026-04-05, only :free, :thinking, :extended are supported
        SUPPORTED_VARIANTS = [':free', ':thinking', ':extended']
        has_unsupported_variant = any(
            model_id.endswith(v) for v in [':nitro', ':floor', ':exacto']
        )
        if has_unsupported_variant:
            # Strip unsupported variant and log warning (rate-limited)
            base_model = model_id.split(':')[0]
            
            # Initialize warning cache if needed (with corruption guard)
            if not hasattr(self, '_variant_warning_cache') or not isinstance(getattr(self, '_variant_warning_cache'), set):
                self._variant_warning_cache = set()
            
            # Only warn once per unique variant model ID
            if model_id not in self._variant_warning_cache:
                self._variant_warning_cache.add(model_id)
                logger.warning(
                    f"Model variant '{model_id}' not supported by OpenRouter yet, "
                    f"using base model '{base_model}'. Supported: {SUPPORTED_VARIANTS}"
                )
            model_id = base_model
        
        # Build request parameters
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        request_params: dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        # Feature: JSON Schema structured output
        if response_schema and task_type and OPENROUTER_OPTS.USE_JSON_SCHEMA_RESPONSES:
            schema = generate_openrouter_schema(task_type.value)
            if schema:
                request_params["response_format"] = schema
                logger.debug(f"Using JSON schema for {task_type.value}")

        # Feature: Native fallback models
        if fallback_models and OPENROUTER_OPTS.USE_NATIVE_FALLBACKS:
            request_params["models"] = fallback_models
            logger.debug(f"Using native fallbacks: {fallback_models}")

        # Feature: Provider sorting strategy
        if task_type and OPENROUTER_OPTS.USE_PROVIDER_SORTING:
            strategy = TASK_PROVIDER_STRATEGIES.get(task_type)
            if strategy:
                request_params["provider"] = {
                    "sort": strategy.sort,
                }
                if strategy.preferred_min_throughput:
                    request_params["provider"]["preferred_min_throughput"] = strategy.preferred_min_throughput
                if strategy.preferred_max_latency:
                    request_params["provider"]["preferred_max_latency"] = strategy.preferred_max_latency
                logger.debug(f"Using provider strategy: {strategy.sort}")

        # Check for reasoning models (strip variant suffix for check)
        base_model_id = model_id.split(":")[0]
        try:
            base_model = Model(base_model_id)
            if self._is_reasoning_model(base_model):
                return await self._call_reasoning_model(
                    client, base_model, prompt, system, max_tokens, temperature
                )
        except ValueError:
            pass  # Model not in enum, treat as standard

        # Standard chat completion with optimizations
        response = await client.chat.completions.create(**request_params)

        choice = response.choices[0]
        usage = response.usage
        
        # Get actual model used (from response for fallbacks)
        actual_model_id = response.model or model_id
        
        # Create Model enum for response (best effort)
        try:
            # Strip any provider suffix OpenRouter might add
            clean_model_id = actual_model_id.split(":")[0]
            response_model = Model(clean_model_id)
        except ValueError:
            # Unknown model, return as string if not a known Model enum value
            response_model = model if isinstance(model, Model) else None

        return APIResponse(
            text=choice.message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            model=response_model,
        )

    def _is_reasoning_model(self, model: Model) -> bool:
        """Check if model is a reasoning model requiring special handling."""
        return ModelRegistry.is_reasoning_model(model.value)

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
