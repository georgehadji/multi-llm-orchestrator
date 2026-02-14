"""
API Clients — Unified interface for OpenAI, Anthropic, Google
=============================================================
Each provider has its own SDK idiom. This module normalizes them
into a single async call_model() interface.

Counterfactual: Without unified interface → vulnerability Ψ:
provider-specific logic leaks into orchestrator, creating
O(n_providers × n_features) maintenance surface.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from .models import Model, get_provider, estimate_cost
from .cache import DiskCache

logger = logging.getLogger("orchestrator.api")


class APIResponse:
    """Normalized response from any provider."""
    __slots__ = ("text", "input_tokens", "output_tokens", "model", "cost_usd",
                 "cached", "latency_ms")

    def __init__(self, text: str, input_tokens: int, output_tokens: int,
                 model: Model, cached: bool = False, latency_ms: float = 0.0):
        self.text = text
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.model = model
        self.cost_usd = 0.0 if cached else estimate_cost(model, input_tokens, output_tokens)
        self.cached = cached
        self.latency_ms = latency_ms


class UnifiedClient:
    """
    Async API client with:
    - Disk caching
    - Retry with exponential backoff
    - Timeout enforcement
    - Concurrency limiting (≤3 simultaneous, per George's laptop spec)
    """

    def __init__(self, cache: Optional[DiskCache] = None,
                 max_concurrency: int = 3):
        self.cache = cache or DiskCache()
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self._clients: dict[str, object] = {}
        self._init_clients()

    def _init_clients(self):
        """Lazy-initialize provider SDKs. Missing keys → provider unavailable."""
        import os
        from dotenv import load_dotenv
        load_dotenv(override=True)

        # OpenAI
        try:
            if os.environ.get("OPENAI_API_KEY"):
                from openai import AsyncOpenAI
                self._clients["openai"] = AsyncOpenAI()
                logger.info("OpenAI client initialized")
        except ImportError:
            logger.warning("openai package not installed")

        # Anthropic
        try:
            if os.environ.get("ANTHROPIC_API_KEY"):
                from anthropic import AsyncAnthropic
                self._clients["anthropic"] = AsyncAnthropic()
                logger.info("Anthropic client initialized")
        except ImportError:
            logger.warning("anthropic package not installed")

        # Google
        try:
            if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
                from google import genai
                import os as _os
                api_key = _os.environ.get("GOOGLE_API_KEY") or _os.environ.get("GEMINI_API_KEY")
                self._clients["google"] = genai.Client(api_key=api_key)
                logger.info("Google GenAI client initialized")
        except ImportError:
            logger.warning("google-genai package not installed")

        # Kimi K2.5 (OpenAI-compatible, via moonshot.cn)
        try:
            if os.environ.get("KIMI_API_KEY"):
                from openai import AsyncOpenAI
                self._clients["kimi"] = AsyncOpenAI(
                    api_key=os.environ.get("KIMI_API_KEY"),
                    base_url="https://api.moonshot.cn/v1",
                )
                logger.info("Kimi K2.5 client initialized")
        except ImportError:
            logger.warning("openai package not installed (required for Kimi K2.5)")

    def is_available(self, model: Model) -> bool:
        provider = get_provider(model)
        return provider in self._clients

    async def call(self, model: Model, prompt: str,
                   system: str = "",
                   max_tokens: int = 1500,
                   temperature: float = 0.3,
                   timeout: int = 60,
                   retries: int = 2) -> APIResponse:
        """
        Unified call with cache check → semaphore → retry → provider dispatch.
        """
        # Cache check
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

        # Execute with concurrency limit
        async with self.semaphore:
            return await self._call_with_retry(
                model, prompt, system, max_tokens, temperature, timeout, retries
            )

    async def _call_with_retry(self, model: Model, prompt: str,
                                system: str, max_tokens: int,
                                temperature: float, timeout: int,
                                retries: int) -> APIResponse:
        last_error = None
        for attempt in range(retries + 1):
            try:
                t0 = time.monotonic()
                response = await asyncio.wait_for(
                    self._dispatch(model, prompt, system, max_tokens, temperature),
                    timeout=timeout
                )
                response.latency_ms = (time.monotonic() - t0) * 1000

                # Cache successful response
                await self.cache.put(
                    model.value, prompt, max_tokens,
                    response.text, response.input_tokens, response.output_tokens,
                    system, temperature
                )
                return response

            except asyncio.TimeoutError:
                logger.warning(f"Timeout calling {model.value} (attempt {attempt + 1})")
                last_error = TimeoutError(f"{model.value} timed out after {timeout}s")
            except Exception as e:
                logger.warning(f"Error calling {model.value}: {e} (attempt {attempt + 1})")
                last_error = e
                if "rate_limit" in str(e).lower() or "429" in str(e):
                    await asyncio.sleep(2 ** attempt)
                    continue

        raise last_error or RuntimeError(f"Failed to call {model.value}")

    async def _dispatch(self, model: Model, prompt: str,
                        system: str, max_tokens: int,
                        temperature: float) -> APIResponse:
        provider = get_provider(model)

        if provider == "openai":
            return await self._call_openai(model, prompt, system, max_tokens, temperature)
        elif provider == "anthropic":
            return await self._call_anthropic(model, prompt, system, max_tokens, temperature)
        elif provider == "google":
            return await self._call_google(model, prompt, system, max_tokens, temperature)
        elif provider == "kimi":
            return await self._call_kimi(model, prompt, system, max_tokens, temperature)
        else:
            raise ValueError(f"Unknown provider for {model.value}")

    async def _call_openai(self, model: Model, prompt: str,
                            system: str, max_tokens: int,
                            temperature: float) -> APIResponse:
        client = self._clients["openai"]
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

    async def _call_anthropic(self, model: Model, prompt: str,
                               system: str, max_tokens: int,
                               temperature: float) -> APIResponse:
        client = self._clients["anthropic"]
        kwargs = {
            "model": model.value,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        response = await client.messages.create(**kwargs)
        text = ""
        for block in response.content:
            if hasattr(block, "text"):
                text += block.text

        return APIResponse(
            text=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=model,
        )

    async def _call_google(self, model: Model, prompt: str,
                            system: str, max_tokens: int,
                            temperature: float) -> APIResponse:
        client = self._clients["google"]
        from google.genai import types

        config = types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
        if system:
            config.system_instruction = system

        # google-genai uses sync API; run in executor
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.models.generate_content(
                model=model.value,
                contents=prompt,
                config=config,
            )
        )

        text = response.text or ""
        # Token counts from usage_metadata
        input_tokens = 0
        output_tokens = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
            output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0

        return APIResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
        )

    async def _call_kimi(self, model: Model, prompt: str,
                         system: str, max_tokens: int,
                         temperature: float) -> APIResponse:
        client = self._clients["kimi"]
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
