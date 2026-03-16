"""
API Clients — Unified interface for OpenAI, Google, Anthropic, DeepSeek, Minimax
=================================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis
Each provider has its own SDK idiom. This module normalizes them
into a single async call_model() interface.

FIX #2: Google client uses asyncio.to_thread() (Python 3.9+)
        instead of deprecated asyncio.get_event_loop().
FIX #3: API keys are never logged; init logs provider availability only.
FIX #9: Rate-limit detection for all providers (not just OpenAI 429).

Providers:
- OpenAI (GPT-4o, GPT-4o-mini)
- Google (Gemini Pro, Gemini Flash)
- Anthropic (Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku)
- DeepSeek (Coder, Reasoner, Coder-V2) - OpenAI-compatible
- Minimax - OpenAI-compatible

"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from .models import Model, get_provider, estimate_cost
from .cache import DiskCache
from .tracing import traced_llm_call

logger = logging.getLogger("orchestrator.api")

# FIX #9: Rate-limit error patterns across providers
_RATE_LIMIT_PATTERNS = (
    "rate_limit", "rate limit", "429", "too many requests",
    "resource_exhausted", "quota", "overloaded",
)


def _is_rate_limit_error(error: Exception) -> bool:
    """Detect rate-limit errors across all providers."""
    err_str = str(error).lower()
    return any(p in err_str for p in _RATE_LIMIT_PATTERNS)


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
    - Retry with exponential backoff (all providers)
    - Timeout enforcement (connect + read timeouts)
    - Concurrency limiting
    """

    # Default timeouts (seconds)
    DEFAULT_CONNECT_TIMEOUT: float = 10.0   # Time to establish connection
    DEFAULT_READ_TIMEOUT: float = 60.0      # Time to read response
    DEFAULT_TOTAL_TIMEOUT: float = 90.0     # Total request timeout

    def __init__(self, cache: Optional[DiskCache] = None,
                 max_concurrency: int = 3,
                 connect_timeout: Optional[float] = None,
                 read_timeout: Optional[float] = None):
        self.cache = cache or DiskCache()
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self._clients: dict[str, object] = {}
        # Timeout configuration
        self._connect_timeout = connect_timeout or self.DEFAULT_CONNECT_TIMEOUT
        self._read_timeout = read_timeout or self.DEFAULT_READ_TIMEOUT
        self._init_clients()

    def _init_clients(self):
        """Lazy-initialize provider SDKs with proper timeout configuration."""
        import os

        # FIX #3: Never log API key values, only provider availability

        # Configure timeouts for OpenAI-compatible clients
        try:
            import httpx
            timeout = httpx.Timeout(
                connect=self._connect_timeout,
                read=self._read_timeout,
                write=self._connect_timeout,
                pool=self._connect_timeout,
            )
        except ImportError:
            # Fallback to float timeout if httpx not available
            timeout = self._read_timeout

        # OpenAI
        try:
            if os.environ.get("OPENAI_API_KEY"):
                from openai import AsyncOpenAI
                self._clients["openai"] = AsyncOpenAI(
                    timeout=timeout,
                    max_retries=0,  # We handle retries ourselves
                )
                logger.info("OpenAI client initialized")
        except ImportError:
            logger.warning("openai package not installed")

        # Google
        try:
            api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
            if api_key:
                from google import genai
                self._clients["google"] = genai.Client(api_key=api_key)
                logger.info("Google GenAI client initialized")
        except ImportError:
            logger.warning("google-genai package not installed")

        # Anthropic Claude (claude.ai) — Native API
        try:
            anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
            if anthropic_key:
                from anthropic import AsyncAnthropic
                self._clients["anthropic"] = AsyncAnthropic(
                    api_key=anthropic_key,
                    timeout=timeout,
                    max_retries=0,  # We handle retries ourselves
                )
                logger.info("Anthropic client initialized")
        except ImportError:
            logger.warning("anthropic package not installed (needed for Claude models)")

        # DeepSeek (platform.deepseek.com) — OpenAI-compatible API
        # Supports deepseek-chat, deepseek-reasoner (R1).
        try:
            deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
            if deepseek_key:
                from openai import AsyncOpenAI
                self._clients["deepseek"] = AsyncOpenAI(
                    api_key=deepseek_key,
                    base_url="https://api.deepseek.com/v1",
                    timeout=timeout,
                    max_retries=0,  # We handle retries ourselves
                )
                logger.info("DeepSeek client initialized")
        except ImportError:
            logger.warning("openai package not installed (needed for DeepSeek)")

        # Minimax (api.minimaxi.chat) — OpenAI-compatible API
        try:
            minimax_key = os.environ.get("MINIMAX_API_KEY")
            if minimax_key:
                from openai import AsyncOpenAI
                self._clients["minimax"] = AsyncOpenAI(
                    api_key=minimax_key,
                    base_url="https://api.minimaxi.chat/v1",
                    timeout=timeout,
                    max_retries=0,
                )
                logger.info("Minimax client initialized")
        except ImportError:
            logger.warning("openai package not installed (needed for Minimax)")
        
        # ═══════════════════════════════════════════════════════
        # NEW PROVIDERS (Added March 2026)
        # ═══════════════════════════════════════════════════════
        
        # Mistral AI (api.mistral.ai) — OpenAI-compatible API
        try:
            mistral_key = os.environ.get("MISTRAL_API_KEY")
            if mistral_key:
                from openai import AsyncOpenAI
                self._clients["mistral"] = AsyncOpenAI(
                    api_key=mistral_key,
                    base_url="https://api.mistral.ai/v1",
                    timeout=timeout,
                    max_retries=0,
                )
                logger.info("Mistral AI client initialized")
        except ImportError:
            logger.warning("openai package not installed (needed for Mistral)")
        
        # xAI Grok (api.x.ai) — OpenAI-compatible API
        try:
            xai_key = os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY")
            if xai_key:
                from openai import AsyncOpenAI
                self._clients["xai"] = AsyncOpenAI(
                    api_key=xai_key,
                    base_url="https://api.x.ai/v1",
                    timeout=timeout,
                    max_retries=0,
                )
                logger.info("xAI client initialized")
        except ImportError:
            logger.warning("openai package not installed (needed for xAI)")
        
        # Cohere (api.cohere.ai) — Native API
        try:
            cohere_key = os.environ.get("COHERE_API_KEY")
            if cohere_key:
                # Cohere has its own SDK but also supports OpenAI-compatible
                # For now, we use the native client
                import cohere
                self._clients["cohere"] = cohere.AsyncClient(api_key=cohere_key)
                logger.info("Cohere client initialized")
        except ImportError:
            logger.warning("cohere package not installed (needed for Cohere)")
        
        # Alibaba Qwen (dashscope-intl.aliyuncs.com) — OpenAI-compatible API
        try:
            dashscope_key = os.environ.get("DASHSCOPE_API_KEY")
            if dashscope_key:
                from openai import AsyncOpenAI
                self._clients["alibaba"] = AsyncOpenAI(
                    api_key=dashscope_key,
                    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
                    timeout=timeout,
                    max_retries=0,
                )
                logger.info("Alibaba Qwen client initialized")
        except ImportError:
            logger.warning("openai package not installed (needed for Alibaba)")
        
        # ByteDance Seed (ark.cn-beijing.volces.com) — OpenAI-compatible API
        try:
            volcengine_key = os.environ.get("ARK_API_KEY") or os.environ.get("VOLCENGINE_API_KEY")
            if volcengine_key:
                from openai import AsyncOpenAI
                self._clients["bytedance"] = AsyncOpenAI(
                    api_key=volcengine_key,
                    base_url="https://ark.cn-beijing.volces.com/api/v3",
                    timeout=timeout,
                    max_retries=0,
                )
                logger.info("ByteDance Seed client initialized")
        except ImportError:
            logger.warning("openai package not installed (needed for ByteDance)")
        
        # Zhipu GLM (api.z.ai) — OpenAI-compatible API
        try:
            zhipu_key = os.environ.get("ZHIPUAI_API_KEY") or os.environ.get("ZHIPU_API_KEY")
            if zhipu_key:
                from openai import AsyncOpenAI
                self._clients["zhipu"] = AsyncOpenAI(
                    api_key=zhipu_key,
                    base_url="https://api.z.ai/api/coding/paas/v4",
                    timeout=timeout,
                    max_retries=0,
                )
                logger.info("Zhipu GLM client initialized")
        except ImportError:
            logger.warning("openai package not installed (needed for Zhipu)")
        
        # Baidu Ernie — Requires special handling (not standard OpenAI)
        # Use via third-party APIs or Novita AI
        try:
            # Support both QIANFAN_ACCESS_KEY/SECRET_KEY and legacy BAIDU_API_KEY
            baidu_access = os.environ.get("QIANFAN_ACCESS_KEY") or os.environ.get("QIANFAN_AK")
            baidu_secret = os.environ.get("QIANFAN_SECRET_KEY") or os.environ.get("QIANFAN_SK")
            baidu_key = os.environ.get("BAIDU_API_KEY")
            if (baidu_access and baidu_secret) or baidu_key:
                # Baidu uses its own API format
                # For now, mark as available but requires custom implementation
                self._clients["baidu"] = "baidu"  # Placeholder
                logger.info("Baidu Ernie configured (requires custom handler)")
        except Exception:
            pass
        
        # Moonshot Kimi (api.moonshot.cn) — OpenAI-compatible API
        try:
            moonshot_key = os.environ.get("MOONSHOT_API_KEY")
            if moonshot_key:
                from openai import AsyncOpenAI
                self._clients["moonshot"] = AsyncOpenAI(
                    api_key=moonshot_key,
                    base_url="https://api.moonshot.cn/v1",
                    timeout=timeout,
                    max_retries=0,
                )
                logger.info("Moonshot Kimi client initialized")
        except ImportError:
            logger.warning("openai package not installed (needed for Moonshot)")
        
        # Tencent Hunyuan — Requires special handling
        try:
            # Support TENCENTCLOUD_SECRET_ID/KEY or HUNYUAN_API_KEY
            tencent_secret_id = os.environ.get("TENCENTCLOUD_SECRET_ID")
            tencent_secret_key = os.environ.get("TENCENTCLOUD_SECRET_KEY")
            tencent_key = os.environ.get("HUNYUAN_API_KEY") or os.environ.get("TENCENT_API_KEY")
            if (tencent_secret_id and tencent_secret_key) or tencent_key:
                self._clients["tencent"] = "tencent"  # Placeholder
                logger.info("Tencent Hunyuan configured (requires custom handler)")
        except Exception:
            pass
        
        # Baichuan — Requires special handling
        try:
            baichuan_key = os.environ.get("BAICHUAN_API_KEY")
            if baichuan_key:
                self._clients["baichuan"] = "baichuan"  # Placeholder
                logger.info("Baichuan configured (requires custom handler)")
        except Exception:
            pass

    def is_available(self, model: Model) -> bool:
        provider = get_provider(model)
        return provider in self._clients

    async def call(self, model: Model, prompt: str,
                   system: str = "",
                   max_tokens: int = 1500,
                   temperature: float = 0.3,
                   timeout: int = 60,
                   retries: int = 2,
                   bypass_cache: bool = False) -> APIResponse:
        """
        Unified call with cache check → semaphore → retry → provider dispatch.
        Set bypass_cache=True to skip the cache lookup (e.g. for decomposition
        calls where a previously-cached bad response should not be reused).
        """
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

                await self.cache.put(
                    model.value, prompt, max_tokens,
                    response.text, response.input_tokens, response.output_tokens,
                    system, temperature
                )
                return response

            except asyncio.TimeoutError:
                logger.warning(f"Timeout calling {model.value} (attempt {attempt + 1})")
                last_error = TimeoutError(f"{model.value} timed out after {timeout}s")
            except asyncio.CancelledError:
                # asyncio.wait_for() cancels the inner coroutine when the timeout
                # fires. httpx may raise CancelledError during its TCP stream
                # cleanup — this is NOT a parent-task cancellation, just a
                # side-effect of the timeout. Convert it to a TimeoutError so
                # the retry loop handles it gracefully instead of propagating.
                elapsed = time.monotonic() - t0
                logger.warning(
                    f"Timeout calling {model.value} (attempt {attempt + 1}) "
                    f"[CancelledError after {elapsed:.1f}s]"
                )
                last_error = TimeoutError(f"{model.value} timed out after {timeout}s")
            except Exception as e:
                logger.warning(f"Error calling {model.value}: {e} (attempt {attempt + 1})")
                last_error = e
                # FIX #9: Detect rate-limit errors from ALL providers
                if _is_rate_limit_error(e):
                    backoff = 2 ** (attempt + 1)
                    logger.info(f"Rate-limited by {model.value}, backing off {backoff}s")
                    await asyncio.sleep(backoff)
                    continue

        raise last_error or RuntimeError(f"Failed to call {model.value}")

    async def _dispatch(self, model: Model, prompt: str,
                        system: str, max_tokens: int,
                        temperature: float) -> APIResponse:
        provider = get_provider(model)

        if provider == "openai":
            return await self._call_openai(model, prompt, system, max_tokens, temperature)
        elif provider == "google":
            return await self._call_google(model, prompt, system, max_tokens, temperature)
        elif provider == "anthropic":
            return await self._call_anthropic(model, prompt, system, max_tokens, temperature)
        elif provider == "deepseek":
            return await self._call_deepseek(model, prompt, system, max_tokens, temperature)
        elif provider == "minimax":
            return await self._call_minimax(model, prompt, system, max_tokens, temperature)
        # ═══════════════════════════════════════════════════════
        # NEW PROVIDERS (March 2026)
        # ═══════════════════════════════════════════════════════
        elif provider in ("mistral", "xai", "alibaba", "bytedance", "zhipu", "moonshot"):
            # All these providers use OpenAI-compatible APIs
            return await self._call_openai_compatible(provider, model, prompt, system, max_tokens, temperature)
        elif provider == "cohere":
            return await self._call_cohere(model, prompt, system, max_tokens, temperature)
        elif provider == "baidu":
            raise NotImplementedError("Baidu Ernie requires custom implementation. Use via Novita AI or similar proxy.")
        elif provider == "tencent":
            raise NotImplementedError("Tencent Hunyuan requires custom implementation.")
        elif provider == "baichuan":
            raise NotImplementedError("Baichuan requires custom implementation.")
        else:
            raise ValueError(f"Unknown provider for {model.value}: {provider}")

    async def _call_openai(self, model: Model, prompt: str,
                            system: str, max_tokens: int,
                            temperature: float) -> APIResponse:
        client = self._clients["openai"]
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # OpenAI models (including o1/o3/o4 series) fix temperature at 1 and
        # reject any explicit temperature parameter — omit it entirely.
        response = await client.chat.completions.create(
            model=model.value,
            messages=messages,
            max_tokens=max_tokens,
        )
        choice = response.choices[0]
        usage = response.usage

        return APIResponse(
            text=choice.message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            model=model,
        )

    async def _call_google(self, model: Model, prompt: str,
                            system: str, max_tokens: int,
                            temperature: float) -> APIResponse:
        """
        FIX #2: Use asyncio.to_thread() instead of deprecated
        asyncio.get_event_loop().run_in_executor(). Compatible with Python 3.9+.
        """
        client = self._clients["google"]
        from google.genai import types

        config = types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
        if system:
            config.system_instruction = system

        def _sync_call():
            return client.models.generate_content(
                model=model.value,
                contents=prompt,
                config=config,
            )

        response = await asyncio.to_thread(_sync_call)

        text = response.text or ""
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

    async def _call_anthropic(self, model: Model, prompt: str,
                               system: str, max_tokens: int,
                               temperature: float) -> APIResponse:
        """
        Anthropic Claude via native Anthropic API.
        
        Notes:
        - Claude models use 'system' parameter (not messages[0] role)
        - max_tokens is required for Anthropic API
        - Input/output token counts available in usage
        """
        client = self._clients["anthropic"]
        
        message = await client.messages.create(
            model=model.value,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system if system else None,
            messages=[{"role": "user", "content": prompt}],
        )
        
        text = ""
        if message.content and len(message.content) > 0:
            text = message.content[0].text if hasattr(message.content[0], 'text') else str(message.content[0])
        
        input_tokens = message.usage.input_tokens if message.usage else 0
        output_tokens = message.usage.output_tokens if message.usage else 0

        return APIResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
        )

    async def _call_minimax(self, model: Model, prompt: str,
                            system: str, max_tokens: int,
                            temperature: float) -> APIResponse:
        """
        Minimax via api.minimaxi.chat OpenAI-compatible endpoint.
        Supports Minimax-4 and other models.
        """
        client = self._clients["minimax"]
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

    async def _call_deepseek(self, model: Model, prompt: str,
                              system: str, max_tokens: int,
                              temperature: float) -> APIResponse:
        """
        DeepSeek via platform.deepseek.com OpenAI-compatible endpoint.

        Supports models:
        - deepseek-chat      : fast, cheap ($0.28/$0.42 per 1M), strong on code
        - deepseek-reasoner  : o1-class reasoning, slower, outputs reasoning_content

        Notes:
        - deepseek-reasoner outputs reasoning_content (CoT) + content. Temperature
          and top_p are IGNORED for reasoner (has no effect). Only deepseek-chat
          supports temperature control.
        - DeepSeek-R1 does not support system prompts for reasoning tasks; if the model
          is reasoner and a system prompt is provided, it is prepended to the user message.
        """
        client = self._clients["deepseek"]
        messages = []

        # DeepSeek-R1 (reasoner) has limited system prompt support in some contexts;
        # prepend system content to user message as a safe fallback.
        if model.value == "deepseek-reasoner" and system:
            messages.append({"role": "user", "content": f"{system}\n\n{prompt}"})
        else:
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

        # Build API call params - deepseek-reasoner ignores temperature/top_p
        api_params = {
            "model": model.value,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if model.value != "deepseek-reasoner":
            api_params["temperature"] = temperature
        
        response = await client.chat.completions.create(**api_params)
        choice = response.choices[0]
        usage = response.usage
        text = choice.message.content or ""

        # DeepSeek-R1 uses reasoning tokens that count against max_tokens but
        # don't appear in content. Raise if we got nothing (budget too low) so
        # the engine retries with a fallback instead of caching an empty response.
        if model.value == "deepseek-reasoner" and not text.strip() and choice.finish_reason == "length":
            raise RuntimeError(
                f"deepseek-reasoner returned empty content with finish_reason='length'. "
                f"max_tokens={max_tokens} was too low for the internal reasoning budget. "
                f"completion_tokens={usage.completion_tokens if usage else '?'}"
            )

        return APIResponse(
            text=text,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            model=model,
        )


    async def _call_openai_compatible(self, provider: str, model: Model, prompt: str,
                                       system: str, max_tokens: int,
                                       temperature: float) -> APIResponse:
        """
        Generic handler for OpenAI-compatible APIs.
        Used by: Mistral, xAI, Alibaba, ByteDance, Zhipu, Moonshot
        """
        client = self._clients[provider]
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
        text = choice.message.content or ""

        return APIResponse(
            text=text,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            model=model,
        )

    async def _call_cohere(self, model: Model, prompt: str,
                           system: str, max_tokens: int,
                           temperature: float) -> APIResponse:
        """
        Cohere native API handler.
        """
        client = self._clients["cohere"]
        
        # Cohere uses a different API format
        response = await client.chat(
            model=model.value,
            message=prompt,
            preamble=system if system else None,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        
        text = response.text or ""
        
        # Cohere usage format differs from OpenAI
        input_tokens = response.meta.tokens.input_tokens if response.meta and response.meta.tokens else 0
        output_tokens = response.meta.tokens.output_tokens if response.meta and response.meta.tokens else 0

        return APIResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=model,
        )
