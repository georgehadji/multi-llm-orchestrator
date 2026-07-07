import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import instructor
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from ..domain.services.config_services import CostService
from ..log_config import get_logger
from ..models import Model, Provider, get_provider

if TYPE_CHECKING:
    from instructor.client import AsyncInstructor

logger = get_logger(__name__)

# Constants
_REQUEST_TIMEOUT_SECONDS = 300
_MAX_RETRIES = 3

# Caching for API clients
_CLIENT_CACHE: dict[str, "AsyncInstructor[AsyncOpenAI]"] = {}

# ----------------------------------------------------------------
# Data Structures
# ----------------------------------------------------------------


@dataclass
class APIResponse:
    """Standardised API response with structured data."""

    text: str
    model: Model
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    cached: bool = False
    raw_response: ChatCompletion | None = None

    def __init__(
        self,
        text: str,
        model: Model,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
        latency_ms: float = 0.0,
        cached: bool = False,
        raw_response: ChatCompletion | None = None,
        **kwargs: Any,
    ):
        self.text = text
        self.model = model

        # Handle new field name aliases if passed as kwargs
        self.input_tokens = kwargs.get("prompt_tokens", input_tokens)
        self.output_tokens = kwargs.get("completion_tokens", output_tokens)

        # Handle cost vs cost_usd
        if "cost" in kwargs:
            self.cost_usd = kwargs["cost"] or 0.0
        else:
            self.cost_usd = cost_usd

        # Handle latency vs latency_ms
        if "latency" in kwargs:
            # If latency is passed in seconds, convert to ms
            self.latency_ms = (
                kwargs["latency"] * 1000.0
                if kwargs["latency"] and kwargs["latency"] < 500
                else (kwargs["latency"] or 0.0)
            )
        else:
            self.latency_ms = latency_ms

        self.cached = kwargs.get("cached", cached)
        self.raw_response = kwargs.get("raw_response", raw_response)

    @property
    def cost(self) -> float:
        return self.cost_usd

    @property
    def latency(self) -> float:
        return self.latency_ms / 1000.0

    @property
    def prompt_tokens(self) -> int:
        return self.input_tokens

    @property
    def completion_tokens(self) -> int:
        return self.output_tokens

    def to_dict(self) -> dict[str, Any]:
        """Convert response to a dictionary for serialization."""
        return {
            "text": self.text,
            "model": self.model.value if hasattr(self.model, "value") else str(self.model),
            "cost": self.cost,
            "latency": self.latency,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
            "latency_ms": self.latency_ms,
            "cached": self.cached,
        }


# ----------------------------------------------------------------
# Exceptions
# ----------------------------------------------------------------


class AuthenticationError(Exception):
    """Raised for API key authentication failures."""


# ----------------------------------------------------------------
# Unified API Client
# ----------------------------------------------------------------


class UnifiedClient:
    """
    Unified, health-aware client for multiple LLM providers via OpenRouter.

    Handles:
    - API key management
    - Automatic retries on transient errors
    - Cost and latency tracking
    - Standardised `APIResponse` object
    - Structured outputs via `instructor`
    """

    _default_client: "AsyncInstructor[AsyncOpenAI] | None" = None
    _clients: ClassVar[dict[str, "AsyncInstructor[AsyncOpenAI]"]] = {}
    _mode: ClassVar[instructor.Mode] = instructor.Mode.JSON

    def __init__(
        self,
        cost_service: CostService | None = None,
        openrouter_api_key: str | None = None,
        cache: Any = None,
        max_concurrency: int | None = None,
        **kwargs: Any,
    ):
        """
        Initialize the unified client.

        Args:
            cost_service: Service for tracking API costs.
            openrouter_api_key: OpenRouter API key. If not provided, it's
                                read from the OPENROUTER_API_KEY env var.
        """
        if cost_service is None:
            from ..domain.services.config_services import CostService
            from .adapters.config_adapter import JsonConfigAdapter

            cost_service = CostService(JsonConfigAdapter())

        self._cost_service = cost_service
        self._api_key = openrouter_api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self._api_key:
            raise AuthenticationError(
                "OpenRouter API key not found. " "Set OPENROUTER_API_KEY environment variable."
            )

        # Per-provider clients, lazily initialized
        self._provider_clients: dict[Provider, "AsyncInstructor[AsyncOpenAI]"] = {}

        # Set up disk cache for compatibility
        from ..cache import DiskCache

        self.cache = cache or DiskCache()

        # Set up circuit breaker for compatibility
        from ..circuit_breaker import CircuitBreaker

        self.circuit_breaker = kwargs.get("circuit_breaker") or CircuitBreaker(
            name="openrouter",
            failure_threshold=5,
            reset_timeout=60.0,
            success_threshold=2,
        )

    async def close(self) -> None:
        """Close all cached HTTP clients and release connection pools."""
        clients = list(self._clients.values())
        if self._default_client is not None:
            clients.append(self._default_client)

        for client in clients:
            try:
                # instructor wraps AsyncOpenAI; close the underlying client.
                await client.client.close()
            except Exception:
                logger.warning("Failed to close LLM client", exc_info=True)

        self._clients.clear()
        self._provider_clients.clear()
        self._default_client = None

    def _init_clients(self) -> None:
        """Shim to support legacy mock-based tests."""
        pass

    def _resolve_model_enum(self, model_str: str) -> Model:
        """Resolve a model string to a Model enum, falling back to GPT_4O_MINI."""
        from ..models import Model as _Model

        for m in _Model:
            if m.value == model_str or m.name == model_str:
                return m
        try:
            return _Model(model_str)
        except ValueError:
            fallback = getattr(_Model, "GPT_4O_MINI", None)
            if fallback is None:
                fallback = list(_Model)[0]
            logger.warning("Unknown model %r; falling back to %s", model_str, fallback.value)
            return fallback

    def is_available(self, model: Model) -> bool:
        """Check if model is available (always True for OpenRouter models)."""
        provider = get_provider(model)
        return provider == "openrouter" or "/" in model.value

    async def call(
        self,
        model: Model | str,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> APIResponse:
        """
        Make an API call to the specified model.

        Args:
            model: The `Model` enum member or model string to use.
            prompt: The user prompt.
            system: The system prompt (optional).
            max_tokens: Maximum tokens for the completion.
            temperature: Sampling temperature.
            **kwargs: Additional parameters for the OpenAI API.

        Returns:
            An `APIResponse` object with the result.
        """
        # Resolve Model enum vs string
        from ..models import Model as _Model

        if model is None:
            model = _Model.GPT_4O_MINI

        if isinstance(model, str):
            model_id = model
            model_enum = self._resolve_model_enum(model)
        else:
            model_enum = model
            model_id = model.value

        bypass_cache = kwargs.pop("bypass_cache", False)
        if not bypass_cache and self.cache:
            cached = await self.cache.get(model_id, prompt, max_tokens, system or "", temperature)
            if cached:
                logger.debug(f"Cache hit for {model_id}")
                return APIResponse(
                    text=cached["response"],
                    input_tokens=cached.get("tokens_input", 0),
                    output_tokens=cached.get("tokens_output", 0),
                    model=model_enum,
                    cost_usd=cached.get("cost", 0.0),
                    latency_ms=cached.get("latency", 0.0),
                    cached=True,
                )

        start_time = asyncio.get_event_loop().time()
        client = await self._get_client_for_model(model_enum)
        messages = [{"role": "user", "content": prompt}]
        if system:
            messages.insert(0, {"role": "system", "content": system})

        timeout = kwargs.pop("timeout", _REQUEST_TIMEOUT_SECONDS)
        # Pop other non-standard arguments from older client interface
        kwargs.pop("retries", None)
        kwargs.pop("task_type", None)
        kwargs.pop("response_schema", None)
        kwargs.pop("fallback_models", None)
        kwargs.pop("policy", None)

        try:
            async with self.circuit_breaker.context():
                dispatch_res = await self._dispatch(
                    client=client,
                    model_id=model_id,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=timeout,
                    **kwargs,
                )
                latency = asyncio.get_event_loop().time() - start_time

                if (
                    isinstance(dispatch_res, APIResponse)
                    or hasattr(dispatch_res, "__mock_self__")
                    or (hasattr(dispatch_res, "text") and not hasattr(dispatch_res, "choices"))
                ):
                    # It's already an APIResponse (or a mocked one from tests)
                    api_response = APIResponse(
                        text=getattr(dispatch_res, "text", ""),
                        model=model_enum,
                        input_tokens=getattr(dispatch_res, "input_tokens", 0),
                        output_tokens=getattr(dispatch_res, "output_tokens", 0),
                        cost_usd=getattr(dispatch_res, "cost_usd", 0.0),
                        latency_ms=getattr(dispatch_res, "latency_ms", 0.0),
                        cached=getattr(dispatch_res, "cached", False),
                        raw_response=getattr(dispatch_res, "raw_response", None),
                    )
                else:
                    # It's a real ChatCompletion response
                    response = dispatch_res
                    completion_text = response.choices[0].message.content or ""
                    prompt_tokens = response.usage.prompt_tokens if response.usage else 0
                    completion_tokens = response.usage.completion_tokens if response.usage else 0

                    cost_rates = self._cost_service.get_cost(model_enum)
                    cost = (
                        prompt_tokens * cost_rates.get("input", 0.0)
                        + completion_tokens * cost_rates.get("output", 0.0)
                    ) / 1_000_000

                    api_response = APIResponse(
                        text=completion_text,
                        model=model_enum,
                        cost=cost,
                        latency=latency,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        raw_response=response,
                    )

                if not bypass_cache and self.cache:
                    await self.cache.put(
                        model_id,
                        prompt,
                        max_tokens,
                        api_response.text,
                        api_response.input_tokens,
                        api_response.output_tokens,
                        system or "",
                        temperature,
                    )

                return api_response
        except Exception as e:
            logger.error("API call to %s failed: %s", model_id, e)
            raise

    async def _dispatch(
        self,
        client: "AsyncInstructor[AsyncOpenAI]",
        model_id: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        timeout: float,
        **kwargs: Any,
    ) -> Any:
        """Surgically isolated method to perform the raw API request, allowing tests to mock it."""
        if "response_model" not in kwargs:
            return await client.client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
                **kwargs,
            )
        else:
            return await client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
                **kwargs,
            )

    async def _get_client_for_model(self, model: Model) -> "AsyncInstructor[AsyncOpenAI]":
        """Get or create an API client for the requested model."""
        provider = get_provider(model)

        # XAI-specific client with fallback
        if provider == "xai":
            xai_api_key = os.environ.get("XAI_API_KEY")
            if xai_api_key:
                logger.info("Using XAI API client for model %s", model.value)
                client = instructor.from_openai(
                    AsyncOpenAI(
                        base_url="https://api.xai.com/v1/",
                        api_key=xai_api_key,
                    ),
                    mode=self._mode,
                )
                self._clients[model] = client
                return client
            else:
                logger.warning(
                    "XAI model specified, but no XAI_API_KEY found. " "Falling back to OpenRouter."
                )

        # Default to OpenRouter client
        if self._default_client is None:
            self._default_client = await self._create_openrouter_client()
        return self._default_client

    async def _create_openrouter_client(self) -> "AsyncInstructor[AsyncOpenAI]":
        """Create a new OpenRouter client."""
        logger.info("Creating new OpenRouter client")
        client = instructor.from_openai(
            AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1/",
                api_key=self._api_key,
            ),
            mode=self._mode,
        )
        return client


async def validate_model_available(model: Model) -> bool:
    """Check if a model is available on OpenRouter."""
    import aiohttp

    url = "https://openrouter.ai/api/v1/models"
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    available_models = [m["id"] for m in data.get("data", [])]
                    return model.value in available_models
                return False
        except Exception:
            return False


_SORT_ALIASES = {":nitro": "throughput", ":floor": "price"}


def _resolve_provider_variant(model_id: str) -> tuple[str, str | None, bool]:
    """Resolve OpenRouter sorting-alias suffixes on a model slug.

    Returns ``(model_id, sort_override, exacto_requested)``:
      - ``:nitro``/``:floor`` are stripped from the slug and mapped to a
        ``provider.sort`` value ("throughput"/"price").
      - ``:exacto`` is kept on the slug (OpenRouter resolves it natively) and
        flagged so the caller does not override it with a task-strategy sort.
      - Endpoint variants and plain slugs are returned unchanged.
    """
    for suffix, sort in _SORT_ALIASES.items():
        if model_id.endswith(suffix):
            return model_id[: -len(suffix)], sort, False
    if model_id.endswith(":exacto"):
        return model_id, None, True
    return model_id, None, False


def _maybe_add_response_healing(request_params: dict, opts) -> None:
    """Add the OpenRouter ``response-healing`` plugin to a request when enabled.

    Only applies to non-streaming structured-output requests (those carrying a
    ``response_format``). The plugin repairs malformed JSON server-side
    (brackets, trailing commas, markdown fences); it cannot fix ``max_tokens``
    truncation. Idempotent and safe when ``opts`` is None.

    ``plugins`` is an OpenRouter-specific field, not a parameter of the OpenAI
    SDK's ``chat.completions.create()`` — so it is routed through ``extra_body``
    (which the SDK merges into the JSON request body) rather than set as a
    top-level kwarg, which would raise ``TypeError``.
    """
    if opts is None or not getattr(opts, "USE_RESPONSE_HEALING", False):
        return
    if "response_format" not in request_params or request_params.get("stream"):
        return
    extra_body = request_params.setdefault("extra_body", {})
    plugins = extra_body.setdefault("plugins", [])
    if not any(isinstance(p, dict) and p.get("id") == "response-healing" for p in plugins):
        plugins.append({"id": "response-healing"})
        logger.debug("Response-healing plugin enabled (via extra_body)")


__all__ = [
    "APIResponse",
    "AuthenticationError",
    "UnifiedClient",
    "validate_model_available",
    "_resolve_provider_variant",
    "_maybe_add_response_healing",
]
