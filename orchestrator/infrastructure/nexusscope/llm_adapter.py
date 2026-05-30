"""Profiling adapter for LLM calls."""

from __future__ import annotations
from typing import Any

from .profiler import NexusScopeProfiler


class ProfiledLLMClient:
    """Wraps an LLM client with profiling."""

    def __init__(self, inner: Any, profiler: NexusScopeProfiler | None = None):
        self._inner = inner
        self._profiler = profiler or NexusScopeProfiler()

    async def call(
        self, model, prompt, system="", max_tokens=1500, temperature=0.3, timeout=120, **kwargs
    ):
        m = model.value if hasattr(model, "value") else str(model)
        async with self._profiler.async_session(f"llm.call.{m}"):
            return await self._inner.call(
                model=model,
                prompt=prompt,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
                **kwargs,
            )
