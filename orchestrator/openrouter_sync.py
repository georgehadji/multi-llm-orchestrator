"""
OpenRouter model sync — fetches current model pricing and capabilities.

Author: Georgios-Chrysovalantis Chatzivantsidis
Description: Queries the OpenRouter /models endpoint and returns structured
data compatible with the COST_TABLE format used in models.py.
Intended to be run monthly (via cron, CI, or manually) to keep pricing current.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"


class OpenRouterSync:
    """
    Fetches model metadata and pricing from the OpenRouter API.

    Usage::

        sync = OpenRouterSync(api_key="sk-or-...")
        models = await sync.fetch_models()
        for m in models:
            entry = sync.to_cost_table_entry(m)
            print(m["id"], entry)
    """

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    async def fetch_models(self) -> list[dict[str, Any]]:
        """
        Fetch the full model list from OpenRouter.

        Returns a list of model dicts, each with at least:
          - id: str
          - name: str
          - pricing: {prompt: str, completion: str}  (cost per token as string)
          - context_length: int
        """
        try:
            import aiohttp
        except ImportError as exc:
            raise ImportError(
                "aiohttp is required for OpenRouterSync. " "Install it with: pip install aiohttp"
            ) from exc

        headers = {"Authorization": f"Bearer {self._api_key}"}
        async with aiohttp.ClientSession() as session:
            async with session.get(_OPENROUTER_MODELS_URL, headers=headers) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(
                        f"OpenRouter /models returned HTTP {resp.status}: {text[:200]}"
                    )
                data = await resp.json()

        return data.get("data", [])

    def to_cost_table_entry(self, model_data: dict[str, Any]) -> dict[str, float]:
        """
        Convert an OpenRouter model dict to a COST_TABLE-compatible entry.

        OpenRouter pricing is per-token (as a string like "0.000003").
        COST_TABLE uses per-million-token costs as floats.

        Returns::

            {"input": <float per 1M tokens>, "output": <float per 1M tokens>}
        """
        pricing = model_data.get("pricing", {})
        try:
            input_per_token = float(pricing.get("prompt") or 0.0)
            output_per_token = float(pricing.get("completion") or 0.0)
        except (TypeError, ValueError):
            input_per_token = 0.0
            output_per_token = 0.0

        return {
            "input": input_per_token * 1_000_000,
            "output": output_per_token * 1_000_000,
        }

    async def build_cost_table_patch(self) -> dict[str, dict[str, float]]:
        """
        Fetch models and return a dict keyed by model ID with COST_TABLE entries.

        Can be used to update or patch the local COST_TABLE::

            patch = await sync.build_cost_table_patch()
            # patch == {"anthropic/claude-3-5-sonnet": {"input": 3.0, "output": 15.0}, ...}
        """
        models = await self.fetch_models()
        result: dict[str, dict[str, float]] = {}
        for model in models:
            model_id = model.get("id", "")
            if model_id:
                result[model_id] = self.to_cost_table_entry(model)
        logger.info("OpenRouter sync: fetched %d models", len(result))
        return result
