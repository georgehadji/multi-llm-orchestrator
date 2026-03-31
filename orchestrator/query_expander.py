"""
QueryExpander — LLM-based query expansion for hybrid search.
=============================================================
Uses DeepSeek-Chat to generate alternative phrasings of a search
query, improving recall in BM25 and vector search.
"""

from __future__ import annotations

import json

from .log_config import get_logger

logger = get_logger(__name__)

_EXPAND_PROMPT = """Generate {n} alternative phrasings for the following search query.
Return ONLY a JSON array of strings. Do not include the original query.

Query: {query}

JSON array:"""


class QueryExpander:
    """
    Expands a search query into multiple alternative phrasings using an LLM.

    Falls back to returning only the original query if the LLM is unavailable.
    """

    def __init__(
        self,
        model: str = "deepseek/deepseek-chat",
        max_variants: int = 3,
    ) -> None:
        self.model = model
        self.max_variants = max_variants

    async def expand(self, query: str) -> list[str]:
        """
        Return [original_query] + up to max_variants LLM-generated alternatives.

        Duplicates are removed; original is always first.
        """
        try:
            variants = await self._call_llm(query)
        except Exception as exc:
            logger.warning("QueryExpander LLM call failed (%s) — using original query only", exc)
            return [query]

        # Deduplicate while preserving order; original always first
        seen: set[str] = {query}
        result = [query]
        for v in variants[: self.max_variants]:
            v = v.strip()
            if v and v not in seen:
                seen.add(v)
                result.append(v)
        return result

    async def _call_llm(self, query: str) -> list[str]:
        """Call DeepSeek-Chat and parse JSON array of variants."""
        from .api_clients import UnifiedClient
        from .models import Model

        client = UnifiedClient()
        prompt = _EXPAND_PROMPT.format(n=self.max_variants, query=query)
        response = await client.call(
            Model(self.model),
            prompt,
            system="You are a search query expansion assistant. Respond ONLY with a valid JSON array of strings.",
            temperature=0.4,
            max_tokens=150,
        )
        content = response.text.strip()
        # Strip markdown code fences if present
        if "```" in content:
            content = content.replace("```json", "").replace("```", "").strip()
        return json.loads(content.strip())
