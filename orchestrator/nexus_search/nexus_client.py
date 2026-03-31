"""
Nexus Search — API Client
==========================
Author: Georgios-Chrysovalantis Chatzivantsidis

Low-level API client for Nexus Search (SearXNG backend).
All SearXNG references are abstracted internally.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any

import aiohttp

from .config import get_config
from .models import (
    NexusConfig,
    SearchResult,
    SearchResults,
    SearchSource,
)

logger = logging.getLogger("orchestrator.nexus_search")


class NexusClient:
    """
    Low-level API client for Nexus Search.

    This client wraps the SearXNG API internally. All SearXNG
    references are abstracted - users only see "Nexus Search".

    Usage:
        client = NexusClient()
        results = await client.search("Python async")
    """

    def __init__(self, config: NexusConfig | None = None):
        """
        Initialize Nexus client.

        Args:
            config: Nexus configuration (uses default if None)
        """
        self.config = config or get_config()
        self.session: aiohttp.ClientSession | None = None
        self._request_lock = asyncio.Lock()

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session

    async def close(self) -> None:
        """Close the client session."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None

    async def health_check(self) -> bool:
        """
        Check if Nexus Search is available.

        Returns:
            True if healthy, False otherwise
        """
        try:
            session = await self._get_session()
            async with session.get(f"{self.config.api_url}/healthz") as response:
                return response.status == 200
        except Exception as e:
            logger.debug(f"Nexus health check failed: {e}")
            return False

    async def search(
        self,
        query: str,
        categories: list[str] | None = None,
        num_results: int | None = None,
        language: str = "en",
        time_range: str | None = None,
    ) -> SearchResults:
        """
        Perform a search query.

        Args:
            query: Search query string
            categories: List of categories (web, academic, tech, news, code)
            num_results: Maximum number of results
            language: Language code
            time_range: Time range filter (day, week, month, year)

        Returns:
            SearchResults with findings
        """
        start_time = asyncio.get_event_loop().time()

        # Build request parameters
        params = {
            "q": query,
            "categories": ",".join(categories or ["general"]),
            "language": language,
            "format": "json",
        }

        if num_results:
            params["limit"] = min(num_results, self.config.max_results)
        else:
            params["limit"] = self.config.max_results

        if time_range:
            params["time_range"] = time_range

        try:
            session = await self._get_session()
            async with session.get(
                f"{self.config.api_url}/search",
                params=params,
            ) as response:
                response.raise_for_status()
                data = await response.json()

                # Parse results
                search_time = (asyncio.get_event_loop().time() - start_time) * 1000
                results = self._parse_results(data, query, search_time)

                logger.debug(
                    f"Nexus search: {query[:50]}... → {len(results)} results in {search_time:.0f}ms"
                )

                return results

        except aiohttp.ClientError as e:
            logger.error(f"Nexus search failed: {e}")
            return SearchResults(
                query=query,
                results=[],
                search_time=(asyncio.get_event_loop().time() - start_time) * 1000,
            )
        except Exception as e:
            logger.error(f"Unexpected search error: {e}")
            return SearchResults(
                query=query,
                results=[],
                search_time=(asyncio.get_event_loop().time() - start_time) * 1000,
            )

    def _parse_results(
        self,
        data: dict[str, Any],
        query: str,
        search_time: float,
    ) -> SearchResults:
        """
        Parse API response into SearchResults.

        Args:
            data: Raw API response
            query: Original query
            search_time: Search time in ms

        Returns:
            Parsed SearchResults
        """
        results = []

        # Parse SearXNG results
        for item in data.get("results", []):
            # Map SearXNG category to SearchSource
            source = self._map_source(item.get("category", "general"))

            # Parse published date if available
            published_date = None
            if "publishedDate" in item:
                try:
                    published_date = datetime.fromisoformat(
                        item["publishedDate"].replace("Z", "+00:00")
                    )
                except (ValueError, TypeError):
                    pass

            result = SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                content=item.get("content", "") or item.get("snippet", ""),
                source=source,
                engine=item.get("engine", ""),
                score=item.get("score", 0.0),
                published_date=published_date,
                metadata={
                    "engines": item.get("engines", []),
                    "positions": item.get("positions", []),
                },
            )
            results.append(result)

        # Get suggestions
        suggestions = data.get("suggestions", [])
        if data.get("correction"):
            suggestions.insert(0, data["correction"])

        # Get total results
        total_results = data.get("number_of_results", len(results))

        # Determine sources from results
        sources = list({r.source for r in results})

        return SearchResults(
            query=query,
            results=results,
            total_results=total_results,
            search_time=search_time,
            sources=sources,
            suggestions=suggestions,
        )

    def _map_source(self, category: str) -> SearchSource:
        """
        Map SearXNG category to SearchSource.

        Args:
            category: SearXNG category

        Returns:
            SearchSource enum value
        """
        mapping = {
            "general": SearchSource.WEB,
            "web": SearchSource.WEB,
            "science": SearchSource.ACADEMIC,
            "scholar": SearchSource.ACADEMIC,
            "pubmed": SearchSource.ACADEMIC,
            "arxiv": SearchSource.ACADEMIC,
            "it": SearchSource.TECH,
            "github": SearchSource.CODE,
            "stackoverflow": SearchSource.CODE,
            "news": SearchSource.NEWS,
        }

        return mapping.get(category, SearchSource.WEB)

    async def get_engines(self) -> list[dict[str, Any]]:
        """
        Get available search engines.

        Returns:
            List of engine configurations
        """
        try:
            session = await self._get_session()
            async with session.get(f"{self.config.api_url}/engine_list") as response:
                response.raise_for_status()
                data = await response.json()
                return data.get("engines", [])
        except Exception as e:
            logger.error(f"Failed to get engines: {e}")
            return []

    async def get_stats(self) -> dict[str, Any]:
        """
        Get Nexus Search statistics.

        Returns:
            Statistics dictionary
        """
        try:
            session = await self._get_session()
            async with session.get(f"{self.config.api_url}/stats") as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}


# Global client instance
_client: NexusClient | None = None


def get_client(config: NexusConfig | None = None) -> NexusClient:
    """
    Get or create Nexus client instance.

    Args:
        config: Optional configuration

    Returns:
        NexusClient instance
    """
    global _client
    if _client is None or (_client and _client.session and _client.session.closed):
        _client = NexusClient(config=config)
    return _client


async def close_client() -> None:
    """Close the global client."""
    global _client
    if _client:
        await _client.close()
        _client = None
