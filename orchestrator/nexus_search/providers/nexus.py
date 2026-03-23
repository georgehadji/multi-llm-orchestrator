"""
Nexus Search — Nexus Provider
==============================
Author: Georgios-Chrysovalantis Chatzivantsidis

Nexus Search provider implementation (SearXNG backend).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import BaseSearchProvider
from ..models import SearchResult, SearchResults, SearchSource
from ..nexus_client import get_client
from ..config import get_config


class NexusProvider(BaseSearchProvider):
    """
    Nexus Search provider.
    
    Provides unified search across multiple sources using
    self-hosted search infrastructure (SearXNG backend).
    
    Supported Sources:
    - web: General web search
    - academic: Scholar, arXiv, PubMed
    - tech: GitHub, Stack Overflow, documentation
    - news: News sources
    - code: Code repositories
    
    Usage:
        provider = NexusProvider()
        results = await provider.search("Python async")
    """
    
    name = "Nexus Search"
    
    SUPPORTED_SOURCES = [
        SearchSource.WEB,
        SearchSource.ACADEMIC,
        SearchSource.TECH,
        SearchSource.NEWS,
        SearchSource.CODE,
    ]
    
    # Map SearchSource to SearXNG categories
    SOURCE_MAPPING = {
        SearchSource.WEB: ["general", "web"],
        SearchSource.ACADEMIC: ["science", "scholar", "pubmed", "arxiv"],
        SearchSource.TECH: ["it"],
        SearchSource.NEWS: ["news"],
        SearchSource.CODE: ["github", "stackoverflow"],
    }
    
    def __init__(self):
        """Initialize Nexus provider."""
        self.config = get_config()
        self._client = None
    
    @property
    def client(self):
        """Get or create Nexus client."""
        if self._client is None:
            from ..nexus_client import get_client
            self._client = get_client(self.config)
        return self._client
    
    async def search(
        self,
        query: str,
        sources: Optional[List[SearchSource]] = None,
        num_results: int = 10,
        language: str = "en",
        time_range: Optional[str] = None,
        **kwargs: Any,
    ) -> SearchResults:
        """
        Search using Nexus.
        
        Args:
            query: Search query
            sources: Sources to search (default: all)
            num_results: Maximum results
            language: Language code
            time_range: Time range filter
            **kwargs: Additional parameters
            
        Returns:
            SearchResults
        """
        # Default to all sources if none specified
        if sources is None:
            sources = self.SUPPORTED_SOURCES.copy()
        
        # Map sources to categories
        categories = []
        for source in sources:
            if source in self.SOURCE_MAPPING:
                categories.extend(self.SOURCE_MAPPING[source])
        
        # Remove duplicates
        categories = list(set(categories))
        
        # Perform search
        results = await self.client.search(
            query=query,
            categories=categories,
            num_results=num_results,
            language=language,
            time_range=time_range,
        )
        
        return results
    
    async def health_check(self) -> bool:
        """Check if Nexus is available."""
        return await self.client.health_check()
    
    async def get_capabilities(self) -> Dict[str, Any]:
        """Get provider capabilities."""
        return {
            "name": self.name,
            "sources": [s.value for s in self.SUPPORTED_SOURCES],
            "max_results": self.config.max_results,
            "rate_limit": self.config.rate_limit,
            "cache_enabled": self.config.cache_enabled,
            "multi_language": True,
            "time_range_filter": True,
        }
    
    async def get_available_sources(self) -> List[SearchSource]:
        """
        Get available sources from Nexus.
        
        Returns:
            List of available sources
        """
        engines = await self.client.get_engines()
        
        # Map engines to sources
        available = set()
        for engine in engines:
            if not engine.get("enabled", True):
                continue
                
            categories = engine.get("categories", [])
            for category in categories:
                source = self._engine_category_to_source(category)
                if source:
                    available.add(source)
        
        return list(available)
    
    def _engine_category_to_source(self, category: str) -> Optional[SearchSource]:
        """Map engine category to SearchSource."""
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
        return mapping.get(category)
