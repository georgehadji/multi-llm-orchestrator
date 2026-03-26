"""
Nexus Search — Core Search Orchestrator
========================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Main entry point for Nexus Search operations.

Optimizations:
- Result deduplication (URL, title, semantic)
- Query caching (TTL-based)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from .models import (
    OptimizationMode,
    ResearchReport,
    SearchResults,
    SearchSource,
)
from .providers.nexus import NexusProvider
from .agents.classifier import QueryClassifier, QueryType, get_classifier
from .agents.researcher import ResearchAgent, get_research_agent
from .config import get_config

# Import optimizations
try:
    from .optimization.deduplication import deduplicate_results
    from .optimization.query_cache import get_query_cache
    OPTIMIZATIONS_ENABLED = True
except ImportError:
    OPTIMIZATIONS_ENABLED = False
    deduplicate_results = None
    get_query_cache = None

logger = logging.getLogger("orchestrator.nexus_search")


class NexusSearchOrchestrator:
    """
    Main orchestrator for Nexus Search operations.
    
    Provides unified interface for:
    - Simple search
    - Deep research
    - Query classification
    
    Usage:
        nexus = NexusSearchOrchestrator()
        
        # Simple search
        results = await nexus.search("Python async")
        
        # Deep research
        report = await nexus.research("Microservices patterns")
    """
    
    def __init__(
        self,
        provider: Optional[NexusProvider] = None,
        auto_classify: bool = True,
    ):
        """
        Initialize Nexus Search orchestrator.
        
        Args:
            provider: Search provider (creates default if None)
            auto_classify: Automatically classify queries
        """
        self.config = get_config()
        self.provider = provider or NexusProvider()
        self.classifier = get_classifier() if auto_classify else None
        self._research_agent: Optional[ResearchAgent] = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize Nexus Search."""
        if self._initialized:
            return
        
        # Check health
        healthy = await self.provider.health_check()
        if not healthy:
            logger.warning("Nexus Search is not available")
        else:
            logger.info("Nexus Search initialized successfully")
        
        self._initialized = True
    
    async def search(
        self,
        query: str,
        sources: Optional[List[SearchSource]] = None,
        optimization: OptimizationMode = OptimizationMode.BALANCED,
        num_results: int = 10,
    ) -> SearchResults:
        """
        Perform a search query with optimizations.

        Optimizations:
        - Query caching (TTL-based)
        - Result deduplication (URL, title, semantic)

        Args:
            query: Search query
            sources: Sources to search (auto-detected if None)
            optimization: Optimization mode
            num_results: Maximum results

        Returns:
            SearchResults (deduplicated)
        """
        # Initialize if needed
        if not self._initialized:
            await self.initialize()

        # Auto-classify to get optimal sources
        if self.classifier and sources is None:
            query_type = await self.classifier.classify(query)
            sources = self.classifier.get_recommended_sources(query_type)
            logger.debug(f"Classified query as {query_type.value}, using sources: {[s.value for s in sources]}")

        # Adjust num_results based on optimization mode
        if optimization == OptimizationMode.SPEED:
            num_results = min(num_results, 5)
        elif optimization == OptimizationMode.QUALITY:
            num_results = min(num_results, 20)

        # Try cache first (if optimizations enabled)
        if OPTIMIZATIONS_ENABLED and get_query_cache:
            cache = get_query_cache()
            cached_results = await cache.get(query, sources or [SearchSource.WEB])
            if cached_results is not None:
                logger.info(f"Cache hit for query: {query[:50]}...")
                return cached_results

        # Perform search
        results = await self.provider.search(
            query=query,
            sources=sources,
            num_results=num_results,
        )

        # Deduplicate results (if optimizations enabled)
        if OPTIMIZATIONS_ENABLED and deduplicate_results:
            original_count = len(results.results)
            results.results = deduplicate_results(results.results)
            results.total_results = len(results.results)
            logger.info(f"Deduplication: {original_count} → {results.total_results} results")

        # Cache results (if optimizations enabled)
        if OPTIMIZATIONS_ENABLED and get_query_cache:
            cache = get_query_cache()
            await cache.set(query, sources or [SearchSource.WEB], results)

        return results
    
    async def research(
        self,
        query: str,
        depth: int = 3,
        sources: Optional[List[SearchSource]] = None,
    ) -> ResearchReport:
        """
        Conduct deep research.
        
        Args:
            query: Research query
            depth: Number of iterations (1-5)
            sources: Sources to search (auto-detected if None)
            
        Returns:
            ResearchReport
        """
        # Initialize if needed
        if not self._initialized:
            await self.initialize()
        
        # Get research agent
        if self._research_agent is None:
            self._research_agent = get_research_agent()
        
        # Conduct research
        report = await self._research_agent.research(
            query=query,
            depth=depth,
            sources=sources,
        )
        
        logger.info(
            f"Research complete: {query[:50]}... → "
            f"{len(report.findings)} findings from {report.source_count} sources"
        )
        
        return report
    
    async def classify(self, query: str) -> QueryType:
        """
        Classify a query.
        
        Args:
            query: Search query
            
        Returns:
            QueryType enum value
        """
        if not self.classifier:
            return QueryType.RESEARCH
        
        return await self.classifier.classify(query)
    
    async def get_status(self) -> Dict[str, Any]:
        """
        Get Nexus Search status.
        
        Returns:
            Status dictionary
        """
        healthy = await self.provider.health_check()
        capabilities = await self.provider.get_capabilities()
        
        return {
            "enabled": self.config.enabled,
            "healthy": healthy,
            "api_url": self.config.api_url,
            "capabilities": capabilities,
        }
    
    async def close(self) -> None:
        """Close Nexus Search."""
        if self._research_agent and hasattr(self._research_agent.provider, 'client'):
            await self._research_agent.provider.client.close()
        if hasattr(self.provider, 'client'):
            await self.provider.client.close()
        self._initialized = False


# Global orchestrator instance
_orchestrator: Optional[NexusSearchOrchestrator] = None


def get_nexus_orchestrator() -> NexusSearchOrchestrator:
    """
    Get or create Nexus Search orchestrator instance.
    
    Returns:
        NexusSearchOrchestrator instance
    """
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = NexusSearchOrchestrator()
    return _orchestrator


# Convenience functions for direct usage

async def search(
    query: str,
    sources: Optional[List[SearchSource]] = None,
    optimization: OptimizationMode = OptimizationMode.BALANCED,
    num_results: int = 10,
) -> SearchResults:
    """
    Search the web using Nexus.
    
    Args:
        query: Search query
        sources: Sources to search
        optimization: Optimization mode
        num_results: Maximum results
        
    Returns:
        SearchResults
    """
    orchestrator = get_nexus_orchestrator()
    return await orchestrator.search(
        query=query,
        sources=sources,
        optimization=optimization,
        num_results=num_results,
    )


async def research(
    query: str,
    depth: int = 3,
    sources: Optional[List[SearchSource]] = None,
) -> ResearchReport:
    """
    Conduct deep research.
    
    Args:
        query: Research query
        depth: Number of iterations
        sources: Sources to search
        
    Returns:
        ResearchReport
    """
    orchestrator = get_nexus_orchestrator()
    return await orchestrator.research(
        query=query,
        depth=depth,
        sources=sources,
    )


async def classify(query: str) -> QueryType:
    """
    Classify a search query.
    
    Args:
        query: Search query
        
    Returns:
        QueryType
    """
    orchestrator = get_nexus_orchestrator()
    return await orchestrator.classify(query)


async def close_nexus() -> None:
    """Close Nexus Search connections."""
    global _orchestrator
    if _orchestrator:
        await _orchestrator.close()
