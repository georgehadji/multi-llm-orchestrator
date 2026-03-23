"""
Nexus Search — Base Provider Interface
=======================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Abstract base class for search providers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..models import SearchResult, SearchResults, SearchSource


class BaseSearchProvider(ABC):
    """
    Abstract base class for search providers.
    
    All search providers must implement this interface.
    """
    
    # Provider name (user-facing)
    name: str = "Base Provider"
    
    # Supported sources
    SUPPORTED_SOURCES: List[SearchSource] = []
    
    @abstractmethod
    async def search(
        self,
        query: str,
        sources: Optional[List[SearchSource]] = None,
        num_results: int = 10,
        **kwargs: Any,
    ) -> SearchResults:
        """
        Perform a search query.
        
        Args:
            query: Search query string
            sources: List of sources to search
            num_results: Maximum number of results
            **kwargs: Additional provider-specific parameters
            
        Returns:
            SearchResults with findings
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if provider is available.
        
        Returns:
            True if healthy, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_capabilities(self) -> Dict[str, Any]:
        """
        Get provider capabilities.
        
        Returns:
            Capabilities dictionary
        """
        pass
    
    def supports_source(self, source: SearchSource) -> bool:
        """
        Check if provider supports a specific source.
        
        Args:
            source: Source to check
            
        Returns:
            True if supported, False otherwise
        """
        return source in self.SUPPORTED_SOURCES
