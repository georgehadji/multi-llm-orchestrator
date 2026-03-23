"""
Nexus Search — Web Search Integration for AI Orchestrator
==========================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Provides unified web search capabilities powered by self-hosted search infrastructure.
All searches are private, tracked-free, and integrated directly into the orchestrator.

Key Features:
- Multi-source search (web, academic, tech, news, code)
- Query classification for optimal search strategy
- Multi-step research agent for deep analysis
- Zero tracking, zero profiling
- Self-hosted (no third-party API costs)

Usage:
    from orchestrator.nexus_search import search, research
    
    # Simple search
    results = await search("Python async best practices")
    
    # Deep research
    report = await research("Microservices architecture patterns 2026")
"""

from .core import NexusSearchOrchestrator, get_nexus_orchestrator
from .models import (
    SearchResult,
    SearchResults,
    SearchSource,
    SearchQuery,
    ResearchReport,
    Finding,
    QueryType,
    OptimizationMode,
)
from .config import NexusConfig, get_config

__all__ = [
    # Main interface
    "NexusSearchOrchestrator",
    "get_nexus_orchestrator",
    
    # Models
    "SearchResult",
    "SearchResults",
    "SearchSource",
    "SearchQuery",
    "ResearchReport",
    "Finding",
    "QueryType",
    "OptimizationMode",
    
    # Configuration
    "NexusConfig",
    "get_config",
]

# Version
__version__ = "1.0.0"
