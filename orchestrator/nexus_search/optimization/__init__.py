"""
Nexus Search — Optimization Module
===================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Optimization modules for Nexus Search:
- Result deduplication (URL, title, semantic)
- Query caching
- Semantic reranking
- Parallel search
- Adaptive depth

Usage:
    from orchestrator.nexus_search.optimization import ResultDeduplicator
    from orchestrator.nexus_search.optimization import QueryCache
    from orchestrator.nexus_search.optimization import SemanticReranker
    from orchestrator.nexus_search.optimization import ParallelSearchExecutor
"""

from .deduplication import ResultDeduplicator, deduplicate_results
from .query_cache import QueryCache, get_query_cache
from .reranker import SemanticReranker, rerank_results
from .parallel_search import ParallelSearchExecutor, search_parallel

__all__ = [
    # Deduplication
    "ResultDeduplicator",
    "deduplicate_results",
    
    # Query Cache
    "QueryCache",
    "get_query_cache",
    
    # Semantic Reranking
    "SemanticReranker",
    "rerank_results",
    
    # Parallel Search
    "ParallelSearchExecutor",
    "search_parallel",
]

# Version
__version__ = "1.0.0"
