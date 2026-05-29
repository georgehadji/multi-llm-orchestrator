"""
BM25Search — Backward-compatibility shim
==========================================
Canonical location: orchestrator/infrastructure/bm25_search.py
Import from there directly for new code; this shim exists for existing callers.
"""
from .infrastructure.bm25_search import (  # noqa: F401
    BM25Search,
    SearchDocument,
    SearchResult,
    get_bm25_search,
)

__all__ = ["BM25Search", "SearchDocument", "SearchResult", "get_bm25_search"]
