"""
Nexus Search — Semantic Reranker
=================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Semantic reranking using sentence embeddings to improve result relevance.

Features:
- Sentence Transformer embeddings
- Cosine similarity scoring
- Lightweight model (all-MiniLM-L6-v2, 80MB)
- Fast reranking (<200ms for 10 results)

Usage:
    from orchestrator.nexus_search.optimization import SemanticReranker

    reranker = SemanticReranker()
    reranked = await reranker.rerank(query, results, top_k=10)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from orchestrator.nexus_search.models import SearchResult

logger = logging.getLogger("orchestrator.nexus_search")


class SemanticReranker:
    """
    Semantic reranker using sentence embeddings.

    Uses Sentence Transformers to encode query and documents,
    then reranks by cosine similarity.

    Model: all-MiniLM-L6-v2 (80MB, fast, good quality)

    Usage:
        reranker = SemanticReranker()
        reranked = reranker.rerank(query, results)
    """

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, model_name: str | None = None):
        """
        Initialize semantic reranker.

        Args:
            model_name: Sentence Transformer model name
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self._model = None
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy initialization of model."""
        if self._initialized:
            return

        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading semantic reranker model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            self._initialized = True
            logger.info("Semantic reranker model loaded successfully")

        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            self._initialized = False
        except Exception as e:
            logger.error(f"Failed to load semantic reranker model: {e}")
            self._initialized = False

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 10,
    ) -> list[SearchResult]:
        """
        Rerank search results by semantic similarity to query.

        Args:
            query: Search query
            results: List of search results to rerank
            top_k: Number of top results to return

        Returns:
            Reranked list of SearchResult (sorted by relevance)
        """
        if len(results) <= 1:
            return results

        # Ensure model is loaded
        self._ensure_initialized()

        if not self._initialized or self._model is None:
            logger.warning("Semantic reranker not initialized, returning original order")
            return results[:top_k]

        try:
            # Prepare documents (combine title and content)
            documents = []
            for result in results:
                doc_text = f"{result.title} {result.content}"
                documents.append(doc_text)

            # Encode query and documents
            query_embedding = self._model.encode(
                query,
                convert_to_tensor=True,
                show_progress_bar=False,
            )
            doc_embeddings = self._model.encode(
                documents,
                convert_to_tensor=True,
                show_progress_bar=False,
            )

            # Calculate cosine similarity
            from sentence_transformers.util import cos_sim

            similarities = []
            for i, doc_emb in enumerate(doc_embeddings):
                sim = cos_sim(query_embedding, doc_emb)[0][0].item()
                similarities.append((i, sim))

            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)

            # Update result scores and reorder
            reranked_results = []
            for _rank, (idx, score) in enumerate(similarities[:top_k], 1):
                result = results[idx]
                result.score = score  # Update score with semantic similarity
                reranked_results.append(result)

            logger.info(
                f"Semantic reranking complete: {len(results)} → {len(reranked_results)} results, "
                f"top score: {reranked_results[0].score:.3f}"
            )

            return reranked_results

        except Exception as e:
            logger.error(f"Semantic reranking failed: {e}")
            # Return original results on error
            return results[:top_k]


def rerank_results(
    query: str,
    results: list[SearchResult],
    top_k: int = 10,
    model_name: str | None = None,
) -> list[SearchResult]:
    """
    Convenience function to rerank search results.

    Args:
        query: Search query
        results: List of search results
        top_k: Number of results to return
        model_name: Optional model name

    Returns:
        Reranked results
    """
    reranker = SemanticReranker(model_name=model_name)
    return reranker.rerank(query, results, top_k)
