"""
Nexus Search — Result Deduplication
====================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Multi-level result deduplication:
1. URL deduplication (exact match)
2. Title hash deduplication (fuzzy)
3. Content similarity (TF-IDF semantic)

Usage:
    from orchestrator.nexus_search.optimization import ResultDeduplicator

    deduplicator = ResultDeduplicator(similarity_threshold=0.85)
    unique_results = deduplicator.deduplicate(search_results)
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from orchestrator.nexus_search.models import SearchResult

logger = logging.getLogger("orchestrator.nexus_search")


class ResultDeduplicator:
    """
    Multi-level result deduplication.

    Levels:
    1. URL deduplication — Exact URL match
    2. Title hash — Near-duplicate titles using MD5 hashing
    3. Content similarity — TF-IDF + cosine similarity for semantic dedup

    Usage:
        deduplicator = ResultDeduplicator(similarity_threshold=0.85)
        unique_results = deduplicator.deduplicate(results)
    """

    def __init__(self, similarity_threshold: float = 0.85):
        """
        Initialize deduplicator.

        Args:
            similarity_threshold: TF-IDF similarity threshold (0-1)
                Higher = more aggressive deduplication
                Lower = more results kept
        """
        self.similarity_threshold = similarity_threshold
        self._tfidf_vectorizer = None

    def deduplicate(self, results: list[SearchResult]) -> list[SearchResult]:
        """
        Remove duplicate results using multi-level deduplication.

        Args:
            results: List of search results (may contain duplicates)

        Returns:
            List of unique results
        """
        if len(results) <= 1:
            return results

        original_count = len(results)

        # Level 1: URL deduplication
        results = self._dedup_by_url(results)
        logger.debug(f"After URL dedup: {len(results)}/{original_count}")

        # Level 2: Title hash deduplication
        results = self._dedup_by_title_hash(results)
        logger.debug(f"After title dedup: {len(results)}/{original_count}")

        # Level 3: Content similarity (only if >5 results)
        if len(results) > 5:
            results = self._dedup_by_content_similarity(results)
            logger.debug(f"After content dedup: {len(results)}/{original_count}")

        duplicate_rate = (
            (original_count - len(results)) / original_count if original_count > 0 else 0
        )
        logger.info(
            f"Deduplication complete: {original_count} → {len(results)} ({duplicate_rate:.1%} removed)"
        )

        return results

    def _dedup_by_url(self, results: list[SearchResult]) -> list[SearchResult]:
        """
        Remove exact URL duplicates.

        Normalizes URLs by:
        - Converting to lowercase
        - Removing trailing slashes
        - Removing www. prefix
        """
        seen_urls = set()
        unique = []

        for result in results:
            # Normalize URL
            url_normalized = result.url.lower().strip()
            url_normalized = url_normalized.rstrip("/")
            url_normalized = url_normalized.replace("www.", "")

            if url_normalized not in seen_urls:
                seen_urls.add(url_normalized)
                unique.append(result)

        return unique

    def _dedup_by_title_hash(self, results: list[SearchResult]) -> list[SearchResult]:
        """
        Remove near-duplicate titles using MD5 hashing.

        Normalizes titles by:
        - Converting to lowercase
        - Stripping whitespace
        - Removing common prefixes/suffixes
        """
        seen_hashes = set()
        unique = []

        for result in results:
            # Normalize title
            title_normalized = result.title.lower().strip()

            # Remove common prefixes
            for prefix in ["review: ", "article: ", "post: ", "guide: "]:
                if title_normalized.startswith(prefix):
                    title_normalized = title_normalized[len(prefix) :]

            # Create hash
            title_hash = hashlib.md5(title_normalized.encode("utf-8")).hexdigest()

            if title_hash not in seen_hashes:
                seen_hashes.add(title_hash)
                unique.append(result)

        return unique

    def _dedup_by_content_similarity(self, results: list[SearchResult]) -> list[SearchResult]:
        """
        Remove semantically similar content using TF-IDF + cosine similarity.

        Args:
            results: List of search results

        Returns:
            List with semantically similar results removed
        """
        if len(results) <= 1:
            return results

        try:
            # Try to import sklearn
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError:
            logger.warning("scikit-learn not installed, skipping semantic dedup")
            return results

        # Extract content for comparison
        contents = []
        for result in results:
            # Combine title and content for better similarity detection
            text = f"{result.title} {result.content}"
            contents.append(text)

        # TF-IDF vectorization
        try:
            vectorizer = TfidfVectorizer(
                stop_words="english",
                max_features=1000,
                ngram_range=(1, 2),
                min_df=1,
            )
            tfidf_matrix = vectorizer.fit_transform(contents)
        except Exception as e:
            logger.warning(f"TF-IDF vectorization failed: {e}")
            return results

        # Calculate similarity and keep unique results
        keep_indices = [0]  # Always keep first result

        for i in range(1, len(results)):
            # Compare with all kept results
            max_similarity = 0.0
            for j in keep_indices:
                similarity = cosine_similarity(tfidf_matrix[i], tfidf_matrix[j])[0][0]
                max_similarity = max(max_similarity, similarity)

            # Keep if below threshold
            if max_similarity < self.similarity_threshold:
                keep_indices.append(i)

        return [results[i] for i in keep_indices]


def deduplicate_results(
    results: list[SearchResult],
    similarity_threshold: float = 0.85,
) -> list[SearchResult]:
    """
    Convenience function to deduplicate search results.

    Args:
        results: List of search results
        similarity_threshold: TF-IDF similarity threshold (0.85 default)

    Returns:
        Deduplicated list of results
    """
    deduplicator = ResultDeduplicator(similarity_threshold=similarity_threshold)
    return deduplicator.deduplicate(results)
