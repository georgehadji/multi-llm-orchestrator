"""
Nexus Search — Query Classifier
================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Classifies queries to determine optimal search strategy.
"""

from __future__ import annotations

import re

from orchestrator.nexus_search.models import QueryType, SearchSource


class QueryClassifier:
    """
    Classifies search queries to determine optimal strategy.

    Query Types:
    - FACTUAL: Simple facts → direct search
    - RESEARCH: Deep research → multi-step
    - TECHNICAL: Code/tech → tech sources
    - ACADEMIC: Academic → scholar/arxiv
    - CREATIVE: Creative → broad search

    Usage:
        classifier = QueryClassifier()
        query_type = await classifier.classify("Python async best practices")
    """

    # Keywords for each query type
    KEYWORDS = {
        QueryType.FACTUAL: [
            "what is", "who is", "when", "where", "define", "meaning",
            "how many", "how much", "year", "date", "born", "capital",
        ],
        QueryType.RESEARCH: [
            "best practices", "guide", "tutorial", "how to", "comparison",
            "vs", "versus", "review", "overview", "trends", "patterns",
            "architecture", "design", "implementation", "strategy",
        ],
        QueryType.TECHNICAL: [
            "code", "example", "api", "library", "framework", "package",
            "install", "dependency", "error", "bug", "debug", "fix",
            "github", "repository", "npm", "pip", "docker", "kubernetes",
        ],
        QueryType.ACADEMIC: [
            "paper", "study", "research", "journal", "citation",
            "doi", "arxiv", "pubmed", "scholar", "thesis", "dissertation",
        ],
        QueryType.CREATIVE: [
            "ideas", "inspiration", "creative", "innovative", "unique",
            "brainstorm", "suggest", "recommend", "explore",
        ],
    }

    # Source mappings for each query type
    SOURCE_MAPPING = {
        QueryType.FACTUAL: [SearchSource.WEB],
        QueryType.RESEARCH: [SearchSource.WEB, SearchSource.TECH, SearchSource.NEWS],
        QueryType.TECHNICAL: [SearchSource.TECH, SearchSource.CODE, SearchSource.WEB],
        QueryType.ACADEMIC: [SearchSource.ACADEMIC, SearchSource.WEB],
        QueryType.CREATIVE: [SearchSource.WEB, SearchSource.NEWS],
    }

    def __init__(self):
        """Initialize classifier."""
        self._compiled_patterns: dict[QueryType, list[re.Pattern]] = {}
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile regex patterns for efficiency."""
        for query_type, keywords in self.KEYWORDS.items():
            patterns = []
            for keyword in keywords:
                # Create case-insensitive pattern
                pattern = re.compile(rf"\b{re.escape(keyword)}\b", re.IGNORECASE)
                patterns.append(pattern)
            self._compiled_patterns[query_type] = patterns

    async def classify(self, query: str) -> QueryType:
        """
        Classify a query.

        Args:
            query: Search query string

        Returns:
            QueryType enum value
        """
        query_lower = query.lower()

        # Score each query type
        scores = dict.fromkeys(QueryType, 0)

        for query_type, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(query_lower):
                    scores[query_type] += 1

        # Return highest scoring type
        max_score = max(scores.values())
        if max_score == 0:
            # Default to RESEARCH for unknown queries
            return QueryType.RESEARCH

        # Get all types with max score
        top_types = [qt for qt, score in scores.items() if score == max_score]

        # Tie-breaking: prefer more specific types
        priority = [
            QueryType.ACADEMIC,
            QueryType.TECHNICAL,
            QueryType.FACTUAL,
            QueryType.RESEARCH,
            QueryType.CREATIVE,
        ]

        for qt in priority:
            if qt in top_types:
                return qt

        return top_types[0]

    def get_recommended_sources(self, query_type: QueryType) -> list[SearchSource]:
        """
        Get recommended sources for query type.

        Args:
            query_type: Classified query type

        Returns:
            List of recommended SearchSource
        """
        return self.SOURCE_MAPPING.get(query_type, [SearchSource.WEB])

    async def classify_and_get_sources(
        self,
        query: str,
    ) -> tuple[QueryType, list[SearchSource]]:
        """
        Classify query and get recommended sources.

        Args:
            query: Search query

        Returns:
            Tuple of (QueryType, List[SearchSource])
        """
        query_type = await self.classify(query)
        sources = self.get_recommended_sources(query_type)
        return query_type, sources


# Global classifier instance
_classifier: QueryClassifier | None = None


def get_classifier() -> QueryClassifier:
    """Get or create classifier instance."""
    global _classifier
    if _classifier is None:
        _classifier = QueryClassifier()
    return _classifier
