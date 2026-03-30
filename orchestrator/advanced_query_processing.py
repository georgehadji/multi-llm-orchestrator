"""
Advanced Query Processing
==========================
Author: Georgios-Chrysovalantis Chatzivantsidis

Advanced query processing features:
- LLM-based query expansion
- Learning query classifier
- Result summarization

Usage:
    from orchestrator.advanced_query_processing import (
        LLMQueryExpander,
        LearningClassifier,
        ResultSummarizer,
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .api_clients import UnifiedClient
from .models import Model

logger = logging.getLogger("orchestrator.advanced_query")


# ─────────────────────────────────────────────
# LLM Query Expander
# ─────────────────────────────────────────────

@dataclass
class ExpandedQuery:
    """An expanded query variant."""
    original: str
    expanded: str
    expansion_type: str  # synonym, rephrase, broaden, narrow
    confidence: float = 1.0


class LLMQueryExpander:
    """
    LLM-based query expansion.

    Uses LLM to generate query variants for better search coverage.

    Expansion Types:
    - Synonym: Replace words with synonyms
    - Rephrase: Rephrase the query
    - Broaden: Make query more general
    - Narrow: Make query more specific

    Usage:
        expander = LLMQueryExpander()
        expansions = await expander.expand("python async best practices")
    """

    EXPANSION_PROMPT = """You are a search query expansion expert. Generate {num_variants} variant queries for the following search query.

Original Query: "{query}"

Generate variants using these strategies:
1. Synonym replacement (replace key terms with synonyms)
2. Rephrasing (same meaning, different wording)
3. Broadening (more general terms)
4. Narrowing (more specific terms)

Return JSON array:
[
    {{"expanded": "...", "type": "synonym"}},
    {{"expanded": "...", "type": "rephrase"}},
    ...
]
"""

    def __init__(self, client: UnifiedClient | None = None):
        """
        Initialize LLM query expander.

        Args:
            client: UnifiedClient for LLM calls
        """
        self.client = client or UnifiedClient()
        self._expansion_model = Model.GPT_4O_MINI  # Fast, cheap model

    async def expand(
        self,
        query: str,
        num_variants: int = 3,
    ) -> list[ExpandedQuery]:
        """
        Expand query using LLM.

        Args:
            query: Original query
            num_variants: Number of variants to generate

        Returns:
            List of ExpandedQuery objects
        """
        prompt = self.EXPANSION_PROMPT.format(
            query=query,
            num_variants=num_variants,
        )

        try:
            response, _ = await self.client.call(
                model=self._expansion_model,
                system_prompt="You are a search query expansion expert. Return ONLY valid JSON.",
                user_prompt=prompt,
                max_tokens=500,
                temperature=0.7,
            )

            # Parse JSON response
            import json
            data = json.loads(response.text)

            expansions = []
            for item in data:
                expansion = ExpandedQuery(
                    original=query,
                    expanded=item.get("expanded", ""),
                    expansion_type=item.get("type", "synonym"),
                    confidence=0.9,
                )
                expansions.append(expansion)

            logger.info(f"Generated {len(expansions)} query expansions")
            return expansions

        except Exception as e:
            logger.warning(f"LLM query expansion failed: {e}")
            # Return empty list on failure (fallback to original query)
            return []

    def expand_with_synonyms(
        self,
        query: str,
    ) -> list[ExpandedQuery]:
        """
        Expand query using predefined synonyms (fast, no LLM).

        Args:
            query: Original query

        Returns:
            List of ExpandedQuery objects
        """
        # Predefined synonyms for common terms
        synonyms = {
            "python": ["python3", "python programming", "python language"],
            "async": ["asynchronous", "concurrent", "non-blocking"],
            "fast": ["fastest", "high-performance", "low-latency"],
            "best": ["top", "recommended", "leading"],
            "tutorial": ["guide", "how-to", "walkthrough"],
            "example": ["sample", "code sample", "snippet"],
            "practice": ["pattern", "approach", "technique"],
        }

        expansions = []
        words = query.lower().split()

        for word in words:
            if word in synonyms:
                for synonym in synonyms[word][:2]:  # Limit to 2 synonyms per word
                    expanded = query.replace(word, synonym)
                    expansions.append(ExpandedQuery(
                        original=query,
                        expanded=expanded,
                        expansion_type="synonym",
                        confidence=0.8,
                    ))

        logger.info(f"Generated {len(expansions)} synonym expansions")
        return expansions


# ─────────────────────────────────────────────
# Learning Classifier
# ─────────────────────────────────────────────

@dataclass
class QueryClassification:
    """Result of query classification."""
    query: str
    category: str
    confidence: float
    subcategories: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class LearningClassifier:
    """
    Learning-based query classifier.

    Uses keyword matching with learning from feedback.
    Improves accuracy over time based on user feedback.

    Categories:
    - factual: Simple facts, definitions
    - research: Deep research, comparisons
    - technical: Code, technical documentation
    - academic: Academic papers, research
    - creative: Brainstorming, ideas

    Usage:
        classifier = LearningClassifier()
        result = await classifier.classify("python async best practices")
    """

    # Initial keyword patterns for each category
    CATEGORY_KEYWORDS = {
        "factual": [
            "what is", "who is", "when", "where", "define", "meaning",
            "how many", "how much", "year", "date", "born", "capital",
        ],
        "research": [
            "best practices", "guide", "tutorial", "how to", "comparison",
            "vs", "versus", "review", "overview", "trends", "patterns",
        ],
        "technical": [
            "code", "example", "api", "library", "framework", "package",
            "install", "dependency", "error", "bug", "debug", "fix",
        ],
        "academic": [
            "paper", "study", "research", "journal", "citation",
            "doi", "arxiv", "pubmed", "scholar", "thesis",
        ],
        "creative": [
            "ideas", "inspiration", "creative", "innovative", "unique",
            "brainstorm", "suggest", "recommend", "explore",
        ],
    }

    def __init__(self):
        """Initialize learning classifier."""
        self.category_keywords = dict(self.CATEGORY_KEYWORDS)
        self.feedback_history: list[dict[str, Any]] = []
        self.accuracy_history: list[float] = []
        self.total_classifications = 0

    async def classify(self, query: str) -> QueryClassification:
        """
        Classify a query.

        Args:
            query: Query to classify

        Returns:
            QueryClassification result
        """
        self.total_classifications += 1
        query_lower = query.lower()

        # Score each category
        scores = dict.fromkeys(self.category_keywords, 0)

        for category, keywords in self.category_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    scores[category] += 1

        # Get highest scoring category
        max_score = max(scores.values())
        if max_score == 0:
            # Default to research for unknown queries
            category = "research"
            confidence = 0.5
        else:
            # Get all categories with max score
            top_categories = [cat for cat, score in scores.items() if score == max_score]
            category = top_categories[0]
            confidence = min(0.9, 0.5 + (max_score * 0.1))

        # Get subcategories (second highest)
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        subcategories = [cat for cat, score in sorted_scores[1:3] if score > 0]

        result = QueryClassification(
            query=query,
            category=category,
            confidence=confidence,
            subcategories=subcategories,
            metadata={"scores": scores},
        )

        logger.debug(f"Classified query: {query[:50]}... → {category} ({confidence:.2f})")

        return result

    def record_feedback(
        self,
        query: str,
        predicted_category: str,
        actual_category: str,
    ):
        """
        Record feedback for learning.

        Args:
            query: Original query
            predicted_category: Predicted category
            actual_category: Actual (correct) category
        """
        feedback = {
            "query": query,
            "predicted": predicted_category,
            "actual": actual_category,
            "correct": predicted_category == actual_category,
        }
        self.feedback_history.append(feedback)

        # Update accuracy
        recent = self.feedback_history[-100:]  # Last 100 feedbacks
        accuracy = sum(1 for f in recent if f["correct"]) / len(recent)
        self.accuracy_history.append(accuracy)

        # Learn from mistakes
        if not feedback["correct"]:
            self._learn_from_mistake(query, predicted_category, actual_category)

        logger.info(f"Recorded feedback: {predicted_category} → {actual_category} (correct: {feedback['correct']})")

    def _learn_from_mistake(
        self,
        query: str,
        predicted: str,
        actual: str,
    ):
        """
        Learn from classification mistake.

        Args:
            query: Original query
            predicted: Predicted category
            actual: Actual category
        """
        # Extract key words from query
        words = query.lower().split()
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being"}
        key_words = [w for w in words if w not in stop_words and len(w) > 3]

        # Add key words to actual category
        if actual in self.category_keywords:
            for word in key_words[:3]:  # Add up to 3 words
                if word not in self.category_keywords[actual]:
                    self.category_keywords[actual].append(word)

        logger.debug(f"Learned from mistake: added keywords to {actual}")

    def get_accuracy(self) -> float:
        """
        Get current classification accuracy.

        Returns:
            Accuracy over last 100 feedbacks (0-1)
        """
        if not self.accuracy_history:
            return 0.0
        return self.accuracy_history[-1]

    def get_stats(self) -> dict[str, Any]:
        """
        Get classifier statistics.

        Returns:
            Dictionary with stats
        """
        recent = self.feedback_history[-100:]
        accuracy = sum(1 for f in recent if f["correct"]) / len(recent) if recent else 0.0

        return {
            "total_classifications": self.total_classifications,
            "feedback_count": len(self.feedback_history),
            "accuracy": accuracy,
            "categories": list(self.category_keywords.keys()),
        }


# ─────────────────────────────────────────────
# Result Summarizer
# ─────────────────────────────────────────────

@dataclass
class SearchResultSummary:
    """Summary of search results."""
    query: str
    summary: str
    key_findings: list[str] = field(default_factory=list)
    sources_count: int = 0
    confidence: float = 1.0


class ResultSummarizer:
    """
    Search result summarizer.

    Generates concise summaries of search results.

    Usage:
        summarizer = ResultSummarizer()
        summary = await summarizer.summarize(query, results)
    """

    SUMMARIZE_PROMPT = """Summarize the following search results for the query "{query}".

Provide:
1. A 2-3 sentence executive summary
2. 3-5 key findings (bullet points)

Search Results:
{results}

Return JSON:
{{
    "summary": "...",
    "key_findings": ["...", "...", "..."]
}}
"""

    def __init__(self, client: UnifiedClient | None = None):
        """
        Initialize result summarizer.

        Args:
            client: UnifiedClient for LLM calls
        """
        self.client = client or UnifiedClient()
        self._summary_model = Model.GPT_4O_MINI

    async def summarize(
        self,
        query: str,
        results: list[Any],  # SearchResult objects
        max_length: int = 200,
    ) -> SearchResultSummary:
        """
        Summarize search results.

        Args:
            query: Original query
            results: List of search results
            max_length: Maximum summary length

        Returns:
            SearchResultSummary
        """
        if not results:
            return SearchResultSummary(
                query=query,
                summary="No results found.",
                key_findings=[],
                sources_count=0,
            )

        # Prepare results text
        results_text = "\n\n".join([
            f"Title: {r.title}\nContent: {r.content[:200]}..."
            for r in results[:5]  # Summarize top 5 results
        ])

        prompt = self.SUMMARIZE_PROMPT.format(
            query=query,
            results=results_text,
        )

        try:
            response, _ = await self.client.call(
                model=self._summary_model,
                system_prompt="You are a research summarizer. Return ONLY valid JSON.",
                user_prompt=prompt,
                max_tokens=500,
                temperature=0.3,
            )

            # Parse JSON response
            import json
            data = json.loads(response.text)

            summary = SearchResultSummary(
                query=query,
                summary=data.get("summary", "")[:max_length],
                key_findings=data.get("key_findings", []),
                sources_count=len(results),
                confidence=0.9,
            )

            logger.info(f"Generated summary: {len(summary.key_findings)} key findings")

            return summary

        except Exception as e:
            logger.warning(f"Result summarization failed: {e}")
            # Return basic summary on failure
            return SearchResultSummary(
                query=query,
                summary=f"Found {len(results)} results for '{query}'.",
                key_findings=[r.title for r in results[:3]],
                sources_count=len(results),
                confidence=0.5,
            )


# ─────────────────────────────────────────────
# Convenience Functions
# ─────────────────────────────────────────────

async def expand_query(
    query: str,
    num_variants: int = 3,
    use_llm: bool = True,
) -> list[ExpandedQuery]:
    """
    Convenience function to expand query.

    Args:
        query: Original query
        num_variants: Number of variants
        use_llm: Use LLM (True) or synonyms (False)

    Returns:
        List of expanded queries
    """
    if use_llm:
        expander = LLMQueryExpander()
        return await expander.expand(query, num_variants)
    else:
        expander = LLMQueryExpander()
        return expander.expand_with_synonyms(query)


async def classify_query(query: str) -> QueryClassification:
    """
    Convenience function to classify query.

    Args:
        query: Query to classify

    Returns:
        QueryClassification result
    """
    classifier = LearningClassifier()
    return await classifier.classify(query)


async def summarize_results(
    query: str,
    results: list[Any],
) -> SearchResultSummary:
    """
    Convenience function to summarize results.

    Args:
        query: Original query
        results: Search results

    Returns:
        SearchResultSummary
    """
    summarizer = ResultSummarizer()
    return await summarizer.summarize(query, results)
