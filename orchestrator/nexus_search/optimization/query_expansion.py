"""
Nexus Search — Query Expansion
===============================
Author: Georgios-Chrysovalantis Chatzivantsidis

Query expansion with synonyms and related terms to improve recall.

Features:
- Domain-specific synonym expansion
- LLM-based query variant generation
- Configurable expansion limits
- Fallback to original query if expansion fails

Usage:
    from orchestrator.nexus_search.optimization import QueryExpander

    expander = QueryExpander()
    expanded = expander.expand("Python async best practices")
    # Returns: ["Python async best practices", "Python asynchronous best practices", ...]
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from orchestrator.models import Model

logger = logging.getLogger("orchestrator.nexus_search")


class QueryExpander:
    """
    Expand queries with synonyms and related terms.

    Uses a combination of:
    1. Predefined synonym dictionary (fast, free)
    2. LLM-based expansion (higher quality, costs tokens)

    Usage:
        expander = QueryExpander()
        expanded = expander.expand("Python async best practices")
    """

    # Domain-specific synonyms for common technical terms
    SYNONYMS = {
        # Programming languages
        "python": ["python3", "python programming", "python language"],
        "javascript": ["js", "ecmascript", "nodejs"],
        "typescript": ["ts", "typescript programming"],
        "java": ["java programming", "jdk"],
        "rust": ["rustlang", "rust programming"],
        "go": ["golang", "go programming"],
        # Async/concurrency
        "async": ["asynchronous", "concurrent", "non-blocking"],
        "synchronous": ["sync", "blocking", "sequential"],
        "parallel": ["concurrent", "multi-threaded", "distributed"],
        "threading": ["multi-threading", "concurrent execution"],
        "multiprocessing": ["multi-processing", "parallel processing"],
        # Quality descriptors
        "fast": ["fastest", "high-performance", "low-latency", "efficient"],
        "slow": ["slowest", "low-performance", "high-latency"],
        "best": ["top", "recommended", "leading", "optimal"],
        "worst": ["worst", "poor", "suboptimal"],
        "good": ["quality", "reliable", "solid"],
        "bad": ["poor", "unreliable", "problematic"],
        # Learning resources
        "tutorial": ["guide", "how-to", "walkthrough", "lesson"],
        "example": ["sample", "code sample", "snippet", "demo"],
        "documentation": ["docs", "reference", "manual"],
        "course": ["class", "training", "bootcamp"],
        # Development
        "development": ["dev", "programming", "coding", "building"],
        "debugging": ["debug", "troubleshooting", "fixing"],
        "testing": ["test", "unit test", "integration test"],
        "deployment": ["deploy", "release", "production"],
        # Architecture
        "architecture": ["design", "pattern", "structure"],
        "microservices": ["micro-services", "service-oriented", "soa"],
        "monolithic": ["monolith", "single-unit"],
        "serverless": ["function-as-a-service", "faas", "cloud functions"],
        # Database
        "database": ["db", "datastore", "storage"],
        "sql": ["relational", "rdbms"],
        "nosql": ["non-relational", "document-store"],
        # Web
        "api": ["rest", "restful", "web service", "endpoint"],
        "frontend": ["client-side", "ui", "user interface"],
        "backend": ["server-side", "server"],
        "web": ["website", "web application", "online"],
        # Security
        "security": ["secure", "safe", "protection"],
        "authentication": ["auth", "login", "sign-in"],
        "authorization": ["authz", "permissions", "access control"],
        # Common actions
        "create": ["build", "make", "generate", "implement"],
        "use": ["using", "utilize", "employ", "leverage"],
        "learn": ["understand", "master", "study"],
        "find": ["discover", "locate", "search"],
    }

    # Query templates for different query types
    QUERY_TEMPLATES = {
        "how_to": ["how to {query}", "{query} tutorial", "{query} guide"],
        "what_is": ["what is {query}", "{query} explanation", "{query} definition"],
        "best_practices": [
            "{query} best practices",
            "{query} recommended approach",
            "{query} patterns",
        ],
        "comparison": ["{query} comparison", "{query} vs alternatives", "{query} alternatives"],
        "example": ["{query} example", "{query} code sample", "{query} implementation"],
    }

    def __init__(
        self,
        max_expansions: int = 3,
        use_llm: bool = False,
        llm_model: Model | None = None,
    ):
        """
        Initialize query expander.

        Args:
            max_expansions: Maximum number of expanded queries to return
            use_llm: Use LLM for higher quality expansion (default: False)
            llm_model: LLM model to use for expansion (if use_llm=True)
        """
        self.max_expansions = max_expansions
        self.use_llm = use_llm
        self.llm_model = llm_model
        self._expansion_count = 0

    def expand(self, query: str) -> list[str]:
        """
        Expand query with synonyms.

        Args:
            query: Original search query

        Returns:
            List of expanded queries (includes original as first item)
        """
        expansions = [query]  # Original query always included

        # Detect query type for template-based expansion
        query_type = self._detect_query_type(query)

        # Apply template-based expansion
        if query_type in self.QUERY_TEMPLATES:
            for template in self.QUERY_TEMPLATES[query_type][:2]:
                expanded = template.format(query=query)
                if expanded != query and len(expansions) < self.max_expansions:
                    expansions.append(expanded)

        # Apply synonym-based expansion
        words = query.lower().split()

        for word in words:
            if word in self.SYNONYMS:
                for synonym in self.SYNONYMS[word][:2]:  # Limit synonyms per word
                    expanded = query.replace(word, synonym)
                    if expanded != query and len(expansions) < self.max_expansions:
                        expansions.append(expanded)
                        logger.debug(
                            f"Expanded '{query}' → '{expanded}' (synonym: {word} → {synonym})"
                        )

                    if len(expansions) >= self.max_expansions:
                        break

            if len(expansions) >= self.max_expansions:
                break

        self._expansion_count += 1
        logger.info(f"Query expansion: '{query}' → {len(expansions)} variants")

        return expansions[: self.max_expansions]

    def _detect_query_type(self, query: str) -> str | None:
        """
        Detect query type for template selection.

        Args:
            query: Search query

        Returns:
            Query type or None if not detected
        """
        query_lower = query.lower()

        # How-to queries
        if query_lower.startswith(("how to", "how do i", "how can i")):
            return "how_to"

        # What-is queries
        if query_lower.startswith(("what is", "what are", "define", "definition of")):
            return "what_is"

        # Best practices
        if "best practice" in query_lower or "best way" in query_lower:
            return "best_practices"

        # Comparison queries
        if any(
            word in query_lower for word in ["vs", "versus", "comparison", "compare", "alternative"]
        ):
            return "comparison"

        # Example queries
        if any(word in query_lower for word in ["example", "sample", "snippet", "demo"]):
            return "example"

        return None

    async def expand_with_llm(self, query: str, num_variants: int = 3) -> list[str]:
        """
        Use LLM to generate query variants.

        More expensive but higher quality expansions.

        Args:
            query: Original search query
            num_variants: Number of variants to generate

        Returns:
            List of expanded queries
        """
        if not self.use_llm or not self.llm_model:
            logger.warning("LLM expansion not configured, falling back to synonym expansion")
            return self.expand(query)

        try:
            from orchestrator.api_clients import get_client

            client = get_client()

            prompt = f"""Generate {num_variants} alternative search queries for: "{query}"

Requirements:
1. Keep the same intent and meaning
2. Use different wording and synonyms
3. Vary the query structure
4. Each variant should be 5-15 words
5. Return ONLY the queries, one per line, no numbering

Example for "Python async best practices":
- asyncio best practices python
- python asynchronous programming patterns
- concurrent python design patterns
"""

            response = await client.call(
                model=self.llm_model,
                prompt=prompt,
                max_tokens=150,
                temperature=0.7,
            )

            # Parse response
            variants = [line.strip() for line in response.text.split("\n") if line.strip()]

            # Filter out empty lines and the original query
            variants = [v for v in variants if v and v.lower() != query.lower()]

            # Include original query as first item
            expansions = [query] + variants[:num_variants]

            logger.info(f"LLM query expansion: '{query}' → {len(expansions)} variants")

            return expansions[: self.max_expansions]

        except Exception as e:
            logger.error(f"LLM query expansion failed: {e}, falling back to synonym expansion")
            return self.expand(query)

    def get_stats(self) -> dict:
        """Get expansion statistics."""
        return {
            "total_expansions": self._expansion_count,
            "max_expansions_per_query": self.max_expansions,
            "llm_enabled": self.use_llm,
        }


# Global instance
_expander: QueryExpander | None = None


def get_expander(use_llm: bool = False, llm_model=None) -> QueryExpander:
    """
    Get or create QueryExpander instance.

    Args:
        use_llm: Use LLM for expansion
        llm_model: LLM model for expansion

    Returns:
        QueryExpander instance
    """
    global _expander
    if _expander is None:
        _expander = QueryExpander(use_llm=use_llm, llm_model=llm_model)
    return _expander
