"""
Nexus Search — Research Agent
==============================
Author: Georgios-Chrysovalantis Chatzivantsidis

Multi-step research agent for deep analysis.
"""

from __future__ import annotations

import time

from orchestrator.nexus_search.models import (
    Finding,
    ResearchReport,
    SearchResult,
    SearchResults,
    SearchSource,
)
from orchestrator.nexus_search.providers.nexus import NexusProvider

from .classifier import get_classifier


class ResearchAgent:
    """
    Multi-step research agent.

    Conducts deep research with multiple iterations:
    1. Initial search
    2. Follow-up searches based on findings
    3. Synthesis of results

    Usage:
        agent = ResearchAgent()
        report = await agent.research("Microservices patterns 2026")
    """

    def __init__(self, provider: NexusProvider | None = None):
        """
        Initialize research agent.

        Args:
            provider: Search provider (creates default if None)
        """
        self.provider = provider or NexusProvider()
        self.classifier = get_classifier()
        self._max_iterations = 3
        self._results_per_iteration = 10

    async def research(
        self,
        query: str,
        depth: int = 3,
        sources: list[SearchSource] | None = None,
    ) -> ResearchReport:
        """
        Conduct deep research.

        Args:
            query: Research query
            depth: Number of iterations (1-5)
            sources: Sources to search (auto-detected if None)

        Returns:
            ResearchReport with findings and summary
        """
        start_time = time.time()

        # Limit depth
        depth = max(1, min(depth, self._max_iterations))

        # Classify query to get optimal sources
        query_type = await self.classifier.classify(query)
        if sources is None:
            sources = self.classifier.get_recommended_sources(query_type)

        # Iterative research
        all_results: list[SearchResults] = []
        follow_up_queries: list[str] = []

        for iteration in range(depth):
            # Build query for this iteration
            if iteration == 0:
                current_query = query
            elif follow_up_queries and iteration <= len(follow_up_queries):
                current_query = follow_up_queries[iteration - 1]
            else:
                break

            # Perform search
            results = await self.provider.search(
                query=current_query,
                sources=sources,
                num_results=self._results_per_iteration,
            )
            all_results.append(results)

            # Generate follow-up queries (if not last iteration)
            if iteration < depth - 1 and results.results:
                follow_ups = await self._generate_follow_up_queries(
                    query, results, iteration
                )
                follow_up_queries.extend(follow_ups)

        # Synthesize findings
        findings = await self._synthesize_findings(all_results)

        # Generate summary
        summary = await self._generate_summary(query, findings)

        # Collect all unique sources
        all_sources = self._collect_sources(all_results)

        total_time = time.time() - start_time

        return ResearchReport(
            query=query,
            findings=findings,
            summary=summary,
            sources=all_sources,
            search_iterations=len(all_results),
            total_time=total_time,
        )

    async def _generate_follow_up_queries(
        self,
        original_query: str,
        results: SearchResults,
        iteration: int,
    ) -> list[str]:
        """
        Generate follow-up queries based on results.

        Args:
            original_query: Original research query
            results: Search results from previous iteration
            iteration: Current iteration number

        Returns:
            List of follow-up queries
        """
        follow_ups = []

        # Extract key topics from top results
        top_results = results.top[:5]

        # Simple approach: use result titles for follow-up
        for result in top_results[:2]:
            # Create follow-up from title
            if result.title and len(result.title) > 10:
                follow_up = f"{original_query} {result.title.split(':')[0]}"
                follow_ups.append(follow_up)

        return follow_ups[:2]  # Limit to 2 follow-ups

    async def _synthesize_findings(
        self,
        results: list[SearchResults],
    ) -> list[Finding]:
        """
        Synthesize findings from all results.

        Args:
            results: List of search results from all iterations

        Returns:
            List of synthesized findings
        """
        findings = []

        # Group results by source
        by_source: dict[str, list[SearchResult]] = {}
        for search_results in results:
            for result in search_results.results:
                source_key = result.source.value
                if source_key not in by_source:
                    by_source[source_key] = []
                by_source[source_key].append(result)

        # Create findings from top results per source
        for source, source_results in by_source.items():
            # Take top 3 results per source
            top = source_results[:3]

            for result in top:
                finding = Finding(
                    content=result.content,
                    sources=[result],
                    confidence=0.8,  # Base confidence
                    category=source,
                )
                findings.append(finding)

        return findings[:10]  # Limit total findings

    async def _generate_summary(
        self,
        query: str,
        findings: list[Finding],
    ) -> str:
        """
        Generate executive summary.

        Args:
            query: Original query
            findings: List of findings

        Returns:
            Summary text
        """
        if not findings:
            return "No findings available."

        # Simple summary: concatenate top findings
        top_findings = findings[:5]
        summary_parts = [f.content for f in top_findings if f.content]

        summary = "Research Summary:\n\n"
        summary += "\n\n".join(summary_parts)

        return summary

    def _collect_sources(
        self,
        results: list[SearchResults],
    ) -> list[SearchResult]:
        """
        Collect all unique sources from results.

        Args:
            results: List of search results

        Returns:
            List of unique sources
        """
        seen_urls = set()
        unique_sources = []

        for search_results in results:
            for result in search_results.results:
                if result.url not in seen_urls:
                    seen_urls.add(result.url)
                    unique_sources.append(result)

        return unique_sources[:20]  # Limit to 20 unique sources


# Global agent instance
_agent: ResearchAgent | None = None


def get_research_agent() -> ResearchAgent:
    """Get or create research agent instance."""
    global _agent
    if _agent is None:
        _agent = ResearchAgent()
    return _agent
