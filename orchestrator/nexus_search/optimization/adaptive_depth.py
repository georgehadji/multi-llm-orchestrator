"""
Nexus Search — Adaptive Research Depth
=======================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Dynamically adjust research depth based on query complexity and result quality.

Features:
- Query type-based depth calculation
- Query complexity analysis (word count, structure)
- Result quality-based early stopping
- Configurable min/max depth limits
- Cost optimization through adaptive iteration

Usage:
    from orchestrator.nexus_search.optimization import AdaptiveDepthController

    controller = AdaptiveDepthController()
    depth = controller.calculate_depth(query, query_type, initial_results)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from orchestrator.nexus_search.models import QueryType, SearchResults

logger = logging.getLogger("orchestrator.nexus_search")


class AdaptiveDepthController:
    """
    Dynamically adjust research depth based on multiple factors.

    Factors considered:
    1. Query type (factual, research, technical, academic, creative)
    2. Query complexity (word count, structure)
    3. Initial result quality (total results, relevance)
    4. User preferences (if available)

    Usage:
        controller = AdaptiveDepthController()
        depth = controller.calculate_depth(query, query_type, initial_results)
    """

    # Base depth by query type
    BASE_DEPTH_MAP = {
        "factual": 1,  # Simple facts - single search sufficient
        "technical": 2,  # Code examples - need some depth
        "creative": 2,  # Brainstorming - moderate exploration
        "research": 3,  # Deep research - multiple iterations
        "academic": 4,  # Academic papers - extensive search
    }

    # Default depth for unknown query types
    DEFAULT_BASE_DEPTH = 2

    def __init__(
        self,
        max_depth: int = 5,
        min_depth: int = 1,
        quality_threshold: float = 0.7,
        result_count_threshold: int = 100,
    ):
        """
        Initialize adaptive depth controller.

        Args:
            max_depth: Maximum research depth (default: 5)
            min_depth: Minimum research depth (default: 1)
            quality_threshold: Quality score threshold for early stopping (0-1)
            result_count_threshold: Result count considered "sufficient"
        """
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.quality_threshold = quality_threshold
        self.result_count_threshold = result_count_threshold
        self._depth_calculations = 0
        self._early_stops = 0

    def calculate_depth(
        self,
        query: str,
        query_type: QueryType | str,
        initial_results: SearchResults | None = None,
    ) -> int:
        """
        Calculate optimal research depth.

        Args:
            query: Research query string
            query_type: Type of query (factual, research, technical, academic, creative)
            initial_results: Initial search results for quality assessment

        Returns:
            Optimal depth (1-5)
        """
        # Get base depth from query type
        if isinstance(query_type, str):
            query_type_str = query_type.lower()
        else:
            query_type_str = query_type.value.lower()

        base_depth = self.BASE_DEPTH_MAP.get(query_type_str, self.DEFAULT_BASE_DEPTH)

        logger.debug(f"Base depth for {query_type_str}: {base_depth}")

        # Adjust by query complexity
        complexity_adjustment = self._analyze_query_complexity(query)
        base_depth += complexity_adjustment
        logger.debug(f"Complexity adjustment: {complexity_adjustment:+d}")

        # Adjust by initial result quality (if available)
        if initial_results:
            quality_adjustment = self._analyze_result_quality(initial_results)
            base_depth += quality_adjustment
            logger.debug(f"Quality adjustment: {quality_adjustment:+d}")

        # Clamp to valid range
        final_depth = max(self.min_depth, min(self.max_depth, base_depth))

        self._depth_calculations += 1
        logger.info(
            f"Adaptive depth calculated: query='{query[:50]}...', "
            f"type={query_type_str}, depth={final_depth}"
        )

        return final_depth

    def _analyze_query_complexity(self, query: str) -> int:
        """
        Analyze query complexity and return depth adjustment.

        Args:
            query: Search query

        Returns:
            Depth adjustment (-1 to +2)
        """
        words = query.split()
        word_count = len(words)

        # Very short queries (< 4 words) may need more exploration
        if word_count < 4:
            return +1  # Increase depth for vague queries

        # Very long queries (> 15 words) are usually specific
        elif word_count > 15:
            return -1  # Decrease depth for specific queries

        # Check for complexity indicators
        complexity_indicators = [
            "compare",
            "versus",
            "vs",  # Comparison queries
            "comprehensive",
            "complete",
            "thorough",  # Comprehensive queries
            "advanced",
            "complex",
            "sophisticated",  # Advanced topics
            "step by step",
            "detailed",
            "in-depth",  # Detailed queries
        ]

        query_lower = query.lower()
        complexity_score = sum(1 for indicator in complexity_indicators if indicator in query_lower)

        # Add depth for complex queries
        if complexity_score >= 2:
            return +2
        elif complexity_score == 1:
            return +1

        return 0

    def _analyze_result_quality(self, results: SearchResults) -> int:
        """
        Analyze initial result quality and return depth adjustment.

        Args:
            results: Initial search results

        Returns:
            Depth adjustment (-2 to +1)
        """
        total_results = results.total_results
        result_count = len(results.results)

        # Many results = less depth needed
        if total_results > self.result_count_threshold * 10:
            logger.debug(f"Many results ({total_results}), reducing depth")
            return -2

        # Moderate results = slight reduction
        elif total_results > self.result_count_threshold:
            logger.debug(f"Good result count ({total_results}), slight depth reduction")
            return -1

        # Few results = more depth needed
        elif total_results < 10:
            logger.debug(f"Few results ({total_results}), increasing depth")
            return +1

        # Check result diversity (multiple sources)
        if hasattr(results, "sources"):
            source_count = len(results.sources)
            if source_count >= 5:
                logger.debug(f"Good source diversity ({source_count} sources)")
                return -1
            elif source_count <= 2:
                logger.debug(f"Limited source diversity ({source_count} sources)")
                return +1

        return 0

    def should_continue(
        self,
        current_iteration: int,
        current_depth: int,
        results_quality: float | None = None,
    ) -> bool:
        """
        Decide whether to continue research iterations.

        Enables early stopping when quality threshold is met.

        Args:
            current_iteration: Current iteration number (0-indexed)
            current_depth: Target depth
            results_quality: Current quality score (0-1), if available

        Returns:
            True if should continue, False if should stop
        """
        # Always stop if max depth reached
        if current_iteration >= current_depth:
            logger.debug(f"Max depth reached ({current_depth} iterations)")
            return False

        # Early stopping if quality threshold met
        if results_quality is not None:
            if results_quality >= self.quality_threshold:
                self._early_stops += 1
                logger.info(
                    f"Early stopping: quality={results_quality:.2f} >= "
                    f"threshold={self.quality_threshold}"
                )
                return False

            logger.debug(
                f"Continuing: quality={results_quality:.2f} < "
                f"threshold={self.quality_threshold}"
            )

        # Continue if quality not yet sufficient
        return True

    def get_stats(self) -> dict:
        """Get depth controller statistics."""
        early_stop_rate = (
            self._early_stops / self._depth_calculations if self._depth_calculations > 0 else 0
        )

        return {
            "depth_calculations": self._depth_calculations,
            "early_stops": self._early_stops,
            "early_stop_rate": f"{early_stop_rate:.1%}",
            "max_depth": self.max_depth,
            "min_depth": self.min_depth,
            "quality_threshold": self.quality_threshold,
        }


# Global instance
_controller: AdaptiveDepthController | None = None


def get_controller(
    max_depth: int = 5,
    min_depth: int = 1,
    quality_threshold: float = 0.7,
) -> AdaptiveDepthController:
    """
    Get or create AdaptiveDepthController instance.

    Args:
        max_depth: Maximum research depth
        min_depth: Minimum research depth
        quality_threshold: Quality threshold for early stopping

    Returns:
        AdaptiveDepthController instance
    """
    global _controller
    if _controller is None:
        _controller = AdaptiveDepthController(
            max_depth=max_depth,
            min_depth=min_depth,
            quality_threshold=quality_threshold,
        )
    return _controller
