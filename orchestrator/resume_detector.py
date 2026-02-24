"""
Resume Detector with ResumeDetector class — Auto-Resume Detection.

Provides both pure functions for keyword extraction and scoring, plus the
ResumeDetector class for integration with StateManager.

Features:
- Keyword extraction from project descriptions
- Match score calculation using Jaccard similarity
- Recency decay scoring (30-day window, not 7-day)
- Match threshold: 0.6
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Optional

# Optional import of StateManager for type hints
try:
    from .state import StateManager
except ImportError:
    StateManager = None


@dataclass(frozen=True)
class ResumeCandidate:
    """A project that could potentially be resumed.

    Attributes:
        project_id: Unique identifier for the project
        description: Original project description
        keywords: Extracted keywords for matching
        recency_score: 0.0–1.0; 1.0 = just created, 0.0 = very old
        similarity_score: 0.0–1.0; how similar to current project
        overall_score: weighted combination for ranking (0.6*similarity + 0.4*recency)
    """
    project_id: str
    description: str
    keywords: list[str]
    recency_score: float
    similarity_score: float
    overall_score: float


def _extract_keywords(text: str | None) -> list[str] | set[str]:
    """Extract meaningful keywords from project description/criteria.

    Rules:
    - Split on whitespace, convert to lowercase
    - Filter out words < 3 characters
    - Filter out common stopwords
    - Return as set (for use in matching)

    Args:
        text: Raw text to extract keywords from

    Returns:
        Set of meaningful keywords
    """
    # Common English stopwords
    stopwords = {
        'the', 'a', 'an', 'is', 'are', 'and', 'or', 'to', 'for', 'of',
        'in', 'on', 'at', 'by', 'with', 'from'
    }

    # Handle empty/None input
    if not text:
        return set()

    # Split, lowercase, and filter
    words = text.lower().split()
    keywords = {
        word for word in words
        if len(word) >= 3 and word not in stopwords
    }

    return keywords


def _recency_factor(created_at_timestamp: float, reference_time: float | None = None) -> float:
    """Calculate recency score 0.0–1.0 based on project age.

    Uses a 30-day decay window (not 7-day).

    Scoring:
    - 1.0 if created within last 24 hours
    - Linear decay: 1.0 - (age_days / 30.0) for 0–30 days
    - 0.0 if older than 30 days

    Args:
        created_at_timestamp: Unix timestamp when the project was created/updated
        reference_time: Current time as Unix timestamp (defaults to now)

    Returns:
        Recency score between 0.0 and 1.0
    """
    if reference_time is None:
        reference_time = time.time()

    # Calculate age in days
    age_seconds = reference_time - created_at_timestamp
    age_days = age_seconds / (24 * 3600)

    # 30-day decay window
    if age_days <= 30:
        # Linear decay from 1.0 to 0.0 over 30 days
        return max(0.0, 1.0 - (age_days / 30.0))
    else:
        # Older than 30 days
        return 0.0


def _calculate_match_score(
    current_keywords: set[str] | list[str],
    prev_description: str | None,
    prev_updated_at: float,
) -> float:
    """Calculate match score between current keywords and previous project.

    Uses Jaccard similarity with 30-day recency decay.

    Args:
        current_keywords: Keywords from current project
        prev_description: Description from previous project
        prev_updated_at: Unix timestamp when previous project was last updated

    Returns:
        Match score between 0.0 and 1.0
    """
    if isinstance(current_keywords, list):
        current_keywords = set(current_keywords)

    # Extract keywords from previous project
    prev_keywords = _extract_keywords(prev_description)
    if isinstance(prev_keywords, list):
        prev_keywords = set(prev_keywords)

    # Calculate Jaccard similarity
    if not current_keywords and not prev_keywords:
        jaccard = 1.0
    elif not current_keywords or not prev_keywords:
        jaccard = 0.0
    else:
        intersection = current_keywords & prev_keywords
        union = current_keywords | prev_keywords
        if len(union) == 0:
            jaccard = 1.0
        else:
            jaccard = len(intersection) / len(union)

    # Calculate recency score (30-day window)
    now = time.time()
    age_seconds = now - prev_updated_at
    age_days = age_seconds / (24 * 3600)

    # Linear decay over 30 days
    if age_days <= 30:
        recency = 1.0 - (age_days / 30.0)
    else:
        recency = 0.0

    # Combined score: 0.6 * similarity + 0.4 * recency
    match_score = 0.6 * jaccard + 0.4 * recency

    return max(0.0, min(1.0, match_score))


def _is_exact_match(keywords_a: list[str] | set[str], keywords_b: list[str] | set[str]) -> bool:
    """Return True if the keyword sets match exactly (order-independent).

    Args:
        keywords_a: First set/list of keywords
        keywords_b: Second set/list of keywords

    Returns:
        True if both contain the same keywords, False otherwise
    """
    set_a = set(keywords_a) if isinstance(keywords_a, list) else keywords_a
    set_b = set(keywords_b) if isinstance(keywords_b, list) else keywords_b
    return set_a == set_b


def _score_candidates(
    current_keywords: list[str] | set[str],
    candidates: list[ResumeCandidate],
) -> list[ResumeCandidate]:
    """Score candidates based on Jaccard similarity and recency.

    Algorithm:
    1. Calculate Jaccard similarity for each candidate
    2. Boost by 0.1 if exact keyword match (capped at 1.0)
    3. Calculate overall_score: 0.6 * similarity + 0.4 * recency
    4. Filter to score > 0.3
    5. Sort by overall_score descending

    Args:
        current_keywords: Keywords from the current project
        candidates: List of resume candidates to score

    Returns:
        Sorted list of candidates with score > 0.3, highest first
    """
    scored_candidates = []
    current_set = set(current_keywords) if isinstance(current_keywords, list) else current_keywords

    for candidate in candidates:
        candidate_set = set(candidate.keywords)

        # Calculate Jaccard similarity: len(intersection) / len(union)
        intersection = current_set & candidate_set
        union = current_set | candidate_set

        if len(union) == 0:
            # Both keyword lists are empty
            jaccard = 1.0
        else:
            jaccard = len(intersection) / len(union)

        # Boost by 0.1 for exact match (capped at 1.0)
        similarity_score = jaccard
        if _is_exact_match(current_keywords, candidate.keywords):
            similarity_score = min(1.0, jaccard + 0.1)

        # Calculate overall score: 0.6 * similarity + 0.4 * recency
        overall_score = 0.6 * similarity_score + 0.4 * candidate.recency_score

        # Filter to score > 0.3
        if overall_score > 0.3:
            # Create new candidate with updated scores
            scored = ResumeCandidate(
                project_id=candidate.project_id,
                description=candidate.description,
                keywords=candidate.keywords,
                recency_score=candidate.recency_score,
                similarity_score=similarity_score,
                overall_score=overall_score,
            )
            scored_candidates.append(scored)

    # Sort by overall_score descending
    scored_candidates.sort(key=lambda c: c.overall_score, reverse=True)

    return scored_candidates


class ResumeDetector:
    """Auto-resume detection for multi-LLM orchestrator.

    Detects if a new project can resume from a previous incomplete project
    by comparing keywords and recency.

    Attributes:
        state_manager: Optional StateManager for database access
        match_threshold: Minimum score to consider a match (default: 0.6)
        recency_decay_days: Window for recency scoring (default: 30)
    """

    def __init__(self, state_manager: Optional[StateManager] = None):
        """Initialize the ResumeDetector.

        Args:
            state_manager: Optional StateManager instance for accessing saved projects
        """
        self.state_manager = state_manager
        self.match_threshold = 0.6
        self.recency_decay_days = 30

    def find_resumable_project(
        self,
        project_description: str,
        success_criteria: str,
    ) -> Optional[dict]:
        """Find a resumable project matching the given description.

        Searches for incomplete projects with similar keywords and returns
        the best match if its score exceeds the threshold.

        Args:
            project_description: Description of the new project
            success_criteria: Success criteria for the new project

        Returns:
            Dict with project info if match found, None otherwise:
            {
                "project_id": str,
                "description": str,
                "similarity": float,
                "recency": float,
                "match_score": float,
            }
        """
        # Extract keywords from current project
        desc_keywords = self._extract_keywords(project_description)
        criteria_keywords = self._extract_keywords(success_criteria)
        combined_keywords = desc_keywords | criteria_keywords

        if not combined_keywords:
            return None

        # If no state manager, cannot find candidates
        if not self.state_manager:
            return None

        # This would be called in an async context via async wrapper
        # For now, return None (would be implemented with async/await)
        return None

    def _extract_keywords(self, text: str | None) -> set[str]:
        """Extract keywords from text.

        Args:
            text: Text to extract keywords from

        Returns:
            Set of keywords
        """
        result = _extract_keywords(text)
        return result if isinstance(result, set) else set(result)

    def _calculate_match_score(
        self,
        current_keywords: set[str],
        prev_description: str | None,
        prev_updated_at: float,
    ) -> float:
        """Calculate match score for a candidate project.

        Args:
            current_keywords: Keywords from current project
            prev_description: Description from candidate project
            prev_updated_at: Unix timestamp of last update

        Returns:
            Match score between 0.0 and 1.0
        """
        return _calculate_match_score(current_keywords, prev_description, prev_updated_at)
