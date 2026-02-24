"""
Resume Detector pure functions — Auto-Resume Detection (Task 1).

Pure functions for detecting and scoring resumable projects based on:
- Keyword extraction from project descriptions
- Recency scoring (0.0-1.0 scale)
- Exact keyword matching
- Jaccard similarity-based candidate scoring

All functions are pure (no I/O, no state mutation).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


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


def _extract_keywords(text: str | None) -> list[str]:
    """Extract meaningful keywords from project description/criteria.
    
    Rules:
    - Split on whitespace, convert to lowercase
    - Filter out words < 3 characters
    - Filter out common stopwords
    - Sort results alphabetically
    - Return [] on empty/None input
    
    Args:
        text: Raw text to extract keywords from
        
    Returns:
        Sorted list of meaningful keywords
    """
    # Common English stopwords
    stopwords = {
        'the', 'a', 'an', 'is', 'are', 'and', 'or', 'to', 'for', 'of',
        'in', 'on', 'at', 'by', 'with', 'from'
    }
    
    # Handle empty/None input
    if not text:
        return []
    
    # Split, lowercase, and filter
    words = text.lower().split()
    keywords = [
        word for word in words
        if len(word) >= 3 and word not in stopwords
    ]
    
    # Return sorted alphabetically
    return sorted(keywords)


def _recency_factor(created_at: datetime, reference_time: datetime | None = None) -> float:
    """Calculate recency score 0.0–1.0 based on project age.
    
    Scoring:
    - 1.0 if created within last 24 hours
    - Linear decay: 1.0 - (age_hours / 168) for 0–7 days
    - 0.1 if older than 7 days
    
    Args:
        created_at: When the project was created
        reference_time: Current time (defaults to now)
        
    Returns:
        Recency score between 0.0 and 1.0
    """
    if reference_time is None:
        reference_time = datetime.now()
    
    # Calculate age in hours
    age_delta = reference_time - created_at
    age_hours = age_delta.total_seconds() / 3600.0
    
    # 7 days = 168 hours
    one_week_hours = 168.0

    if age_hours <= one_week_hours:
        # Linear decay from 1.0 to 0.0 over 7 days
        return max(0.0, 1.0 - (age_hours / one_week_hours))
    else:
        # Older than 7 days
        return 0.1


def _is_exact_match(keywords_a: list[str], keywords_b: list[str]) -> bool:
    """Return True if the keyword lists match exactly (order-independent).
    
    Args:
        keywords_a: First list of keywords
        keywords_b: Second list of keywords
        
    Returns:
        True if both lists contain the same keywords, False otherwise
    """
    # Convert to sets for order-independent comparison
    return set(keywords_a) == set(keywords_b)


def _score_candidates(
    current_keywords: list[str],
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
    current_set = set(current_keywords)
    
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
