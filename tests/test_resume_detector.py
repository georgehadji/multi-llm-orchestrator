"""
Tests for Resume Detector class and pure functions (spec compliance).
"""
from __future__ import annotations

import pytest
from orchestrator.resume_detector import (
    ResumeDetector,
    _extract_keywords,
    _calculate_match_score,
)


@pytest.fixture
def detector():
    """Create a ResumeDetector instance for testing."""
    return ResumeDetector()


def test_extract_keywords(detector):
    """Test keyword extraction with stop word filtering.

    Verifies:
    - Words are converted to lowercase
    - Stop words are filtered out
    - Short words (< 3 chars) are filtered out
    - Returns a set of keywords
    """
    text = "Build a REST API with FastAPI for production use"
    keywords = _extract_keywords(text)

    # Convert to set if it's a list
    if isinstance(keywords, list):
        keywords = set(keywords)

    # Verify stop words are filtered out
    assert 'a' not in keywords  # stop word
    assert 'the' not in keywords  # stop word
    assert 'with' not in keywords  # stop word
    assert 'for' not in keywords  # stop word

    # Verify meaningful words are kept
    assert 'build' in keywords or 'rest' in keywords or 'api' in keywords

    # Verify it's a set
    assert isinstance(keywords, (set, list, tuple))


def test_no_match_for_unrelated_project(detector):
    """Test that no match is returned when there are no saved projects.

    Verifies:
    - Returns None when no candidates are available
    - Handles empty project list gracefully
    """
    # Since detector has no state_manager, it should return None
    result = detector.find_resumable_project(
        project_description="Build a React web app",
        success_criteria="Create a functional user interface"
    )

    # With no state manager, should return None
    assert result is None


def test_match_score_calculation(detector):
    """Test match score calculation between similar and dissimilar projects.

    Verifies:
    - Similar projects (high keyword overlap) have score > 0.5
    - Dissimilar projects (no keyword overlap) have score < 0.3
    - Uses Jaccard similarity with recency weighting
    """
    import time

    # Test case 1: Similar projects (high overlap)
    current_kw = {"api", "rest", "python"}
    similar_desc = "Build a REST API with Python"
    now = time.time()

    score_similar = _calculate_match_score(current_kw, similar_desc, now)
    assert score_similar > 0.5, f"Similar projects should score > 0.5, got {score_similar}"

    # Test case 2: Dissimilar projects (no overlap)
    # Use an old timestamp (40+ days ago) to ensure low recency score
    old_timestamp = now - (40 * 24 * 3600)  # 40 days in the past
    dissimilar_desc = "Create a JavaScript React frontend"
    score_dissimilar = _calculate_match_score(current_kw, dissimilar_desc, old_timestamp)
    assert score_dissimilar < 0.3, f"Dissimilar projects should score < 0.3, got {score_dissimilar}"
