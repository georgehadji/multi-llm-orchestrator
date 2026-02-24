"""
Tests for Resume Detector pure functions (Task 1).
"""
from __future__ import annotations

from datetime import datetime, timedelta
import pytest

from orchestrator.resume_detector import (
    ResumeCandidate,
    _extract_keywords,
    _is_exact_match,
    _recency_factor,
    _score_candidates,
)


@pytest.fixture
def now():
    return datetime(2025, 2, 24, 12, 0, 0)


def test_extract_keywords_complex_text():
    text = "Build a REST API with FastAPI for production use"
    keywords = _extract_keywords(text)
    expected = ["api", "build", "fastapi", "production", "rest", "use"]
    assert keywords == expected


def test_extract_keywords_empty_input():
    assert _extract_keywords("") == []
    assert _extract_keywords(None) == []


def test_recency_factor_recent(now):
    created_at = now - timedelta(hours=2)
    score = _recency_factor(created_at, reference_time=now)
    assert 0.96 < score < 0.99


def test_recency_factor_2_days(now):
    created_at = now - timedelta(hours=48)
    score = _recency_factor(created_at, reference_time=now)
    assert 0.70 < score < 0.73


def test_recency_factor_1_day(now):
    created_at = now - timedelta(hours=24)
    score = _recency_factor(created_at, reference_time=now)
    assert 0.84 < score < 0.87


def test_recency_factor_just_created(now):
    created_at = now
    score = _recency_factor(created_at, reference_time=now)
    assert score == 1.0


def test_recency_factor_8_days_old(now):
    created_at = now - timedelta(days=8)
    score = _recency_factor(created_at, reference_time=now)
    assert score == 0.1


def test_recency_factor_7_days_old(now):
    created_at = now - timedelta(days=7)
    score = _recency_factor(created_at, reference_time=now)
    assert 0.0 <= score <= 0.01


def test_is_exact_match_same_order():
    keywords_a = ["api", "rest", "fastapi"]
    keywords_b = ["api", "rest", "fastapi"]
    assert _is_exact_match(keywords_a, keywords_b) is True


def test_is_exact_match_different_order():
    keywords_a = ["api", "rest"]
    keywords_b = ["rest", "api"]
    assert _is_exact_match(keywords_a, keywords_b) is True


def test_is_exact_match_subset():
    keywords_a = ["api"]
    keywords_b = ["api", "rest"]
    assert _is_exact_match(keywords_a, keywords_b) is False


def test_is_exact_match_superset():
    keywords_a = ["api", "rest", "fastapi"]
    keywords_b = ["api", "rest"]
    assert _is_exact_match(keywords_a, keywords_b) is False


def test_is_exact_match_empty():
    keywords_a = []
    keywords_b = []
    assert _is_exact_match(keywords_a, keywords_b) is True


def test_is_exact_match_one_empty():
    keywords_a = ["api"]
    keywords_b = []
    assert _is_exact_match(keywords_a, keywords_b) is False


def test_score_candidates_ranking_and_filtering(now):
    current_keywords = ["api", "fastapi", "python"]
    
    c1 = ResumeCandidate(
        project_id="proj1",
        description="Build REST API with FastAPI",
        keywords=["api", "fastapi", "python"],
        recency_score=0.9,
        similarity_score=0.0,
        overall_score=0.0,
    )
    
    c2 = ResumeCandidate(
        project_id="proj2",
        description="Python data science project",
        keywords=["python", "data", "science"],
        recency_score=0.5,
        similarity_score=0.0,
        overall_score=0.0,
    )
    
    c3 = ResumeCandidate(
        project_id="proj3",
        description="JavaScript web app",
        keywords=["javascript", "react", "node"],
        recency_score=0.05,
        similarity_score=0.0,
        overall_score=0.0,
    )
    
    candidates = [c1, c2, c3]
    scored = _score_candidates(current_keywords, candidates)
    
    assert len(scored) >= 1
    assert scored[0].project_id in ["proj1", "proj2"]
    
    for candidate in scored:
        assert candidate.overall_score > 0.3
    
    if any(c.project_id == "proj1" for c in scored):
        proj1 = next(c for c in scored if c.project_id == "proj1")
        assert proj1.similarity_score == 1.0
        assert proj1.overall_score >= 0.66


def test_score_candidates_jaccard_similarity():
    current_keywords = ["api", "rest", "python"]
    
    candidate = ResumeCandidate(
        project_id="proj",
        description="Build REST API",
        keywords=["api", "rest", "fastapi"],
        recency_score=0.5,
        similarity_score=0.0,
        overall_score=0.0,
    )
    
    scored = _score_candidates(current_keywords, [candidate])
    
    assert len(scored) == 1
    assert scored[0].similarity_score == 0.5


def test_score_candidates_exact_match_boost():
    current_keywords = ["api", "rest"]
    
    candidate = ResumeCandidate(
        project_id="proj",
        description="REST API project",
        keywords=["api", "rest"],
        recency_score=0.6,
        similarity_score=0.0,
        overall_score=0.0,
    )
    
    scored = _score_candidates(current_keywords, [candidate])
    
    assert scored[0].similarity_score == 1.0


def test_score_candidates_no_overlap():
    current_keywords = ["api", "rest", "python"]
    
    candidate = ResumeCandidate(
        project_id="proj",
        description="JavaScript frontend",
        keywords=["react", "javascript", "typescript"],
        recency_score=0.2,
        similarity_score=0.0,
        overall_score=0.0,
    )
    
    scored = _score_candidates(current_keywords, [candidate])
    assert len(scored) == 0


def test_score_candidates_sorted_descending():
    current_keywords = ["python", "data"]
    
    candidates = [
        ResumeCandidate(
            project_id="proj_low",
            description="Low score project",
            keywords=["javascript"],
            recency_score=0.1,
            similarity_score=0.0,
            overall_score=0.0,
        ),
        ResumeCandidate(
            project_id="proj_high",
            description="High score project",
            keywords=["python", "data"],
            recency_score=0.9,
            similarity_score=0.0,
            overall_score=0.0,
        ),
        ResumeCandidate(
            project_id="proj_mid",
            description="Mid score project",
            keywords=["python"],
            recency_score=0.5,
            similarity_score=0.0,
            overall_score=0.0,
        ),
    ]
    
    scored = _score_candidates(current_keywords, candidates)
    
    for i in range(len(scored) - 1):
        assert scored[i].overall_score >= scored[i + 1].overall_score
    
    if any(c.project_id == "proj_high" for c in scored):
        assert scored[0].project_id == "proj_high"


def test_resume_candidate_immutable():
    candidate = ResumeCandidate(
        project_id="proj1",
        description="Test project",
        keywords=["test"],
        recency_score=0.5,
        similarity_score=0.8,
        overall_score=0.65,
    )
    
    with pytest.raises(Exception):
        candidate.project_id = "proj2"


def test_resume_candidate_creation():
    candidate = ResumeCandidate(
        project_id="test-123",
        description="A test project description",
        keywords=["test", "sample", "demo"],
        recency_score=0.75,
        similarity_score=0.85,
        overall_score=0.80,
    )
    
    assert candidate.project_id == "test-123"
    assert candidate.description == "A test project description"
    assert candidate.keywords == ["test", "sample", "demo"]
    assert candidate.recency_score == 0.75
    assert candidate.similarity_score == 0.85
    assert candidate.overall_score == 0.80
