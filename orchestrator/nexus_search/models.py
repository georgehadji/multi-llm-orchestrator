"""
Nexus Search — Data Models
===========================
Author: Georgios-Chrysovalantis Chatzivantsidis

Data models for Nexus Search integration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datetime import datetime


class SearchSource(str, Enum):
    """Available search sources."""
    WEB = "web"
    ACADEMIC = "academic"
    TECH = "tech"
    NEWS = "news"
    CODE = "code"


class QueryType(str, Enum):
    """Query classification types."""
    FACTUAL = "factual"       # Simple facts → direct search
    RESEARCH = "research"     # Deep research → multi-step
    TECHNICAL = "technical"   # Code/tech → tech sources
    ACADEMIC = "academic"     # Academic → scholar/arxiv
    CREATIVE = "creative"     # Creative → broad search


class OptimizationMode(str, Enum):
    """Search optimization modes."""
    SPEED = "speed"         # Fastest results
    BALANCED = "balanced"   # Balance of speed and quality
    QUALITY = "quality"     # Best quality (may be slower)


@dataclass
class SearchResult:
    """
    Individual search result.

    Attributes:
        title: Result title
        url: Source URL
        content: Snippet or summary
        source: Search source (web, academic, etc.)
        engine: Specific engine that returned result
        score: Relevance score (0-1)
        published_date: Publication date if available
        metadata: Additional metadata
    """
    title: str
    url: str
    content: str
    source: SearchSource = SearchSource.WEB
    engine: str = ""
    score: float = 0.0
    published_date: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "url": self.url,
            "content": self.content,
            "source": self.source.value,
            "engine": self.engine,
            "score": self.score,
            "published_date": self.published_date.isoformat() if self.published_date else None,
            "metadata": self.metadata,
        }


@dataclass
class SearchResults:
    """
    Collection of search results.

    Attributes:
        query: Original search query
        results: List of search results
        total_results: Total number of results
        search_time: Time taken for search (ms)
        sources: Sources that were searched
        suggestions: Related search suggestions
        metadata: Additional metadata
    """
    query: str
    results: list[SearchResult] = field(default_factory=list)
    total_results: int = 0
    search_time: float = 0.0
    sources: list[SearchSource] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self):
        return iter(self.results)

    def __getitem__(self, index: int) -> SearchResult:
        return self.results[index]

    @property
    def top(self) -> list[SearchResult]:
        """Get top 5 results."""
        return self.results[:5]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "results": [r.to_dict() for r in self.results],
            "total_results": self.total_results,
            "search_time": self.search_time,
            "sources": [s.value for s in self.sources],
            "suggestions": self.suggestions,
        }


@dataclass
class Finding:
    """
    A research finding with supporting evidence.

    Attributes:
        content: The finding content
        sources: Supporting search results
        confidence: Confidence score (0-1)
        category: Finding category
    """
    content: str
    sources: list[SearchResult] = field(default_factory=list)
    confidence: float = 1.0
    category: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "sources": [s.to_dict() for s in self.sources],
            "confidence": self.confidence,
            "category": self.category,
        }


@dataclass
class ResearchReport:
    """
    Comprehensive research report.

    Attributes:
        query: Original research query
        findings: List of research findings
        summary: Executive summary
        sources: All sources used
        search_iterations: Number of search iterations
        total_time: Total research time (seconds)
    """
    query: str
    findings: list[Finding] = field(default_factory=list)
    summary: str = ""
    sources: list[SearchResult] = field(default_factory=list)
    search_iterations: int = 0
    total_time: float = 0.0

    @property
    def source_count(self) -> int:
        """Total number of unique sources."""
        return len(self.sources)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "findings": [f.to_dict() for f in self.findings],
            "summary": self.summary,
            "sources": [s.to_dict() for s in self.sources],
            "search_iterations": self.search_iterations,
            "total_time": self.total_time,
        }


@dataclass
class SearchQuery:
    """
    Structured search query.

    Attributes:
        query: Search query string
        sources: Sources to search
        language: Language code (en, el, etc.)
        num_results: Maximum number of results
        time_range: Time range filter
        optimization: Optimization mode
    """
    query: str
    sources: list[SearchSource] = field(default_factory=lambda: [SearchSource.WEB])
    language: str = "en"
    num_results: int = 10
    time_range: str | None = None  # "day", "week", "month", "year"
    optimization: OptimizationMode = OptimizationMode.BALANCED

    def to_params(self) -> dict[str, Any]:
        """Convert to API parameters."""
        params = {
            "q": self.query,
            "categories": [s.value for s in self.sources],
            "language": self.language,
            "limit": self.num_results,
            "format": "json",
        }

        if self.time_range:
            params["time_range"] = self.time_range

        return params


@dataclass
class NexusConfig:
    """
    Nexus Search configuration.

    Attributes:
        enabled: Enable/disable Nexus Search
        api_url: Nexus API URL
        timeout: Request timeout (seconds)
        max_results: Maximum results per query
        rate_limit: Queries per minute
        cache_enabled: Enable result caching
        cache_ttl: Cache TTL (seconds)
    """
    enabled: bool = True
    api_url: str = "http://nexus-search:8080"
    timeout: int = 30
    max_results: int = 20
    rate_limit: int = 60  # queries per minute
    cache_enabled: bool = True
    cache_ttl: int = 3600  # 1 hour

    @classmethod
    def from_env(cls) -> NexusConfig:
        """Create config from environment variables."""
        import os

        return cls(
            enabled=os.getenv("NEXUS_SEARCH_ENABLED", "true").lower() == "true",
            api_url=os.getenv("NEXUS_API_URL", "http://nexus-search:8080"),
            timeout=int(os.getenv("NEXUS_TIMEOUT", "30")),
            max_results=int(os.getenv("NEXUS_MAX_RESULTS", "20")),
            rate_limit=int(os.getenv("NEXUS_RATE_LIMIT", "60")),
            cache_enabled=os.getenv("NEXUS_CACHE_ENABLED", "true").lower() == "true",
            cache_ttl=int(os.getenv("NEXUS_CACHE_TTL", "3600")),
        )
