"""
Tests for Nexus Search Module
==============================
Author: Georgios-Chrysovalantis Chatzivantsidis

Tests for the Nexus Search integration including:
- Models and data structures
- Query classification
- Search functionality (mocked)
- Research agent
- Configuration
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from orchestrator.nexus_search.models import (
    SearchResult,
    SearchResults,
    SearchSource,
    SearchQuery,
    Finding,
    ResearchReport,
    QueryType,
    OptimizationMode,
    NexusConfig,
)
from orchestrator.nexus_search.config import get_config, configure, reset_config
from orchestrator.nexus_search.agents.classifier import QueryClassifier, get_classifier
from orchestrator.nexus_search.agents.researcher import ResearchAgent, get_research_agent

# ─────────────────────────────────────────────
# Model Tests
# ─────────────────────────────────────────────


class TestSearchResult:
    """Tests for SearchResult model."""

    def test_create_search_result(self):
        """Test creating a search result."""
        result = SearchResult(
            title="Test Result",
            url="https://example.com",
            content="Test content",
            source=SearchSource.WEB,
            engine="google",
            score=0.95,
        )

        assert result.title == "Test Result"
        assert result.url == "https://example.com"
        assert result.content == "Test content"
        assert result.source == SearchSource.WEB
        assert result.engine == "google"
        assert result.score == 0.95

    def test_search_result_to_dict(self):
        """Test converting search result to dictionary."""
        result = SearchResult(
            title="Test",
            url="https://example.com",
            content="Content",
            source=SearchSource.TECH,
        )

        data = result.to_dict()

        assert data["title"] == "Test"
        assert data["url"] == "https://example.com"
        assert data["source"] == "tech"

    def test_search_result_default_values(self):
        """Test default values for search result."""
        result = SearchResult(
            title="Test",
            url="https://example.com",
            content="Content",
        )

        assert result.source == SearchSource.WEB
        assert result.engine == ""
        assert result.score == 0.0
        assert result.published_date is None


class TestSearchResults:
    """Tests for SearchResults collection."""

    def test_create_search_results(self):
        """Test creating search results collection."""
        results = SearchResults(
            query="test query",
            results=[
                SearchResult(title="Result 1", url="https://example.com/1", content="Content 1"),
                SearchResult(title="Result 2", url="https://example.com/2", content="Content 2"),
            ],
            total_results=2,
            search_time=123.45,
        )

        assert results.query == "test query"
        assert len(results) == 2
        assert results.total_results == 2
        assert results.search_time == 123.45

    def test_search_results_indexing(self):
        """Test indexing and iteration."""
        results = SearchResults(
            query="test",
            results=[
                SearchResult(
                    title=f"Result {i}", url=f"https://example.com/{i}", content=f"Content {i}"
                )
                for i in range(5)
            ],
        )

        # Test indexing
        assert results[0].title == "Result 0"
        assert results[4].title == "Result 4"

        # Test iteration
        titles = [r.title for r in results]
        assert len(titles) == 5

    def test_search_results_top(self):
        """Test top property returns first 5 results."""
        results = SearchResults(
            query="test",
            results=[
                SearchResult(
                    title=f"Result {i}", url=f"https://example.com/{i}", content=f"Content {i}"
                )
                for i in range(10)
            ],
        )

        assert len(results.top) == 5
        assert results.top[0].title == "Result 0"

    def test_search_results_to_dict(self):
        """Test converting search results to dictionary."""
        results = SearchResults(
            query="test",
            results=[
                SearchResult(title="Result", url="https://example.com", content="Content"),
            ],
            sources=[SearchSource.WEB, SearchSource.TECH],
        )

        data = results.to_dict()

        assert data["query"] == "test"
        assert len(data["results"]) == 1
        assert data["sources"] == ["web", "tech"]


class TestFinding:
    """Tests for Finding model."""

    def test_create_finding(self):
        """Test creating a finding."""
        finding = Finding(
            content="Test finding",
            sources=[SearchResult(title="Source", url="https://example.com", content="Content")],
            confidence=0.85,
            category="tech",
        )

        assert finding.content == "Test finding"
        assert finding.confidence == 0.85
        assert finding.category == "tech"
        assert len(finding.sources) == 1

    def test_finding_to_dict(self):
        """Test converting finding to dictionary."""
        finding = Finding(
            content="Test",
            confidence=0.9,
            category="test",
        )

        data = finding.to_dict()

        assert data["content"] == "Test"
        assert data["confidence"] == 0.9
        assert data["category"] == "test"


class TestResearchReport:
    """Tests for ResearchReport model."""

    def test_create_research_report(self):
        """Test creating a research report."""
        report = ResearchReport(
            query="test query",
            findings=[
                Finding(content="Finding 1", confidence=0.9),
                Finding(content="Finding 2", confidence=0.8),
            ],
            summary="Test summary",
            search_iterations=3,
            total_time=5.5,
        )

        assert report.query == "test query"
        assert len(report.findings) == 2
        assert report.summary == "Test summary"
        assert report.search_iterations == 3
        assert report.total_time == 5.5

    def test_research_report_source_count(self):
        """Test source_count property."""
        report = ResearchReport(
            query="test",
            sources=[
                SearchResult(title="Source 1", url="https://example.com/1", content="Content 1"),
                SearchResult(title="Source 2", url="https://example.com/2", content="Content 2"),
            ],
        )

        assert report.source_count == 2

    def test_research_report_to_dict(self):
        """Test converting research report to dictionary."""
        report = ResearchReport(
            query="test",
            findings=[Finding(content="Finding")],
            summary="Summary",
        )

        data = report.to_dict()

        assert data["query"] == "test"
        assert len(data["findings"]) == 1
        assert data["summary"] == "Summary"


class TestSearchQuery:
    """Tests for SearchQuery model."""

    def test_create_search_query(self):
        """Test creating a search query."""
        query = SearchQuery(
            query="test query",
            sources=[SearchSource.WEB, SearchSource.TECH],
            language="en",
            num_results=10,
            optimization=OptimizationMode.BALANCED,
        )

        assert query.query == "test query"
        assert len(query.sources) == 2
        assert query.language == "en"
        assert query.num_results == 10
        assert query.optimization == OptimizationMode.BALANCED

    def test_search_query_to_params(self):
        """Test converting search query to API parameters."""
        query = SearchQuery(
            query="test",
            sources=[SearchSource.WEB],
            language="el",
            num_results=5,
            time_range="week",
        )

        params = query.to_params()

        assert params["q"] == "test"
        assert params["categories"] == ["web"]
        assert params["language"] == "el"
        assert params["limit"] == 5
        assert params["time_range"] == "week"

    def test_search_query_defaults(self):
        """Test default values for search query."""
        query = SearchQuery(query="test")

        assert query.sources == [SearchSource.WEB]
        assert query.language == "en"
        assert query.num_results == 10
        assert query.optimization == OptimizationMode.BALANCED
        assert query.time_range is None


class TestNexusConfig:
    """Tests for NexusConfig model."""

    def test_create_config(self):
        """Test creating a config."""
        config = NexusConfig(
            enabled=True,
            api_url="http://localhost:8080",
            timeout=60,
            max_results=30,
        )

        assert config.enabled is True
        assert config.api_url == "http://localhost:8080"
        assert config.timeout == 60
        assert config.max_results == 30

    def test_config_from_env(self):
        """Test creating config from environment variables."""
        with patch.dict(
            "os.environ",
            {
                "NEXUS_SEARCH_ENABLED": "false",
                "NEXUS_API_URL": "http://custom:8080",
                "NEXUS_TIMEOUT": "90",
                "NEXUS_MAX_RESULTS": "50",
                "NEXUS_RATE_LIMIT": "120",
                "NEXUS_CACHE_ENABLED": "false",
                "NEXUS_CACHE_TTL": "7200",
            },
        ):
            config = NexusConfig.from_env()

            assert config.enabled is False
            assert config.api_url == "http://custom:8080"
            assert config.timeout == 90
            assert config.max_results == 50
            assert config.rate_limit == 120
            assert config.cache_enabled is False
            assert config.cache_ttl == 7200

    def test_config_defaults(self):
        """Test default config values."""
        config = NexusConfig()

        assert config.enabled is True
        assert config.api_url == "http://nexus-search:8080"
        assert config.timeout == 30
        assert config.max_results == 20
        assert config.rate_limit == 60
        assert config.cache_enabled is True
        assert config.cache_ttl == 3600


# ─────────────────────────────────────────────
# Configuration Tests
# ─────────────────────────────────────────────


class TestConfiguration:
    """Tests for configuration management."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()

    def teardown_method(self):
        """Reset config after each test."""
        reset_config()

    def test_get_config_default(self):
        """Test getting default config."""
        config = get_config()

        assert config.enabled is True
        assert config.api_url == "http://nexus-search:8080"

    def test_configure_updates_config(self):
        """Test that configure updates config."""
        config = configure(
            enabled=False,
            api_url="http://custom:8080",
            timeout=60,
        )

        assert config.enabled is False
        assert config.api_url == "http://custom:8080"
        assert config.timeout == 60

    def test_configure_persists(self):
        """Test that configure persists config."""
        configure(api_url="http://first:8080")
        config1 = get_config()

        configure(api_url="http://second:8080")
        config2 = get_config()

        assert config1.api_url == "http://second:8080"
        assert config2.api_url == "http://second:8080"

    def test_reset_config(self):
        """Test resetting config."""
        configure(api_url="http://custom:8080")
        reset_config()

        config = get_config()
        assert config.api_url == "http://nexus-search:8080"


# ─────────────────────────────────────────────
# Query Classifier Tests
# ─────────────────────────────────────────────


class TestQueryClassifier:
    """Tests for query classification."""

    def test_classify_factual(self):
        """Test classifying factual queries."""
        classifier = QueryClassifier()

        factual_queries = [
            "What is Python?",
            "Who created Python?",
            "When was Python released?",
            "Define async/await",
            "How many Python versions exist?",
        ]

        for query in factual_queries:
            query_type = asyncio.run(classifier.classify(query))
            assert query_type == QueryType.FACTUAL, f"Expected FACTUAL for: {query}"

    def test_classify_research(self):
        """Test classifying research queries."""
        classifier = QueryClassifier()

        research_queries = [
            "Python async best practices",
            "Microservices architecture guide",
            "How to build FastAPI service",
            "Comparison of ORMs",
            "Design patterns overview",
        ]

        for query in research_queries:
            query_type = asyncio.run(classifier.classify(query))
            assert query_type == QueryType.RESEARCH, f"Expected RESEARCH for: {query}"

    def test_classify_technical(self):
        """Test classifying technical queries."""
        classifier = QueryClassifier()

        technical_queries = [
            "Python code example",
            "FastAPI API library",
            "Install pytest package",
            "Debug asyncio error",
            "GitHub repository",
        ]

        for query in technical_queries:
            query_type = asyncio.run(classifier.classify(query))
            assert query_type == QueryType.TECHNICAL, f"Expected TECHNICAL for: {query}"

    def test_classify_academic(self):
        """Test classifying academic queries."""
        classifier = QueryClassifier()

        academic_queries = [
            "Python research paper",
            "Async/await study",
            "arxiv machine learning",
            "PubMed database",
            "Google scholar citation",
        ]

        for query in academic_queries:
            query_type = asyncio.run(classifier.classify(query))
            assert query_type == QueryType.ACADEMIC, f"Expected ACADEMIC for: {query}"

    def test_classify_creative(self):
        """Test classifying creative queries."""
        classifier = QueryClassifier()

        creative_queries = [
            "Ideas for Python project",
            "Creative coding inspiration",
            "Innovative UI design",
            "Brainstorm features",
            "Suggest improvements",
        ]

        for query in creative_queries:
            query_type = asyncio.run(classifier.classify(query))
            assert query_type == QueryType.CREATIVE, f"Expected CREATIVE for: {query}"

    def test_get_recommended_sources(self):
        """Test getting recommended sources for query type."""
        classifier = QueryClassifier()

        # Factual → WEB
        sources = classifier.get_recommended_sources(QueryType.FACTUAL)
        assert SearchSource.WEB in sources

        # Technical → TECH, CODE
        sources = classifier.get_recommended_sources(QueryType.TECHNICAL)
        assert SearchSource.TECH in sources
        assert SearchSource.CODE in sources

        # Academic → ACADEMIC
        sources = classifier.get_recommended_sources(QueryType.ACADEMIC)
        assert SearchSource.ACADEMIC in sources

    def test_classify_and_get_sources(self):
        """Test classifying and getting sources in one call."""
        classifier = QueryClassifier()

        query_type, sources = asyncio.run(
            classifier.classify_and_get_sources("Python async best practices")
        )

        assert query_type == QueryType.RESEARCH
        assert len(sources) > 0

    def test_get_classifier_singleton(self):
        """Test get_classifier returns singleton."""
        classifier1 = get_classifier()
        classifier2 = get_classifier()

        assert classifier1 is classifier2


# ─────────────────────────────────────────────
# Research Agent Tests (Mocked)
# ─────────────────────────────────────────────


class TestResearchAgent:
    """Tests for research agent."""

    @pytest.mark.asyncio
    async def test_research_basic(self):
        """Test basic research flow."""
        # Mock the provider
        mock_provider = AsyncMock()
        mock_provider.search.return_value = SearchResults(
            query="test query",
            results=[
                SearchResult(title="Result 1", url="https://example.com/1", content="Content 1"),
                SearchResult(title="Result 2", url="https://example.com/2", content="Content 2"),
            ],
            total_results=2,
        )

        agent = ResearchAgent(provider=mock_provider)
        report = await agent.research("test query", depth=2)

        assert report.query == "test query"
        assert report.search_iterations >= 1
        assert len(report.sources) > 0

    @pytest.mark.asyncio
    async def test_research_with_custom_depth(self):
        """Test research with custom depth."""
        mock_provider = AsyncMock()
        mock_provider.search.return_value = SearchResults(
            query="test",
            results=[SearchResult(title="Result", url="https://example.com", content="Content")],
        )

        agent = ResearchAgent(provider=mock_provider)
        report = await agent.research("test query", depth=5)

        # Depth should be limited to max_iterations (3)
        assert report.search_iterations <= 3

    @pytest.mark.asyncio
    async def test_research_with_custom_sources(self):
        """Test research with custom sources."""
        mock_provider = AsyncMock()
        mock_provider.search.return_value = SearchResults(
            query="test",
            results=[SearchResult(title="Result", url="https://example.com", content="Content")],
            sources=[SearchSource.TECH],
        )

        agent = ResearchAgent(provider=mock_provider)
        report = await agent.research(
            "test query",
            sources=[SearchSource.TECH, SearchSource.CODE],
        )

        # Verify provider was called with correct sources
        mock_provider.search.assert_called()

    @pytest.mark.asyncio
    async def test_research_generates_findings(self):
        """Test that research generates findings."""
        mock_provider = AsyncMock()
        mock_provider.search.return_value = SearchResults(
            query="test",
            results=[
                SearchResult(title="Result", url="https://example.com", content="Detailed content"),
            ],
        )

        agent = ResearchAgent(provider=mock_provider)
        report = await agent.research("test query")

        assert len(report.findings) > 0
        assert report.findings[0].content is not None

    @pytest.mark.asyncio
    async def test_research_generates_summary(self):
        """Test that research generates summary."""
        mock_provider = AsyncMock()
        mock_provider.search.return_value = SearchResults(
            query="test",
            results=[SearchResult(title="Result", url="https://example.com", content="Content")],
        )

        agent = ResearchAgent(provider=mock_provider)
        report = await agent.research("test query")

        assert report.summary != ""
        assert "Research Summary" in report.summary or len(report.summary) > 0

    def test_get_research_agent_singleton(self):
        """Test get_research_agent returns singleton."""
        agent1 = get_research_agent()
        agent2 = get_research_agent()

        assert agent1 is agent2


# ─────────────────────────────────────────────
# Integration Tests (Mocked)
# ─────────────────────────────────────────────


class TestNexusSearchIntegration:
    """Integration tests for Nexus Search."""

    @pytest.mark.asyncio
    async def test_search_integration(self):
        """Test search integration."""
        with patch("orchestrator.nexus_search.core.get_nexus_orchestrator") as mock_get:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.search.return_value = SearchResults(
                query="test",
                results=[
                    SearchResult(title="Result", url="https://example.com", content="Content")
                ],
            )
            mock_get.return_value = mock_orchestrator

            from orchestrator.nexus_search import search

            results = await search("test query")

            assert len(results) > 0
            mock_orchestrator.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_research_integration(self):
        """Test research integration."""
        with patch("orchestrator.nexus_search.core.get_nexus_orchestrator") as mock_get:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.research.return_value = ResearchReport(
                query="test",
                findings=[Finding(content="Finding")],
                summary="Summary",
            )
            mock_get.return_value = mock_orchestrator

            from orchestrator.nexus_search import research

            report = await research("test query")

            assert len(report.findings) > 0
            mock_orchestrator.research.assert_called_once()

    @pytest.mark.asyncio
    async def test_classify_integration(self):
        """Test classify integration."""
        with patch("orchestrator.nexus_search.core.get_nexus_orchestrator") as mock_get:
            mock_orchestrator = AsyncMock()
            mock_orchestrator.classify.return_value = QueryType.RESEARCH
            mock_get.return_value = mock_orchestrator

            from orchestrator.nexus_search import classify

            query_type = await classify("test query")

            assert query_type == QueryType.RESEARCH
            mock_orchestrator.classify.assert_called_once()


# ─────────────────────────────────────────────
# Enum Tests
# ─────────────────────────────────────────────


class TestEnums:
    """Tests for enum types."""

    def test_search_source_values(self):
        """Test SearchSource enum values."""
        assert SearchSource.WEB.value == "web"
        assert SearchSource.ACADEMIC.value == "academic"
        assert SearchSource.TECH.value == "tech"
        assert SearchSource.NEWS.value == "news"
        assert SearchSource.CODE.value == "code"

    def test_query_type_values(self):
        """Test QueryType enum values."""
        assert QueryType.FACTUAL.value == "factual"
        assert QueryType.RESEARCH.value == "research"
        assert QueryType.TECHNICAL.value == "technical"
        assert QueryType.ACADEMIC.value == "academic"
        assert QueryType.CREATIVE.value == "creative"

    def test_optimization_mode_values(self):
        """Test OptimizationMode enum values."""
        assert OptimizationMode.SPEED.value == "speed"
        assert OptimizationMode.BALANCED.value == "balanced"
        assert OptimizationMode.QUALITY.value == "quality"


# ─────────────────────────────────────────────
# Run Tests
# ─────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
