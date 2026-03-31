"""
Tests for Model Performance Knowledge Graph.
"""

import pytest
import asyncio
from datetime import datetime
from pathlib import Path
import tempfile
import shutil

from orchestrator.knowledge_graph import (
    PerformanceKnowledgeGraph,
    NodeType,
    EdgeType,
    Node,
    Edge,
)
from orchestrator.models import Model, TaskType
from orchestrator.feedback_loop import (
    ProductionOutcome,
    CodebaseFingerprint,
    OutcomeStatus,
)


class TestPerformanceKnowledgeGraph:
    """Test suite for knowledge graph."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp = tempfile.mkdtemp()
        yield Path(temp)
        shutil.rmtree(temp)

    @pytest.fixture
    def kg(self, temp_dir):
        """Create knowledge graph instance."""
        return PerformanceKnowledgeGraph(storage_path=temp_dir)

    @pytest.fixture
    def sample_outcome(self):
        """Create sample production outcome."""
        return ProductionOutcome(
            project_id="test-project",
            deployment_id="deploy-001",
            task_type=TaskType.CODE_GEN,
            model_used=Model.DEEPSEEK_CHAT,
            generated_code_hash="abc123",
            status=OutcomeStatus.SUCCESS,
        )

    @pytest.fixture
    def sample_fingerprint(self):
        """Create sample codebase fingerprint."""
        return CodebaseFingerprint(
            languages=["python"],
            framework="fastapi",
            patterns=["repository", "dependency-injection"],
            complexity_score=0.6,
        )

    def test_initialization(self, kg):
        """Test knowledge graph initialization."""
        assert kg is not None
        assert kg.storage_path.exists()

    def test_add_node(self, kg):
        """Test adding a node."""
        node = Node(
            id="test-node",
            type=NodeType.PATTERN,
            properties={"pattern": "repository"},
        )
        kg._add_node_internal(node)

        retrieved = kg._get_node("test-node")
        assert retrieved is not None
        assert retrieved.id == "test-node"
        assert retrieved.type == NodeType.PATTERN

    def test_add_edge(self, kg):
        """Test adding an edge."""
        # Add nodes first
        source = Node(id="source", type=NodeType.MODEL, properties={})
        target = Node(id="target", type=NodeType.PATTERN, properties={})
        kg._add_node_internal(source)
        kg._add_node_internal(target)

        # Add edge
        edge = Edge(
            source="source",
            target="target",
            type=EdgeType.EXCELS_AT,
            weight=0.8,
        )
        kg._add_edge_internal(edge)

        # Verify
        edges = kg._get_outgoing_edges("source")
        assert len(edges) == 1
        assert edges[0].weight == 0.8

    @pytest.mark.asyncio
    async def test_add_performance_outcome(self, kg, sample_outcome, sample_fingerprint):
        """Test adding a production outcome."""
        result = await kg.add_performance_outcome(
            sample_outcome,
            sample_fingerprint,
        )

        assert result["nodes_created"] > 0
        assert result["total_nodes"] >= result["nodes_created"]

    @pytest.mark.asyncio
    async def test_find_similar_patterns(self, kg, sample_outcome, sample_fingerprint):
        """Test finding similar patterns."""
        # Add some data first
        await kg.add_performance_outcome(sample_outcome, sample_fingerprint)

        # Search for similar
        query = CodebaseFingerprint(
            languages=["python"],
            framework="fastapi",
            patterns=["repository"],  # Overlapping pattern
        )

        matches = await kg.find_similar_patterns(query, top_k=5)

        # Should find matches
        assert isinstance(matches, list)

    @pytest.mark.asyncio
    async def test_recommend_models(self, kg, sample_outcome, sample_fingerprint):
        """Test model recommendation."""
        # Add data
        await kg.add_performance_outcome(sample_outcome, sample_fingerprint)

        # Get recommendations
        recs = await kg.recommend_models(
            task_type=TaskType.CODE_GEN,
            fingerprint=sample_fingerprint,
            top_k=3,
        )

        assert isinstance(recs, list)
        if recs:
            assert "model_name" in recs[0]
            assert "score" in recs[0]

    def test_graph_stats(self, kg):
        """Test getting graph statistics."""
        stats = kg.get_graph_stats()

        assert "total_nodes" in stats
        assert "total_edges" in stats
        assert "query_count" in stats

    def test_persistence(self, temp_dir):
        """Test saving and loading graph."""
        # Create and populate
        kg1 = PerformanceKnowledgeGraph(storage_path=temp_dir)
        node = Node(id="persistent", type=NodeType.PATTERN, properties={})
        kg1._add_node_internal(node)
        kg1._save_graph()

        # Load in new instance
        kg2 = PerformanceKnowledgeGraph(storage_path=temp_dir)
        loaded = kg2._get_node("persistent")

        assert loaded is not None
        assert loaded.id == "persistent"


class TestNodeAndEdge:
    """Test Node and Edge dataclasses."""

    def test_node_to_dict(self):
        """Test node serialization."""
        node = Node(
            id="test",
            type=NodeType.MODEL,
            properties={"name": "test-model"},
        )

        data = node.to_dict()
        assert data["id"] == "test"
        assert data["type"] == "model"

    def test_edge_to_dict(self):
        """Test edge serialization."""
        edge = Edge(
            source="a",
            target="b",
            type=EdgeType.EXCELS_AT,
            weight=0.9,
        )

        data = edge.to_dict()
        assert data["source"] == "a"
        assert data["target"] == "b"
        assert data["weight"] == 0.9


class TestCodebaseFingerprint:
    """Test CodebaseFingerprint similarity."""

    def test_similarity_identical(self):
        """Test similarity between identical fingerprints."""
        fp1 = CodebaseFingerprint(
            languages=["python", "javascript"],
            framework="fastapi",
            patterns=["repository"],
        )
        fp2 = CodebaseFingerprint(
            languages=["python", "javascript"],
            framework="fastapi",
            patterns=["repository"],
        )

        sim = fp1.similarity(fp2)
        assert sim > 0.8  # Should be very similar

    def test_similarity_different(self):
        """Test similarity between different fingerprints."""
        fp1 = CodebaseFingerprint(
            languages=["python"],
            framework="fastapi",
            patterns=["repository"],
        )
        fp2 = CodebaseFingerprint(
            languages=["rust"],
            framework="actix",
            patterns=["actor-model"],
        )

        sim = fp1.similarity(fp2)
        assert sim < 0.5  # Should be less similar


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
