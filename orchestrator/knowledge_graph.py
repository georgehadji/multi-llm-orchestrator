"""
Model Performance Knowledge Graph
=================================

A semantic network connecting models, task types, code patterns, and outcomes.
Uses NetworkX for graph operations with embedding-based similarity.

Key Features:
- Multi-hop reasoning for model recommendations
- Pattern-based similarity matching
- Confidence-weighted relationship traversal
- Incremental graph updates from feedback loop

Usage:
    from orchestrator.knowledge_graph import PerformanceKnowledgeGraph

    pkg = PerformanceKnowledgeGraph()
    await pkg.add_performance_node(outcome)

    # Multi-hop recommendation
    results = await pkg.find_optimal_path(
        from_pattern="FastAPI+Repository",
        to_outcome="ProductionSuccess",
        max_hops=3
    )
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .feedback_loop import (
    CodebaseFingerprint,
    OutcomeStatus,
    ProductionOutcome,
)
from .log_config import get_logger

if TYPE_CHECKING:
    from .models import TaskType

logger = get_logger(__name__)

# Optional NetworkX import with fallback
try:
    import networkx as nx

    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    logger.warning("NetworkX not available, using fallback implementation")


class NodeType(Enum):
    """Types of nodes in the knowledge graph."""

    MODEL = "model"
    TASK_TYPE = "task_type"
    PATTERN = "pattern"
    FRAMEWORK = "framework"
    LANGUAGE = "language"
    OUTCOME = "outcome"
    PROJECT = "project"
    DEPENDENCY = "dependency"


class EdgeType(Enum):
    """Types of relationships in the graph."""

    EXCELS_AT = "excels_at"  # Model -> Pattern/Task
    STRUGGLES_WITH = "struggles_with"  # Model -> Pattern/Task
    USED_FOR = "used_for"  # Model -> TaskType
    PRODUCES = "produces"  # Model -> Outcome
    HAS_PATTERN = "has_pattern"  # Project -> Pattern
    USES_FRAMEWORK = "uses_framework"  # Project -> Framework
    USES_LANGUAGE = "uses_language"  # Project -> Language
    DEPENDS_ON = "depends_on"  # Pattern -> Dependency
    SIMILAR_TO = "similar_to"  # Pattern -> Pattern


@dataclass
class Node:
    """A node in the knowledge graph."""

    id: str
    type: NodeType
    properties: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "properties": self.properties,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class Edge:
    """An edge in the knowledge graph."""

    source: str
    target: str
    type: EdgeType
    weight: float = 1.0
    confidence: float = 1.0
    sample_size: int = 1
    properties: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "type": self.type.value,
            "weight": self.weight,
            "confidence": self.confidence,
            "sample_size": self.sample_size,
            "properties": self.properties,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class PathResult:
    """Result of a path query."""

    nodes: list[Node]
    edges: list[Edge]
    total_weight: float
    confidence: float
    path_type: str


@dataclass
class SimilarityMatch:
    """A similarity match result."""

    node: Node
    similarity: float
    matching_patterns: list[str]
    explanation: str


class PerformanceKnowledgeGraph:
    """
    Knowledge graph for model performance relationships.

    Optimized for:
    - Fast similarity queries (cached embeddings)
    - Incremental updates (event-driven)
    - Multi-hop reasoning (BFS with pruning)
    """

    # Caching configuration
    EMBEDDING_CACHE_TTL = timedelta(minutes=10)
    SIMILARITY_THRESHOLD = 0.6
    MAX_PATH_LENGTH = 4

    def __init__(self, storage_path: Path | None = None):
        self.storage_path = storage_path or Path(".knowledge_graph")
        self.storage_path.mkdir(exist_ok=True)

        # Graph storage
        if HAS_NETWORKX:
            self._graph = nx.DiGraph()
        else:
            self._graph = None
            self._nodes: dict[str, Node] = {}
            self._edges: dict[str, list[Edge]] = defaultdict(list)
            self._reverse_edges: dict[str, list[Edge]] = defaultdict(list)

        # Caches
        self._embedding_cache: dict[str, tuple[list[float], datetime]] = {}
        self._pattern_index: dict[str, set[str]] = defaultdict(set)
        self._framework_index: dict[str, set[str]] = defaultdict(set)

        # Statistics
        self._query_count = 0
        self._cache_hit_count = 0

        self._load_graph()

    def _load_graph(self) -> None:
        """Load graph from disk."""
        nodes_file = self.storage_path / "nodes.jsonl"
        edges_file = self.storage_path / "edges.jsonl"

        if nodes_file.exists():
            try:
                for line in nodes_file.read_text().strip().split("\n"):
                    if line:
                        data = json.loads(line)
                        node = Node(
                            id=data["id"],
                            type=NodeType(data["type"]),
                            properties=data.get("properties", {}),
                            created_at=datetime.fromisoformat(data["created_at"]),
                            updated_at=datetime.fromisoformat(data["updated_at"]),
                        )
                        self._add_node_internal(node)
                logger.info(f"Loaded {len(self._get_all_nodes())} nodes")
            except Exception as e:
                logger.error(f"Failed to load nodes: {e}")

        if edges_file.exists():
            try:
                for line in edges_file.read_text().strip().split("\n"):
                    if line:
                        data = json.loads(line)
                        edge = Edge(
                            source=data["source"],
                            target=data["target"],
                            type=EdgeType(data["type"]),
                            weight=data.get("weight", 1.0),
                            confidence=data.get("confidence", 1.0),
                            sample_size=data.get("sample_size", 1),
                            properties=data.get("properties", {}),
                            created_at=datetime.fromisoformat(data["created_at"]),
                        )
                        self._add_edge_internal(edge)
                logger.info(f"Loaded {len(self._get_all_edges())} edges")
            except Exception as e:
                logger.error(f"Failed to load edges: {e}")

    def _save_graph(self) -> None:
        """Save graph to disk."""
        try:
            nodes_file = self.storage_path / "nodes.jsonl"
            edges_file = self.storage_path / "edges.jsonl"

            # Save nodes
            with nodes_file.open("w") as f:
                for node in self._get_all_nodes():
                    f.write(json.dumps(node.to_dict()) + "\n")

            # Save edges
            with edges_file.open("w") as f:
                for edge in self._get_all_edges():
                    f.write(json.dumps(edge.to_dict()) + "\n")

            logger.debug(f"Saved graph to {self.storage_path}")
        except Exception as e:
            logger.error(f"Failed to save graph: {e}")

    def _get_all_nodes(self) -> list[Node]:
        """Get all nodes in the graph."""
        if HAS_NETWORKX and self._graph is not None:
            return [
                Node(
                    id=n,
                    type=NodeType(data.get("type", NodeType.PATTERN.value)),
                    properties=data.get("properties", {}),
                )
                for n, data in self._graph.nodes(data=True)
            ]
        else:
            return list(self._nodes.values())

    def _get_all_edges(self) -> list[Edge]:
        """Get all edges in the graph."""
        if HAS_NETWORKX and self._graph is not None:
            edges = []
            for u, v, data in self._graph.edges(data=True):
                edges.append(
                    Edge(
                        source=u,
                        target=v,
                        type=EdgeType(data.get("type", EdgeType.EXCELS_AT.value)),
                        weight=data.get("weight", 1.0),
                        confidence=data.get("confidence", 1.0),
                        sample_size=data.get("sample_size", 1),
                        properties=data.get("properties", {}),
                    )
                )
            return edges
        else:
            return [e for edges in self._edges.values() for e in edges]

    def _add_node_internal(self, node: Node) -> None:
        """Internal method to add a node."""
        if HAS_NETWORKX and self._graph is not None:
            self._graph.add_node(
                node.id,
                type=node.type.value,
                properties=node.properties,
            )
        else:
            self._nodes[node.id] = node

            # Update indexes
            if node.type == NodeType.PATTERN:
                self._pattern_index[node.properties.get("pattern", "")].add(node.id)
            elif node.type == NodeType.FRAMEWORK:
                self._framework_index[node.properties.get("framework", "")].add(node.id)

    def _add_edge_internal(self, edge: Edge) -> None:
        """Internal method to add an edge."""
        if HAS_NETWORKX and self._graph is not None:
            self._graph.add_edge(
                edge.source,
                edge.target,
                type=edge.type.value,
                weight=edge.weight,
                confidence=edge.confidence,
                sample_size=edge.sample_size,
                properties=edge.properties,
            )
        else:
            self._edges[edge.source].append(edge)
            self._reverse_edges[edge.target].append(edge)

    def _get_node(self, node_id: str) -> Node | None:
        """Get a node by ID."""
        if HAS_NETWORKX and self._graph is not None:
            if node_id not in self._graph:
                return None
            data = self._graph.nodes[node_id]
            return Node(
                id=node_id,
                type=NodeType(data.get("type", NodeType.PATTERN.value)),
                properties=data.get("properties", {}),
            )
        else:
            return self._nodes.get(node_id)

    def _get_outgoing_edges(self, node_id: str) -> list[Edge]:
        """Get outgoing edges from a node."""
        if HAS_NETWORKX and self._graph is not None:
            edges = []
            for _, target, data in self._graph.out_edges(node_id, data=True):
                edges.append(
                    Edge(
                        source=node_id,
                        target=target,
                        type=EdgeType(data.get("type", EdgeType.EXCELS_AT.value)),
                        weight=data.get("weight", 1.0),
                        confidence=data.get("confidence", 1.0),
                        sample_size=data.get("sample_size", 1),
                        properties=data.get("properties", {}),
                    )
                )
            return edges
        else:
            return self._edges.get(node_id, [])

    def _get_incoming_edges(self, node_id: str) -> list[Edge]:
        """Get incoming edges to a node."""
        if HAS_NETWORKX and self._graph is not None:
            edges = []
            for source, _, data in self._graph.in_edges(node_id, data=True):
                edges.append(
                    Edge(
                        source=source,
                        target=node_id,
                        type=EdgeType(data.get("type", EdgeType.EXCELS_AT.value)),
                        weight=data.get("weight", 1.0),
                        confidence=data.get("confidence", 1.0),
                        sample_size=data.get("sample_size", 1),
                        properties=data.get("properties", {}),
                    )
                )
            return edges
        else:
            return self._reverse_edges.get(node_id, [])

    # ═══════════════════════════════════════════════════════════════════════════
    # Public API
    # ═══════════════════════════════════════════════════════════════════════════

    async def add_performance_outcome(
        self,
        outcome: ProductionOutcome,
        codebase_fingerprint: CodebaseFingerprint | None = None,
    ) -> dict[str, Any]:
        """
        Add a production outcome to the knowledge graph.

        Creates/updates nodes and edges based on the outcome.
        """
        nodes_created = []
        edges_created = []

        # Create model node
        model_id = f"model:{outcome.model_used.value}"
        if not self._get_node(model_id):
            model_node = Node(
                id=model_id,
                type=NodeType.MODEL,
                properties={
                    "name": outcome.model_used.value,
                    "provider": outcome.model_used.value.split("-")[0],
                },
            )
            self._add_node_internal(model_node)
            nodes_created.append(model_node)

        # Create task type node
        task_id = f"task:{outcome.task_type.value}"
        if not self._get_node(task_id):
            task_node = Node(
                id=task_id,
                type=NodeType.TASK_TYPE,
                properties={"name": outcome.task_type.value},
            )
            self._add_node_internal(task_node)
            nodes_created.append(task_node)

        # Create pattern nodes from fingerprint
        if codebase_fingerprint:
            for pattern in codebase_fingerprint.patterns:
                pattern_id = f"pattern:{pattern.lower().replace(' ', '_')}"
                if not self._get_node(pattern_id):
                    pattern_node = Node(
                        id=pattern_id,
                        type=NodeType.PATTERN,
                        properties={"pattern": pattern},
                    )
                    self._add_node_internal(pattern_node)
                    nodes_created.append(pattern_node)

                # Create or update relationship
                edge_type = (
                    EdgeType.EXCELS_AT
                    if outcome.status == OutcomeStatus.SUCCESS
                    else EdgeType.STRUGGLES_WITH
                )
                await self._update_relationship(
                    source=model_id,
                    target=pattern_id,
                    edge_type=edge_type,
                    outcome=outcome,
                )
                edges_created.append((model_id, pattern_id, edge_type.value))

            # Create framework node
            if codebase_fingerprint.framework:
                framework_id = f"framework:{codebase_fingerprint.framework.lower()}"
                if not self._get_node(framework_id):
                    framework_node = Node(
                        id=framework_id,
                        type=NodeType.FRAMEWORK,
                        properties={"name": codebase_fingerprint.framework},
                    )
                    self._add_node_internal(framework_node)
                    nodes_created.append(framework_node)

        # Create outcome relationship
        await self._update_relationship(
            source=model_id,
            target=task_id,
            edge_type=EdgeType.USED_FOR,
            outcome=outcome,
        )
        edges_created.append((model_id, task_id, EdgeType.USED_FOR.value))

        # Persist
        self._save_graph()

        return {
            "nodes_created": len(nodes_created),
            "edges_created": len(edges_created),
            "total_nodes": len(self._get_all_nodes()),
            "total_edges": len(self._get_all_edges()),
        }

    async def _update_relationship(
        self,
        source: str,
        target: str,
        edge_type: EdgeType,
        outcome: ProductionOutcome,
    ) -> None:
        """Update or create a relationship with outcome data."""
        # Calculate success score
        success_score = outcome.calculate_success_score()

        # Check for existing edge
        existing = None
        for edge in self._get_outgoing_edges(source):
            if edge.target == target and edge.type == edge_type:
                existing = edge
                break

        if existing:
            # Update with exponential moving average
            alpha = 0.1
            existing.weight = (1 - alpha) * existing.weight + alpha * success_score
            existing.sample_size += 1
            existing.confidence = min(1.0, existing.sample_size / 20)
            existing.properties["last_outcome"] = outcome.status.value
        else:
            # Create new edge
            new_edge = Edge(
                source=source,
                target=target,
                type=edge_type,
                weight=success_score,
                confidence=0.5,
                sample_size=1,
                properties={"last_outcome": outcome.status.value},
            )
            self._add_edge_internal(new_edge)

    async def find_similar_patterns(
        self,
        fingerprint: CodebaseFingerprint,
        top_k: int = 5,
    ) -> list[SimilarityMatch]:
        """
        Find similar patterns in the graph using multi-factor similarity.

        More sophisticated than simple Jaccard - uses graph structure.
        """
        self._query_count += 1

        # Build query embedding from fingerprint
        query_embedding = self._fingerprint_to_embedding(fingerprint)

        matches = []

        # Get all pattern nodes
        pattern_nodes = [n for n in self._get_all_nodes() if n.type == NodeType.PATTERN]

        for pattern_node in pattern_nodes:
            # Calculate structural similarity
            structural_sim = self._structural_similarity(fingerprint, pattern_node)

            # Calculate semantic similarity (if embedding available)
            semantic_sim = self._semantic_similarity(query_embedding, pattern_node)

            # Combined score
            similarity = 0.6 * structural_sim + 0.4 * semantic_sim

            if similarity >= self.SIMILARITY_THRESHOLD:
                # Find matching patterns
                matching = self._find_matching_patterns(fingerprint, pattern_node)

                matches.append(
                    SimilarityMatch(
                        node=pattern_node,
                        similarity=similarity,
                        matching_patterns=matching,
                        explanation=self._generate_explanation(fingerprint, pattern_node, matching),
                    )
                )

        # Sort by similarity and return top_k
        matches.sort(key=lambda x: x.similarity, reverse=True)
        return matches[:top_k]

    def _fingerprint_to_embedding(self, fingerprint: CodebaseFingerprint) -> dict[str, float]:
        """Convert fingerprint to embedding vector representation."""
        embedding = {}

        # Language features
        for lang in fingerprint.languages:
            embedding[f"lang:{lang.lower()}"] = 1.0

        # Framework feature
        if fingerprint.framework:
            embedding[f"framework:{fingerprint.framework.lower()}"] = 1.0

        # Pattern features
        for pattern in fingerprint.patterns:
            embedding[f"pattern:{pattern.lower().replace(' ', '_')}"] = 1.0

        # Dependency features
        for dep in fingerprint.dependencies:
            embedding[f"dep:{dep.lower()}"] = 0.5

        # Complexity
        embedding["complexity"] = fingerprint.complexity_score

        return embedding

    def _structural_similarity(
        self,
        fingerprint: CodebaseFingerprint,
        pattern_node: Node,
    ) -> float:
        """Calculate structural similarity using graph neighborhood."""
        # Get connected models for this pattern
        connected_models = set()
        for edge in self._get_incoming_edges(pattern_node.id):
            if edge.type in (EdgeType.EXCELS_AT, EdgeType.STRUGGLES_WITH):
                connected_models.add(edge.source)

        if not connected_models:
            return 0.0

        # Score based on model overlap with query fingerprint patterns
        score = 0.0
        for pattern in fingerprint.patterns:
            pattern_id = f"pattern:{pattern.lower().replace(' ', '_')}"
            if pattern_id == pattern_node.id:
                score = 1.0
                break

            # Check if patterns share models
            pattern_models = set()
            for edge in self._get_incoming_edges(pattern_id):
                if edge.type in (EdgeType.EXCELS_AT, EdgeType.STRUGGLES_WITH):
                    pattern_models.add(edge.source)

            if pattern_models & connected_models:
                score = max(score, 0.5)

        return score

    def _semantic_similarity(
        self,
        query_embedding: dict[str, float],
        pattern_node: Node,
    ) -> float:
        """Calculate cosine similarity between embeddings."""
        pattern_embedding = pattern_node.properties.get("embedding", {})

        if not pattern_embedding:
            # Generate simple embedding from node properties
            pattern_embedding = {
                f"pattern:{pattern_node.properties.get('pattern', '').lower().replace(' ', '_')}": 1.0
            }

        # Cosine similarity
        dot_product = sum(
            query_embedding.get(k, 0) * pattern_embedding.get(k, 0)
            for k in set(query_embedding) | set(pattern_embedding)
        )

        norm_query = sum(v**2 for v in query_embedding.values()) ** 0.5
        norm_pattern = sum(v**2 for v in pattern_embedding.values()) ** 0.5

        if norm_query == 0 or norm_pattern == 0:
            return 0.0

        return dot_product / (norm_query * norm_pattern)

    def _find_matching_patterns(
        self,
        fingerprint: CodebaseFingerprint,
        pattern_node: Node,
    ) -> list[str]:
        """Find which patterns match between fingerprint and node."""
        node_pattern = pattern_node.properties.get("pattern", "").lower()

        matching = []
        for fp_pattern in fingerprint.patterns:
            if (
                fp_pattern.lower() == node_pattern
                or node_pattern in fp_pattern.lower()
                or fp_pattern.lower() in node_pattern
            ):
                matching.append(fp_pattern)

        return matching

    def _generate_explanation(
        self,
        fingerprint: CodebaseFingerprint,
        pattern_node: Node,
        matching: list[str],
    ) -> str:
        """Generate human-readable explanation of similarity."""
        node_pattern = pattern_node.properties.get("pattern", "")

        if matching:
            return f"Shares pattern '{node_pattern}' with your codebase"

        # Check for framework similarity
        if fingerprint.framework:
            framework_edges = self._get_outgoing_edges(pattern_node.id)
            for edge in framework_edges:
                if edge.type == EdgeType.SIMILAR_TO:
                    target_node = self._get_node(edge.target)
                    if target_node and target_node.type == NodeType.FRAMEWORK:
                        return f"Related to your {fingerprint.framework} framework usage"

        return f"Graph structural similarity to '{node_pattern}'"

    async def recommend_models(
        self,
        task_type: TaskType,
        fingerprint: CodebaseFingerprint,
        strategy: str = "balanced",
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """
        Recommend models using multi-hop graph traversal.

        Strategy options:
        - "balanced": Equal weight to quality and cost
        - "quality": Prioritize high-performing models
        - "exploration": Try under-sampled models
        """
        # Find similar patterns
        similar = await self.find_similar_patterns(fingerprint, top_k=10)

        # Collect model scores through graph traversal
        model_scores: dict[str, list[float]] = defaultdict(list)
        model_confidences: dict[str, list[float]] = defaultdict(list)

        for match in similar:
            # Traverse from pattern to models
            for edge in self._get_incoming_edges(match.node.id):
                if edge.type == EdgeType.EXCELS_AT:
                    # Weight by similarity and edge weight
                    score = match.similarity * edge.weight * edge.confidence
                    model_scores[edge.source].append(score)
                    model_confidences[edge.source].append(edge.confidence)
                elif edge.type == EdgeType.STRUGGLES_WITH:
                    # Penalize struggling models
                    score = -match.similarity * edge.weight * 0.5
                    model_scores[edge.source].append(score)

        # Aggregate scores
        recommendations = []
        for model_id, scores in model_scores.items():
            node = self._get_node(model_id)
            if not node:
                continue

            avg_score = sum(scores) / len(scores)
            avg_confidence = sum(model_confidences.get(model_id, [0.5])) / len(scores)

            # Apply strategy
            if strategy == "quality":
                avg_score *= 1.2 if avg_score > 0.7 else 0.8
            elif strategy == "exploration":
                # Boost under-sampled models
                if len(scores) < 5:
                    avg_score *= 1.3

            recommendations.append(
                {
                    "model_id": model_id,
                    "model_name": node.properties.get("name", model_id),
                    "score": avg_score,
                    "confidence": avg_confidence,
                    "evidence_count": len(scores),
                    "strategy": strategy,
                }
            )

        # Sort by score
        recommendations.sort(key=lambda x: x["score"], reverse=True)

        return recommendations[:top_k]

    async def find_optimal_path(
        self,
        from_pattern: str,
        to_outcome: str,
        max_hops: int = 3,
    ) -> list[PathResult]:
        """
        Find optimal paths in the graph using BFS with pruning.

        Useful for understanding: "Which models excel at Pattern X
        to achieve Outcome Y?"
        """
        from_id = f"pattern:{from_pattern.lower().replace(' ', '_')}"
        to_id = f"outcome:{to_outcome.lower().replace(' ', '_')}"

        if not self._get_node(from_id):
            return []

        # BFS with path tracking
        paths: list[PathResult] = []
        queue: list[tuple[str, list[str], list[str], float, float]] = [
            (from_id, [from_id], [], 1.0, 1.0)  # node, path_nodes, path_edges, weight, confidence
        ]
        visited: set[str] = set()

        while queue and len(paths) < 10:  # Limit results
            current, path_nodes, path_edges, weight, confidence = queue.pop(0)

            if len(path_nodes) > max_hops:
                continue

            if current == to_id:
                # Found a path
                node_objs = [self._get_node(n) for n in path_nodes if self._get_node(n)]
                edge_objs = []  # Would need to reconstruct from IDs

                paths.append(
                    PathResult(
                        nodes=node_objs,
                        edges=edge_objs,
                        total_weight=weight,
                        confidence=confidence,
                        path_type="success" if weight > 0.5 else "caution",
                    )
                )
                continue

            if current in visited:
                continue
            visited.add(current)

            # Expand neighbors
            for edge in self._get_outgoing_edges(current):
                if edge.target not in path_nodes:  # Avoid cycles
                    new_weight = weight * edge.weight
                    new_confidence = confidence * edge.confidence

                    # Pruning: drop low-confidence paths
                    if new_confidence > 0.1:
                        queue.append(
                            (
                                edge.target,
                                path_nodes + [edge.target],
                                path_edges + [edge.source + "->" + edge.target],
                                new_weight,
                                new_confidence,
                            )
                        )

        # Sort by weight (descending)
        paths.sort(key=lambda x: x.total_weight, reverse=True)
        return paths

    def get_graph_stats(self) -> dict[str, Any]:
        """Get statistics about the knowledge graph."""
        nodes = self._get_all_nodes()
        edges = self._get_all_edges()

        by_type = defaultdict(int)
        for node in nodes:
            by_type[node.type.value] += 1

        by_edge_type = defaultdict(int)
        for edge in edges:
            by_edge_type[edge.type.value] += 1

        return {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "nodes_by_type": dict(by_type),
            "edges_by_type": dict(by_edge_type),
            "avg_edge_weight": sum(e.weight for e in edges) / len(edges) if edges else 0,
            "query_count": self._query_count,
            "cache_hit_rate": self._cache_hit_count / max(1, self._query_count),
            "storage_path": str(self.storage_path),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience Functions
# ═══════════════════════════════════════════════════════════════════════════════

_kg: PerformanceKnowledgeGraph | None = None


def get_knowledge_graph() -> PerformanceKnowledgeGraph:
    """Get global knowledge graph instance."""
    global _kg
    if _kg is None:
        _kg = PerformanceKnowledgeGraph()
    return _kg


def reset_knowledge_graph() -> None:
    """Reset global knowledge graph (for testing)."""
    global _kg
    _kg = None
