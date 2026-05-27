"""
KnowledgeGraph — Relational learning from task patterns
==========================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Optimization A-3: Builds a graph of (TaskType, Model, Method, Score)
with edges for "used_with", "produced_score", "failed_on".
Enables queries like "best method for this task type" and
"most common failure mode for this model".
"""

from __future__ import annotations
import os

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("orchestrator.learning.knowledge_graph")


@dataclass
class KnowledgeNode:
    """A node in the knowledge graph."""

    id: str
    type: str  # "task_type", "model", "method"
    label: str


@dataclass
class KnowledgeEdge:
    """A relationship between two nodes."""

    source: str
    target: str
    relation: str  # "used_with", "produced_score", "failed_on"
    weight: float = 1.0


class KnowledgeGraph:
    """Relational knowledge graph for cross-task learning."""

    def __init__(self) -> None:
        self.nodes: dict[str, KnowledgeNode] = {}
        self.edges: list[KnowledgeEdge] = []

    def add_node(self, node_id: str, node_type: str, label: str) -> KnowledgeNode:
        if node_id not in self.nodes:
            self.nodes[node_id] = KnowledgeNode(id=node_id, type=node_type, label=label)
        return self.nodes[node_id]

    def add_edge(self, source: str, target: str, relation: str, weight: float = 1.0) -> None:
        self.edges.append(
            KnowledgeEdge(source=source, target=target, relation=relation, weight=weight)
        )

    def record_success(self, task_type: str, model: str, method: str, score: float) -> None:
        """Record a successful execution as graph edges."""
        self.add_node(f"tt:{task_type}", "task_type", task_type)
        self.add_node(f"model:{model}", "model", model)
        self.add_node(f"method:{method}", "method", method)
        self.add_edge(f"model:{model}", f"tt:{task_type}", "used_with")
        self.add_edge(f"method:{method}", f"tt:{task_type}", "used_with")
        self.add_edge(f"model:{model}", f"tt:{task_type}", "produced_score", weight=score)
        self.add_edge(f"method:{method}", f"tt:{task_type}", "produced_score", weight=score)

    def best_method_for(self, task_type: str) -> str | None:
        """Find the method with highest average score for a task type."""
        scores: dict[str, list[float]] = {}
        for edge in self.edges:
            if edge.relation == "produced_score" and edge.target == f"tt:{task_type}":
                if edge.source.startswith("method:"):
                    method = edge.source.replace("method:", "")
                    scores.setdefault(method, []).append(edge.weight)
        if not scores:
            return None
        return max(scores, key=lambda m: sum(scores[m]) / len(scores[m]) if scores[m] else 0)

    def failures_for_model(self, model: str) -> list[str]:
        """Get task types where the model has failed."""
        failed: list[str] = []
        for edge in self.edges:
            if edge.relation == "failed_on" and edge.source == f"model:{model}":
                failed.append(edge.target.replace("tt:", ""))
        return failed

    def save(self, path: str | None = None) -> None:
        import json

        p = path or os.path.join(os.path.expanduser("~"), ".orchestrator", "knowledge_graph.json")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        data = {
            "nodes": {k: v.__dict__ for k, v in self.nodes.items()},
            "edges": [e.__dict__ for e in self.edges],
        }
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)

    @classmethod
    def load(cls, path: str | None = None) -> "KnowledgeGraph":
        import json, os

        p = path or os.path.join(os.path.expanduser("~"), ".orchestrator", "knowledge_graph.json")
        if not os.path.exists(p):
            return cls()
        try:
            with open(p, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            kg = cls()
            for nid, ndata in data.get("nodes", {}).items():
                kg.nodes[nid] = KnowledgeNode(**ndata)
            for edata in data.get("edges", []):
                kg.edges.append(KnowledgeEdge(**edata))
            return kg
        except (json.JSONDecodeError, KeyError, TypeError):
            return cls()

    def find_similar_projects(self, goal: str, limit: int = 3) -> list[dict]:
        """Find past projects with similar goals. Level 3: cross-project transfer."""
        results = []
        goal_lower = goal.lower()
        goal_words = set(goal_lower.split())
        for nid, node in self.nodes.items():
            if node.type == "task_type" and node.label in ("code_gen", "reasoning"):
                overlap = goal_words & set(node.label.split("_"))
                if overlap:
                    # All edges in the graph have task_type as TARGET (not source),
                    # because record_success() adds edges as model→tt:X and method→tt:X.
                    # Collect the average produced_score across all models/methods that
                    # have been used for this task_type.
                    score_weights = [
                        e.weight
                        for e in self.edges
                        if e.target == nid and e.relation == "produced_score"
                    ]
                    if score_weights:
                        avg_score = sum(score_weights) / len(score_weights)
                        results.append(
                            {
                                "name": node.label,
                                "architecture": [e.source for e in self.edges if e.target == nid],
                                "score": avg_score,
                            }
                        )
        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:limit]
