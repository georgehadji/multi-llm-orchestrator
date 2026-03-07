"""
Knowledge Management System
===========================
Centralized learning repository with semantic search.

Features:
- Vector storage for code snippets and solutions
- Knowledge graph for concept relationships
- Pattern matching for similar problems
- Auto-generated documentation

Usage:
    from orchestrator.knowledge_base import KnowledgeBase
    
    kb = KnowledgeBase()
    await kb.learn_from_project(project_id, artifacts)
    similar = await kb.find_similar_problems(current_task)
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
from enum import Enum
import asyncio

from .log_config import get_logger
from .performance import cached, LRUCache

logger = get_logger(__name__)


class KnowledgeType(Enum):
    """Types of knowledge artifacts."""
    CODE_SNIPPET = "code_snippet"
    SOLUTION = "solution"
    BUGFIX = "bugfix"
    PATTERN = "pattern"
    ARCHITECTURE = "architecture"
    DECISION = "decision"
    LESSON = "lesson_learned"


@dataclass
class KnowledgeArtifact:
    """A single piece of knowledge."""
    id: str
    type: KnowledgeType
    title: str
    content: str
    context: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    source_project: Optional[str] = None
    source_task: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    embedding: Optional[List[float]] = None
    similarity_score: float = 0.0
    usage_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeArtifact":
        data["type"] = KnowledgeType(data["type"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Pattern:
    """Recognized pattern from multiple artifacts."""
    id: str
    name: str
    description: str
    artifacts: List[str] = field(default_factory=list)
    frequency: int = 0
    confidence: float = 0.0
    related_patterns: List[str] = field(default_factory=list)


class KnowledgeBase:
    """
    Central knowledge repository with semantic search.
    
    Architecture:
    - Local JSON storage for artifacts
    - In-memory vector index for similarity
    - LRU cache for frequent queries
    - Async indexing for performance
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path(".knowledge")
        self.storage_path.mkdir(exist_ok=True)
        
        self._artifacts: Dict[str, KnowledgeArtifact] = {}
        self._vector_index: Optional[Any] = None
        self._query_cache = LRUCache(max_size=1000, default_ttl=300)
        self._patterns: Dict[str, Pattern] = {}
        
        # Lazy-loaded embedding model
        self._embedding_model: Optional[Callable] = None
        
        # Load existing knowledge
        self._load_index()
    
    def _load_index(self):
        """Load existing knowledge from disk."""
        index_file = self.storage_path / "index.json"
        if index_file.exists():
            try:
                with open(index_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for item in data.get("artifacts", []):
                        artifact = KnowledgeArtifact.from_dict(item)
                        self._artifacts[artifact.id] = artifact
                logger.info(f"Loaded {len(self._artifacts)} knowledge artifacts")
            except Exception as e:
                logger.warning(f"Failed to load knowledge index: {e}")
    
    async def _get_embedding_model(self):
        """Lazy-load embedding model."""
        if self._embedding_model is None:
            try:
                # Try sentence-transformers
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Loaded sentence-transformers model")
            except ImportError:
                # Fallback to simple hashing
                logger.warning("sentence-transformers not available, using hash fallback")
                self._embedding_model = self._hash_embedding
        return self._embedding_model
    
    def _hash_embedding(self, text: str) -> List[float]:
        """Fallback embedding using hash."""
        # Simple but deterministic embedding
        hash_val = hashlib.sha256(text.encode()).hexdigest()
        return [int(hash_val[i:i+8], 16) / 2**32 for i in range(0, 64, 8)]
    
    async def _compute_embedding(self, text: str) -> List[float]:
        """Compute embedding vector for text."""
        model = await self._get_embedding_model()
        
        if callable(model) and not hasattr(model, 'encode'):
            return model(text)
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(None, model.encode, text)
        return embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0
    
    async def add_artifact(
        self,
        type: KnowledgeType,
        title: str,
        content: str,
        context: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        source_project: Optional[str] = None,
        source_task: Optional[str] = None,
    ) -> KnowledgeArtifact:
        """Add new knowledge artifact."""
        # Generate ID
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        artifact_id = f"{type.value}_{content_hash}_{int(time.time())}"
        
        # Compute embedding
        embedding = await self._compute_embedding(f"{title} {content}")
        
        artifact = KnowledgeArtifact(
            id=artifact_id,
            type=type,
            title=title,
            content=content,
            context=context or {},
            tags=tags or [],
            source_project=source_project,
            source_task=source_task,
            embedding=embedding,
        )
        
        self._artifacts[artifact_id] = artifact
        await self._persist_index()
        
        logger.info(f"Added knowledge artifact: {artifact_id}")
        return artifact
    
    async def find_similar(
        self,
        query: str,
        type_filter: Optional[KnowledgeType] = None,
        top_k: int = 5,
        min_similarity: float = 0.5,
    ) -> List[KnowledgeArtifact]:
        """Find similar knowledge artifacts."""
        # Check cache
        cache_key = f"sim_{hashlib.sha256(query.encode()).hexdigest()[:16]}_{type_filter}_{top_k}"
        cached = await self._query_cache.get(cache_key)
        if cached:
            return [KnowledgeArtifact.from_dict(a) for a in cached]
        
        if not self._artifacts:
            return []
        
        # Compute query embedding
        query_embedding = await self._compute_embedding(query)
        
        # Calculate similarities
        scored = []
        for artifact in self._artifacts.values():
            if type_filter and artifact.type != type_filter:
                continue
            
            if artifact.embedding:
                similarity = self._cosine_similarity(query_embedding, artifact.embedding)
                if similarity >= min_similarity:
                    artifact.similarity_score = similarity
                    scored.append((similarity, artifact))
        
        # Sort by similarity
        scored.sort(reverse=True)
        results = [a for _, a in scored[:top_k]]
        
        # Cache results
        await self._query_cache.set(cache_key, [a.to_dict() for a in results], ttl=300)
        
        return results
    
    async def search(
        self,
        query: str,
        tags: Optional[List[str]] = None,
        type_filter: Optional[KnowledgeType] = None,
    ) -> List[KnowledgeArtifact]:
        """Search knowledge base with filters."""
        results = []
        query_lower = query.lower()
        
        for artifact in self._artifacts.values():
            # Type filter
            if type_filter and artifact.type != type_filter:
                continue
            
            # Tag filter
            if tags and not any(t in artifact.tags for t in tags):
                continue
            
            # Text search
            if (query_lower in artifact.title.lower() or 
                query_lower in artifact.content.lower() or
                query_lower in ' '.join(artifact.tags)):
                results.append(artifact)
        
        # Sort by usage (popularity)
        results.sort(key=lambda a: a.usage_count, reverse=True)
        
        return results
    
    async def learn_from_project(
        self,
        project_id: str,
        artifacts_dir: Path,
        decisions: List[Dict[str, Any]],
    ) -> List[KnowledgeArtifact]:
        """Extract knowledge from completed project."""
        created = []
        
        # Learn from decisions
        for decision in decisions:
            artifact = await self.add_artifact(
                type=KnowledgeType.DECISION,
                title=decision.get("title", "Architecture Decision"),
                content=decision.get("rationale", ""),
                context={
                    "alternatives": decision.get("alternatives", []),
                    "consequences": decision.get("consequences", []),
                },
                tags=["decision", project_id] + decision.get("tags", []),
                source_project=project_id,
            )
            created.append(artifact)
        
        # Learn from code patterns
        if artifacts_dir.exists():
            for code_file in artifacts_dir.rglob("*.py"):
                try:
                    content = code_file.read_text(encoding='utf-8')
                    # Extract functions/classes as snippets
                    artifact = await self.add_artifact(
                        type=KnowledgeType.CODE_SNIPPET,
                        title=f"Pattern: {code_file.name}",
                        content=content[:2000],  # Limit size
                        context={"file": str(code_file), "language": "python"},
                        tags=["code", project_id, "python"],
                        source_project=project_id,
                    )
                    created.append(artifact)
                except Exception as e:
                    logger.debug(f"Failed to process {code_file}: {e}")
        
        # Update patterns
        await self._update_patterns()
        
        logger.info(f"Learned {len(created)} artifacts from project {project_id}")
        return created
    
    async def _update_patterns(self):
        """Update recognized patterns from artifacts."""
        # Simple pattern detection based on tags
        tag_groups: Dict[str, List[str]] = {}
        
        for artifact in self._artifacts.values():
            for tag in artifact.tags:
                if tag not in tag_groups:
                    tag_groups[tag] = []
                tag_groups[tag].append(artifact.id)
        
        # Create patterns for frequently occurring tags
        for tag, artifact_ids in tag_groups.items():
            if len(artifact_ids) >= 3:  # Minimum frequency
                pattern_id = f"pattern_{tag}"
                self._patterns[pattern_id] = Pattern(
                    id=pattern_id,
                    name=f"Pattern: {tag}",
                    description=f"Recurring pattern found in {len(artifact_ids)} artifacts",
                    artifacts=artifact_ids,
                    frequency=len(artifact_ids),
                    confidence=min(1.0, len(artifact_ids) / 10),
                )
    
    async def get_recommendations(
        self,
        current_task: str,
        project_context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Get knowledge-based recommendations."""
        recommendations = []
        
        # Find similar past solutions
        similar = await self.find_similar(current_task, top_k=3)
        
        for artifact in similar:
            recommendations.append({
                "type": "similar_solution",
                "artifact_id": artifact.id,
                "title": artifact.title,
                "relevance": f"{artifact.similarity_score:.1%}",
                "content_preview": artifact.content[:200] + "...",
                "source_project": artifact.source_project,
            })
        
        # Check for patterns
        for pattern in self._patterns.values():
            if pattern.confidence > 0.7:
                recommendations.append({
                    "type": "pattern",
                    "pattern_id": pattern.id,
                    "name": pattern.name,
                    "frequency": pattern.frequency,
                    "confidence": f"{pattern.confidence:.1%}",
                })
        
        return recommendations
    
    async def record_usage(self, artifact_id: str):
        """Record that an artifact was used."""
        if artifact_id in self._artifacts:
            self._artifacts[artifact_id].usage_count += 1
            await self._persist_index()
    
    async def _persist_index(self):
        """Save knowledge index to disk."""
        index_file = self.storage_path / "index.json"
        
        data = {
            "updated_at": datetime.now().isoformat(),
            "artifact_count": len(self._artifacts),
            "artifacts": [a.to_dict() for a in self._artifacts.values()],
            "patterns": [
                {
                    "id": p.id,
                    "name": p.name,
                    "frequency": p.frequency,
                    "confidence": p.confidence,
                }
                for p in self._patterns.values()
            ],
        }
        
        # Write atomically
        temp_file = index_file.with_suffix('.tmp')
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        temp_file.replace(index_file)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        by_type = {}
        for a in self._artifacts.values():
            by_type[a.type.value] = by_type.get(a.type.value, 0) + 1
        
        return {
            "total_artifacts": len(self._artifacts),
            "by_type": by_type,
            "patterns_recognized": len(self._patterns),
            "storage_path": str(self.storage_path),
            "cache_stats": self._query_cache.get_stats(),
        }


# Global knowledge base instance
_knowledge_base: Optional[KnowledgeBase] = None


def get_knowledge_base(storage_path: Optional[Path] = None) -> KnowledgeBase:
    """Get global knowledge base instance."""
    global _knowledge_base
    if _knowledge_base is None:
        _knowledge_base = KnowledgeBase(storage_path)
    return _knowledge_base
