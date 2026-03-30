"""
Transfer Learning for Meta-Optimization
========================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Cross-project transfer learning that applies successful patterns from past
projects to new, similar projects. Enables the orchestrator to learn from
experience and accelerate optimization convergence.

KEY CONCEPTS:
- Project Embedding: Vector representation of project characteristics
- Similarity Engine: Finds similar projects for pattern transfer
- Transfer Patterns: Learned strategies that transfer between projects
- Validation: Ensures transfers are appropriate before application

USAGE:
    from orchestrator.transfer_learning import TransferLearningEngine

    transfer_engine = TransferLearningEngine(archive)

    # Find transferable patterns for current project
    patterns = await transfer_engine.find_transferable_patterns(
        current_project_id="new-project-123"
    )

    # Validate and apply patterns
    for pattern in patterns:
        if await transfer_engine.validate_transfer(pattern, context):
            proposal = await transfer_engine.apply_pattern(pattern)
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .meta_orchestrator import (
    ExecutionArchive,
    ProjectTrajectory,
    StrategyProposal,
    StrategyType,
)

logger = logging.getLogger("orchestrator.transfer")


# ─────────────────────────────────────────────
# Enums & Constants
# ─────────────────────────────────────────────

class PatternType(str, Enum):
    """Types of transferable patterns."""
    MODEL_ROUTING = "model_routing"
    BUDGET_ALLOCATION = "budget_allocation"
    TASK_TEMPLATE = "task_template"
    EXECUTION_STRATEGY = "execution_strategy"
    VALIDATOR_CONFIG = "validator_config"


class TransferStatus(str, Enum):
    """Status of a transfer pattern."""
    ACTIVE = "active"
    APPLIED = "applied"
    REJECTED = "rejected"
    EXPIRED = "expired"


# Similarity thresholds
DEFAULT_MIN_SIMILARITY = 0.7
DEFAULT_TRANSFER_CONFIDENCE = 0.8
MIN_SUCCESS_COUNT = 3  # Minimum successes before pattern is transferable


# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────

@dataclass
class ProjectFeatures:
    """Extracted features from a project for embedding."""
    project_id: str

    # Technology stack (one-hot encoded)
    tech_stack: dict[str, bool] = field(default_factory=dict)

    # Complexity metrics
    total_tasks: int = 0
    avg_task_iterations: float = 0.0
    total_cost: float = 0.0
    total_time: float = 0.0

    # Task type distribution
    task_types: dict[str, int] = field(default_factory=dict)

    # Success metrics
    success_rate: float = 0.0
    avg_score: float = 0.0

    # Model usage
    models_used: dict[str, int] = field(default_factory=dict)

    def to_vector(self) -> list[float]:
        """Convert features to normalized vector for similarity computation."""
        vector = []

        # Tech stack (normalized)
        for val in self.tech_stack.values():
            vector.append(1.0 if val else 0.0)

        # Complexity metrics (normalized to 0-1)
        vector.append(min(self.total_tasks / 100, 1.0))
        vector.append(min(self.avg_task_iterations / 5, 1.0))
        vector.append(min(self.total_cost / 10, 1.0))
        vector.append(min(self.total_time / 3600, 1.0))

        # Task type distribution (normalized)
        total_tasks = sum(self.task_types.values()) or 1
        for task_type in ["code_generation", "code_review", "reasoning", "evaluation"]:
            vector.append(self.task_types.get(task_type, 0) / total_tasks)

        # Success metrics
        vector.append(self.success_rate)
        vector.append(self.avg_score)

        return vector


@dataclass
class ProjectEmbedding:
    """Vector representation of a project."""
    project_id: str
    embedding: list[float]
    features: ProjectFeatures
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "project_id": self.project_id,
            "embedding": self.embedding,
            "features": {
                "project_id": self.features.project_id,
                "tech_stack": self.features.tech_stack,
                "total_tasks": self.features.total_tasks,
                "avg_task_iterations": self.features.avg_task_iterations,
                "total_cost": self.features.total_cost,
                "total_time": self.features.total_time,
                "task_types": self.features.task_types,
                "success_rate": self.features.success_rate,
                "avg_score": self.features.avg_score,
                "models_used": self.features.models_used,
            },
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ProjectEmbedding:
        features = ProjectFeatures(
            project_id=data["features"]["project_id"],
            tech_stack=data["features"]["tech_stack"],
            total_tasks=data["features"]["total_tasks"],
            avg_task_iterations=data["features"]["avg_task_iterations"],
            total_cost=data["features"]["total_cost"],
            total_time=data["features"]["total_time"],
            task_types=data["features"]["task_types"],
            success_rate=data["features"]["success_rate"],
            avg_score=data["features"]["avg_score"],
            models_used=data["features"]["models_used"],
        )

        return cls(
            project_id=data["project_id"],
            embedding=data["embedding"],
            features=features,
            created_at=data.get("created_at", time.time()),
        )


@dataclass
class TransferPattern:
    """A pattern that can transfer between projects."""
    pattern_id: str
    pattern_type: PatternType
    source_projects: list[str]
    pattern_data: dict[str, Any]

    # Performance tracking
    success_count: int = 0
    transfer_count: int = 0
    confidence: float = 0.5

    # Metadata
    created_at: float = field(default_factory=time.time)
    last_applied_at: float | None = None
    status: TransferStatus = TransferStatus.ACTIVE

    # Description
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type.value,
            "source_projects": self.source_projects,
            "pattern_data": self.pattern_data,
            "success_count": self.success_count,
            "transfer_count": self.transfer_count,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "last_applied_at": self.last_applied_at,
            "status": self.status.value,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TransferPattern:
        return cls(
            pattern_id=data["pattern_id"],
            pattern_type=PatternType(data["pattern_type"]),
            source_projects=data["source_projects"],
            pattern_data=data["pattern_data"],
            success_count=data.get("success_count", 0),
            transfer_count=data.get("transfer_count", 0),
            confidence=data.get("confidence", 0.5),
            created_at=data.get("created_at", time.time()),
            last_applied_at=data.get("last_applied_at"),
            status=TransferStatus(data.get("status", "active")),
            description=data.get("description", ""),
        )

    def record_success(self):
        """Record successful transfer."""
        self.success_count += 1
        self.transfer_count += 1
        self.last_applied_at = time.time()
        # Update confidence using EMA
        self.confidence = 0.9 * self.confidence + 0.1 * 1.0

    def record_failure(self):
        """Record failed transfer."""
        self.transfer_count += 1
        # Update confidence using EMA
        self.confidence = 0.9 * self.confidence + 0.1 * 0.0


@dataclass
class TransferValidation:
    """Result of transfer validation."""
    is_valid: bool
    confidence: float
    risks: list[str]
    recommendations: list[str]

    def to_dict(self) -> dict:
        return {
            "is_valid": self.is_valid,
            "confidence": self.confidence,
            "risks": self.risks,
            "recommendations": self.recommendations,
        }


# ─────────────────────────────────────────────
# Project Embedder
# ─────────────────────────────────────────────

class ProjectEmbedder:
    """
    Generate embeddings for projects based on characteristics.

    Converts project trajectories into normalized feature vectors
    suitable for similarity computation.
    """

    def __init__(self):
        self._tech_keywords = [
            "python", "javascript", "typescript", "react", "vue", "angular",
            "fastapi", "django", "flask", "express", "nextjs",
            "postgresql", "mongodb", "redis", "mysql",
            "docker", "kubernetes", "aws", "azure", "gcp",
        ]

    def embed(self, trajectory: ProjectTrajectory) -> ProjectEmbedding:
        """
        Generate embedding vector from project characteristics.

        Args:
            trajectory: Project trajectory to embed

        Returns:
            Project embedding with features and vector
        """
        features = self._extract_features(trajectory)
        vector = features.to_vector()

        return ProjectEmbedding(
            project_id=trajectory.project_id,
            embedding=vector,
            features=features,
        )

    def _extract_features(self, trajectory: ProjectTrajectory) -> ProjectFeatures:
        """Extract features from project trajectory."""
        features = ProjectFeatures(project_id=trajectory.project_id)

        # Extract from project description
        desc_lower = trajectory.project_description.lower()
        for keyword in self._tech_keywords:
            features.tech_stack[keyword] = keyword in desc_lower

        # Extract from task records
        if trajectory.task_records:
            features.total_tasks = len(trajectory.task_records)

            # Task type distribution
            for record in trajectory.task_records:
                task_type = record.task_type
                features.task_types[task_type] = features.task_types.get(task_type, 0) + 1

                # Model usage
                model = record.model_used
                features.models_used[model] = features.models_used.get(model, 0) + 1

            # Success metrics
            successes = sum(1 for r in trajectory.task_records if r.success)
            features.success_rate = successes / len(trajectory.task_records)
            features.avg_score = sum(r.score for r in trajectory.task_records) / len(trajectory.task_records)

            # Complexity metrics
            features.total_cost = trajectory.total_cost
            features.total_time = trajectory.total_time

        return features


# ─────────────────────────────────────────────
# Similarity Engine
# ─────────────────────────────────────────────

class SimilarityEngine:
    """
    Compute similarity between projects for transfer learning.

    Uses cosine similarity on project embeddings.
    """

    def __init__(self, min_similarity: float = DEFAULT_MIN_SIMILARITY):
        self.min_similarity = min_similarity
        self._embeddings: dict[str, ProjectEmbedding] = {}

    def add_embedding(self, embedding: ProjectEmbedding):
        """Add embedding to index."""
        self._embeddings[embedding.project_id] = embedding

    def find_similar(
        self,
        project_id: str,
        min_similarity: float | None = None,
        limit: int = 10,
    ) -> list[tuple[str, float]]:
        """
        Find projects similar to the given project.

        Args:
            project_id: Project to find similar projects for
            min_similarity: Minimum similarity threshold
            limit: Maximum number of results

        Returns:
            List of (project_id, similarity) tuples
        """
        if project_id not in self._embeddings:
            return []

        threshold = min_similarity or self.min_similarity
        target = self._embeddings[project_id]

        similarities = []
        for other_id, other in self._embeddings.items():
            if other_id == project_id:
                continue

            sim = self._cosine_similarity(target.embedding, other.embedding)
            if sim >= threshold:
                similarities.append((other_id, sim))

        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:limit]

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if not vec1 or not vec2:
            return 0.0

        # Dot product
        dot = sum(a * b for a, b in zip(vec1, vec2, strict=False))

        # Magnitudes
        mag1 = math.sqrt(sum(a * a for a in vec1))
        mag2 = math.sqrt(sum(b * b for b in vec2))

        if mag1 == 0 or mag2 == 0:
            return 0.0

        return dot / (mag1 * mag2)


# ─────────────────────────────────────────────
# Pattern Miner
# ─────────────────────────────────────────────

class PatternMiner:
    """
    Mine transferable patterns from project history.

    Identifies successful strategies that appear across multiple projects.
    """

    def __init__(self, archive: ExecutionArchive):
        self.archive = archive
        self._patterns: dict[str, TransferPattern] = {}

    def mine_patterns(self) -> list[TransferPattern]:
        """
        Mine patterns from archive.

        Returns:
            List of discovered patterns
        """
        patterns = []

        # Mine model routing patterns
        patterns.extend(self._mine_routing_patterns())

        # Mine budget patterns
        patterns.extend(self._mine_budget_patterns())

        # Mine template patterns
        patterns.extend(self._mine_template_patterns())

        # Store patterns
        for pattern in patterns:
            self._patterns[pattern.pattern_id] = pattern

        return patterns

    def _mine_routing_patterns(self) -> list[TransferPattern]:
        """Mine model routing patterns."""
        patterns = []

        # Group by task type
        task_type_models: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for record in self.archive._records:
            if record.success:
                task_type_models[record.task_type][record.model_used].append(record.score)

        # Find best model per task type
        for task_type, model_scores in task_type_models.items():
            best_model = None
            best_avg_score = 0.0

            for model, scores in model_scores.items():
                if len(scores) >= MIN_SUCCESS_COUNT:
                    avg_score = sum(scores) / len(scores)
                    if avg_score > best_avg_score:
                        best_avg_score = avg_score
                        best_model = model

            if best_model:
                pattern = TransferPattern(
                    pattern_id=f"routing_{task_type}_{int(time.time())}",
                    pattern_type=PatternType.MODEL_ROUTING,
                    source_projects=[],  # Would populate from archive
                    pattern_data={
                        "task_type": task_type,
                        "recommended_model": best_model,
                        "avg_score": best_avg_score,
                    },
                    description=f"Use {best_model} for {task_type} tasks",
                )
                patterns.append(pattern)

        return patterns

    def _mine_budget_patterns(self) -> list[TransferPattern]:
        """Mine budget allocation patterns."""
        patterns = []

        # Analyze cost by task type
        task_type_costs: dict[str, list[float]] = defaultdict(list)

        for record in self.archive._records:
            task_type_costs[record.task_type].append(record.cost_usd)

        # Find optimal budget factors
        for task_type, costs in task_type_costs.items():
            if len(costs) >= MIN_SUCCESS_COUNT:
                avg_cost = sum(costs) / len(costs)
                p90_cost = sorted(costs)[int(len(costs) * 0.9)]

                pattern = TransferPattern(
                    pattern_id=f"budget_{task_type}_{int(time.time())}",
                    pattern_type=PatternType.BUDGET_ALLOCATION,
                    source_projects=[],
                    pattern_data={
                        "task_type": task_type,
                        "avg_cost": avg_cost,
                        "p90_cost": p90_cost,
                        "recommended_factor": p90_cost / max(avg_cost, 0.001),
                    },
                    description=f"Budget factor for {task_type}: {p90_cost/avg_cost:.2f}x average",
                )
                patterns.append(pattern)

        return patterns

    def _mine_template_patterns(self) -> list[TransferPattern]:
        """Mine template configuration patterns."""
        # Would analyze successful template variants
        # For now, return empty
        return []

    def get_pattern(self, pattern_id: str) -> TransferPattern | None:
        """Get pattern by ID."""
        return self._patterns.get(pattern_id)


# ─────────────────────────────────────────────
# Transfer Validator
# ─────────────────────────────────────────────

class TransferValidator:
    """
    Validate if a pattern transfer is appropriate.

    Checks compatibility between source pattern and target context.
    """

    def validate_transfer(
        self,
        pattern: TransferPattern,
        target_context: dict[str, Any],
    ) -> TransferValidation:
        """
        Validate if pattern can transfer to target context.

        Args:
            pattern: Pattern to validate
            target_context: Target project context

        Returns:
            Validation result with confidence and risks
        """
        risks = []
        recommendations = []

        # Check pattern confidence
        if pattern.confidence < DEFAULT_TRANSFER_CONFIDENCE:
            risks.append(f"Low pattern confidence: {pattern.confidence:.2f}")

        # Check pattern type-specific validation
        if pattern.pattern_type == PatternType.MODEL_ROUTING:
            validation = self._validate_routing_pattern(pattern, target_context)
            risks.extend(validation["risks"])
            recommendations.extend(validation["recommendations"])

        elif pattern.pattern_type == PatternType.BUDGET_ALLOCATION:
            validation = self._validate_budget_pattern(pattern, target_context)
            risks.extend(validation["risks"])
            recommendations.extend(validation["recommendations"])

        # Calculate overall confidence
        confidence = pattern.confidence
        if risks:
            confidence *= (1.0 - 0.1 * len(risks))

        is_valid = len(risks) == 0 or (len(risks) == 1 and confidence >= 0.7)

        return TransferValidation(
            is_valid=is_valid,
            confidence=confidence,
            risks=risks,
            recommendations=recommendations,
        )

    def _validate_routing_pattern(
        self,
        pattern: TransferPattern,
        target_context: dict[str, Any],
    ) -> dict[str, Any]:
        """Validate model routing pattern."""
        risks = []
        recommendations = []

        task_type = pattern.pattern_data.get("task_type")
        pattern.pattern_data.get("recommended_model")

        # Check if task type matches
        if target_context.get("task_type") != task_type:
            risks.append(f"Task type mismatch: expected {task_type}")

        # Check if model is available
        # (would check against available models in production)

        return {"risks": risks, "recommendations": recommendations}

    def _validate_budget_pattern(
        self,
        pattern: TransferPattern,
        target_context: dict[str, Any],
    ) -> dict[str, Any]:
        """Validate budget allocation pattern."""
        risks = []
        recommendations = []

        pattern.pattern_data.get("task_type")
        recommended_factor = pattern.pattern_data.get("recommended_factor", 1.0)

        # Check if factor is reasonable
        if recommended_factor > 3.0:
            risks.append(f"High budget factor: {recommended_factor:.2f}x")
            recommendations.append("Consider gradual budget increase")

        return {"risks": risks, "recommendations": recommendations}


# ─────────────────────────────────────────────
# Transfer Learning Engine
# ─────────────────────────────────────────────

class TransferLearningEngine:
    """
    Main transfer learning orchestrator.

    Coordinates embedding, similarity search, pattern mining, and validation.
    """

    def __init__(
        self,
        archive: ExecutionArchive,
        storage_path: Path | None = None,
    ):
        self.archive = archive

        self._storage_path = storage_path or (
            Path.home() / ".orchestrator_cache" / "transfer_learning"
        )
        self._storage_path.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self._embedder = ProjectEmbedder()
        self._similarity_engine = SimilarityEngine()
        self._pattern_miner = PatternMiner(archive)
        self._validator = TransferValidator()

        # Load embeddings and patterns
        self._embeddings: dict[str, ProjectEmbedding] = {}
        self._patterns: dict[str, TransferPattern] = {}

        self._load_state()

    def _load_state(self):
        """Load embeddings and patterns from disk."""
        embeddings_file = self._storage_path / "embeddings.jsonl"
        patterns_file = self._storage_path / "patterns.jsonl"

        # Load embeddings
        if embeddings_file.exists():
            try:
                with open(embeddings_file) as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            embedding = ProjectEmbedding.from_dict(data)
                            self._embeddings[embedding.project_id] = embedding
                            self._similarity_engine.add_embedding(embedding)

                logger.info(f"Loaded {len(self._embeddings)} embeddings")
            except Exception as e:
                logger.warning(f"Failed to load embeddings: {e}")

        # Load patterns
        if patterns_file.exists():
            try:
                with open(patterns_file) as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            pattern = TransferPattern.from_dict(data)
                            self._patterns[pattern.pattern_id] = pattern

                logger.info(f"Loaded {len(self._patterns)} patterns")
            except Exception as e:
                logger.warning(f"Failed to load patterns: {e}")

    def _save_state(self):
        """Save embeddings and patterns to disk."""
        embeddings_file = self._storage_path / "embeddings.jsonl"
        patterns_file = self._storage_path / "patterns.jsonl"

        # Save embeddings
        with open(embeddings_file, "w") as f:
            for embedding in self._embeddings.values():
                f.write(json.dumps(embedding.to_dict()) + "\n")

        # Save patterns
        with open(patterns_file, "w") as f:
            for pattern in self._patterns.values():
                f.write(json.dumps(pattern.to_dict()) + "\n")

    async def index_project(self, trajectory: ProjectTrajectory):
        """
        Index a project for transfer learning.

        Args:
            trajectory: Project trajectory to index
        """
        # Generate embedding
        embedding = self._embedder.embed(trajectory)
        self._embeddings[trajectory.project_id] = embedding
        self._similarity_engine.add_embedding(embedding)

        # Mine new patterns
        new_patterns = self._pattern_miner.mine_patterns()
        for pattern in new_patterns:
            if pattern.pattern_id not in self._patterns:
                self._patterns[pattern.pattern_id] = pattern

        # Save state
        self._save_state()

        logger.debug(f"Indexed project {trajectory.project_id}")

    async def find_transferable_patterns(
        self,
        current_project_id: str,
        min_similarity: float = DEFAULT_MIN_SIMILARITY,
    ) -> list[TransferPattern]:
        """
        Find patterns transferable to current project.

        Args:
            current_project_id: Current project to find patterns for
            min_similarity: Minimum similarity threshold

        Returns:
            List of transferable patterns
        """
        # Find similar projects
        similar_projects = self._similarity_engine.find_similar(
            current_project_id,
            min_similarity=min_similarity,
        )

        if not similar_projects:
            logger.info(f"No similar projects found for {current_project_id}")
            return []

        # Collect patterns from similar projects
        candidate_patterns = []
        for pattern in self._patterns.values():
            if pattern.status == TransferStatus.ACTIVE:
                candidate_patterns.append(pattern)

        logger.info(f"Found {len(candidate_patterns)} candidate patterns")
        return candidate_patterns

    async def validate_transfer(
        self,
        pattern: TransferPattern,
        target_context: dict[str, Any],
    ) -> TransferValidation:
        """
        Validate if pattern can transfer to target.

        Args:
            pattern: Pattern to validate
            target_context: Target context

        Returns:
            Validation result
        """
        return self._validator.validate_transfer(pattern, target_context)

    async def apply_pattern(
        self,
        pattern: TransferPattern,
        target_context: dict[str, Any],
    ) -> StrategyProposal | None:
        """
        Apply a pattern to generate a strategy proposal.

        Args:
            pattern: Pattern to apply
            target_context: Target context

        Returns:
            Strategy proposal, or None if pattern doesn't apply
        """
        # Validate first
        validation = await self.validate_transfer(pattern, target_context)

        if not validation.is_valid:
            logger.warning(
                f"Pattern {pattern.pattern_id} validation failed: "
                f"{validation.risks}"
            )
            return None

        # Generate proposal based on pattern type
        if pattern.pattern_type == PatternType.MODEL_ROUTING:
            return self._create_routing_proposal(pattern, target_context)
        elif pattern.pattern_type == PatternType.BUDGET_ALLOCATION:
            return self._create_budget_proposal(pattern, target_context)

        return None

    def _create_routing_proposal(
        self,
        pattern: TransferPattern,
        target_context: dict[str, Any],
    ) -> StrategyProposal:
        """Create model routing proposal from pattern."""
        task_type = pattern.pattern_data.get("task_type")
        recommended_model = pattern.pattern_data.get("recommended_model")

        return StrategyProposal(
            proposal_id=f"transfer_{pattern.pattern_id}_{int(time.time())}",
            strategy_type=StrategyType.MODEL_ROUTING,
            description=f"Transfer: {pattern.description}",
            current_config={
                "task_type": task_type,
                "routing": "adaptive",
            },
            proposed_config={
                "task_type": task_type,
                "routing": "fixed",
                "model": recommended_model,
            },
            expected_improvement=pattern.confidence * 0.1,  # Up to 10%
            confidence=pattern.confidence,
            evidence=[
                f"Pattern succeeded {pattern.success_count} times",
                f"Transferred {pattern.transfer_count} times",
            ],
        )

    def _create_budget_proposal(
        self,
        pattern: TransferPattern,
        target_context: dict[str, Any],
    ) -> StrategyProposal:
        """Create budget allocation proposal from pattern."""
        task_type = pattern.pattern_data.get("task_type")
        recommended_factor = pattern.pattern_data.get("recommended_factor", 1.0)

        return StrategyProposal(
            proposal_id=f"transfer_{pattern.pattern_id}_{int(time.time())}",
            strategy_type=StrategyType.BUDGET_ALLOCATION,
            description=f"Transfer: {pattern.description}",
            current_config={
                "task_type": task_type,
                "budget_factor": 1.0,
            },
            proposed_config={
                "task_type": task_type,
                "budget_factor": recommended_factor,
            },
            expected_improvement=pattern.confidence * 0.05,  # Up to 5%
            confidence=pattern.confidence,
            evidence=[
                f"Pattern succeeded {pattern.success_count} times",
                f"P90 cost factor: {recommended_factor:.2f}x",
            ],
        )

    def get_stats(self) -> dict[str, Any]:
        """Get transfer learning statistics."""
        return {
            "indexed_projects": len(self._embeddings),
            "active_patterns": len([
                p for p in self._patterns.values()
                if p.status == TransferStatus.ACTIVE
            ]),
            "total_patterns": len(self._patterns),
            "avg_pattern_confidence": (
                sum(p.confidence for p in self._patterns.values()) /
                max(len(self._patterns), 1)
            ),
        }


# ─────────────────────────────────────────────
# Convenience Functions
# ─────────────────────────────────────────────

_transfer_engine: TransferLearningEngine | None = None


def get_transfer_engine() -> TransferLearningEngine | None:
    """Get global transfer engine instance."""
    return _transfer_engine


def initialize_transfer_engine(
    archive: ExecutionArchive,
    storage_path: Path | None = None,
) -> TransferLearningEngine:
    """Initialize global transfer engine."""
    global _transfer_engine
    _transfer_engine = TransferLearningEngine(archive, storage_path)
    return _transfer_engine


def reset_transfer_engine() -> None:
    """Reset global transfer engine (for testing)."""
    global _transfer_engine
    _transfer_engine = None
