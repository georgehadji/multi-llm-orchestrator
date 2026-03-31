"""
Outcome-Weighted Router with Production Feedback
================================================

Routes tasks to models based on proven production outcomes, not just
cost/latency estimates. Creates a learning system that improves over time.

Key Features:
- Production-weighted model scoring
- Codebase-specific routing preferences
- Hybrid routing (blends multiple strategies)
- Feedback loop integration
- Nash-stable decision making

Usage:
    from orchestrator.outcome_router import OutcomeWeightedRouter

    router = OutcomeWeightedRouter()
    model = await router.select_model(
        task=my_task,
        codebase_fingerprint=my_fingerprint,
    )
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from .adaptive_router import AdaptiveRouter, ModelState
from .feedback_loop import CodebaseFingerprint, FeedbackLoop
from .leaderboard import ModelLeaderboard, get_leaderboard
from .log_config import get_logger
from .models import COST_TABLE, ROUTING_TABLE, Model, Task, TaskType
from .plugins import get_plugin_registry

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Routing Strategies
# ═══════════════════════════════════════════════════════════════════════════════


class RoutingStrategy(Enum):
    """Available routing strategies."""

    COST_OPTIMIZED = "cost_optimized"  # Minimize cost
    QUALITY_OPTIMIZED = "quality_optimized"  # Maximize quality
    BALANCED = "balanced"  # Balance cost/quality
    PRODUCTION_WEIGHTED = "production_weighted"  # Use production outcomes
    CODEBASE_SPECIFIC = "codebase_specific"  # Match similar codebases
    EXPLORATION = "exploration"  # Try under-sampled models


@dataclass
class ModelScore:
    """Comprehensive scoring for a model."""

    model: Model

    # Base scores (0.0 - 1.0)
    cost_score: float = 0.5
    quality_score: float = 0.5
    latency_score: float = 0.5
    reliability_score: float = 0.5

    # Production-weighted scores
    production_score: float = 0.5
    codebase_specific_score: float = 0.5

    # Metadata
    confidence: float = 1.0  # How confident we are in these scores
    sample_size: int = 0  # Number of production samples

    @property
    def composite_score(self) -> float:
        """
        Calculate composite score using Nash-stable weighting.

        Weight distribution optimized for long-term stability:
        - Production outcomes: 40% (learned from real usage)
        - Quality estimate: 25% (benchmark-based)
        - Cost efficiency: 20% (budget optimization)
        - Reliability: 15% (consistency)
        """
        # Weight production score more heavily when we have data
        production_weight = min(0.4, 0.1 + (self.sample_size / 100) * 0.3)

        quality_weight = 0.25
        cost_weight = 0.20
        reliability_weight = 0.15

        # Adjust remaining weight
        remaining = 1.0 - production_weight
        total_other = quality_weight + cost_weight + reliability_weight

        quality_weight = quality_weight / total_other * remaining
        cost_weight = cost_weight / total_other * remaining
        reliability_weight = reliability_weight / total_other * remaining

        return (
            production_weight * self.production_score
            + quality_weight * self.quality_score
            + cost_weight * self.cost_score
            + reliability_weight * self.reliability_score
        )


@dataclass
class RoutingContext:
    """Context for routing decisions."""

    task: Task
    task_type: TaskType
    budget_remaining: float
    budget_total: float
    codebase_fingerprint: CodebaseFingerprint | None = None
    strategy: RoutingStrategy = RoutingStrategy.BALANCED
    previous_model: Model | None = None  # For fallback
    required_capabilities: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# Outcome-Weighted Router
# ═══════════════════════════════════════════════════════════════════════════════


class OutcomeWeightedRouter:
    """
    Production-outcome-weighted model router.

    This router creates Nash stability by:
    1. Learning from actual production outcomes (not just estimates)
    2. Building codebase-specific preferences
    3. Balancing exploration vs exploitation
    4. Providing increasing returns to scale
    """

    # Exploration parameters
    EXPLORATION_RATE = 0.15  # 15% of traffic to exploration
    MIN_SAMPLES_FOR_CONFIDENCE = 10

    def __init__(
        self,
        feedback_loop: FeedbackLoop | None = None,
        leaderboard: ModelLeaderboard | None = None,
        adaptive_router: AdaptiveRouter | None = None,
    ):
        self.feedback = feedback_loop or FeedbackLoop()
        self.leaderboard = leaderboard or get_leaderboard()
        self.adaptive = adaptive_router or AdaptiveRouter()

        # Caches
        self._score_cache: dict[tuple[Model, TaskType], ModelScore] = {}
        self._last_cache_update = datetime.min
        self._cache_ttl = timedelta(minutes=5)

    async def select_model(
        self,
        context: RoutingContext,
    ) -> tuple[Model, dict[str, Any]]:
        """
        Select the best model for a task.

        Returns:
            (selected_model, decision_metadata)
        """
        # Get candidates from routing table
        candidates = ROUTING_TABLE.get(context.task_type, list(Model))

        # Filter by adaptive router health
        healthy_candidates = [m for m in candidates if self.adaptive.is_available(m)]

        if not healthy_candidates:
            logger.warning("No healthy models available, using fallback")
            healthy_candidates = candidates  # Use all as fallback

        # Score all candidates
        scores = {model: self._score_model(model, context) for model in healthy_candidates}

        # Apply strategy-specific adjustments
        scores = self._apply_strategy(scores, context)

        # Check for exploration opportunity
        if random.random() < self.EXPLORATION_RATE:
            explore_model = self._select_exploration_candidate(scores, context)
            if explore_model:
                logger.info(f"Exploration: trying {explore_model.value}")
                return explore_model, {
                    "strategy": "exploration",
                    "scores": {m.value: s.composite_score for m, s in scores.items()},
                }

        # Run plugin routers
        plugin_suggestions = await self._get_plugin_suggestions(context, healthy_candidates)
        scores = self._blend_plugin_suggestions(scores, plugin_suggestions)

        # Select best model
        best_model = max(scores.keys(), key=lambda m: scores[m].composite_score)
        best_score = scores[best_model]

        # Log decision
        logger.info(
            f"Selected {best_model.value} for {context.task_type.value} "
            f"(score: {best_score.composite_score:.3f}, "
            f"production: {best_score.production_score:.3f}, "
            f"samples: {best_score.sample_size})"
        )

        return best_model, {
            "strategy": context.strategy.value,
            "composite_score": best_score.composite_score,
            "production_score": best_score.production_score,
            "confidence": best_score.confidence,
            "all_scores": {
                m.value: {
                    "composite": s.composite_score,
                    "production": s.production_score,
                    "quality": s.quality_score,
                }
                for m, s in scores.items()
            },
        }

    def _score_model(self, model: Model, context: RoutingContext) -> ModelScore:
        """Calculate comprehensive score for a model."""
        cache_key = (model, context.task_type)

        # Check cache
        if cache_key in self._score_cache:
            if datetime.utcnow() - self._last_cache_update < self._cache_ttl:
                return self._score_cache[cache_key]

        score = ModelScore(model=model)

        # 1. Production score from feedback loop
        production_score, sample_size = self._get_production_score(model, context)
        score.production_score = production_score
        score.sample_size = sample_size
        score.confidence = min(1.0, sample_size / self.MIN_SAMPLES_FOR_CONFIDENCE)

        # 2. Codebase-specific score
        if context.codebase_fingerprint:
            score.codebase_specific_score = self.feedback.get_model_score(
                model,
                context.task_type,
                context.codebase_fingerprint,
            )

        # 3. Cost score (inverse of cost, normalized)
        score.cost_score = self._calculate_cost_score(model, context)

        # 4. Quality score from leaderboard
        score.quality_score = self._get_quality_score(model, context)

        # 5. Latency score from adaptive router
        score.latency_score = self._get_latency_score(model)

        # 6. Reliability score
        score.reliability_score = self._get_reliability_score(model, context)

        # Cache
        self._score_cache[cache_key] = score
        self._last_cache_update = datetime.utcnow()

        return score

    def _get_production_score(
        self,
        model: Model,
        context: RoutingContext,
    ) -> tuple[float, int]:
        """Get production-weighted score for a model."""
        # Try codebase-specific first
        if context.codebase_fingerprint:
            score = self.feedback.get_model_score(
                model,
                context.task_type,
                context.codebase_fingerprint,
            )
            # If we have codebase-specific data, use it
            if score != 0.5:  # Non-default score
                # Get sample size estimate
                fp_hash = self._hash_fingerprint(context.codebase_fingerprint)
                outcomes = self.feedback._codebase_outcomes.get(fp_hash, [])
                relevant = [
                    o
                    for o in outcomes
                    if o.model_used == model and o.task_type == context.task_type
                ]
                return score, len(relevant)

        # Fall back to global score
        score = self.feedback.get_model_score(model, context.task_type)

        # Estimate sample size from performance records
        record = self.feedback._performance_records.get((model, context.task_type))
        sample_size = record.total_deployments if record else 0

        return score, sample_size

    def _hash_fingerprint(self, fingerprint: CodebaseFingerprint) -> str:
        """Hash a codebase fingerprint."""
        import hashlib
        import json

        data = json.dumps(
            {
                "languages": sorted(fingerprint.languages),
                "framework": fingerprint.framework,
                "patterns": sorted(fingerprint.patterns),
            },
            sort_keys=True,
        )
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def _calculate_cost_score(self, model: Model, context: RoutingContext) -> float:
        """Calculate cost efficiency score."""
        costs = COST_TABLE.get(model, {"input": 1.0, "output": 1.0})
        avg_cost = (costs["input"] + costs["output"]) / 2

        # Find max cost for normalization
        max_cost = (
            max((c["input"] + c["output"]) / 2 for c in COST_TABLE.values()) if COST_TABLE else 10.0
        )

        # Inverse score (lower cost = higher score)
        score = 1.0 - (avg_cost / max_cost)

        # Adjust for budget constraints
        if context.budget_total > 0:
            budget_pct = context.budget_remaining / context.budget_total
            if budget_pct < 0.2:  # Low budget - prefer cheaper models
                score *= 2.0 if avg_cost < 1.0 else 0.5

        return max(0.0, min(1.0, score))

    def _get_quality_score(self, model: Model, context: RoutingContext) -> float:
        """Get quality score from leaderboard."""
        summary = self.leaderboard._summaries.get(model)
        if summary:
            type_scores = summary.by_task_type.get(context.task_type, {})
            return type_scores.get("avg_quality", summary.avg_quality)

        # Default to neutral if no benchmark data
        return 0.5

    def _get_latency_score(self, model: Model) -> float:
        """Get latency score from adaptive router."""
        # Lower latency = higher score
        # Assume good latency < 2000ms
        default_latency = 2000.0
        latency = self.adaptive._latencies.get(model, default_latency)
        return max(0.0, 1.0 - (latency / 5000))

    def _get_reliability_score(self, model: Model, context: RoutingContext) -> float:
        """Get reliability score based on failure rates."""
        record = self.feedback._performance_records.get((model, context.task_type))
        if record and record.total_deployments > 0:
            return record.success_rate

        # Default: check adaptive router state
        state = self.adaptive.get_state(model)
        if state == ModelState.DISABLED:
            return 0.0
        elif state == ModelState.DEGRADED:
            return 0.3

        return 0.7  # Unknown but available

    def _apply_strategy(
        self,
        scores: dict[Model, ModelScore],
        context: RoutingContext,
    ) -> dict[Model, ModelScore]:
        """Apply strategy-specific adjustments to scores."""
        if context.strategy == RoutingStrategy.COST_OPTIMIZED:
            for score in scores.values():
                score.cost_score = min(1.0, score.cost_score * 1.5)

        elif context.strategy == RoutingStrategy.QUALITY_OPTIMIZED:
            for score in scores.values():
                score.quality_score = min(1.0, score.quality_score * 1.5)

        elif context.strategy == RoutingStrategy.PRODUCTION_WEIGHTED:
            for score in scores.values():
                # Boost production score significantly
                score.production_score = min(1.0, score.production_score * 1.3)

        elif context.strategy == RoutingStrategy.EXPLORATION:
            # Will be handled separately
            pass

        return scores

    def _select_exploration_candidate(
        self,
        scores: dict[Model, ModelScore],
        context: RoutingContext,
    ) -> Model | None:
        """Select a model for exploration."""
        # Find models with few samples but decent potential
        candidates = [
            (model, score)
            for model, score in scores.items()
            if score.sample_size < self.MIN_SAMPLES_FOR_CONFIDENCE
            and score.quality_score > 0.4  # Minimum quality threshold
        ]

        if not candidates:
            return None

        # Weight by inverse sample size (prefer less-sampled models)
        weights = [1.0 / (1 + s.sample_size) for _, s in candidates]
        total_weight = sum(weights)

        if total_weight == 0:
            return None

        # Weighted random selection
        r = random.uniform(0, total_weight)
        cumulative = 0
        for (model, _), weight in zip(candidates, weights, strict=False):
            cumulative += weight
            if r <= cumulative:
                return model

        return candidates[-1][0]

    async def _get_plugin_suggestions(
        self,
        context: RoutingContext,
        candidates: list[Model],
    ) -> list[tuple[Model, float, float]]:  # (model, score, weight)
        """Get suggestions from router plugins."""
        registry = get_plugin_registry()
        plugins = registry.get_routers()

        suggestions = []

        for plugin in plugins:
            try:
                plugin_suggestions = plugin.suggest_models(
                    context.task,
                    candidates,
                    {"strategy": context.strategy.value},
                )

                for sugg in plugin_suggestions:
                    suggestions.append((sugg.model, sugg.confidence, plugin.get_weight()))

            except Exception as e:
                logger.error(f"Plugin {plugin.metadata.name} failed: {e}")

        return suggestions

    def _blend_plugin_suggestions(
        self,
        scores: dict[Model, ModelScore],
        suggestions: list[tuple[Model, float, float]],
    ) -> dict[Model, ModelScore]:
        """Blend plugin suggestions into scores."""
        if not suggestions:
            return scores

        # Group by model
        by_model: dict[Model, list[tuple[float, float]]] = {}
        for model, score, weight in suggestions:
            if model not in by_model:
                by_model[model] = []
            by_model[model].append((score, weight))

        # Blend into existing scores
        for model, sugg_list in by_model.items():
            if model not in scores:
                continue

            # Weighted average of suggestions
            total_weight = sum(w for _, w in sugg_list)
            if total_weight == 0:
                continue

            avg_sugg_score = sum(s * w for s, w in sugg_list) / total_weight

            # Blend with existing production score
            score = scores[model]
            score.production_score = 0.8 * score.production_score + 0.2 * avg_sugg_score

        return scores

    def get_nash_stability_report(self) -> dict[str, Any]:
        """
        Generate a report on Nash stability of current routing.

        This helps understand why users shouldn't switch to competitors.
        """
        total_samples = sum(
            r.total_deployments for r in self.feedback._performance_records.values()
        )

        codebase_count = len(self.feedback._codebase_outcomes)

        # Calculate information advantage
        model_scores = {}
        for (model, task_type), record in self.feedback._performance_records.items():
            if record.total_deployments > 0:
                key = f"{model.value}:{task_type.value}"
                model_scores[key] = {
                    "deployments": record.total_deployments,
                    "success_rate": record.success_rate,
                    "confidence": min(1.0, record.total_deployments / 50),
                }

        return {
            "total_production_samples": total_samples,
            "unique_codebases_learned": codebase_count,
            "model_task_combinations": len(model_scores),
            "information_advantage": {
                "description": "Competitors lack your specific production history",
                "switching_cost_estimate": f"~{total_samples} production samples would need re-learning",
                "codebase_specific_knowledge": codebase_count,
            },
            "top_performing_combinations": sorted(
                model_scores.items(),
                key=lambda x: x[1]["success_rate"],
                reverse=True,
            )[:10],
            "exploration_status": {
                "exploration_rate": self.EXPLORATION_RATE,
                "underexplored_models": [
                    m.value
                    for m in Model
                    if m not in {k[0] for k in self.feedback._performance_records}
                ],
            },
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience Functions
# ═══════════════════════════════════════════════════════════════════════════════

_router: OutcomeWeightedRouter | None = None


def get_outcome_router() -> OutcomeWeightedRouter:
    """Get global outcome-weighted router."""
    global _router
    if _router is None:
        _router = OutcomeWeightedRouter()
    return _router


def reset_outcome_router() -> None:
    """Reset global router (for testing)."""
    global _router
    _router = None


# Factory for creating routing context from task
def create_routing_context(
    task: Task,
    budget_remaining: float,
    budget_total: float,
    strategy: RoutingStrategy = RoutingStrategy.BALANCED,
    codebase_fingerprint: CodebaseFingerprint | None = None,
) -> RoutingContext:
    """Create a routing context for a task."""
    return RoutingContext(
        task=task,
        task_type=task.task_type,
        budget_remaining=budget_remaining,
        budget_total=budget_total,
        codebase_fingerprint=codebase_fingerprint,
        strategy=strategy,
    )
