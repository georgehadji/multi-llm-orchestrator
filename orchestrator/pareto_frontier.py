"""
Predictive Cost-Quality Frontier API
====================================

Pareto-optimal model recommendations with confidence intervals.
Provides multi-objective optimization for cost, quality, and latency.

Key Features:
- Pareto frontier computation with dominance checking
- Probabilistic forecasting with confidence intervals
- Multi-objective optimization (cost, quality, latency, reliability)
- Dominance-based ranking
- Uncertainty quantification

Usage:
    from orchestrator.pareto_frontier import CostQualityFrontier

    frontier = CostQualityFrontier()

    results = await frontier.get_pareto_frontier(
        task_type=TaskType.CODE_GEN,
        objectives=[Objective.QUALITY, Objective.COST, Objective.LATENCY],
        fingerprint=codebase_fingerprint,
    )

    # Returns ranked list with confidence intervals:
    # [
    #   {"model": "deepseek-chat", "quality": 0.85, "cost": 0.02,
    #    "confidence": 0.92, "ci_lower": 0.78, "ci_upper": 0.91},
    #   ...
    # ]
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any

from .feedback_loop import CodebaseFingerprint, FeedbackLoop
from .leaderboard import ModelLeaderboard, get_leaderboard
from .log_config import get_logger
from .models import COST_TABLE, Model, TaskType

logger = get_logger(__name__)


class Objective(Enum):
    """Optimization objectives."""
    COST = "cost"              # Minimize
    QUALITY = "quality"        # Maximize
    LATENCY = "latency"        # Minimize
    RELIABILITY = "reliability"  # Maximize
    EFFICIENCY = "efficiency"  # Maximize (quality per dollar)


class OptimizationDirection(Enum):
    """Direction of optimization."""
    MINIMIZE = auto()
    MAXIMIZE = auto()


OBJECTIVE_DIRECTIONS = {
    Objective.COST: OptimizationDirection.MINIMIZE,
    Objective.QUALITY: OptimizationDirection.MAXIMIZE,
    Objective.LATENCY: OptimizationDirection.MINIMIZE,
    Objective.RELIABILITY: OptimizationDirection.MAXIMIZE,
    Objective.EFFICIENCY: OptimizationDirection.MAXIMIZE,
}


@dataclass
class ModelPrediction:
    """Predicted performance for a model."""
    model: Model

    # Predicted values
    quality: float
    cost: float
    latency_ms: float
    reliability: float
    efficiency: float  # quality per dollar

    # Confidence intervals (95%)
    quality_ci: tuple[float, float] = (0, 1)
    cost_ci: tuple[float, float] = (0, 0)

    # Confidence and uncertainty
    confidence: float = 0.5
    uncertainty: float = 0.5

    # Evidence
    sample_size: int = 0
    evidence_sources: list[str] = field(default_factory=list)

    # Metadata
    prediction_timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.value,
            "quality": round(self.quality, 3),
            "cost": round(self.cost, 4),
            "latency_ms": round(self.latency_ms, 1),
            "reliability": round(self.reliability, 3),
            "efficiency": round(self.efficiency, 2),
            "confidence": round(self.confidence, 3),
            "uncertainty": round(self.uncertainty, 3),
            "quality_ci": [round(self.quality_ci[0], 3), round(self.quality_ci[1], 3)],
            "sample_size": self.sample_size,
            "evidence_sources": self.evidence_sources,
        }


@dataclass
class ParetoPoint:
    """A point on the Pareto frontier."""
    prediction: ModelPrediction

    # Objective values (normalized 0-1)
    objective_values: dict[Objective, float]

    # Dominance info
    dominated_by_count: int = 0
    dominates: list[str] = field(default_factory=list)

    # Pareto rank (0 = best frontier)
    pareto_rank: int = 0

    # Recommendation metadata
    is_recommended: bool = False
    recommendation_reason: str = ""


@dataclass
class FrontierRecommendation:
    """A recommendation from the frontier analysis."""
    model: Model
    rank: int

    # Predicted metrics
    quality: float
    cost: float
    latency_ms: float

    # Confidence
    confidence: float
    quality_ci_lower: float
    quality_ci_upper: float

    # Pareto info
    is_pareto_optimal: bool
    dominates_count: int

    # Trade-off analysis
    trade_offs: dict[str, Any]

    # Recommendation
    best_for: list[str]
    recommendation: str


class CostQualityFrontier:
    """
    Predictive cost-quality frontier with confidence intervals.

    Optimized for:
    - Fast Pareto frontier computation (incremental updates)
    - Accurate uncertainty quantification (Bayesian confidence)
    - Multi-objective trade-off analysis
    """

    # Configuration
    CONFIDENCE_LEVEL = 0.95
    MIN_SAMPLES_FOR_CI = 5
    MAX_HISTORY_AGE_DAYS = 30

    def __init__(
        self,
        feedback_loop: FeedbackLoop | None = None,
        leaderboard: ModelLeaderboard | None = None,
        storage_path: Path | None = None,
    ):
        self.storage_path = storage_path or Path(".pareto_frontier")
        self.storage_path.mkdir(exist_ok=True)

        self.feedback = feedback_loop or FeedbackLoop()
        self.leaderboard = leaderboard or get_leaderboard()

        # Cache for predictions
        self._prediction_cache: dict[str, tuple[ModelPrediction, datetime]] = {}
        self._cache_ttl = timedelta(minutes=5)

        # Historical predictions for trend analysis
        self._history: dict[str, list[ModelPrediction]] = defaultdict(list)

        self._load_history()

    def _load_history(self) -> None:
        """Load historical predictions."""
        history_file = self.storage_path / "history.json"
        if history_file.exists():
            try:
                data = json.loads(history_file.read_text())
                # Load last 30 days only
                cutoff = datetime.utcnow() - timedelta(days=30)

                for key, predictions in data.items():
                    for pred_data in predictions:
                        pred = ModelPrediction(
                            model=Model(pred_data["model"]),
                            quality=pred_data["quality"],
                            cost=pred_data["cost"],
                            latency_ms=pred_data["latency_ms"],
                            reliability=pred_data["reliability"],
                            efficiency=pred_data["efficiency"],
                            prediction_timestamp=datetime.fromisoformat(
                                pred_data["prediction_timestamp"]
                            ),
                        )
                        if pred.prediction_timestamp > cutoff:
                            self._history[key].append(pred)

                logger.info(f"Loaded prediction history for {len(self._history)} keys")
            except Exception as e:
                logger.error(f"Failed to load history: {e}")

    def _save_history(self) -> None:
        """Save historical predictions."""
        try:
            data = {}
            for key, predictions in self._history.items():
                data[key] = [{
                    "model": p.model.value,
                    "quality": p.quality,
                    "cost": p.cost,
                    "latency_ms": p.latency_ms,
                    "reliability": p.reliability,
                    "efficiency": p.efficiency,
                    "prediction_timestamp": p.prediction_timestamp.isoformat(),
                } for p in predictions[-100:]]  # Keep last 100 per key

            history_file = self.storage_path / "history.json"
            history_file.write_text(json.dumps(data, indent=2, default=str))
        except Exception as e:
            logger.error(f"Failed to save history: {e}")

    async def get_pareto_frontier(
        self,
        task_type: TaskType,
        objectives: list[Objective],
        fingerprint: CodebaseFingerprint | None = None,
        budget_constraint: float | None = None,
        min_confidence: float = 0.3,
    ) -> list[FrontierRecommendation]:
        """
        Get Pareto-optimal model recommendations.

        Args:
            task_type: Type of task
            objectives: List of objectives to optimize
            fingerprint: Optional codebase fingerprint for context
            budget_constraint: Optional max budget constraint
            min_confidence: Minimum confidence threshold

        Returns:
            Ranked list of frontier recommendations
        """
        # Generate predictions for all models
        predictions = await self._predict_all_models(
            task_type, fingerprint, objectives
        )

        # Filter by budget constraint
        if budget_constraint is not None:
            predictions = [p for p in predictions if p.cost <= budget_constraint]

        # Filter by confidence
        predictions = [p for p in predictions if p.confidence >= min_confidence]

        if not predictions:
            logger.warning("No models meet constraints")
            return []

        # Normalize objective values
        normalized = self._normalize_objectives(predictions, objectives)

        # Compute Pareto frontier
        pareto_points = self._compute_pareto_frontier(normalized, objectives)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            pareto_points, objectives, budget_constraint
        )

        # Cache and save
        cache_key = f"{task_type.value}:{fingerprint._hash_fingerprint() if fingerprint else 'none'}"
        for pred in predictions:
            self._history[cache_key].append(pred)
        self._save_history()

        return recommendations

    async def _predict_all_models(
        self,
        task_type: TaskType,
        fingerprint: CodebaseFingerprint | None,
        objectives: list[Objective],
    ) -> list[ModelPrediction]:
        """Generate predictions for all models."""
        predictions = []

        for model in Model:
            pred = await self._predict_model(model, task_type, fingerprint, objectives)
            if pred:
                predictions.append(pred)

        return predictions

    async def _predict_model(
        self,
        model: Model,
        task_type: TaskType,
        fingerprint: CodebaseFingerprint | None,
        objectives: list[Objective],
    ) -> ModelPrediction | None:
        """Generate prediction for a single model."""
        # Check cache
        cache_key = f"{model.value}:{task_type.value}:{fingerprint._hash_fingerprint() if fingerprint else 'none'}"
        if cache_key in self._prediction_cache:
            cached, timestamp = self._prediction_cache[cache_key]
            if datetime.utcnow() - timestamp < self._cache_ttl:
                return cached

        # Gather evidence
        evidence_sources = []
        samples = []

        # 1. Production feedback data
        feedback_score, feedback_samples = self._get_feedback_prediction(
            model, task_type, fingerprint
        )
        if feedback_samples > 0:
            evidence_sources.append("production_feedback")
            samples.append(feedback_samples)

        # 2. Benchmark data
        benchmark_score, benchmark_samples = self._get_benchmark_prediction(
            model, task_type
        )
        if benchmark_samples > 0:
            evidence_sources.append("benchmarks")
            samples.append(benchmark_samples)

        # 3. Cost table
        cost = self._estimate_cost(model)

        # 4. Latency estimate
        latency = self._estimate_latency(model)

        # Combine predictions (weighted by sample size)
        total_samples = sum(samples) if samples else 0

        if total_samples == 0:
            # No data - use defaults with high uncertainty
            quality = 0.5
            confidence = 0.1
            uncertainty = 0.5
        else:
            # Weighted combination
            weights = [s / total_samples for s in samples]
            quality = 0.0

            if feedback_samples > 0:
                quality += weights[0] * feedback_score
            if benchmark_samples > 0:
                idx = 1 if feedback_samples > 0 else 0
                quality += weights[idx] * benchmark_score

            # Confidence based on sample size
            confidence = min(1.0, total_samples / 30)
            uncertainty = 1.0 / math.sqrt(1 + total_samples)

        # Calculate efficiency
        efficiency = quality / cost if cost > 0 else quality * 100

        # Calculate reliability
        reliability = self._estimate_reliability(model, task_type, total_samples)

        # Calculate confidence intervals
        quality_ci = self._calculate_confidence_interval(
            quality, total_samples, uncertainty
        )

        prediction = ModelPrediction(
            model=model,
            quality=quality,
            cost=cost,
            latency_ms=latency,
            reliability=reliability,
            efficiency=efficiency,
            quality_ci=quality_ci,
            confidence=confidence,
            uncertainty=uncertainty,
            sample_size=total_samples,
            evidence_sources=evidence_sources,
        )

        # Cache
        self._prediction_cache[cache_key] = (prediction, datetime.utcnow())

        return prediction

    def _get_feedback_prediction(
        self,
        model: Model,
        task_type: TaskType,
        fingerprint: CodebaseFingerprint | None,
    ) -> tuple[float, int]:
        """Get prediction from production feedback."""
        score = self.feedback.get_model_score(model, task_type, fingerprint)

        # Get sample size
        record = self.feedback._performance_records.get((model, task_type))
        sample_size = record.total_deployments if record else 0

        return score, sample_size

    def _get_benchmark_prediction(
        self,
        model: Model,
        task_type: TaskType,
    ) -> tuple[float, int]:
        """Get prediction from benchmarks."""
        summary = self.leaderboard._summaries.get(model)
        if not summary:
            return 0.5, 0

        type_scores = summary.by_task_type.get(task_type, {})
        score = type_scores.get("avg_quality", summary.avg_quality)

        return score, summary.benchmark_count

    def _estimate_cost(self, model: Model) -> float:
        """Estimate cost per 1K tokens."""
        costs = COST_TABLE.get(model, {"input": 0, "output": 0})
        # Average of input and output cost per 1K tokens
        return ((costs["input"] + costs["output"]) / 2) / 1000

    def _estimate_latency(self, model: Model) -> float:
        """Estimate latency in milliseconds."""
        # Default estimates based on model characteristics
        latency_table = {
            Model.DEEPSEEK_CHAT: 1200,
            Model.GEMINI_FLASH_LITE: 600,
            Model.GPT_4O_MINI: 1000,
            Model.LLAMA_3_3_70B: 1500,
            Model.PHI_4: 1800,
            Model.GPT_4O: 2500,
            Model.GEMINI_PRO: 3000,
        }

        # Check if we have actual measurements
        summary = self.leaderboard._summaries.get(model)
        if summary and summary.avg_latency_ms > 0:
            return summary.avg_latency_ms

        return latency_table.get(model, 2000)

    def _estimate_reliability(
        self,
        model: Model,
        task_type: TaskType,
        sample_size: int,
    ) -> float:
        """Estimate reliability score."""
        record = self.feedback._performance_records.get((model, task_type))
        if record and record.total_deployments > 0:
            return record.success_rate

        # Default based on sample size
        return min(0.9, 0.5 + sample_size / 100)

    def _calculate_confidence_interval(
        self,
        estimate: float,
        sample_size: int,
        uncertainty: float,
    ) -> tuple[float, float]:
        """Calculate 95% confidence interval."""
        if sample_size < self.MIN_SAMPLES_FOR_CI:
            # Wide interval for low samples
            margin = 0.3
        else:
            # Normal approximation
            margin = 1.96 * uncertainty / math.sqrt(sample_size)

        return (
            max(0, estimate - margin),
            min(1, estimate + margin),
        )

    def _normalize_objectives(
        self,
        predictions: list[ModelPrediction],
        objectives: list[Objective],
    ) -> list[tuple[ModelPrediction, dict[Objective, float]]]:
        """Normalize objective values to 0-1 range."""
        if not predictions:
            return []

        # Find min/max for each objective
        ranges = {}
        for obj in objectives:
            values = []
            for pred in predictions:
                if obj == Objective.QUALITY:
                    values.append(pred.quality)
                elif obj == Objective.COST:
                    values.append(pred.cost)
                elif obj == Objective.LATENCY:
                    values.append(pred.latency_ms)
                elif obj == Objective.RELIABILITY:
                    values.append(pred.reliability)
                elif obj == Objective.EFFICIENCY:
                    values.append(pred.efficiency)

            if values:
                ranges[obj] = (min(values), max(values))
            else:
                ranges[obj] = (0, 1)

        # Normalize
        result = []
        for pred in predictions:
            normalized = {}
            for obj in objectives:
                min_val, max_val = ranges[obj]
                range_size = max_val - min_val if max_val != min_val else 1

                if obj == Objective.QUALITY:
                    raw = pred.quality
                elif obj == Objective.COST:
                    raw = pred.cost
                elif obj == Objective.LATENCY:
                    raw = pred.latency_ms
                elif obj == Objective.RELIABILITY:
                    raw = pred.reliability
                elif obj == Objective.EFFICIENCY:
                    raw = pred.efficiency

                # Normalize
                norm = (raw - min_val) / range_size

                # Invert for minimization objectives
                if OBJECTIVE_DIRECTIONS[obj] == OptimizationDirection.MINIMIZE:
                    norm = 1 - norm

                normalized[obj] = max(0, min(1, norm))

            result.append((pred, normalized))

        return result

    def _compute_pareto_frontier(
        self,
        normalized: list[tuple[ModelPrediction, dict[Objective, float]]],
        objectives: list[Objective],
    ) -> list[ParetoPoint]:
        """Compute Pareto frontier using dominance checking."""
        points = []

        for pred, obj_values in normalized:
            point = ParetoPoint(
                prediction=pred,
                objective_values=obj_values,
            )
            points.append(point)

        # Dominance checking
        for i, point_i in enumerate(points):
            for j, point_j in enumerate(points):
                if i == j:
                    continue

                # Check if i dominates j
                if self._dominates(point_i, point_j, objectives):
                    point_i.dominates.append(point_j.prediction.model.value)
                # Check if j dominates i
                elif self._dominates(point_j, point_i, objectives):
                    point_i.dominated_by_count += 1

        # Assign Pareto ranks
        remaining = set(range(len(points)))
        rank = 0

        while remaining:
            # Find non-dominated points in remaining set
            frontier = []
            for idx in remaining:
                if points[idx].dominated_by_count == 0:
                    frontier.append(idx)

            if not frontier:
                break

            # Assign rank
            for idx in frontier:
                points[idx].pareto_rank = rank
                remaining.remove(idx)

                # Decrease dominated count for dominated points
                for other_idx in remaining:
                    if points[idx].prediction.model.value in [
                        points[other_idx].prediction.model.value
                    ] or points[idx].prediction.model.value in list(points[other_idx].dominates):
                        points[other_idx].dominated_by_count -= 1

            rank += 1

        # Sort by rank
        points.sort(key=lambda p: p.pareto_rank)

        return points

    def _dominates(
        self,
        point_a: ParetoPoint,
        point_b: ParetoPoint,
        objectives: list[Objective],
    ) -> bool:
        """Check if point A dominates point B."""
        at_least_as_good = True
        strictly_better = False

        for obj in objectives:
            val_a = point_a.objective_values.get(obj, 0)
            val_b = point_b.objective_values.get(obj, 0)

            if val_a < val_b:
                at_least_as_good = False
                break
            elif val_a > val_b:
                strictly_better = True

        return at_least_as_good and strictly_better

    def _generate_recommendations(
        self,
        pareto_points: list[ParetoPoint],
        objectives: list[Objective],
        budget_constraint: float | None,
    ) -> list[FrontierRecommendation]:
        """Generate human-readable recommendations."""
        recommendations = []

        for i, point in enumerate(pareto_points):
            pred = point.prediction

            # Determine best use case
            best_for = []
            if Objective.COST in objectives and pred.cost < 0.01:
                best_for.append("cost_sensitive")
            if Objective.QUALITY in objectives and pred.quality > 0.8:
                best_for.append("high_quality")
            if Objective.LATENCY in objectives and pred.latency_ms < 1000:
                best_for.append("low_latency")
            if Objective.EFFICIENCY in objectives and pred.efficiency > 50:
                best_for.append("cost_efficient")

            # Generate recommendation text
            if point.pareto_rank == 0:
                recommendation = f"Pareto-optimal: Best balance of {', '.join(o.value for o in objectives)}"
            elif pred.confidence > 0.8:
                recommendation = "High-confidence choice with proven performance"
            elif pred.cost < 0.005:
                recommendation = "Budget-friendly option"
            else:
                recommendation = "Alternative with different trade-offs"

            # Trade-off analysis
            trade_offs = {}
            if i > 0:
                better = pareto_points[0]
                if pred.cost > better.prediction.cost:
                    trade_offs["cost_increase"] = pred.cost - better.prediction.cost
                if pred.quality < better.prediction.quality:
                    trade_offs["quality_decrease"] = better.prediction.quality - pred.quality

            rec = FrontierRecommendation(
                model=pred.model,
                rank=i + 1,
                quality=pred.quality,
                cost=pred.cost,
                latency_ms=pred.latency_ms,
                confidence=pred.confidence,
                quality_ci_lower=pred.quality_ci[0],
                quality_ci_upper=pred.quality_ci[1],
                is_pareto_optimal=point.pareto_rank == 0,
                dominates_count=len(point.dominates),
                trade_offs=trade_offs,
                best_for=best_for,
                recommendation=recommendation,
            )

            recommendations.append(rec)

        return recommendations

    def compare_models(
        self,
        model_a: Model,
        model_b: Model,
        task_type: TaskType,
        fingerprint: CodebaseFingerprint | None = None,
    ) -> dict[str, Any]:
        """Compare two models with statistical significance."""
        # Get predictions
        pred_a = asyncio.run(self._predict_model(
            model_a, task_type, fingerprint, list(Objective)
        ))
        pred_b = asyncio.run(self._predict_model(
            model_b, task_type, fingerprint, list(Objective)
        ))

        if not pred_a or not pred_b:
            return {"error": "Could not generate predictions"}

        # Calculate differences
        quality_diff = pred_a.quality - pred_b.quality
        cost_diff = pred_a.cost - pred_b.cost

        # Determine significance
        quality_significant = abs(quality_diff) > (
            pred_a.uncertainty + pred_b.uncertainty
        )

        # Determine winner per objective
        winners = {}
        if pred_a.quality > pred_b.quality:
            winners["quality"] = model_a.value
        else:
            winners["quality"] = model_b.value

        if pred_a.cost < pred_b.cost:
            winners["cost"] = model_a.value
        else:
            winners["cost"] = model_b.value

        if pred_a.efficiency > pred_b.efficiency:
            winners["efficiency"] = model_a.value
        else:
            winners["efficiency"] = model_b.value

        return {
            "model_a": pred_a.to_dict(),
            "model_b": pred_b.to_dict(),
            "differences": {
                "quality": round(quality_diff, 3),
                "cost": round(cost_diff, 4),
                "efficiency": round(pred_a.efficiency - pred_b.efficiency, 2),
            },
            "significant_difference": quality_significant,
            "winners": winners,
            "recommendation": (
                f"{model_a.value} is better for quality"
                if quality_diff > 0.1 and quality_significant
                else f"{model_b.value} is more cost-effective"
                if cost_diff < -0.01
                else "Both models are comparable"
            ),
        }

    def get_frontier_stats(self) -> dict[str, Any]:
        """Get statistics about frontier computations."""
        return {
            "cache_size": len(self._prediction_cache),
            "history_entries": sum(len(h) for h in self._history.values()),
            "tracked_models": len({
                p.model for history in self._history.values() for p in history
            }),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience Functions
# ═══════════════════════════════════════════════════════════════════════════════

_frontier: CostQualityFrontier | None = None


def get_cost_quality_frontier() -> CostQualityFrontier:
    """Get global cost-quality frontier."""
    global _frontier
    if _frontier is None:
        _frontier = CostQualityFrontier()
    return _frontier


def reset_cost_quality_frontier() -> None:
    """Reset global frontier (for testing)."""
    global _frontier
    _frontier = None
