"""
Cross-Project Transfer Learning
================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Paradigm Shift: Extract patterns across ALL completed projects → Apply to new projects

Current State: Memory Bank stores decisions per project (isolated)
Future State: Cross-project pattern extraction (provably better over time)

After 50 projects, the orchestrator "knows" that:
- FastAPI projects achieve higher scores with DeepSeek (pattern)
- Authentication tasks fail 40% more often with GPT-4o-mini (anti-pattern)
- Projects with >15 tasks always need 2+ repair cycles at task 8+ (threshold pattern)

Usage:
    from orchestrator.cross_project_learning import CrossProjectLearning

    learning = CrossProjectLearning()
    insights = await learning.extract_insights()
    learning.inject_into_routing(router)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .log_config import get_logger
from .models import Model, TaskType

logger = get_logger(__name__)


@dataclass
class Insight:
    """
    A learned insight from cross-project analysis.

    Attributes:
        type: Type of insight (model_affinity, failure_predictor, scaling_threshold)
        description: Human-readable description
        action: Recommended action to take
        confidence: Confidence score 0.0-1.0 based on sample size
        sample_size: Number of data points supporting this insight
        metadata: Additional metadata
    """

    type: str
    description: str
    action: str
    confidence: float = 0.0
    sample_size: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "type": self.type,
            "description": self.description,
            "action": self.action,
            "confidence": self.confidence,
            "sample_size": self.sample_size,
            "metadata": self.metadata,
        }


@dataclass
class ModelTaskScore:
    """Aggregated score for a model on a specific task type."""

    model: Model
    task_type: TaskType
    avg_score: float
    sample_size: int
    total_cost: float
    avg_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.value,
            "task_type": self.task_type.value,
            "avg_score": self.avg_score,
            "sample_size": self.sample_size,
            "total_cost": self.total_cost,
            "avg_tokens": self.avg_tokens,
        }


@dataclass
class FailurePattern:
    """Pattern that correlates with failures."""

    regex: str
    failure_rate: float
    sample_size: int
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "regex": self.regex,
            "failure_rate": self.failure_rate,
            "sample_size": self.sample_size,
            "description": self.description,
        }


@dataclass
class ScalingThreshold:
    """Threshold where project size correlates with repair cycles."""

    threshold: int
    avg_repairs_below: float
    avg_repairs_above: float
    sample_size: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "avg_repairs_below": self.avg_repairs_below,
            "avg_repairs_above": self.avg_repairs_above,
            "sample_size": self.sample_size,
        }


class CrossProjectLearning:
    """
    Extract patterns across all completed projects.

    This creates a competitive moat: the orchestrator becomes provably
    better over time as it learns from more projects.

    Key capabilities:
    1. Model affinity: Which models work best for which task types
    2. Failure predictors: Task descriptions that correlate with failures
    3. Scaling thresholds: Project size vs repair cycles
    4. Cost patterns: Actual vs estimated costs by task type
    """

    def __init__(self, patterns_dir: Path | None = None):
        """
        Initialize cross-project learning.

        Args:
            patterns_dir: Directory to store learned patterns
        """
        self.patterns_dir = patterns_dir or Path(".orchestrator/patterns")
        self.patterns_path = self.patterns_dir / "learned_patterns.json"

        # Ensure directory exists
        self.patterns_dir.mkdir(parents=True, exist_ok=True)

        # Load existing patterns
        self.insights: list[Insight] = []
        self._load_patterns()

        logger.info(f"Cross-project learning initialized with {len(self.insights)} insights")

    def _load_patterns(self) -> None:
        """Load previously learned patterns from disk."""
        if self.patterns_path.exists():
            try:
                with self.patterns_path.open("r") as f:
                    data = json.load(f)
                    self.insights = [Insight(**item) for item in data.get("insights", [])]
                logger.info(f"Loaded {len(self.insights)} patterns from disk")
            except Exception as e:
                logger.warning(f"Failed to load patterns: {e}")

    def _save_patterns(self) -> None:
        """Save learned patterns to disk."""
        try:
            with self.patterns_path.open("w") as f:
                json.dump(
                    {
                        "insights": [i.to_dict() for i in self.insights],
                        "last_updated": datetime.now().isoformat(),
                    },
                    f,
                    indent=2,
                )
            logger.info(f"Saved {len(self.insights)} patterns to disk")
        except Exception as e:
            logger.error(f"Failed to save patterns: {e}")

    async def extract_insights(
        self,
        all_traces: list[dict[str, Any]] | None = None,
    ) -> list[Insight]:
        """
        Extract insights from all completed projects.

        Args:
            all_traces: List of project traces (if None, load from disk)

        Returns:
            List of insights with confidence scores
        """
        if all_traces is None:
            all_traces = await self._load_all_traces()

        if not all_traces:
            logger.warning("No traces found for cross-project analysis")
            return []

        logger.info(f"Analyzing {len(all_traces)} project traces...")

        insights = []

        # ═══════════════════════════════════════════════════════
        # Pattern 1: Model Affinity (which models work best for which tasks)
        # ═══════════════════════════════════════════════════════
        model_task_scores = self._aggregate_model_task_scores(all_traces)
        for task_type, model_scores in model_task_scores.items():
            if len(model_scores) >= 2:  # Need at least 2 models to compare
                best = max(model_scores, key=lambda m: m.avg_score)
                if best.sample_size >= 5:  # Confident after 5 samples
                    insights.append(
                        Insight(
                            type="model_affinity",
                            description=f"{best.model.value} scores {best.avg_score:.2f} avg on {task_type.value} tasks",
                            action=f"Route {task_type.value} tasks to {best.model.value} by default",
                            confidence=min(
                                1.0, best.sample_size / 20
                            ),  # Max confidence at 20 samples
                            sample_size=best.sample_size,
                            metadata=best.to_dict(),
                        )
                    )
                    logger.info(
                        f"  Model affinity: {best.model.value} → {task_type.value} "
                        f"(score={best.avg_score:.2f}, n={best.sample_size})"
                    )

        # ═══════════════════════════════════════════════════════
        # Pattern 2: Failure Predictors (task descriptions that correlate with failures)
        # ═══════════════════════════════════════════════════════
        failure_patterns = self._extract_failure_patterns(all_traces)
        for pattern in failure_patterns:
            if pattern.sample_size >= 3:  # Confident after 3 samples
                insights.append(
                    Insight(
                        type="failure_predictor",
                        description=f"Tasks matching '{pattern.regex}' fail {pattern.failure_rate:.0%} of the time",
                        action=f"Add extra verification or use premium model for tasks matching '{pattern.regex}'",
                        confidence=min(1.0, pattern.sample_size / 10),
                        sample_size=pattern.sample_size,
                        metadata=pattern.to_dict(),
                    )
                )
                logger.info(
                    f"  Failure pattern: '{pattern.regex}' → "
                    f"{pattern.failure_rate:.0%} failure rate (n={pattern.sample_size})"
                )

        # ═══════════════════════════════════════════════════════
        # Pattern 3: Scaling Thresholds (project size vs repair cycles)
        # ═══════════════════════════════════════════════════════
        size_repair = self._correlate_size_repairs(all_traces)
        if size_repair.sample_size >= 5:
            insights.append(
                Insight(
                    type="scaling_threshold",
                    description=f"Projects with >{size_repair.threshold} tasks need {size_repair.avg_repairs_above:.1f}x more repairs",
                    action=f"Auto-increase repair_attempts for projects with >{size_repair.threshold} tasks",
                    confidence=min(1.0, size_repair.sample_size / 15),
                    sample_size=size_repair.sample_size,
                    metadata=size_repair.to_dict(),
                )
            )
            logger.info(
                f"  Scaling threshold: >{size_repair.threshold} tasks → "
                f"{size_repair.avg_repairs_above:.1f}x more repairs (n={size_repair.sample_size})"
            )

        # ═══════════════════════════════════════════════════════
        # Pattern 4: Cost Predictors (actual vs estimated by task type)
        # ═══════════════════════════════════════════════════════
        cost_patterns = self._extract_cost_patterns(all_traces)
        for task_type, cost_data in cost_patterns.items():
            if cost_data["sample_size"] >= 5:
                ratio = cost_data["avg_actual"] / max(0.01, cost_data["avg_estimated"])
                if ratio > 1.3 or ratio < 0.7:  # Significant deviation
                    insights.append(
                        Insight(
                            type="cost_predictor",
                            description=f"{task_type.value} tasks cost {ratio:.1f}x estimated",
                            action=f"Adjust cost estimates for {task_type.value} tasks by {ratio:.1f}x",
                            confidence=min(1.0, cost_data["sample_size"] / 20),
                            sample_size=cost_data["sample_size"],
                            metadata={
                                "task_type": task_type.value,
                                "avg_actual": cost_data["avg_actual"],
                                "avg_estimated": cost_data["avg_estimated"],
                                "ratio": ratio,
                            },
                        )
                    )
                    logger.info(
                        f"  Cost pattern: {task_type.value} → "
                        f"{ratio:.1f}x estimated (n={cost_data['sample_size']})"
                    )

        # Merge with existing insights (prefer higher confidence)
        self._merge_insights(insights)

        # Save to disk
        self._save_patterns()

        logger.info(f"Extracted {len(insights)} new insights (total: {len(self.insights)})")
        return insights

    async def _load_all_traces(self) -> list[dict[str, Any]]:
        """Load all project traces from disk."""
        traces_dir = Path(".orchestrator/traces")
        traces = []

        if not traces_dir.exists():
            return traces

        for trace_file in traces_dir.glob("*.json"):
            try:
                with trace_file.open("r") as f:
                    trace = json.load(f)
                    traces.append(trace)
            except Exception as e:
                logger.warning(f"Failed to load trace {trace_file}: {e}")

        return traces

    def _aggregate_model_task_scores(
        self,
        all_traces: list[dict[str, Any]],
    ) -> dict[TaskType, list[ModelTaskScore]]:
        """
        Aggregate scores by model and task type.

        Returns:
            Dict mapping task_type → list of ModelTaskScore
        """
        # Collect scores per (model, task_type)
        scores: dict[tuple[Model, TaskType], list[float]] = {}
        costs: dict[tuple[Model, TaskType], float] = {}
        tokens: dict[tuple[Model, TaskType], list[int]] = {}

        for trace in all_traces:
            tasks = trace.get("tasks", [])
            for task in tasks:
                model_str = task.get("model_used", "")
                task_type_str = task.get("type", "")
                score = task.get("score", 0.0)
                cost = task.get("cost_usd", 0.0)
                tokens_used = task.get("tokens_used", {}).get("output", 0)

                try:
                    model = Model(model_str)
                    task_type = TaskType(task_type_str)
                except (ValueError, AttributeError):
                    continue

                key = (model, task_type)
                if key not in scores:
                    scores[key] = []
                    costs[key] = 0.0
                    tokens[key] = []

                scores[key].append(score)
                costs[key] += cost
                tokens[key].append(tokens_used)

        # Aggregate into ModelTaskScore
        result: dict[TaskType, list[ModelTaskScore]] = {}
        for (model, task_type), score_list in scores.items():
            if task_type not in result:
                result[task_type] = []

            result[task_type].append(
                ModelTaskScore(
                    model=model,
                    task_type=task_type,
                    avg_score=sum(score_list) / len(score_list),
                    sample_size=len(score_list),
                    total_cost=costs[(model, task_type)],
                    avg_tokens=(
                        sum(tokens[(model, task_type)]) // len(tokens[(model, task_type)])
                        if tokens[(model, task_type)]
                        else 0
                    ),
                )
            )

        return result

    def _extract_failure_patterns(
        self,
        all_traces: list[dict[str, Any]],
    ) -> list[FailurePattern]:
        """
        Extract task description patterns that correlate with failures.

        Returns:
            List of FailurePattern with regex and failure rates
        """
        # Collect failed vs successful task prompts
        failed_prompts: list[str] = []
        success_prompts: list[str] = []

        for trace in all_traces:
            tasks = trace.get("tasks", [])
            for task in tasks:
                prompt = task.get("prompt", "")
                score = task.get("score", 0.0)
                status = task.get("status", "")

                if not prompt:
                    continue

                if score < 0.7 or status == "FAILED":
                    failed_prompts.append(prompt)
                else:
                    success_prompts.append(prompt)

        # Extract common patterns from failed prompts
        patterns = []

        # Look for keywords that correlate with failures
        keywords = [
            "authentication",
            "auth",
            "login",
            "jwt",
            "oauth",
            "database",
            "sql",
            "migration",
            "async",
            "concurrent",
            "parallel",
            "validation",
            "schema",
            "pydantic",
            "api",
            "rest",
            "endpoint",
        ]

        for keyword in keywords:
            failed_count = sum(1 for p in failed_prompts if keyword.lower() in p.lower())
            success_count = sum(1 for p in success_prompts if keyword.lower() in p.lower())
            total = failed_count + success_count

            if total >= 3:  # Minimum sample size
                failure_rate = failed_count / total
                if failure_rate > 0.4:  # >40% failure rate
                    patterns.append(
                        FailurePattern(
                            regex=keyword,
                            failure_rate=failure_rate,
                            sample_size=total,
                            description=f"Tasks containing '{keyword}' fail {failure_rate:.0%} of the time",
                        )
                    )

        return patterns

    def _correlate_size_repairs(
        self,
        all_traces: list[dict[str, Any]],
    ) -> ScalingThreshold:
        """
        Correlate project size with repair cycles.

        Returns:
            ScalingThreshold with threshold and repair multipliers
        """
        # Collect (num_tasks, num_repairs) pairs
        data: list[tuple[int, int]] = []

        for trace in all_traces:
            tasks = trace.get("tasks", [])
            num_tasks = len(tasks)

            # Count total repair cycles (iterations > 1)
            num_repairs = sum(1 for task in tasks if task.get("iterations", 1) > 1)

            data.append((num_tasks, num_repairs))

        if not data:
            return ScalingThreshold(
                threshold=10,
                avg_repairs_below=0.5,
                avg_repairs_above=1.5,
                sample_size=0,
            )

        # Find optimal threshold (try 5, 10, 15, 20)
        best_threshold = 10
        best_ratio = 0.0

        for threshold in [5, 10, 15, 20]:
            below = [r for s, r in data if s <= threshold]
            above = [r for s, r in data if s > threshold]

            if below and above:
                avg_below = sum(below) / len(below)
                avg_above = sum(above) / len(above)

                if avg_below > 0:
                    ratio = avg_above / avg_below
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_threshold = threshold

        # Calculate final stats
        below = [r for s, r in data if s <= best_threshold]
        above = [r for s, r in data if s > best_threshold]

        return ScalingThreshold(
            threshold=best_threshold,
            avg_repairs_below=sum(below) / len(below) if below else 0.0,
            avg_repairs_above=sum(above) / len(above) if above else 0.0,
            sample_size=len(data),
        )

    def _extract_cost_patterns(
        self,
        all_traces: list[dict[str, Any]],
    ) -> dict[TaskType, dict[str, float]]:
        """
        Extract actual vs estimated cost patterns by task type.

        Returns:
            Dict mapping task_type → cost statistics
        """
        costs: dict[TaskType, dict[str, list[float]]] = {}

        for trace in all_traces:
            tasks = trace.get("tasks", [])
            for task in tasks:
                task_type_str = task.get("type", "")
                cost = task.get("cost_usd", 0.0)
                estimated = task.get("estimated_cost", cost)  # If no estimate, use actual

                try:
                    task_type = TaskType(task_type_str)
                except (ValueError, AttributeError):
                    continue

                if task_type not in costs:
                    costs[task_type] = {"actual": [], "estimated": []}

                costs[task_type]["actual"].append(cost)
                costs[task_type]["estimated"].append(estimated)

        # Aggregate
        result = {}
        for task_type, cost_data in costs.items():
            actual = cost_data["actual"]
            estimated = cost_data["estimated"]

            result[task_type] = {
                "avg_actual": sum(actual) / len(actual) if actual else 0.0,
                "avg_estimated": sum(estimated) / len(estimated) if estimated else 0.0,
                "sample_size": len(actual),
            }

        return result

    def _merge_insights(self, new_insights: list[Insight]) -> None:
        """
        Merge new insights with existing ones.

        Prefer higher confidence insights for duplicate types.
        """
        for new_insight in new_insights:
            # Find existing insight of same type and description
            existing_idx = None
            for i, existing in enumerate(self.insights):
                if (
                    existing.type == new_insight.type
                    and existing.description == new_insight.description
                ):
                    existing_idx = i
                    break

            if existing_idx is not None:
                # Update if new has higher confidence
                if new_insight.confidence > self.insights[existing_idx].confidence:
                    self.insights[existing_idx] = new_insight
            else:
                # Add new insight
                self.insights.append(new_insight)

    def inject_into_routing(self, router) -> None:
        """
        Apply learned patterns to routing decisions.

        Args:
            router: ModelRouter instance to inject patterns into
        """
        for insight in self.insights:
            if insight.type == "model_affinity" and insight.confidence > 0.7:
                # Extract model and task type from metadata
                metadata = insight.metadata
                if "model" in metadata and "task_type" in metadata:
                    try:
                        model = Model(metadata["model"])
                        task_type = TaskType(metadata["task_type"])
                        router.add_preference(task_type, model, insight.confidence)
                        logger.info(
                            f"Injected routing preference: {task_type.value} → {model.value} "
                            f"(confidence={insight.confidence:.2f})"
                        )
                    except (ValueError, AttributeError) as e:
                        logger.warning(f"Failed to inject insight: {e}")

            elif insight.type == "failure_predictor" and insight.confidence > 0.7:
                # Add to router's high-risk patterns
                if hasattr(router, "add_high_risk_pattern"):
                    router.add_high_risk_pattern(
                        insight.metadata.get("regex", ""),
                        insight.metadata.get("failure_rate", 0.0),
                    )
                    logger.info(
                        f"Injected failure predictor: {insight.metadata.get('regex', '')} "
                        f"(rate={insight.metadata.get('failure_rate', 0.0):.0%})"
                    )

        logger.info(f"Injected {len(self.insights)} insights into routing")

    def get_insights(self) -> list[Insight]:
        """Get all learned insights."""
        return self.insights

    def clear_insights(self) -> None:
        """Clear all learned insights (for testing)."""
        self.insights = []
        if self.patterns_path.exists():
            self.patterns_path.unlink()
        logger.info("Cleared all learned insights")


__all__ = [
    "CrossProjectLearning",
    "Insight",
    "ModelTaskScore",
    "FailurePattern",
    "ScalingThreshold",
]
