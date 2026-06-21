"""
OpenRouter Optimization A/B Testing
====================================
Author: Georgios-Chrysovalantis Chatzivantsidis

A/B testing framework specifically for OpenRouter optimizations.
Tracks metrics like parsing error rates, latency, cost, and quality
to determine if optimizations should be rolled out.

USAGE:
    from orchestrator.openrouter_ab_testing import OpenRouterABTester

    ab_tester = OpenRouterABTester()

    # Check if this request should use optimized path
    use_optimization = ab_tester.should_use_optimization(
        project_id="proj_123",
        optimization="json_schema"
    )

    # Record metrics after completion
    ab_tester.record_metrics(
        project_id="proj_123",
        optimization="json_schema",
        metrics={
            "parsing_error": False,
            "latency_ms": 1200,
            "cost_usd": 0.05,
            "token_count": 500
        }
    )
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger("orchestrator.openrouter_ab_testing")


class OptimizationType(str, Enum):
    """Types of OpenRouter optimizations that can be A/B tested."""

    JSON_SCHEMA = "json_schema"  # response_format: json_schema
    MODEL_VARIANTS = "model_variants"  # :nitro, :thinking variants
    NATIVE_FALLBACKS = "native_fallbacks"  # OpenRouter models array
    PROVIDER_SORTING = "provider_sorting"  # throughput/latency/price sorting


class Variant(str, Enum):
    """A/B test variants."""

    CONTROL = "control"  # Standard behavior
    TREATMENT = "treatment"  # Optimized behavior


@dataclass
class OptimizationMetrics:
    """Metrics collected for optimization comparison."""

    # Quality metrics
    parsing_error: bool = False
    validation_error: bool = False
    retry_count: int = 0

    # Performance metrics
    latency_ms: float = 0.0
    tokens_per_second: float = 0.0

    # Cost metrics
    cost_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0

    # Success metrics
    success: bool = True
    fallback_triggered: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "parsing_error": self.parsing_error,
            "validation_error": self.validation_error,
            "retry_count": self.retry_count,
            "latency_ms": self.latency_ms,
            "tokens_per_second": self.tokens_per_second,
            "cost_usd": self.cost_usd,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "success": self.success,
            "fallback_triggered": self.fallback_triggered,
        }


@dataclass
class ExperimentResult:
    """Results of an A/B test experiment."""

    optimization: OptimizationType
    control_samples: int = 0
    treatment_samples: int = 0

    # Aggregated metrics
    control_error_rate: float = 0.0
    treatment_error_rate: float = 0.0

    control_avg_latency_ms: float = 0.0
    treatment_avg_latency_ms: float = 0.0

    control_avg_cost: float = 0.0
    treatment_avg_cost: float = 0.0

    # Statistical results
    error_rate_improvement: float = 0.0  # Negative is better (reduction)
    latency_improvement: float = 0.0  # Negative is better (reduction)
    cost_improvement: float = 0.0  # Negative is better (reduction)

    # Recommendation
    should_adopt: bool = False
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "optimization": self.optimization.value,
            "control_samples": self.control_samples,
            "treatment_samples": self.treatment_samples,
            "control_error_rate": self.control_error_rate,
            "treatment_error_rate": self.treatment_error_rate,
            "control_avg_latency_ms": self.control_avg_latency_ms,
            "treatment_avg_latency_ms": self.treatment_avg_latency_ms,
            "control_avg_cost": self.control_avg_cost,
            "treatment_avg_cost": self.treatment_avg_cost,
            "error_rate_improvement": self.error_rate_improvement,
            "latency_improvement": self.latency_improvement,
            "cost_improvement": self.cost_improvement,
            "should_adopt": self.should_adopt,
            "confidence": self.confidence,
        }


class OpenRouterABTester:
    """
    A/B testing for OpenRouter optimizations.

    Tracks metrics and determines if optimizations improve outcomes.
    Uses consistent hashing for deterministic variant assignment.
    """

    DEFAULT_TRAFFIC_SPLIT = 0.1  # 10% to treatment initially
    MIN_SAMPLES = 50  # Minimum samples before analysis

    def __init__(
        self,
        db_path: str | Path | None = None,
        traffic_split: float = DEFAULT_TRAFFIC_SPLIT,
    ):
        """
        Initialize A/B tester.

        Args:
            db_path: Path to SQLite database for metrics storage
            traffic_split: Fraction of traffic to route to treatment (0.0-1.0)
        """
        self.traffic_split = traffic_split

        if db_path is None:
            db_path = Path(".orchestrator/openrouter_ab_test.db")

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_db()
        logger.info(f"OpenRouter A/B tester initialized (split: {traffic_split:.0%})")

    def _init_db(self) -> None:
        """Initialize SQLite database for metrics storage."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ab_test_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    project_id TEXT NOT NULL,
                    optimization TEXT NOT NULL,
                    variant TEXT NOT NULL,
                    metrics_json TEXT NOT NULL
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_optimization_variant 
                ON ab_test_records(optimization, variant)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_project 
                ON ab_test_records(project_id)
            """)

            conn.commit()

    def should_use_optimization(
        self,
        project_id: str,
        optimization: OptimizationType | str,
    ) -> bool:
        """
        Determine if this request should use the optimized path.

        Uses consistent hashing for deterministic assignment.

        Args:
            project_id: Unique project identifier
            optimization: Type of optimization being tested

        Returns:
            True if treatment (optimization enabled), False if control
        """
        opt_type = (
            optimization
            if isinstance(optimization, OptimizationType)
            else OptimizationType(optimization)
        )

        # Consistent hashing for deterministic assignment
        hash_input = f"{project_id}:{opt_type.value}"
        hash_value = int(hashlib.md5(hash_input.encode(), usedforsecurity=False).hexdigest(), 16)

        # Map to 0-1 range
        normalized = (hash_value % 10000) / 10000.0

        is_treatment = normalized < self.traffic_split

        variant = Variant.TREATMENT if is_treatment else Variant.CONTROL
        logger.debug(f"A/B test assignment: {project_id} -> {opt_type.value} -> {variant.value}")

        return is_treatment

    def record_metrics(
        self,
        project_id: str,
        optimization: OptimizationType | str,
        metrics: OptimizationMetrics | dict[str, Any],
        variant: Variant | str | None = None,
    ) -> None:
        """
        Record metrics for an A/B test sample.

        Args:
            project_id: Unique project identifier
            optimization: Type of optimization
            metrics: Metrics collected for this sample
            variant: Variant assignment (if known, else auto-detected)
        """
        opt_type = (
            optimization
            if isinstance(optimization, OptimizationType)
            else OptimizationType(optimization)
        )

        if isinstance(metrics, dict):
            metrics = OptimizationMetrics(**metrics)

        if variant is None:
            # Recompute variant from project_id
            is_treatment = self.should_use_optimization(project_id, opt_type)
            variant = Variant.TREATMENT if is_treatment else Variant.CONTROL
        elif isinstance(variant, str):
            variant = Variant(variant)

        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                """
                INSERT INTO ab_test_records (timestamp, project_id, optimization, variant, metrics_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    time.time(),
                    project_id,
                    opt_type.value,
                    variant.value,
                    json.dumps(metrics.to_dict()),
                ),
            )
            conn.commit()

        logger.debug(f"Recorded metrics for {project_id} ({opt_type.value}/{variant.value})")

    def analyze_experiment(
        self,
        optimization: OptimizationType | str,
    ) -> ExperimentResult:
        """
        Analyze A/B test results for an optimization.

        Args:
            optimization: Type of optimization to analyze

        Returns:
            ExperimentResult with statistical analysis
        """
        opt_type = (
            optimization
            if isinstance(optimization, OptimizationType)
            else OptimizationType(optimization)
        )

        # Aggregate metrics from database
        with sqlite3.connect(str(self.db_path)) as conn:
            # Control metrics
            control_row = conn.execute(
                """
                SELECT 
                    COUNT(*),
                    AVG(CAST(json_extract(metrics_json, '$.parsing_error') AS INTEGER)),
                    AVG(CAST(json_extract(metrics_json, '$.latency_ms') AS REAL)),
                    AVG(CAST(json_extract(metrics_json, '$.cost_usd') AS REAL))
                FROM ab_test_records
                WHERE optimization = ? AND variant = 'control'
                """,
                (opt_type.value,),
            ).fetchone()

            # Treatment metrics
            treatment_row = conn.execute(
                """
                SELECT 
                    COUNT(*),
                    AVG(CAST(json_extract(metrics_json, '$.parsing_error') AS INTEGER)),
                    AVG(CAST(json_extract(metrics_json, '$.latency_ms') AS REAL)),
                    AVG(CAST(json_extract(metrics_json, '$.cost_usd') AS REAL))
                FROM ab_test_records
                WHERE optimization = ? AND variant = 'treatment'
                """,
                (opt_type.value,),
            ).fetchone()

        control_samples = control_row[0] or 0
        treatment_samples = treatment_row[0] or 0

        result = ExperimentResult(
            optimization=opt_type,
            control_samples=control_samples,
            treatment_samples=treatment_samples,
            control_error_rate=control_row[1] or 0.0,
            treatment_error_rate=treatment_row[1] or 0.0,
            control_avg_latency_ms=control_row[2] or 0.0,
            treatment_avg_latency_ms=treatment_row[2] or 0.0,
            control_avg_cost=control_row[3] or 0.0,
            treatment_avg_cost=treatment_row[3] or 0.0,
        )

        # Calculate improvements (negative is better for error rate, latency, cost)
        if control_samples > 0 and treatment_samples > 0:
            # Handle zero control error rate specially
            if result.control_error_rate == 0:
                # If treatment also has 0 errors, no improvement to calculate
                # If treatment has errors, that's a degradation
                result.error_rate_improvement = (
                    0.0 if result.treatment_error_rate == 0 else float("-inf")
                )
            else:
                result.error_rate_improvement = (
                    result.treatment_error_rate - result.control_error_rate
                ) / result.control_error_rate
            result.latency_improvement = (
                result.treatment_avg_latency_ms - result.control_avg_latency_ms
            ) / max(result.control_avg_latency_ms, 1.0)
            result.cost_improvement = (result.treatment_avg_cost - result.control_avg_cost) / max(
                result.control_avg_cost, 0.001
            )

            # Simple adoption criteria (can be made more sophisticated)
            has_enough_samples = (
                control_samples >= self.MIN_SAMPLES and treatment_samples >= self.MIN_SAMPLES
            )

            # Handle -inf (treatment worse than perfect control) and normal improvement
            if result.error_rate_improvement == float("-inf"):
                error_rate_better = False  # Treatment has errors, control had none - not better
            else:
                error_rate_better = result.error_rate_improvement < -0.1  # >10% improvement
            latency_acceptable = result.latency_improvement < 0.2  # <20% degradation
            cost_acceptable = result.cost_improvement < 0.15  # <15% cost increase

            result.should_adopt = (
                has_enough_samples and error_rate_better and latency_acceptable and cost_acceptable
            )

            # Confidence based on sample size (simplified)
            min_n = min(control_samples, treatment_samples)
            result.confidence = min(min_n / (self.MIN_SAMPLES * 2), 1.0)

        return result

    def get_summary(self) -> dict[str, ExperimentResult]:
        """
        Get summary of all active experiments.

        Returns:
            Dict mapping optimization type to experiment result
        """
        results = {}
        for opt_type in OptimizationType:
            result = self.analyze_experiment(opt_type)
            if result.control_samples > 0 or result.treatment_samples > 0:
                results[opt_type.value] = result
        return results

    def clear_experiment(self, optimization: OptimizationType | str) -> None:
        """
        Clear experiment data for an optimization.

        Args:
            optimization: Type of optimization to clear
        """
        opt_type = (
            optimization
            if isinstance(optimization, OptimizationType)
            else OptimizationType(optimization)
        )

        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("DELETE FROM ab_test_records WHERE optimization = ?", (opt_type.value,))
            conn.commit()

        logger.info(f"Cleared experiment data for {opt_type.value}")


# Global instance for convenience
_default_tester: OpenRouterABTester | None = None


def get_ab_tester() -> OpenRouterABTester:
    """Get or create default A/B tester instance."""
    global _default_tester
    if _default_tester is None:
        _default_tester = OpenRouterABTester()
    return _default_tester
