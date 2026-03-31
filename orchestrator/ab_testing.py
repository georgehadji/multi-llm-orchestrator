"""
A/B Testing Engine for Meta-Optimization
=========================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Live A/B testing for strategy proposals. Routes traffic between control
and treatment groups, collects metrics, and performs statistical analysis
to determine if proposals should be adopted.

Inspired by standard A/B testing frameworks, adapted for AI Orchestrator.

USAGE:
    from orchestrator.ab_testing import ABTestingEngine, ExperimentConfig

    ab_engine = ABTestingEngine(archive)

    # Create experiment for a proposal
    experiment = await ab_engine.create_experiment(
        proposal,
        traffic_split=0.1,  # 10% to treatment
        min_samples=30
    )

    # Route executions
    group = await ab_engine.route_execution(project_id)

    # Record outcomes
    await ab_engine.record_outcome(experiment.id, group, metrics)

    # Analyze results
    result = await ab_engine.analyze_results(experiment.id)
    if result.recommendation == "adopt":
        await ab_engine.apply_winning_variant(experiment.id)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .meta_orchestrator import ExecutionArchive, StrategyProposal

logger = logging.getLogger("orchestrator.ab_testing")


# ─────────────────────────────────────────────
# Enums & Constants
# ─────────────────────────────────────────────


class ExperimentStatus(str, Enum):
    """Status of an A/B experiment."""

    RUNNING = "running"
    COMPLETED = "completed"
    PAUSED = "paused"
    INVALIDATED = "invalidated"


class Recommendation(str, Enum):
    """Statistical recommendation from A/B test."""

    ADOPT = "adopt"  # Treatment significantly better
    REJECT = "reject"  # Treatment significantly worse or no difference
    INCONCLUSIVE = "inconclusive"  # Insufficient data or unclear result


class Variant(str, Enum):
    """Experiment variants."""

    CONTROL = "control"  # Current strategy (A)
    TREATMENT = "treatment"  # Proposed strategy (B)


# Statistical constants
DEFAULT_SIGNIFICANCE_LEVEL = 0.05  # p-value threshold (alpha)
DEFAULT_POWER = 0.80  # Statistical power (1 - beta)
DEFAULT_MIN_SAMPLES = 30  # Minimum samples per variant for analysis


# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────


@dataclass
class MetricSummary:
    """Summary statistics for a metric."""

    mean: float
    std: float
    count: int
    min: float
    max: float

    @classmethod
    def from_samples(cls, samples: list[float]) -> MetricSummary:
        if not samples:
            return cls(mean=0.0, std=0.0, count=0, min=0.0, max=0.0)

        n = len(samples)
        mean = sum(samples) / n
        variance = sum((x - mean) ** 2 for x in samples) / max(n - 1, 1)
        std = math.sqrt(variance)

        return cls(
            mean=mean,
            std=std,
            count=n,
            min=min(samples),
            max=max(samples),
        )


@dataclass
class ExperimentOutcome:
    """Outcome recorded for an experiment."""

    outcome_id: str
    experiment_id: str
    variant: Variant
    project_id: str
    success: bool
    score: float
    cost_usd: float
    latency_ms: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "outcome_id": self.outcome_id,
            "experiment_id": self.experiment_id,
            "variant": self.variant.value,
            "project_id": self.project_id,
            "success": self.success,
            "score": self.score,
            "cost_usd": self.cost_usd,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ExperimentOutcome:
        return cls(
            outcome_id=data["outcome_id"],
            experiment_id=data["experiment_id"],
            variant=Variant(data["variant"]),
            project_id=data["project_id"],
            success=data["success"],
            score=data["score"],
            cost_usd=data["cost_usd"],
            latency_ms=data["latency_ms"],
            timestamp=data.get("timestamp", time.time()),
        )


@dataclass
class ExperimentResult:
    """Statistical analysis result of an experiment."""

    experiment_id: str
    control_metrics: MetricSummary
    treatment_metrics: MetricSummary

    # Statistical test results
    p_value: float  # Probability of observing this difference by chance
    confidence_level: float  # 1 - p_value
    effect_size: float  # Cohen's d (standardized difference)

    # Confidence interval for difference
    confidence_interval: tuple[float, float]  # (lower, upper)

    # Recommendation
    recommendation: Recommendation
    reasoning: str

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "control_metrics": {
                "mean": self.control_metrics.mean,
                "std": self.control_metrics.std,
                "count": self.control_metrics.count,
            },
            "treatment_metrics": {
                "mean": self.treatment_metrics.mean,
                "std": self.treatment_metrics.std,
                "count": self.treatment_metrics.count,
            },
            "p_value": self.p_value,
            "confidence_level": self.confidence_level,
            "effect_size": self.effect_size,
            "confidence_interval": list(self.confidence_interval),
            "recommendation": self.recommendation.value,
            "reasoning": self.reasoning,
        }


@dataclass
class Experiment:
    """An A/B test experiment."""

    experiment_id: str
    proposal: StrategyProposal
    traffic_split: float  # 0.1 = 10% to treatment
    min_samples: int
    significance_level: float
    start_time: float
    status: ExperimentStatus = ExperimentStatus.RUNNING

    # Tracking
    outcomes: list[ExperimentOutcome] = field(default_factory=list)
    control_count: int = 0
    treatment_count: int = 0

    # Results (populated after analysis)
    result: ExperimentResult | None = None
    completed_at: float | None = None

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "proposal": self.proposal.to_dict(),
            "traffic_split": self.traffic_split,
            "min_samples": self.min_samples,
            "significance_level": self.significance_level,
            "start_time": self.start_time,
            "status": self.status.value,
            "control_count": self.control_count,
            "treatment_count": self.treatment_count,
            "result": self.result.to_dict() if self.result else None,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Experiment:
        experiment = cls(
            experiment_id=data["experiment_id"],
            proposal=StrategyProposal(**data["proposal"]),
            traffic_split=data["traffic_split"],
            min_samples=data["min_samples"],
            significance_level=data["significance_level"],
            start_time=data["start_time"],
            status=ExperimentStatus(data["status"]),
            control_count=data.get("control_count", 0),
            treatment_count=data.get("treatment_count", 0),
            completed_at=data.get("completed_at"),
        )

        # Load outcomes
        experiment.outcomes = [ExperimentOutcome.from_dict(o) for o in data.get("outcomes", [])]

        # Load result if present
        if data.get("result"):
            experiment.result = ExperimentResult(**data["result"])

        return experiment


# ─────────────────────────────────────────────
# Statistical Analysis
# ─────────────────────────────────────────────


class StatisticalAnalyzer:
    """
    Statistical analysis for A/B tests.

    Implements two-sample t-test for comparing means between
    control and treatment groups.
    """

    @staticmethod
    def two_sample_t_test(
        control_samples: list[float],
        treatment_samples: list[float],
    ) -> tuple[float, float]:
        """
        Perform two-sample t-test.

        Returns:
            (t_statistic, p_value)
        """
        n1 = len(control_samples)
        n2 = len(treatment_samples)

        if n1 < 2 or n2 < 2:
            return 0.0, 1.0  # Insufficient data

        # Calculate means
        mean1 = sum(control_samples) / n1
        mean2 = sum(treatment_samples) / n2

        # Calculate variances
        var1 = sum((x - mean1) ** 2 for x in control_samples) / (n1 - 1)
        var2 = sum((x - mean2) ** 2 for x in treatment_samples) / (n2 - 1)

        # Calculate t-statistic (Welch's t-test, unequal variances)
        se = math.sqrt(var1 / n1 + var2 / n2)
        if se == 0:
            return 0.0, 1.0

        t_stat = (mean1 - mean2) / se

        # Degrees of freedom (Welch-Satterthwaite equation)
        df_num = (var1 / n1 + var2 / n2) ** 2
        df_den = (var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1)
        df = df_num / df_den if df_den > 0 else n1 + n2 - 2

        # Calculate p-value (two-tailed)
        p_value = StatisticalAnalyzer._t_distribution_cdf(abs(t_stat), df)
        p_value = 2 * (1 - p_value)  # Two-tailed

        return t_stat, p_value

    @staticmethod
    def _t_distribution_cdf(t: float, df: float) -> float:
        """
        Approximate CDF of t-distribution.

        Uses approximation for computational efficiency.
        For production, consider scipy.stats.t.cdf
        """
        # Simple approximation using normal distribution for large df
        if df > 30:
            # Use normal approximation
            return 0.5 * (1 + math.erf(t / math.sqrt(2)))

        # For small df, use beta distribution approximation
        x = df / (df + t * t)
        return 0.5 * StatisticalAnalyzer._incomplete_beta(df / 2, 0.5, x)

    @staticmethod
    def _incomplete_beta(a: float, b: float, x: float) -> float:
        """Approximate incomplete beta function."""
        # Simple approximation for t-distribution CDF
        # This is a simplified version; for production use scipy.special.betainc
        if x <= 0:
            return 0.0
        if x >= 1:
            return 1.0

        # Use continued fraction approximation (simplified)
        return x**a * (1 - x) ** b / a

    @staticmethod
    def cohens_d(
        control_samples: list[float],
        treatment_samples: list[float],
    ) -> float:
        """
        Calculate Cohen's d effect size.

        Interpretation:
          0.2 = small effect
          0.5 = medium effect
          0.8 = large effect
        """
        n1 = len(control_samples)
        n2 = len(treatment_samples)

        if n1 < 2 or n2 < 2:
            return 0.0

        mean1 = sum(control_samples) / n1
        mean2 = sum(treatment_samples) / n2

        # Pooled standard deviation
        var1 = sum((x - mean1) ** 2 for x in control_samples) / (n1 - 1)
        var2 = sum((x - mean2) ** 2 for x in treatment_samples) / (n2 - 1)

        pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        if pooled_std == 0:
            return 0.0

        return (mean1 - mean2) / pooled_std

    @staticmethod
    def confidence_interval(
        samples: list[float],
        confidence: float = 0.95,
    ) -> tuple[float, float]:
        """Calculate confidence interval for mean."""
        n = len(samples)
        if n < 2:
            return (0.0, 0.0)

        mean = sum(samples) / n
        std = math.sqrt(sum((x - mean) ** 2 for x in samples) / (n - 1))

        # Z-score for confidence level (approximation)
        z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
        z = z_scores.get(confidence, 1.96)

        margin = z * std / math.sqrt(n)

        return (mean - margin, mean + margin)


# ─────────────────────────────────────────────
# A/B Testing Engine
# ─────────────────────────────────────────────


class ABTestingEngine:
    """
    A/B testing engine for strategy proposals.

    Manages experiment lifecycle, traffic routing, and statistical analysis.
    """

    def __init__(
        self,
        archive: ExecutionArchive,
        storage_path: Path | None = None,
    ):
        self.archive = archive
        self._storage_path = storage_path or (
            Path.home() / ".orchestrator_cache" / "ab_experiments"
        )
        self._storage_path.mkdir(parents=True, exist_ok=True)

        self._experiments: dict[str, Experiment] = {}
        self._analyzer = StatisticalAnalyzer()
        self._lock = asyncio.Lock()

        self._load_experiments()

    def _load_experiments(self):
        """Load experiments from disk."""
        experiments_file = self._storage_path / "experiments.jsonl"
        if not experiments_file.exists():
            return

        try:
            with open(experiments_file) as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        experiment = Experiment.from_dict(data)
                        self._experiments[experiment.experiment_id] = experiment

            logger.info(f"Loaded {len(self._experiments)} experiments from disk")
        except Exception as e:
            logger.warning(f"Failed to load experiments: {e}")

    def _persist_experiment(self, experiment: Experiment):
        """Persist experiment to disk."""
        experiments_file = self._storage_path / "experiments.jsonl"

        # Read existing, update/add, write back
        experiments = []
        if experiments_file.exists():
            with open(experiments_file) as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        if data["experiment_id"] != experiment.experiment_id:
                            experiments.append(data)

        experiments.append(experiment.to_dict())

        with open(experiments_file, "w") as f:
            for exp in experiments:
                f.write(json.dumps(exp) + "\n")

    async def create_experiment(
        self,
        proposal: StrategyProposal,
        traffic_split: float = 0.1,
        min_samples: int = DEFAULT_MIN_SAMPLES,
        significance_level: float = DEFAULT_SIGNIFICANCE_LEVEL,
    ) -> Experiment:
        """
        Create a new A/B test experiment.

        Args:
            proposal: The strategy proposal to test
            traffic_split: Fraction of traffic to treatment (0.1 = 10%)
            min_samples: Minimum samples per variant for analysis
            significance_level: P-value threshold for significance

        Returns:
            Created experiment
        """
        async with self._lock:
            experiment_id = f"exp_{proposal.proposal_id}_{int(time.time())}"

            experiment = Experiment(
                experiment_id=experiment_id,
                proposal=proposal,
                traffic_split=traffic_split,
                min_samples=min_samples,
                significance_level=significance_level,
                start_time=time.time(),
            )

            self._experiments[experiment_id] = experiment
            self._persist_experiment(experiment)

            logger.info(f"Created experiment {experiment_id} for proposal {proposal.proposal_id}")
            return experiment

    async def route_execution(self, project_id: str) -> Variant:
        """
        Route a project to control or treatment group.

        Uses consistent hashing to ensure same project always gets same variant.

        Args:
            project_id: Project identifier

        Returns:
            Assigned variant (CONTROL or TREATMENT)
        """
        # Find active experiments
        active_experiments = [
            exp for exp in self._experiments.values() if exp.status == ExperimentStatus.RUNNING
        ]

        if not active_experiments:
            return Variant.CONTROL

        # Use most recent experiment for routing
        experiment = max(active_experiments, key=lambda e: e.start_time)

        # Consistent hashing based on project_id
        hash_value = int(
            hashlib.sha256(f"{project_id}:{experiment.experiment_id}".encode()).hexdigest(), 16
        )
        normalized = (hash_value % 1000) / 1000  # 0.0 to 1.0

        if normalized < experiment.traffic_split:
            return Variant.TREATMENT
        else:
            return Variant.CONTROL

    async def record_outcome(
        self,
        experiment_id: str,
        variant: Variant,
        project_id: str,
        success: bool,
        score: float,
        cost_usd: float,
        latency_ms: float,
    ) -> ExperimentOutcome:
        """
        Record an outcome for an experiment.

        Args:
            experiment_id: Experiment identifier
            variant: Which variant was tested
            project_id: Project that was executed
            success: Whether execution succeeded
            score: Quality score (0-1)
            cost_usd: Actual cost
            latency_ms: Execution latency

        Returns:
            Recorded outcome
        """
        async with self._lock:
            if experiment_id not in self._experiments:
                raise ValueError(f"Experiment {experiment_id} not found")

            experiment = self._experiments[experiment_id]

            outcome = ExperimentOutcome(
                outcome_id=f"out_{int(time.time() * 1000)}",
                experiment_id=experiment_id,
                variant=variant,
                project_id=project_id,
                success=success,
                score=score,
                cost_usd=cost_usd,
                latency_ms=latency_ms,
            )

            experiment.outcomes.append(outcome)

            if variant == Variant.CONTROL:
                experiment.control_count += 1
            else:
                experiment.treatment_count += 1

            self._persist_experiment(experiment)

            logger.debug(
                f"Recorded outcome for {experiment_id}: "
                f"{variant.value} success={success} score={score:.3f}"
            )

            return outcome

    async def analyze_results(self, experiment_id: str) -> ExperimentResult | None:
        """
        Analyze results of an experiment.

        Performs statistical analysis if minimum samples are met.

        Args:
            experiment_id: Experiment identifier

        Returns:
            Analysis result, or None if insufficient data
        """
        async with self._lock:
            if experiment_id not in self._experiments:
                raise ValueError(f"Experiment {experiment_id} not found")

            experiment = self._experiments[experiment_id]

            # Check minimum samples
            if (
                experiment.control_count < experiment.min_samples
                or experiment.treatment_count < experiment.min_samples
            ):
                logger.info(
                    f"Insufficient samples for {experiment_id}: "
                    f"control={experiment.control_count}, "
                    f"treatment={experiment.treatment_count}, "
                    f"min={experiment.min_samples}"
                )
                return None

            # Extract scores by variant
            control_scores = [o.score for o in experiment.outcomes if o.variant == Variant.CONTROL]
            treatment_scores = [
                o.score for o in experiment.outcomes if o.variant == Variant.TREATMENT
            ]

            # Perform t-test
            t_stat, p_value = self._analyzer.two_sample_t_test(control_scores, treatment_scores)

            # Calculate effect size
            effect_size = self._analyzer.cohens_d(control_scores, treatment_scores)

            # Calculate confidence interval for difference
            control_mean = sum(control_scores) / len(control_scores)
            treatment_mean = sum(treatment_scores) / len(treatment_scores)
            diff = treatment_mean - control_mean

            # Pooled confidence interval
            all_scores = control_scores + treatment_scores
            ci_margin = (
                self._analyzer.confidence_interval(all_scores)[1]
                - self._analyzer.confidence_interval(all_scores)[0]
            )
            ci_margin = ci_margin / 2
            confidence_interval = (diff - ci_margin, diff + ci_margin)

            # Determine recommendation
            if p_value < experiment.significance_level:
                if effect_size < 0:
                    recommendation = Recommendation.ADOPT
                    reasoning = (
                        f"Treatment significantly better (p={p_value:.4f}, "
                        f"d={abs(effect_size):.3f})"
                    )
                else:
                    recommendation = Recommendation.REJECT
                    reasoning = (
                        f"Treatment significantly worse (p={p_value:.4f}, " f"d={effect_size:.3f})"
                    )
            else:
                recommendation = Recommendation.REJECT  # Default to reject if not significant
                reasoning = (
                    f"No significant difference (p={p_value:.4f}, " f"d={abs(effect_size):.3f})"
                )

            # Create result
            result = ExperimentResult(
                experiment_id=experiment_id,
                control_metrics=MetricSummary.from_samples(control_scores),
                treatment_metrics=MetricSummary.from_samples(treatment_scores),
                p_value=p_value,
                confidence_level=1 - p_value,
                effect_size=abs(effect_size),
                confidence_interval=confidence_interval,
                recommendation=recommendation,
                reasoning=reasoning,
            )

            experiment.result = result
            experiment.status = ExperimentStatus.COMPLETED
            experiment.completed_at = time.time()

            self._persist_experiment(experiment)

            logger.info(
                f"Experiment {experiment_id} complete: " f"{recommendation.value} - {reasoning}"
            )

            return result

    async def get_active_experiments(self) -> list[Experiment]:
        """Get all running experiments."""
        return [exp for exp in self._experiments.values() if exp.status == ExperimentStatus.RUNNING]

    async def get_experiment(self, experiment_id: str) -> Experiment | None:
        """Get experiment by ID."""
        return self._experiments.get(experiment_id)

    async def pause_experiment(self, experiment_id: str) -> bool:
        """Pause a running experiment."""
        async with self._lock:
            if experiment_id not in self._experiments:
                return False

            experiment = self._experiments[experiment_id]
            if experiment.status != ExperimentStatus.RUNNING:
                return False

            experiment.status = ExperimentStatus.PAUSED
            self._persist_experiment(experiment)

            logger.info(f"Paused experiment {experiment_id}")
            return True

    async def invalidate_experiment(self, experiment_id: str, reason: str) -> bool:
        """Invalidate an experiment (e.g., due to contamination)."""
        async with self._lock:
            if experiment_id not in self._experiments:
                return False

            experiment = self._experiments[experiment_id]
            experiment.status = ExperimentStatus.INVALIDATED
            self._persist_experiment(experiment)

            logger.info(f"Invalidated experiment {experiment_id}: {reason}")
            return True

    def get_experiment_stats(self) -> dict[str, Any]:
        """Get overall experiment statistics."""
        return {
            "total_experiments": len(self._experiments),
            "active_experiments": len(
                [e for e in self._experiments.values() if e.status == ExperimentStatus.RUNNING]
            ),
            "completed_experiments": len(
                [e for e in self._experiments.values() if e.status == ExperimentStatus.COMPLETED]
            ),
            "total_outcomes": sum(len(e.outcomes) for e in self._experiments.values()),
        }


# ─────────────────────────────────────────────
# Advanced A/B Testing Features
# ─────────────────────────────────────────────


@dataclass
class SequentialTestConfig:
    """Configuration for sequential testing with early stopping."""

    min_samples: int = 10
    max_samples: int = 1000
    stopping_threshold: float = 0.95  # Confidence for early stop
    check_interval: int = 5  # Check every N samples


class SequentialABTest:
    """
    A/B test with early stopping capability.

    Monitors experiment progress and stops early if result is clear,
    saving time and resources.
    """

    def __init__(self, config: SequentialTestConfig | None = None):
        self.config = config or SequentialTestConfig()

    async def check_early_stopping(
        self,
        experiment: Experiment,
    ) -> tuple[bool, str]:
        """
        Check if experiment should stop early.

        Args:
            experiment: Experiment to check

        Returns:
            (should_stop, reason)
        """
        total_samples = experiment.control_count + experiment.treatment_count

        # Check minimum samples
        if total_samples < self.config.min_samples:
            return False, f"Insufficient samples ({total_samples} < {self.config.min_samples})"

        # Check maximum samples
        if total_samples >= self.config.max_samples:
            return True, f"Maximum samples reached ({self.config.max_samples})"

        # Check if enough data for analysis
        if (
            experiment.control_count >= self.config.min_samples
            and experiment.treatment_count >= self.config.min_samples
        ):

            # Perform interim analysis
            control_scores = [o.score for o in experiment.outcomes if o.variant == Variant.CONTROL]
            treatment_scores = [
                o.score for o in experiment.outcomes if o.variant == Variant.TREATMENT
            ]

            if control_scores and treatment_scores:
                t_stat, p_value = StatisticalAnalyzer.two_sample_t_test(
                    control_scores, treatment_scores
                )

                # Check for clear winner
                if p_value < (1 - self.config.stopping_threshold):
                    control_mean = sum(control_scores) / len(control_scores)
                    treatment_mean = sum(treatment_scores) / len(treatment_scores)

                    if treatment_mean > control_mean:
                        return True, f"Clear winner: treatment (p={p_value:.4f})"
                    else:
                        return True, f"Clear winner: control (p={p_value:.4f})"

        return False, f"Continue monitoring ({total_samples} samples)"


@dataclass
class BanditArm:
    """A bandit arm with Thompson Sampling."""

    arm_id: str
    successes: int = 0
    failures: int = 0

    @property
    def alpha(self) -> float:
        """Beta distribution alpha parameter."""
        return self.successes + 1

    @property
    def beta(self) -> float:
        """Beta distribution beta parameter."""
        return self.failures + 1

    def sample(self) -> float:
        """Sample from posterior distribution."""
        import random

        return random.betavariate(self.alpha, self.beta)

    def update(self, success: bool):
        """Update arm with outcome."""
        if success:
            self.successes += 1
        else:
            self.failures += 1


class MultiArmedBandit:
    """
    Thompson Sampling for dynamic traffic allocation.

    Automatically allocates more traffic to better-performing variants
    while maintaining exploration.
    """

    def __init__(self):
        self._arms: dict[str, BanditArm] = {}

    def add_arm(self, arm_id: str):
        """Add a new arm to the bandit."""
        self._arms[arm_id] = BanditArm(arm_id=arm_id)

    def allocate_traffic(
        self,
        total_traffic: float = 1.0,
    ) -> dict[str, float]:
        """
        Allocate traffic based on Thompson Sampling.

        Args:
            total_traffic: Total traffic to allocate (default 1.0 = 100%)

        Returns:
            Dictionary mapping arm_id to traffic allocation
        """
        if not self._arms:
            return {}

        # Sample from each arm's posterior
        samples = {arm_id: arm.sample() for arm_id, arm in self._arms.items()}

        # Find winner (highest sample)
        winner = max(samples.keys(), key=lambda k: samples[k])

        # Allocate traffic (epsilon-greedy with Thompson Sampling)
        epsilon = 0.1  # 10% exploration
        allocations = {}

        for arm_id in self._arms:
            if arm_id == winner:
                allocations[arm_id] = total_traffic * (1 - epsilon + epsilon / len(self._arms))
            else:
                allocations[arm_id] = total_traffic * (epsilon / len(self._arms))

        return allocations

    def update_arm(self, arm_id: str, success: bool):
        """Update arm with outcome."""
        if arm_id in self._arms:
            self._arms[arm_id].update(success)

    def get_stats(self) -> dict[str, Any]:
        """Get bandit statistics."""
        return {
            arm_id: {
                "successes": arm.successes,
                "failures": arm.failures,
                "success_rate": arm.successes / max(arm.successes + arm.failures, 1),
            }
            for arm_id, arm in self._arms.items()
        }


class CUPEDAdjustment:
    """
    Controlled-Experiment Using Pre-Experiment Data (CUPED).

    Reduces variance in A/B test metrics by adjusting for pre-experiment
    covariates, improving sensitivity to detect true effects.
    """

    def __init__(self, theta: float | None = None):
        """
        Initialize CUPED adjustment.

        Args:
            theta: Variance reduction parameter. If None, estimated from data.
        """
        self.theta = theta

    def adjust_metrics(
        self,
        treatment: list[float],
        control: list[float],
        pre_experiment: list[float],
    ) -> tuple[list[float], list[float]]:
        """
        Adjust metrics using CUPED.

        Args:
            treatment: Treatment group metrics
            control: Control group metrics
            pre_experiment: Pre-experiment covariate (same for both groups)

        Returns:
            (adjusted_treatment, adjusted_control)
        """
        if len(treatment) != len(pre_experiment) or len(control) != len(pre_experiment):
            raise ValueError("All lists must have same length")

        # Estimate theta if not provided
        if self.theta is None:
            self.theta = self._estimate_theta(control, pre_experiment)

        # Calculate means
        mean_pre = sum(pre_experiment) / len(pre_experiment)

        # Adjust metrics
        adjusted_treatment = [
            y - self.theta * (x - mean_pre) for y, x in zip(treatment, pre_experiment, strict=False)
        ]

        adjusted_control = [
            y - self.theta * (x - mean_pre) for y, x in zip(control, pre_experiment, strict=False)
        ]

        return adjusted_treatment, adjusted_control

    def _estimate_theta(
        self,
        metrics: list[float],
        covariate: list[float],
    ) -> float:
        """
        Estimate theta from covariance.

        theta = Cov(Y, X) / Var(X)
        """
        n = len(metrics)
        if n < 2:
            return 0.0

        mean_y = sum(metrics) / n
        mean_x = sum(covariate) / n

        # Covariance
        cov = sum((y - mean_y) * (x - mean_x) for y, x in zip(metrics, covariate, strict=False)) / (
            n - 1
        )

        # Variance of covariate
        var_x = sum((x - mean_x) ** 2 for x in covariate) / (n - 1)

        if var_x == 0:
            return 0.0

        return cov / var_x

    def variance_reduction(
        self,
        original: list[float],
        adjusted: list[float],
    ) -> float:
        """
        Calculate variance reduction achieved.

        Returns:
            Percentage variance reduction (0-1)
        """
        if len(original) < 2:
            return 0.0

        var_original = sum((x - sum(original) / len(original)) ** 2 for x in original) / (
            len(original) - 1
        )
        var_adjusted = sum((x - sum(adjusted) / len(adjusted)) ** 2 for x in adjusted) / (
            len(adjusted) - 1
        )

        if var_original == 0:
            return 0.0

        return 1 - (var_adjusted / var_original)
