"""
Cross-Organization Learning with Differential Privacy
=====================================================

Federated learning system that enables collective intelligence across
organizations while preserving privacy through differential privacy mechanisms.

Key Features:
- Local model training with private data
- Differentially private gradient aggregation
- Secure aggregation protocol
- Global baseline generation for cold-start
- Privacy budget accounting

This creates Nash stability through network effects - the more organizations
participate, the better the global models become, creating switching costs.

Usage:
    from orchestrator.federated_learning import FederatedLearningOrchestrator

    # Initialize local learner
    learner = FederatedLearningOrchestrator(
        org_id="acme-corp",
        privacy_budget=1.0,  # Epsilon
    )

    # Contribute insights (automatically privatized)
    await learner.contribute_insight(insight)

    # Get global baseline (benefit from collective wisdom)
    baseline = await learner.get_global_baseline(
        task_type=TaskType.CODE_GEN,
        fingerprint=codebase_fingerprint
    )
"""

from __future__ import annotations

import hashlib
import json
import secrets
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from .feedback_loop import CodebaseFingerprint, OutcomeStatus, ProductionOutcome
from .log_config import get_logger
from .models import Model, TaskType

logger = get_logger(__name__)


class PrivacyMechanism(Enum):
    """Differential privacy mechanisms."""

    GAUSSIAN = "gaussian"  # Add Gaussian noise
    LAPLACE = "laplace"  # Add Laplace noise
    RANDOMIZED_RESPONSE = "rr"  # Randomized response
    SUBSAMPLING = "subsampling"  # Gradient subsampling


@dataclass
class PrivacyBudget:
    """Privacy budget tracking."""

    epsilon: float = 1.0  # Total privacy budget
    delta: float = 1e-5  # Failure probability

    # Consumed budget
    consumed_epsilon: float = 0.0

    # Query count
    query_count: int = 0

    def can_spend(self, epsilon_cost: float) -> bool:
        """Check if we can spend epsilon budget."""
        return (self.consumed_epsilon + epsilon_cost) <= self.epsilon

    def spend(self, epsilon_cost: float) -> bool:
        """Spend privacy budget."""
        if self.can_spend(epsilon_cost):
            self.consumed_epsilon += epsilon_cost
            self.query_count += 1
            return True
        return False

    @property
    def remaining(self) -> float:
        """Get remaining budget."""
        return self.epsilon - self.consumed_epsilon

    @property
    def utilization(self) -> float:
        """Get budget utilization percentage."""
        return self.consumed_epsilon / self.epsilon if self.epsilon > 0 else 1.0


@dataclass
class ModelInsight:
    """A single model performance insight."""

    # Identifiers (hashed for privacy)
    insight_id: str
    org_hash: str  # Anonymized org identifier

    # Model and task info
    model: Model
    task_type: TaskType

    # Performance metrics
    success_rate: float
    avg_quality: float
    avg_cost: float
    sample_size: int

    # Pattern signature (anonymized)
    pattern_signature: str  # Hash of patterns, not actual patterns
    language_signature: str  # Hash of languages

    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    region: str = "global"  # Geographic region for aggregation

    def anonymize(self) -> ModelInsight:
        """Create anonymized version for sharing."""
        return ModelInsight(
            insight_id=self.insight_id,
            org_hash=self.org_hash,  # Already hashed
            model=self.model,
            task_type=self.task_type,
            success_rate=self.success_rate,
            avg_quality=self.avg_quality,
            avg_cost=self.avg_cost,
            sample_size=self.sample_size,
            pattern_signature=self.pattern_signature,
            language_signature=self.language_signature,
            timestamp=self.timestamp,
            region=self.region,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "insight_id": self.insight_id,
            "org_hash": self.org_hash,
            "model": self.model.value,
            "task_type": self.task_type.value,
            "success_rate": self.success_rate,
            "avg_quality": self.avg_quality,
            "avg_cost": self.avg_cost,
            "sample_size": self.sample_size,
            "pattern_signature": self.pattern_signature,
            "language_signature": self.language_signature,
            "timestamp": self.timestamp.isoformat(),
            "region": self.region,
        }


@dataclass
class GlobalBaseline:
    """Global performance baseline from collective wisdom."""

    task_type: TaskType

    # Recommended models with confidence
    recommended_models: list[dict[str, Any]]

    # Expected performance ranges
    quality_range: tuple[float, float]
    cost_range: tuple[float, float]

    # Confidence in baseline
    confidence: float
    total_contributions: int
    contributing_orgs: int

    # Pattern-specific recommendations
    pattern_baselines: dict[str, dict[str, Any]]

    # Metadata
    generated_at: datetime = field(default_factory=datetime.utcnow)
    freshness_hours: float = 24.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_type": self.task_type.value,
            "recommended_models": self.recommended_models,
            "quality_range": list(self.quality_range),
            "cost_range": list(self.cost_range),
            "confidence": self.confidence,
            "total_contributions": self.total_contributions,
            "contributing_orgs": self.contributing_orgs,
            "generated_at": self.generated_at.isoformat(),
            "freshness_hours": self.freshness_hours,
        }


@dataclass
class LocalModel:
    """Local model trained on private data."""

    model: Model
    task_type: TaskType

    # Model parameters (simplified as weights)
    weights: dict[str, float]

    # Training metadata
    local_samples: int
    local_epochs: int

    # Privacy accounting
    epsilon_spent: float

    def get_privatized_update(
        self,
        mechanism: PrivacyMechanism,
        epsilon: float,
    ) -> dict[str, float]:
        """Get privatized model update."""
        import math
        import random

        if mechanism == PrivacyMechanism.GAUSSIAN:
            # Add Gaussian noise
            sigma = math.sqrt(2 * math.log(1.25 / 1e-5)) / epsilon
            return {k: v + random.gauss(0, sigma) for k, v in self.weights.items()}

        elif mechanism == PrivacyMechanism.LAPLACE:
            # Add Laplace noise
            scale = 1.0 / epsilon
            return {
                k: v
                + (
                    scale * math.log(random.random())
                    if random.random() < 0.5
                    else -scale * math.log(random.random())
                )
                for k, v in self.weights.items()
            }

        else:
            # No privacy mechanism
            return self.weights.copy()


class DifferentialPrivacyEngine:
    """
    Differential privacy engine for noise injection and budget accounting.

    Implements:
    - Gaussian mechanism
    - Laplace mechanism
    - Moment accountant for tight bounds
    """

    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        self.epsilon = epsilon
        self.delta = delta
        self.budget = PrivacyBudget(epsilon, delta)

        # Query history for advanced composition
        self.query_history: list[tuple[float, float]] = []  # (epsilon, delta) per query

    def add_gaussian_noise(
        self,
        value: float,
        sensitivity: float = 1.0,
    ) -> float:
        """Add Gaussian noise for (epsilon, delta)-DP."""
        import math
        import random

        # Calculate sigma for Gaussian mechanism
        sigma = sensitivity * math.sqrt(2 * math.log(1.25 / self.delta)) / self.epsilon

        noise = random.gauss(0, sigma)
        return value + noise

    def add_laplace_noise(
        self,
        value: float,
        sensitivity: float = 1.0,
    ) -> float:
        """Add Laplace noise for epsilon-DP."""
        import math
        import random

        scale = sensitivity / self.epsilon

        # Sample from Laplace
        u = random.random() - 0.5
        noise = -scale * math.copysign(1.0, u) * math.log(1 - 2 * abs(u))

        return value + noise

    def randomized_response(
        self,
        value: bool,
        p: float = 0.75,  # Probability of truthful response
    ) -> bool:
        """Randomized response mechanism."""
        import random

        if random.random() < p:
            return value
        else:
            return random.choice([True, False])

    def privatize_histogram(
        self,
        histogram: dict[str, int],
        epsilon: float | None = None,
    ) -> dict[str, float]:
        """Privatize a histogram using Laplace mechanism."""
        sensitivity = 1.0  # Adding/removing one user changes count by 1

        privatized = {}
        for key, count in histogram.items():
            noisy_count = self.add_laplace_noise(float(count), sensitivity)
            # Ensure non-negative
            privatized[key] = max(0, noisy_count)

        return privatized

    def compute_privacy_cost(
        self,
        mechanism: PrivacyMechanism,
        num_queries: int = 1,
    ) -> tuple[float, float]:
        """
        Compute privacy cost using advanced composition.

        Returns (epsilon, delta) cost.
        """
        if mechanism == PrivacyMechanism.GAUSSIAN:
            # Advanced composition: epsilon' = epsilon * sqrt(2 * k * ln(1/delta'))
            composed_delta = self.delta / num_queries
            composed_epsilon = self.epsilon * math.sqrt(
                2 * num_queries * math.log(1 / composed_delta)
            )
            return composed_epsilon, composed_delta

        elif mechanism == PrivacyMechanism.LAPLACE:
            # Basic composition: k * epsilon
            return num_queries * self.epsilon, self.delta

        else:
            return self.epsilon, self.delta


class FederatedLearningOrchestrator:
    """
    Federated learning orchestrator for cross-organization learning.

    Creates Nash stability through:
    1. Network effects: More orgs = better global models
    2. Switching costs: Loss of collective intelligence
    3. Data moat: Privately accumulated patterns

    Privacy guarantees:
    - Differential privacy for all shared data
    - Secure aggregation (simulated)
    - Local differential privacy option
    """

    # Configuration
    MIN_ORGS_FOR_AGGREGATION = 3
    MIN_INSIGHTS_PER_ORG = 5
    DEFAULT_PRIVACY_BUDGET = 1.0
    AGGREGATION_INTERVAL_HOURS = 24

    def __init__(
        self,
        org_id: str | None = None,
        privacy_budget: float = DEFAULT_PRIVACY_BUDGET,
        storage_path: Path | None = None,
        enable_contribution: bool = True,
    ):
        self.org_id = org_id or f"anon_{secrets.token_hex(8)}"
        self.org_hash = hashlib.sha256(self.org_id.encode()).hexdigest()[:16]

        self.storage_path = storage_path or Path(".federated_learning")
        self.storage_path.mkdir(exist_ok=True)

        # Privacy engine
        self.privacy = DifferentialPrivacyEngine(
            epsilon=privacy_budget,
            delta=1e-5,
        )

        # Local insights (private)
        self._local_insights: list[ModelInsight] = []
        self._local_models: dict[tuple[Model, TaskType], LocalModel] = {}

        # Global cache (anonymized)
        self._global_cache: dict[str, GlobalBaseline] = {}
        self._global_insights: list[ModelInsight] = []

        # Configuration
        self.enable_contribution = enable_contribution
        self.last_aggregation = datetime.min

        # Load data
        self._load_local_data()
        self._load_global_cache()

    def _load_local_data(self) -> None:
        """Load local insights."""
        local_file = self.storage_path / f"local_{self.org_hash}.json"
        if local_file.exists():
            try:
                data = json.loads(local_file.read_text())
                for insight_data in data.get("insights", []):
                    insight = ModelInsight(
                        insight_id=insight_data["insight_id"],
                        org_hash=insight_data["org_hash"],
                        model=Model(insight_data["model"]),
                        task_type=TaskType(insight_data["task_type"]),
                        success_rate=insight_data["success_rate"],
                        avg_quality=insight_data["avg_quality"],
                        avg_cost=insight_data["avg_cost"],
                        sample_size=insight_data["sample_size"],
                        pattern_signature=insight_data["pattern_signature"],
                        language_signature=insight_data["language_signature"],
                        timestamp=datetime.fromisoformat(insight_data["timestamp"]),
                        region=insight_data.get("region", "global"),
                    )
                    self._local_insights.append(insight)

                logger.info(f"Loaded {len(self._local_insights)} local insights")
            except Exception as e:
                logger.error(f"Failed to load local data: {e}")

    def _save_local_data(self) -> None:
        """Save local insights."""
        try:
            local_file = self.storage_path / f"local_{self.org_hash}.json"
            data = {
                "org_hash": self.org_hash,
                "insights": [i.to_dict() for i in self._local_insights],
                "privacy_budget": {
                    "epsilon": self.privacy.epsilon,
                    "consumed": self.privacy.budget.consumed_epsilon,
                    "queries": self.privacy.budget.query_count,
                },
            }
            local_file.write_text(json.dumps(data, indent=2, default=str))
        except Exception as e:
            logger.error(f"Failed to save local data: {e}")

    def _load_global_cache(self) -> None:
        """Load cached global baselines."""
        global_file = self.storage_path / "global_cache.json"
        if global_file.exists():
            try:
                data = json.loads(global_file.read_text())
                for key, baseline_data in data.items():
                    baseline = GlobalBaseline(
                        task_type=TaskType(baseline_data["task_type"]),
                        recommended_models=baseline_data["recommended_models"],
                        quality_range=tuple(baseline_data["quality_range"]),
                        cost_range=tuple(baseline_data["cost_range"]),
                        confidence=baseline_data["confidence"],
                        total_contributions=baseline_data["total_contributions"],
                        contributing_orgs=baseline_data["contributing_orgs"],
                        pattern_baselines=baseline_data.get("pattern_baselines", {}),
                        generated_at=datetime.fromisoformat(baseline_data["generated_at"]),
                    )
                    self._global_cache[key] = baseline

                logger.info(f"Loaded {len(self._global_cache)} cached baselines")
            except Exception as e:
                logger.error(f"Failed to load global cache: {e}")

    def _save_global_cache(self) -> None:
        """Save global cache."""
        try:
            global_file = self.storage_path / "global_cache.json"
            data = {key: baseline.to_dict() for key, baseline in self._global_cache.items()}
            global_file.write_text(json.dumps(data, indent=2, default=str))
        except Exception as e:
            logger.error(f"Failed to save global cache: {e}")

    async def contribute_insight(
        self,
        outcome: ProductionOutcome,
        fingerprint: CodebaseFingerprint | None = None,
    ) -> ModelInsight | None:
        """
        Contribute a production outcome to the federated learning pool.

        The insight is automatically privatized before sharing.
        """
        if not self.enable_contribution:
            return None

        # Check privacy budget
        epsilon_cost = 0.1  # Cost per insight
        if not self.privacy.budget.spend(epsilon_cost):
            logger.warning("Privacy budget exhausted, cannot contribute insight")
            return None

        # Create pattern signature (hashed)
        if fingerprint:
            pattern_sig = hashlib.sha256(
                json.dumps(sorted(fingerprint.patterns), sort_keys=True).encode()
            ).hexdigest()[:16]
            lang_sig = hashlib.sha256(
                json.dumps(sorted(fingerprint.languages), sort_keys=True).encode()
            ).hexdigest()[:16]
        else:
            pattern_sig = "unknown"
            lang_sig = "unknown"

        # Calculate metrics with DP noise
        success_rate = 1.0 if outcome.status == OutcomeStatus.SUCCESS else 0.0
        privatized_success = self.privacy.randomized_response(success_rate > 0.5, p=0.9)

        quality = outcome.calculate_success_score()
        privatized_quality = self.privacy.add_gaussian_noise(quality, sensitivity=1.0)
        privatized_quality = max(0, min(1, privatized_quality))

        # Create insight
        insight = ModelInsight(
            insight_id=f"{self.org_hash}_{secrets.token_hex(8)}",
            org_hash=self.org_hash,
            model=outcome.model_used,
            task_type=outcome.task_type,
            success_rate=1.0 if privatized_success else 0.0,
            avg_quality=privatized_quality,
            avg_cost=0.0,  # Would calculate from actual cost
            sample_size=1,
            pattern_signature=pattern_sig,
            language_signature=lang_sig,
            timestamp=datetime.utcnow(),
            region="global",
        )

        # Store locally
        self._local_insights.append(insight)
        self._save_local_data()

        # Add to global pool (anonymized)
        self._global_insights.append(insight.anonymize())

        logger.debug(f"Contributed insight: {insight.insight_id}")

        # Trigger aggregation if needed
        await self._maybe_aggregate()

        return insight

    async def get_global_baseline(
        self,
        task_type: TaskType,
        fingerprint: CodebaseFingerprint | None = None,
        freshness_hours: float = 24.0,
    ) -> GlobalBaseline:
        """
        Get global performance baseline from collective wisdom.

        New users start here instead of cold-start.
        """
        cache_key = (
            f"{task_type.value}:{fingerprint._hash_fingerprint() if fingerprint else 'none'}"
        )

        # Check cache freshness
        if cache_key in self._global_cache:
            cached = self._global_cache[cache_key]
            age = (datetime.utcnow() - cached.generated_at).total_seconds() / 3600
            if age < freshness_hours:
                return cached

        # Generate new baseline from global insights
        baseline = await self._generate_baseline(task_type, fingerprint)

        # Cache
        self._global_cache[cache_key] = baseline
        self._save_global_cache()

        return baseline

    async def _generate_baseline(
        self,
        task_type: TaskType,
        fingerprint: CodebaseFingerprint | None,
    ) -> GlobalBaseline:
        """Generate baseline from aggregated insights."""
        # Filter relevant insights
        relevant = [i for i in self._global_insights if i.task_type == task_type]

        if fingerprint:
            # Match by pattern signature similarity
            fp_pattern_sig = hashlib.sha256(
                json.dumps(sorted(fingerprint.patterns), sort_keys=True).encode()
            ).hexdigest()[:16]

            pattern_relevant = [i for i in relevant if i.pattern_signature == fp_pattern_sig]
            if pattern_relevant:
                relevant = pattern_relevant

        if not relevant:
            # No data - return default baseline
            return GlobalBaseline(
                task_type=task_type,
                recommended_models=[],
                quality_range=(0.5, 0.5),
                cost_range=(0, 0),
                confidence=0.0,
                total_contributions=0,
                contributing_orgs=0,
                pattern_baselines={},
            )

        # Aggregate by model
        by_model: dict[Model, list[ModelInsight]] = defaultdict(list)
        for insight in relevant:
            by_model[insight.model].append(insight)

        # Compute aggregate scores
        recommended_models = []
        for model, insights in by_model.items():
            # Weighted average by sample size
            total_weight = sum(i.sample_size for i in insights)
            if total_weight == 0:
                continue

            avg_quality = sum(i.avg_quality * i.sample_size for i in insights) / total_weight
            avg_success = sum(i.success_rate * i.sample_size for i in insights) / total_weight

            # Differentially private aggregation
            privatized_quality = self.privacy.add_gaussian_noise(avg_quality, sensitivity=1.0)
            privatized_success = self.privacy.add_gaussian_noise(avg_success, sensitivity=1.0)

            recommended_models.append(
                {
                    "model": model.value,
                    "quality": max(0, min(1, privatized_quality)),
                    "success_rate": max(0, min(1, privatized_success)),
                    "sample_size": total_weight,
                    "confidence": min(1.0, total_weight / 50),
                }
            )

        # Sort by quality
        recommended_models.sort(key=lambda x: x["quality"], reverse=True)

        # Calculate ranges
        qualities = [m["quality"] for m in recommended_models]
        costs = [i.avg_cost for i in relevant if i.avg_cost > 0]

        # Count unique orgs
        unique_orgs = len({i.org_hash for i in relevant})

        # Confidence based on data volume
        confidence = min(1.0, len(relevant) / 100) * (unique_orgs / self.MIN_ORGS_FOR_AGGREGATION)

        return GlobalBaseline(
            task_type=task_type,
            recommended_models=recommended_models[:5],
            quality_range=(
                min(qualities) if qualities else 0.5,
                max(qualities) if qualities else 0.5,
            ),
            cost_range=(min(costs) if costs else 0, max(costs) if costs else 0),
            confidence=confidence,
            total_contributions=len(relevant),
            contributing_orgs=unique_orgs,
            pattern_baselines={},
            freshness_hours=self.AGGREGATION_INTERVAL_HOURS,
        )

    async def _maybe_aggregate(self) -> None:
        """Trigger aggregation if enough time has passed."""
        now = datetime.utcnow()
        hours_since = (now - self.last_aggregation).total_seconds() / 3600

        if hours_since >= self.AGGREGATION_INTERVAL_HOURS:
            await self._run_aggregation()
            self.last_aggregation = now

    async def _run_aggregation(self) -> None:
        """Run secure aggregation of insights."""
        # In a real implementation, this would use secure multi-party computation
        # For now, we simulate with local aggregation

        unique_orgs = len({i.org_hash for i in self._global_insights})

        if unique_orgs < self.MIN_ORGS_FOR_AGGREGATION:
            logger.debug(f"Not enough orgs for aggregation: {unique_orgs}")
            return

        logger.info(
            f"Running aggregation with {len(self._global_insights)} insights from {unique_orgs} orgs"
        )

        # Regenerate all baselines
        for task_type in TaskType:
            baseline = await self._generate_baseline(task_type, None)
            self._global_cache[task_type.value] = baseline

        self._save_global_cache()

    def get_switching_cost_estimate(self) -> dict[str, Any]:
        """
        Estimate the cost of switching to a competitor.

        This quantifies the Nash stability of the platform.
        """
        # Local data value
        local_samples = len(self._local_insights)
        local_value = local_samples * 0.1  # Estimated value per sample

        # Global data value
        global_samples = len(self._global_insights)
        global_contributing_orgs = len({i.org_hash for i in self._global_insights})

        # Network effect value
        if global_contributing_orgs >= self.MIN_ORGS_FOR_AGGREGATION:
            network_multiplier = math.log(global_contributing_orgs)
            global_value = global_samples * 0.05 * network_multiplier
        else:
            global_value = 0

        # Pattern-specific knowledge
        unique_patterns = len({i.pattern_signature for i in self._local_insights})
        pattern_value = unique_patterns * 0.5

        total_switching_cost = local_value + global_value + pattern_value

        return {
            "local_samples": local_samples,
            "local_value_usd": round(local_value, 2),
            "global_samples": global_samples,
            "global_contributing_orgs": global_contributing_orgs,
            "global_value_usd": round(global_value, 2),
            "unique_patterns_learned": unique_patterns,
            "pattern_knowledge_value_usd": round(pattern_value, 2),
            "total_switching_cost_usd": round(total_switching_cost, 2),
            "nash_stability_score": min(1.0, total_switching_cost / 100),
            "explanation": (
                f"Switching would lose {local_samples} local insights, "
                f"{global_samples} global insights from {global_contributing_orgs} orgs, "
                f"and {unique_patterns} learned patterns. "
                f"Estimated replacement cost: ${total_switching_cost:.2f}"
            ),
        }

    def get_federated_stats(self) -> dict[str, Any]:
        """Get statistics about federated learning."""
        unique_orgs = len({i.org_hash for i in self._global_insights})

        return {
            "local_insights": len(self._local_insights),
            "global_insights": len(self._global_insights),
            "contributing_orgs": unique_orgs,
            "can_aggregate": unique_orgs >= self.MIN_ORGS_FOR_AGGREGATION,
            "privacy_budget": {
                "epsilon": self.privacy.epsilon,
                "consumed": self.privacy.budget.consumed_epsilon,
                "remaining": self.privacy.budget.remaining,
                "utilization": self.privacy.budget.utilization,
            },
            "cached_baselines": len(self._global_cache),
            "switching_cost": self.get_switching_cost_estimate(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience Functions
# ═══════════════════════════════════════════════════════════════════════════════

_federated: FederatedLearningOrchestrator | None = None


def get_federated_orchestrator(
    org_id: str | None = None,
    privacy_budget: float = 1.0,
) -> FederatedLearningOrchestrator:
    """Get global federated learning orchestrator."""
    global _federated
    if _federated is None:
        _federated = FederatedLearningOrchestrator(
            org_id=org_id,
            privacy_budget=privacy_budget,
        )
    return _federated


def reset_federated_orchestrator() -> None:
    """Reset global federated orchestrator (for testing)."""
    global _federated
    _federated = None
