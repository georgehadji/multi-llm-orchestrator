"""
Production Feedback Loop
========================

Captures real-world outcomes of generated code and feeds them back into
the knowledge base and routing decisions.

Key Features:
- Webhook endpoint for production deployment notifications
- SDK for capturing runtime errors and metrics
- Automatic model routing weight adjustment based on outcomes
- Codebase-specific learning

Usage:
    from orchestrator.feedback_loop import FeedbackLoop, ProductionOutcome

    loop = FeedbackLoop()
    await loop.record_outcome(ProductionOutcome(
        project_id="webgl-dj",
        deployment_id="prod-123",
        runtime_errors=[...],
        performance_metrics={"p95_latency_ms": 120},
    ))
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .knowledge_base import KnowledgeArtifact, KnowledgeBase, KnowledgeType
from .log_config import get_logger
from .models import Model, TaskType
from .plugins import FeedbackPayload, get_plugin_registry

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════════════════════════


class OutcomeStatus(Enum):
    """Status of a production outcome."""

    SUCCESS = "success"
    PARTIAL = "partial"  # Some issues but functional
    FAILURE = "failure"
    ROLLED_BACK = "rolled_back"


@dataclass
class RuntimeError:
    """A runtime error from production."""

    error_type: str
    message: str
    stack_trace: str | None = None
    file_path: str | None = None
    line_number: int | None = None
    count: int = 1  # How many times this error occurred
    first_seen: datetime = field(default_factory=datetime.utcnow)
    last_seen: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PerformanceMetrics:
    """Performance metrics from production."""

    p50_latency_ms: float | None = None
    p95_latency_ms: float | None = None
    p99_latency_ms: float | None = None
    throughput_rps: float | None = None
    cpu_percent: float | None = None
    memory_mb: float | None = None
    error_rate: float = 0.0
    uptime_percent: float = 100.0
    custom_metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class UserFeedback:
    """Explicit user feedback."""

    rating: int  # 1-5
    comment: str | None = None
    reported_issues: list[str] = field(default_factory=list)
    would_recommend: bool | None = None


@dataclass
class CodebaseFingerprint:
    """Fingerprint of a codebase for similarity matching."""

    languages: list[str] = field(default_factory=list)
    framework: str | None = None
    patterns: list[str] = field(default_factory=list)
    complexity_score: float = 0.5
    dependencies: list[str] = field(default_factory=list)

    def similarity(self, other: CodebaseFingerprint) -> float:
        """Calculate similarity to another fingerprint (0.0 - 1.0)."""
        score = 0.0

        # Language overlap
        if self.languages and other.languages:
            common = set(self.languages) & set(other.languages)
            total = set(self.languages) | set(other.languages)
            score += 0.3 * len(common) / len(total) if total else 0

        # Framework match
        if self.framework and other.framework:
            score += 0.3 if self.framework == other.framework else 0

        # Pattern overlap
        if self.patterns and other.patterns:
            common = set(self.patterns) & set(other.patterns)
            total = set(self.patterns) | set(other.patterns)
            score += 0.2 * len(common) / len(total) if total else 0

        # Dependency overlap
        if self.dependencies and other.dependencies:
            common = set(self.dependencies) & set(other.dependencies)
            total = set(self.dependencies) | set(other.dependencies)
            score += 0.2 * len(common) / len(total) if total else 0

        return score


@dataclass
class ProductionOutcome:
    """Complete production outcome data."""

    project_id: str
    deployment_id: str
    task_type: TaskType
    model_used: Model
    generated_code_hash: str  # Hash of the generated code

    status: OutcomeStatus
    runtime_errors: list[RuntimeError] = field(default_factory=list)
    performance_metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    user_feedback: UserFeedback | None = None

    codebase_fingerprint: CodebaseFingerprint | None = None
    deployment_timestamp: datetime = field(default_factory=datetime.utcnow)
    observation_period_hours: float = 24.0

    def calculate_success_score(self) -> float:
        """
        Calculate overall success score (0.0 - 1.0).

        Combines error rate, performance, and user feedback.
        """
        score = 1.0

        # Error penalty
        if self.status == OutcomeStatus.SUCCESS:
            score *= 1.0
        elif self.status == OutcomeStatus.PARTIAL:
            score *= 0.7
        elif self.status == OutcomeStatus.FAILURE:
            score *= 0.3
        elif self.status == OutcomeStatus.ROLLED_BACK:
            score *= 0.0

        # Runtime errors penalty
        total_errors = sum(e.count for e in self.runtime_errors)
        if total_errors > 0:
            score *= max(0.5, 1.0 - (total_errors * 0.1))

        # User feedback
        if self.user_feedback:
            score *= self.user_feedback.rating / 5.0

        return max(0.0, min(1.0, score))


@dataclass
class ModelPerformanceRecord:
    """Aggregated performance record for a model on a task type."""

    model: Model
    task_type: TaskType

    total_deployments: int = 0
    success_count: int = 0
    partial_count: int = 0
    failure_count: int = 0
    rollback_count: int = 0

    avg_success_score: float = 0.5
    total_runtime_errors: int = 0

    # EMA of various metrics
    latency_p95_ema: float = 0.0
    error_rate_ema: float = 0.0

    last_updated: datetime = field(default_factory=datetime.utcnow)

    def update(self, outcome: ProductionOutcome) -> None:
        """Update record with a new outcome."""
        self.total_deployments += 1

        if outcome.status == OutcomeStatus.SUCCESS:
            self.success_count += 1
        elif outcome.status == OutcomeStatus.PARTIAL:
            self.partial_count += 1
        elif outcome.status == OutcomeStatus.FAILURE:
            self.failure_count += 1
        elif outcome.status == OutcomeStatus.ROLLED_BACK:
            self.rollback_count += 1

        # Update success score EMA
        success_score = outcome.calculate_success_score()
        alpha = 0.1  # EMA smoothing factor
        self.avg_success_score = (1 - alpha) * self.avg_success_score + alpha * success_score

        # Update error count
        self.total_runtime_errors += sum(e.count for e in outcome.runtime_errors)

        # Update latency EMA
        if outcome.performance_metrics.p95_latency_ms:
            if self.latency_p95_ema == 0:
                self.latency_p95_ema = outcome.performance_metrics.p95_latency_ms
            else:
                self.latency_p95_ema = (
                    1 - alpha
                ) * self.latency_p95_ema + alpha * outcome.performance_metrics.p95_latency_ms

        # Update error rate EMA
        self.error_rate_ema = (
            1 - alpha
        ) * self.error_rate_ema + alpha * outcome.performance_metrics.error_rate

        self.last_updated = datetime.utcnow()

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_deployments == 0:
            return 0.5
        return (self.success_count + 0.5 * self.partial_count) / self.total_deployments


# ═══════════════════════════════════════════════════════════════════════════════
# Feedback Loop Core
# ═══════════════════════════════════════════════════════════════════════════════


class FeedbackLoop:
    """
    Production feedback loop system.

    Captures outcomes, updates model performance records,
    and feeds insights into knowledge base.
    """

    EMA_ALPHA = 0.1  # Exponential moving average smoothing

    def __init__(
        self,
        storage_path: Path | None = None,
        knowledge_base: KnowledgeBase | None = None,
    ):
        self.storage_path = storage_path or Path(".feedback")
        self.storage_path.mkdir(exist_ok=True)

        self.kb = knowledge_base or KnowledgeBase()

        # In-memory performance records: (model, task_type) -> ModelPerformanceRecord
        self._performance_records: dict[tuple[Model, TaskType], ModelPerformanceRecord] = {}

        # Codebase-specific outcomes: codebase_hash -> List[outcomes]
        self._codebase_outcomes: dict[str, list[ProductionOutcome]] = defaultdict(list)

        # Callbacks for outcome processing
        self._callbacks: list[Callable[[ProductionOutcome], None]] = []

        self._load_records()

    def _load_records(self) -> None:
        """Load performance records from disk."""
        records_file = self.storage_path / "performance_records.json"
        if records_file.exists():
            try:
                data = json.loads(records_file.read_text())
                for _key, record_data in data.items():
                    model = Model(record_data["model"])
                    task_type = TaskType(record_data["task_type"])
                    record = ModelPerformanceRecord(
                        model=model,
                        task_type=task_type,
                        **{k: v for k, v in record_data.items() if k not in ("model", "task_type")},
                    )
                    self._performance_records[(model, task_type)] = record
            except Exception as e:
                logger.error(f"Failed to load performance records: {e}")

    def _save_records(self) -> None:
        """Save performance records to disk."""
        records_file = self.storage_path / "performance_records.json"
        try:
            data = {}
            for (model, task_type), record in self._performance_records.items():
                record_dict = asdict(record)
                record_dict["model"] = model.value
                record_dict["task_type"] = task_type.value
                data[f"{model.value}:{task_type.value}"] = record_dict
            records_file.write_text(json.dumps(data, indent=2, default=str))
        except Exception as e:
            logger.error(f"Failed to save performance records: {e}")

    async def record_outcome(self, outcome: ProductionOutcome) -> dict[str, Any]:
        """
        Record a production outcome.

        This is the main entry point for the feedback loop.
        """
        logger.info(
            f"Recording outcome for {outcome.project_id}: "
            f"{outcome.status.value} (score: {outcome.calculate_success_score():.2f})"
        )

        # 1. Update performance record
        record = self._get_or_create_record(outcome.model_used, outcome.task_type)
        record.update(outcome)

        # 2. Store codebase-specific outcome
        if outcome.codebase_fingerprint:
            fp_hash = self._hash_fingerprint(outcome.codebase_fingerprint)
            self._codebase_outcomes[fp_hash].append(outcome)

        # 3. Create knowledge artifact
        await self._create_knowledge_artifact(outcome)

        # 4. Run plugin processors
        await self._run_plugin_processors(outcome)

        # 5. Notify callbacks
        for callback in self._callbacks:
            try:
                callback(outcome)
            except Exception as e:
                logger.error(f"Callback error: {e}")

        # 6. Persist
        self._save_records()

        return {
            "success": True,
            "model": outcome.model_used.value,
            "success_score": outcome.calculate_success_score(),
            "updated_record": {
                "total_deployments": record.total_deployments,
                "success_rate": record.success_rate,
                "avg_score": record.avg_success_score,
            },
        }

    def _get_or_create_record(
        self,
        model: Model,
        task_type: TaskType,
    ) -> ModelPerformanceRecord:
        """Get or create a performance record."""
        key = (model, task_type)
        if key not in self._performance_records:
            self._performance_records[key] = ModelPerformanceRecord(
                model=model,
                task_type=task_type,
            )
        return self._performance_records[key]

    def _hash_fingerprint(self, fingerprint: CodebaseFingerprint) -> str:
        """Create a hash of codebase fingerprint."""
        data = json.dumps(
            {
                "languages": sorted(fingerprint.languages),
                "framework": fingerprint.framework,
                "patterns": sorted(fingerprint.patterns),
            },
            sort_keys=True,
        )
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    async def _create_knowledge_artifact(self, outcome: ProductionOutcome) -> None:
        """Create a knowledge artifact from the outcome."""
        if outcome.status != OutcomeStatus.FAILURE and not outcome.runtime_errors:
            return  # Only learn from failures

        artifact = KnowledgeArtifact(
            id=f"outcome-{outcome.deployment_id}",
            type=KnowledgeType.LESSON,
            title=f"Production issue: {outcome.project_id}",
            content=self._format_lesson(outcome),
            context={
                "model": outcome.model_used.value,
                "task_type": outcome.task_type.value,
                "status": outcome.status.value,
                "errors": [
                    {"type": e.error_type, "message": e.message} for e in outcome.runtime_errors
                ],
            },
            tags=["production", "feedback", outcome.status.value],
            source_project=outcome.project_id,
        )

        await self.kb.add_artifact(artifact)

    def _format_lesson(self, outcome: ProductionOutcome) -> str:
        """Format a lesson learned from an outcome."""
        lines = [
            f"# Production Issue: {outcome.project_id}",
            "",
            f"**Model**: {outcome.model_used.value}",
            f"**Task Type**: {outcome.task_type.value}",
            f"**Status**: {outcome.status.value}",
            "",
            "## Errors",
        ]

        for error in outcome.runtime_errors:
            lines.append(f"- **{error.error_type}**: {error.message}")

        return "\n".join(lines)

    async def _run_plugin_processors(self, outcome: ProductionOutcome) -> None:
        """Run feedback plugin processors."""
        registry = get_plugin_registry()
        plugins = registry.get_feedback_processors()

        payload = FeedbackPayload(
            project_id=outcome.project_id,
            deployment_id=outcome.deployment_id,
            task_type=outcome.task_type,
            model_used=outcome.model_used,
            generated_code="",  # Could load from storage
            runtime_errors=[asdict(e) for e in outcome.runtime_errors],
            performance_metrics=asdict(outcome.performance_metrics),
            user_rating=outcome.user_feedback.rating if outcome.user_feedback else None,
        )

        for plugin in plugins:
            if plugin.should_process(payload):
                try:
                    await plugin.process_feedback(payload)
                except Exception as e:
                    logger.error(f"Plugin {plugin.metadata.name} failed: {e}")

    def get_model_score(
        self,
        model: Model,
        task_type: TaskType,
        codebase_fingerprint: CodebaseFingerprint | None = None,
    ) -> float:
        """
        Get production-weighted score for a model.

        Returns value 0.0 - 1.0 representing expected success probability.
        """
        # Base score from performance records
        record = self._performance_records.get((model, task_type))
        if record and record.total_deployments > 0:
            base_score = record.avg_success_score
        else:
            base_score = 0.5  # Unknown - neutral

        # Adjust for codebase similarity if fingerprint provided
        if codebase_fingerprint:
            codebase_score = self._get_codebase_adjusted_score(
                model, task_type, codebase_fingerprint
            )
            # Blend global and codebase-specific scores
            base_score = 0.7 * base_score + 0.3 * codebase_score

        return base_score

    def _get_codebase_adjusted_score(
        self,
        model: Model,
        task_type: TaskType,
        fingerprint: CodebaseFingerprint,
    ) -> float:
        """Get score adjusted for similar codebases."""
        scores = []
        weights = []

        for _fp_hash, outcomes in self._codebase_outcomes.items():
            for outcome in outcomes:
                if outcome.model_used != model or outcome.task_type != task_type:
                    continue

                if outcome.codebase_fingerprint:
                    similarity = fingerprint.similarity(outcome.codebase_fingerprint)
                    if similarity > 0.5:  # Only consider reasonably similar
                        scores.append(outcome.calculate_success_score())
                        weights.append(similarity)

        if not scores:
            return 0.5  # No data - neutral

        # Weighted average by similarity
        total_weight = sum(weights)
        if total_weight == 0:
            return 0.5

        return sum(s * w for s, w in zip(scores, weights, strict=False)) / total_weight

    def get_best_models_for_codebase(
        self,
        task_type: TaskType,
        fingerprint: CodebaseFingerprint,
        top_n: int = 3,
    ) -> list[tuple[Model, float]]:
        """Get best models for a specific codebase fingerprint."""
        from .models import Model  # Import all models

        scores = []
        for model in Model:
            score = self.get_model_score(model, task_type, fingerprint)
            scores.append((model, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]

    def on_outcome(self, callback: Callable[[ProductionOutcome], None]) -> None:
        """Register a callback for new outcomes."""
        self._callbacks.append(callback)

    def get_performance_summary(self) -> dict[str, Any]:
        """Get summary of all performance records."""
        total_deployments = sum(r.total_deployments for r in self._performance_records.values())

        if not total_deployments:
            return {"status": "no_data"}

        model_rankings = [
            {
                "model": r.model.value,
                "task_type": r.task_type.value,
                "deployments": r.total_deployments,
                "success_rate": r.success_rate,
                "avg_score": r.avg_success_score,
            }
            for r in sorted(
                self._performance_records.values(),
                key=lambda x: x.avg_success_score,
                reverse=True,
            )
        ]

        return {
            "total_deployments": total_deployments,
            "unique_models": len({r.model for r in self._performance_records.values()}),
            "model_rankings": model_rankings[:10],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SDK for External Applications
# ═══════════════════════════════════════════════════════════════════════════════


class FeedbackSDK:
    """
    SDK for sending feedback from deployed applications.

    Lightweight client that can be embedded in generated code.
    """

    def __init__(self, endpoint_url: str, api_key: str):
        self.endpoint_url = endpoint_url.rstrip("/")
        self.api_key = api_key
        self._buffer: list[dict] = []
        self._buffer_size = 10

    async def report_error(
        self,
        deployment_id: str,
        error_type: str,
        message: str,
        **kwargs,
    ) -> bool:
        """Report a runtime error."""
        error = {
            "deployment_id": deployment_id,
            "error_type": error_type,
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
            **kwargs,
        }
        self._buffer.append(error)

        if len(self._buffer) >= self._buffer_size:
            return await self._flush()
        return True

    async def submit_user_feedback(
        self,
        deployment_id: str,
        rating: int,
        comment: str | None = None,
    ) -> bool:
        """Submit user feedback."""
        feedback = {
            "deployment_id": deployment_id,
            "rating": rating,
            "comment": comment,
            "timestamp": datetime.utcnow().isoformat(),
        }
        return await self._send(feedback, "/feedback/user")

    async def _flush(self) -> bool:
        """Flush buffered errors."""
        if not self._buffer:
            return True

        success = await self._send({"errors": self._buffer}, "/feedback/errors")
        if success:
            self._buffer.clear()
        return success

    async def _send(self, data: dict, path: str) -> bool:
        """Send data to feedback endpoint."""
        try:
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.endpoint_url}{path}",
                    json=data,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=5.0,
                )
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to send feedback: {e}")
            return False


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience Functions
# ═══════════════════════════════════════════════════════════════════════════════

_feedback_loop: FeedbackLoop | None = None


def get_feedback_loop() -> FeedbackLoop:
    """Get global feedback loop instance."""
    global _feedback_loop
    if _feedback_loop is None:
        _feedback_loop = FeedbackLoop()
    return _feedback_loop


def reset_feedback_loop() -> None:
    """Reset global feedback loop (for testing)."""
    global _feedback_loop
    _feedback_loop = None
