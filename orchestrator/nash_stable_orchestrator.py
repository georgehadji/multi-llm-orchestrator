"""
Nash-Stable Multi-LLM Orchestrator
===================================

Production-ready orchestrator integrating all four strategic features:
1. Model Performance Knowledge Graph
2. Adaptive Prompt Template System
3. Predictive Cost-Quality Frontier API
4. Cross-Organization Federated Learning

This orchestrator creates Nash stability through accumulated intelligence
that compounds over time, creating significant switching costs.

Usage:
    from orchestrator.nash_stable_orchestrator import NashStableOrchestrator

    orchestrator = NashStableOrchestrator()

    # All features work automatically
    result = await orchestrator.run_project(
        project_description="Build a FastAPI service",
        success_criteria="All tests pass",
        budget=5.0,
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .adaptive_templates import (
    get_adaptive_template_system,
)
from .engine import Orchestrator
from .federated_learning import (
    get_federated_orchestrator,
)
from .feedback_loop import CodebaseFingerprint, ProductionOutcome
from .knowledge_graph import get_knowledge_graph
from .log_config import get_logger
from .models import Budget, Model, Task, TaskType
from .pareto_frontier import (
    Objective,
    get_cost_quality_frontier,
)

if TYPE_CHECKING:
    from pathlib import Path

logger = get_logger(__name__)


@dataclass
class NashStabilityMetrics:
    """Metrics tracking Nash stability of the system."""
    # Knowledge accumulation
    knowledge_graph_nodes: int = 0
    knowledge_graph_edges: int = 0
    patterns_learned: int = 0

    # Template optimization
    templates_tested: int = 0
    template_convergence_rate: float = 0.0

    # Frontier accuracy
    predictions_made: int = 0
    prediction_accuracy: float = 0.0

    # Federated learning
    local_insights: int = 0
    global_insights: int = 0
    contributing_orgs: int = 0
    switching_cost_usd: float = 0.0

    # Composite stability score
    nash_stability_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "knowledge_graph": {
                "nodes": self.knowledge_graph_nodes,
                "edges": self.knowledge_graph_edges,
                "patterns_learned": self.patterns_learned,
            },
            "template_optimization": {
                "templates_tested": self.templates_tested,
                "convergence_rate": self.template_convergence_rate,
            },
            "frontier_accuracy": {
                "predictions_made": self.predictions_made,
                "accuracy": self.prediction_accuracy,
            },
            "federated_learning": {
                "local_insights": self.local_insights,
                "global_insights": self.global_insights,
                "contributing_orgs": self.contributing_orgs,
            },
            "switching_cost_usd": self.switching_cost_usd,
            "nash_stability_score": self.nash_stability_score,
        }


class NashStableOrchestrator:
    """
    Production orchestrator with Nash stability through:

    1. KNOWLEDGE GRAPH: Multi-hop reasoning for model recommendations
    2. ADAPTIVE TEMPLATES: Self-improving prompts via A/B testing
    3. PARETO FRONTIER: Optimal cost-quality trade-offs with confidence
    4. FEDERATED LEARNING: Collective intelligence with privacy

    Switching costs accumulate through:
    - Local pattern knowledge (your specific codebases)
    - Template optimizations (your specific use cases)
    - Global baseline (collective intelligence)
    - Historical predictions (calibrated to your patterns)
    """

    def __init__(
        self,
        budget: Budget | None = None,
        org_id: str | None = None,
        privacy_budget: float = 1.0,
        enable_federation: bool = True,
    ):
        # Base orchestrator
        self.base_orchestrator = Orchestrator(budget=budget)

        # Strategic components
        self.knowledge_graph = get_knowledge_graph()
        self.adaptive_templates = get_adaptive_template_system()
        self.pareto_frontier = get_cost_quality_frontier()
        self.federated = get_federated_orchestrator(
            org_id=org_id,
            privacy_budget=privacy_budget,
        ) if enable_federation else None

        # Metrics
        self.metrics = NashStabilityMetrics()
        self._prediction_results: list[tuple[float, float]] = []  # (predicted, actual)

    async def run_project(
        self,
        project_description: str,
        success_criteria: str,
        budget: float | None = None,
        output_dir: Path | None = None,
        enable_learning: bool = True,
    ) -> dict[str, Any]:
        """
        Run a project with all Nash-stable features enabled.

        Args:
            project_description: Description of the project
            success_criteria: Success criteria for the project
            budget: Optional budget override
            output_dir: Output directory
            enable_learning: Whether to learn from this run

        Returns:
            Project result with Nash stability metrics
        """
        logger.info("Starting Nash-stable project execution")

        # Step 1: Get global baseline (federated learning)
        baseline = None
        if self.federated:
            baseline = await self.federated.get_global_baseline(
                task_type=TaskType.CODE_GEN,
            )
            logger.info(f"Global baseline confidence: {baseline.confidence:.2f}")

        # Step 2: Get Pareto frontier recommendations
        frontier = await self.pareto_frontier.get_pareto_frontier(
            task_type=TaskType.CODE_GEN,
            objectives=[Objective.QUALITY, Objective.COST, Objective.EFFICIENCY],
            budget_constraint=budget,
        )

        if frontier:
            top_recommendation = frontier[0]
            logger.info(
                f"Top recommendation: {top_recommendation.model.value} "
                f"(quality: {top_recommendation.quality:.2f}, "
                f"cost: ${top_recommendation.cost:.4f}, "
                f"confidence: {top_recommendation.confidence:.2f})"
            )

        # Step 3: Run base orchestrator
        result = await self.base_orchestrator.run_project(
            project_description=project_description,
            success_criteria=success_criteria,
            output_dir=output_dir,
        )

        # Step 4: Learn from results
        if enable_learning:
            await self._learn_from_run(result, project_description)

        # Step 5: Update metrics
        self._update_metrics()

        # Step 6: Return enhanced result
        return {
            **result,
            "nash_stability": {
                "metrics": self.metrics.to_dict(),
                "frontier_recommendations": [r.__dict__ for r in frontier[:3]] if frontier else [],
                "global_baseline_confidence": baseline.confidence if baseline else 0.0,
            },
        }

    async def run_task_with_adaptive_template(
        self,
        task: Task,
        model: Model,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Run a single task with adaptive template selection.

        Automatically selects the best template variant based on
        historical performance and A/B testing.
        """
        context = context or {}

        # Select template
        template, metadata = await self.adaptive_templates.select_template(
            task_type=task.task_type,
            model=model,
            context=context,
        )

        logger.info(f"Selected template: {template.name} ({metadata['strategy']})")

        # Render template
        rendered = template.render(
            task=task.description,
            language=context.get("language", "python"),
            criteria=context.get("criteria", ""),
        )

        # Execute task (would integrate with actual execution)
        # For now, return template info
        return {
            "template_name": template.name,
            "template_style": template.style.value,
            "selection_strategy": metadata["strategy"],
            "rendered_prompt": rendered,
            "confidence": metadata.get("composite_score", 0.5),
        }

    async def get_model_recommendation(
        self,
        task_type: TaskType,
        fingerprint: CodebaseFingerprint | None = None,
        budget: float | None = None,
    ) -> dict[str, Any]:
        """
        Get comprehensive model recommendation using all systems.

        Combines:
        - Knowledge graph similarity matching
        - Pareto frontier optimization
        - Global baseline from federation
        """
        recommendations = []

        # 1. Knowledge graph recommendations
        kg_recs = await self.knowledge_graph.recommend_models(
            task_type=task_type,
            fingerprint=fingerprint or CodebaseFingerprint(),
            top_k=3,
        )

        for rec in kg_recs:
            recommendations.append({
                "source": "knowledge_graph",
                "model": rec["model_name"],
                "score": rec["score"],
                "confidence": rec["confidence"],
                "evidence_count": rec["evidence_count"],
            })

        # 2. Pareto frontier recommendations
        frontier = await self.pareto_frontier.get_pareto_frontier(
            task_type=task_type,
            objectives=[Objective.QUALITY, Objective.COST],
            fingerprint=fingerprint,
            budget_constraint=budget,
        )

        for rec in frontier[:3]:
            recommendations.append({
                "source": "pareto_frontier",
                "model": rec.model.value,
                "quality": rec.quality,
                "cost": rec.cost,
                "confidence": rec.confidence,
                "is_pareto_optimal": rec.is_pareto_optimal,
            })

        # 3. Global baseline
        if self.federated:
            baseline = await self.federated.get_global_baseline(
                task_type=task_type,
                fingerprint=fingerprint,
            )

            for model_rec in baseline.recommended_models[:3]:
                recommendations.append({
                    "source": "global_baseline",
                    "model": model_rec["model"],
                    "quality": model_rec["quality"],
                    "confidence": model_rec["confidence"],
                    "sample_size": model_rec["sample_size"],
                })

        # Aggregate scores
        model_scores: dict[str, list[float]] = defaultdict(list)
        model_confidences: dict[str, list[float]] = defaultdict(list)

        for rec in recommendations:
            model = rec.get("model", "")
            if "score" in rec:
                score = rec["score"]
            elif "quality" in rec:
                score = rec["quality"]
            else:
                continue

            model_scores[model].append(score)
            model_confidences[model].append(rec.get("confidence", 0.5))

        # Calculate aggregated recommendations
        aggregated = []
        for model, scores in model_scores.items():
            avg_score = sum(scores) / len(scores)
            avg_confidence = sum(model_confidences[model]) / len(scores)

            aggregated.append({
                "model": model,
                "aggregated_score": avg_score,
                "confidence": avg_confidence,
                "source_count": len(scores),
            })

        aggregated.sort(key=lambda x: x["aggregated_score"], reverse=True)

        return {
            "recommendations": aggregated[:5],
            "all_sources": recommendations,
            "top_recommendation": aggregated[0] if aggregated else None,
        }

    async def _learn_from_run(
        self,
        result: dict[str, Any],
        project_description: str,
    ) -> None:
        """Learn from a completed run across all systems."""
        # Extract outcome info
        tasks = result.get("tasks", [])

        for task_result in tasks:
            model = task_result.get("model")
            if not model:
                continue

            # Create outcome
            outcome = ProductionOutcome(
                project_id=result.get("project_id", "unknown"),
                deployment_id=f"{result.get('project_id', 'unknown')}_{task_result.get('task_id', 'unknown')}",
                task_type=TaskType(task_result.get("task_type", "CODE_GEN")),
                model_used=Model(model),
                generated_code_hash=task_result.get("code_hash", ""),
                status=task_result.get("status", "partial"),
            )

            # Update knowledge graph
            await self.knowledge_graph.add_performance_outcome(outcome)

            # Contribute to federation
            if self.federated:
                await self.federated.contribute_insight(outcome)

            # Report template result
            if "template_name" in task_result:
                await self.adaptive_templates.report_result(
                    task_type=TaskType(task_result.get("task_type", "CODE_GEN")),
                    model=Model(model),
                    variant_name=task_result["template_name"],
                    score=task_result.get("quality_score", 0.5),
                    success=task_result.get("status") == "success",
                )

    def _update_metrics(self) -> None:
        """Update Nash stability metrics."""
        # Knowledge graph metrics
        kg_stats = self.knowledge_graph.get_graph_stats()
        self.metrics.knowledge_graph_nodes = kg_stats["total_nodes"]
        self.metrics.knowledge_graph_edges = kg_stats["total_edges"]
        self.metrics.patterns_learned = kg_stats["nodes_by_type"].get("pattern", 0)

        # Template metrics
        template_stats = self.adaptive_templates.get_template_stats()
        self.metrics.templates_tested = template_stats["total_variants"]
        if template_stats["top_performers"]:
            confidences = [p["confidence"] for p in template_stats["top_performers"]]
            self.metrics.template_convergence_rate = sum(confidences) / len(confidences)

        # Frontier metrics
        frontier_stats = self.pareto_frontier.get_frontier_stats()
        self.metrics.predictions_made = frontier_stats["cache_size"]

        # Federated metrics
        if self.federated:
            fed_stats = self.federated.get_federated_stats()
            self.metrics.local_insights = fed_stats["local_insights"]
            self.metrics.global_insights = fed_stats["global_insights"]
            self.metrics.contributing_orgs = fed_stats["contributing_orgs"]
            self.metrics.switching_cost_usd = fed_stats["switching_cost"]["total_switching_cost_usd"]

        # Calculate composite Nash stability score
        # Formula: weighted combination of all factors
        weights = {
            "knowledge": 0.25,
            "templates": 0.20,
            "frontier": 0.15,
            "federated": 0.40,
        }

        knowledge_score = min(1.0, self.metrics.knowledge_graph_edges / 100)
        template_score = self.metrics.template_convergence_rate
        frontier_score = min(1.0, self.metrics.predictions_made / 50)
        federated_score = fed_stats.get("nash_stability_score", 0) if self.federated else 0

        self.metrics.nash_stability_score = (
            weights["knowledge"] * knowledge_score +
            weights["templates"] * template_score +
            weights["frontier"] * frontier_score +
            weights["federated"] * federated_score
        )

    def get_nash_stability_report(self) -> dict[str, Any]:
        """
        Generate comprehensive Nash stability report.

        This report explains why users shouldn't switch to competitors.
        """
        self._update_metrics()

        switching_cost = self.federated.get_switching_cost_estimate() if self.federated else {}

        return {
            "nash_stability_score": self.metrics.nash_stability_score,
            "interpretation": self._interpret_stability_score(self.metrics.nash_stability_score),
            "switching_cost_analysis": switching_cost,
            "accumulated_assets": {
                "knowledge_graph_relationships": self.metrics.knowledge_graph_edges,
                "learned_patterns": self.metrics.patterns_learned,
                "optimized_templates": self.metrics.templates_tested,
                "calibrated_predictions": self.metrics.predictions_made,
                "local_insights": self.metrics.local_insights,
                "global_insights_contributed": self.metrics.global_insights,
            },
            "competitive_moat": {
                "description": "Your accumulated intelligence creates switching costs",
                "estimated_replacement_time": f"{self.metrics.knowledge_graph_nodes * 0.5:.0f} hours",
                "estimated_replacement_cost": f"${switching_cost.get('total_switching_cost_usd', 0):.2f}",
            },
            "recommendations": [
                "Continue using the system to accumulate more knowledge",
                "Contribute insights to global pool for better recommendations",
                "Use Pareto frontier for optimal cost-quality trade-offs",
            ],
        }

    def _interpret_stability_score(self, score: float) -> str:
        """Interpret the Nash stability score."""
        if score < 0.2:
            return "Early stage - minimal switching costs accumulated"
        elif score < 0.4:
            return "Growing - some knowledge accumulated"
        elif score < 0.6:
            return "Moderate - meaningful switching costs exist"
        elif score < 0.8:
            return "Strong - significant competitive moat"
        else:
            return "Dominant - very high switching costs, Nash stable"

    async def compare_with_competitor(
        self,
        competitor_name: str = "generic_competitor",
    ) -> dict[str, Any]:
        """
        Compare accumulated intelligence with a hypothetical competitor.

        Demonstrates the competitive advantage of Nash stability.
        """
        our_metrics = self.metrics

        # Competitor starts with zero accumulated knowledge
        competitor_metrics = {
            "knowledge_graph_edges": 0,
            "patterns_learned": 0,
            "templates_tested": 0,
            "predictions_made": 0,
            "local_insights": 0,
            "global_insights": 0,
        }

        return {
            "comparison": {
                "our_platform": {
                    "knowledge_relationships": our_metrics.knowledge_graph_edges,
                    "learned_patterns": our_metrics.patterns_learned,
                    "template_variants": our_metrics.templates_tested,
                    "calibrated_predictions": our_metrics.predictions_made,
                    "global_intelligence": our_metrics.global_insights,
                },
                "competitor": competitor_metrics,
            },
            "advantage_analysis": {
                "knowledge_advantage": our_metrics.knowledge_graph_edges,
                "prediction_accuracy_advantage": f"{our_metrics.prediction_accuracy * 100:.1f}%",
                "cost_efficiency_advantage": "Unknown without usage data",
            },
            "switching_disadvantage": {
                "lost_knowledge": our_metrics.knowledge_graph_edges,
                "lost_predictions": our_metrics.predictions_made,
                "lost_global_intelligence": our_metrics.global_insights,
                "time_to_recover": f"{our_metrics.knowledge_graph_edges * 0.5:.0f} hours",
            },
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience Functions
# ═══════════════════════════════════════════════════════════════════════════════

_nash_stable: NashStableOrchestrator | None = None


def get_nash_stable_orchestrator(
    budget: Budget | None = None,
    org_id: str | None = None,
) -> NashStableOrchestrator:
    """Get global Nash-stable orchestrator."""
    global _nash_stable
    if _nash_stable is None:
        _nash_stable = NashStableOrchestrator(
            budget=budget,
            org_id=org_id,
        )
    return _nash_stable


def reset_nash_stable_orchestrator() -> None:
    """Reset global Nash-stable orchestrator."""
    global _nash_stable
    _nash_stable = None
