"""
ARA Pipeline — Method Selector
===============================
Author: Georgios-Chrysovalantis Chatzivantsidis

Intelligent method selection for ARA reasoning pipelines.
Uses rule-based classification + LLM optimization to select the optimal reasoning method.
"""

from __future__ import annotations

import json
import logging
from enum import Enum
from typing import TYPE_CHECKING, Any

from .ara_pipelines import ReasoningMethod
from .models import Task, TaskType

if TYPE_CHECKING:
    from .api_clients import UnifiedClient

logger = logging.getLogger("orchestrator")


class ComplexityLevel(str, Enum):
    """Task complexity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskLevel(str, Enum):
    """Risk levels for tasks."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MethodSelection:
    """Result of method selection process."""

    def __init__(
        self,
        method: ReasoningMethod,
        confidence: float,
        rationale: str,
        alternative_methods: list[ReasoningMethod],
        estimated_cost_multiplier: float,
        estimated_time_multiplier: float,
    ):
        self.method = method
        self.confidence = confidence
        self.rationale = rationale
        self.alternative_methods = alternative_methods
        self.estimated_cost_multiplier = estimated_cost_multiplier
        self.estimated_time_multiplier = estimated_time_multiplier

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method.value,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "alternative_methods": [m.value for m in self.alternative_methods],
            "estimated_cost_multiplier": self.estimated_cost_multiplier,
            "estimated_time_multiplier": self.estimated_time_multiplier,
        }


# ─────────────────────────────────────────────
# Method Selection Rules
# ─────────────────────────────────────────────

# Rule-based method selection matrix
# Format: (task_type, complexity, risk) → recommended methods
METHOD_SELECTION_RULES = {
    # Code Generation
    (TaskType.CODE_GEN, ComplexityLevel.LOW, RiskLevel.LOW): [
        ReasoningMethod.MULTI_PERSPECTIVE,
        ReasoningMethod.ITERATIVE,
    ],
    (TaskType.CODE_GEN, ComplexityLevel.MEDIUM, RiskLevel.LOW): [
        ReasoningMethod.MULTI_PERSPECTIVE,
        ReasoningMethod.ITERATIVE,
    ],
    (TaskType.CODE_GEN, ComplexityLevel.HIGH, RiskLevel.MEDIUM): [
        ReasoningMethod.ITERATIVE,
        ReasoningMethod.MULTI_PERSPECTIVE,
    ],
    (TaskType.CODE_GEN, ComplexityLevel.CRITICAL, RiskLevel.HIGH): [
        ReasoningMethod.JURY,
        ReasoningMethod.PRE_MORTEM,
    ],
    # Code Review
    (TaskType.CODE_REVIEW, ComplexityLevel.LOW, RiskLevel.LOW): [
        ReasoningMethod.MULTI_PERSPECTIVE,
    ],
    (TaskType.CODE_REVIEW, ComplexityLevel.MEDIUM, RiskLevel.MEDIUM): [
        ReasoningMethod.MULTI_PERSPECTIVE,
        ReasoningMethod.SCIENTIFIC,
    ],
    (TaskType.CODE_REVIEW, ComplexityLevel.HIGH, RiskLevel.HIGH): [
        ReasoningMethod.JURY,
        ReasoningMethod.PRE_MORTEM,
    ],
    # Reasoning
    (TaskType.REASONING, ComplexityLevel.LOW, RiskLevel.LOW): [
        ReasoningMethod.MULTI_PERSPECTIVE,
        ReasoningMethod.SOCRATIC,
    ],
    (TaskType.REASONING, ComplexityLevel.MEDIUM, RiskLevel.MEDIUM): [
        ReasoningMethod.DEBATE,
        ReasoningMethod.DIALECTICAL,
    ],
    (TaskType.REASONING, ComplexityLevel.HIGH, RiskLevel.HIGH): [
        ReasoningMethod.BAYESIAN,
        ReasoningMethod.DELPHI,
    ],
    (TaskType.REASONING, ComplexityLevel.CRITICAL, RiskLevel.CRITICAL): [
        ReasoningMethod.DELPHI,
        ReasoningMethod.JURY,
    ],
    # Creative Writing
    (TaskType.WRITING, ComplexityLevel.LOW, RiskLevel.LOW): [
        ReasoningMethod.MULTI_PERSPECTIVE,
        ReasoningMethod.ANALOGICAL,
    ],
    (TaskType.WRITING, ComplexityLevel.MEDIUM, RiskLevel.LOW): [
        ReasoningMethod.ANALOGICAL,
        ReasoningMethod.ITERATIVE,
    ],
    (TaskType.WRITING, ComplexityLevel.HIGH, RiskLevel.MEDIUM): [
        ReasoningMethod.ANALOGICAL,
        ReasoningMethod.DIALECTICAL,
    ],
    # Data Extraction
    (TaskType.DATA_EXTRACT, ComplexityLevel.LOW, RiskLevel.LOW): [
        ReasoningMethod.MULTI_PERSPECTIVE,
    ],
    (TaskType.DATA_EXTRACT, ComplexityLevel.MEDIUM, RiskLevel.MEDIUM): [
        ReasoningMethod.SCIENTIFIC,
        ReasoningMethod.RESEARCH,
    ],
    (TaskType.DATA_EXTRACT, ComplexityLevel.HIGH, RiskLevel.HIGH): [
        ReasoningMethod.RESEARCH,
        ReasoningMethod.BAYESIAN,
    ],
    # Summarization
    (TaskType.SUMMARIZE, ComplexityLevel.LOW, RiskLevel.LOW): [
        ReasoningMethod.MULTI_PERSPECTIVE,
    ],
    (TaskType.SUMMARIZE, ComplexityLevel.MEDIUM, RiskLevel.LOW): [
        ReasoningMethod.MULTI_PERSPECTIVE,
        ReasoningMethod.SOCRATIC,
    ],
    # Evaluation
    (TaskType.EVALUATE, ComplexityLevel.LOW, RiskLevel.LOW): [
        ReasoningMethod.MULTI_PERSPECTIVE,
    ],
    (TaskType.EVALUATE, ComplexityLevel.MEDIUM, RiskLevel.MEDIUM): [
        ReasoningMethod.JURY,
        ReasoningMethod.SCIENTIFIC,
    ],
    (TaskType.EVALUATE, ComplexityLevel.HIGH, RiskLevel.HIGH): [
        ReasoningMethod.JURY,
        ReasoningMethod.DELPHI,
    ],
}

# Cost multipliers per method (relative to baseline)
METHOD_COST_MULTIPLIERS = {
    ReasoningMethod.MULTI_PERSPECTIVE: 4.0,  # 4 parallel perspectives
    ReasoningMethod.ITERATIVE: 2.0,  # Up to 3 rounds
    ReasoningMethod.DEBATE: 2.5,  # 2 sides + judge
    ReasoningMethod.RESEARCH: 1.5,  # +search API calls
    ReasoningMethod.JURY: 5.0,  # 4 generators + 3 critics + verifier
    ReasoningMethod.SCIENTIFIC: 2.0,  # Hypothesize + test + evaluate
    ReasoningMethod.SOCRATIC: 1.5,  # Multiple Q&A rounds
    ReasoningMethod.PRE_MORTEM: 1.8,  # 4 phases
    ReasoningMethod.BAYESIAN: 2.2,  # Priors + likelihoods + posteriors + sensitivity
    ReasoningMethod.DIALECTICAL: 2.0,  # Thesis + antithesis + contradictions + aufhebung
    ReasoningMethod.ANALOGICAL: 1.9,  # Abstraction + search + mapping + transfer
    ReasoningMethod.DELPHI: 3.5,  # 4 experts × 2 rounds + aggregation
}

# Time multipliers per method (relative to baseline)
METHOD_TIME_MULTIPLIERS = {
    ReasoningMethod.MULTI_PERSPECTIVE: 1.4,  # Parallel execution
    ReasoningMethod.ITERATIVE: 1.3,
    ReasoningMethod.DEBATE: 1.6,
    ReasoningMethod.RESEARCH: 1.2,
    ReasoningMethod.JURY: 1.8,
    ReasoningMethod.SCIENTIFIC: 1.5,
    ReasoningMethod.SOCRATIC: 1.3,
    ReasoningMethod.PRE_MORTEM: 1.4,
    ReasoningMethod.BAYESIAN: 1.5,
    ReasoningMethod.DIALECTICAL: 1.5,
    ReasoningMethod.ANALOGICAL: 1.4,
    ReasoningMethod.DELPHI: 1.7,
}

# Keyword-based method hints
METHOD_KEYWORDS = {
    ReasoningMethod.PRE_MORTEM: [
        "risk",
        "failure",
        "critical",
        "production",
        "safety",
        "reliability",
        "mission-critical",
        "high-stakes",
        "financial",
        "security",
    ],
    ReasoningMethod.ANALOGICAL: [
        "innovative",
        "creative",
        "novel",
        "breakthrough",
        "inspire",
        "cross-domain",
        "analogy",
        "metaphor",
        "similar to",
    ],
    ReasoningMethod.BAYESIAN: [
        "uncertainty",
        "probability",
        "confidence",
        "likelihood",
        "risk quantification",
        "bayesian",
        "prior",
        "posterior",
        "estimate",
    ],
    ReasoningMethod.DEBATE: [
        "trade-off",
        "decision",
        "choose",
        "versus",
        "compare",
        "architecture",
        "pattern",
        "strategy",
    ],
    ReasoningMethod.DIALECTICAL: [
        "conflict",
        "contradiction",
        "philosophical",
        "policy",
        "ethics",
        "values",
        "principle",
        "ideology",
    ],
    ReasoningMethod.DELPHI: [
        "prediction",
        "forecast",
        "expert",
        "consensus",
        "estimate",
        "future",
        "trend",
        "projection",
    ],
    ReasoningMethod.ITERATIVE: [
        "optimize",
        "improve",
        "refine",
        "best",
        "optimal",
        "performance",
        "efficiency",
        "quality",
    ],
    ReasoningMethod.JURY: [
        "critical",
        "high-stakes",
        "multiple stakeholders",
        "compliance",
        "audit",
        "review board",
        "panel",
    ],
    ReasoningMethod.RESEARCH: [
        "current",
        "latest",
        "recent",
        "study",
        "evidence",
        "empirical",
        "factual",
        "verify",
        "fact-check",
    ],
    ReasoningMethod.SCIENTIFIC: [
        "hypothesis",
        "experiment",
        "test",
        "validate",
        "scientific",
        "algorithm",
        "technical",
        "research",
    ],
    ReasoningMethod.SOCRATIC: [
        "clarify",
        "understand",
        "explore",
        "question",
        "ambiguous",
        "vague",
        "unclear requirements",
    ],
}


class MethodSelector:
    """
    Intelligent method selector for ARA reasoning pipelines.

    Uses a two-phase approach:
    1. Rule-based classification (fast, deterministic)
    2. LLM optimization (optional, for complex cases)
    """

    def __init__(self, client: UnifiedClient | None = None):
        self.client = client
        self._method_keywords = METHOD_KEYWORDS

    def select_method(
        self,
        task: Task,
        complexity: ComplexityLevel = ComplexityLevel.MEDIUM,
        risk_level: RiskLevel = RiskLevel.MEDIUM,
        use_llm_optimization: bool = True,
        budget_constraint: float | None = None,
    ) -> MethodSelection:
        """
        Select the optimal reasoning method for a task.

        Args:
            task: The task to select method for
            complexity: Task complexity level
            risk_level: Risk level associated with the task
            use_llm_optimization: Whether to use LLM for optimization
            budget_constraint: Optional budget constraint (overrides cost considerations)

        Returns:
            MethodSelection with recommended method and metadata
        """
        # Phase 1: Rule-based selection
        rule_based_methods = self._rule_based_selection(task.type, complexity, risk_level)

        # Phase 2: Keyword-based refinement
        keyword_methods = self._keyword_based_selection(task.prompt)

        # Combine rankings
        ranked_methods = self._combine_rankings(rule_based_methods, keyword_methods)

        # Select top method
        top_method = ranked_methods[0] if ranked_methods else ReasoningMethod.MULTI_PERSPECTIVE

        # Phase 3: LLM optimization (optional)
        if use_llm_optimization and self.client:
            optimized_method = self._llm_optimize_selection(
                task, top_method, ranked_methods[:3], complexity, risk_level
            )
            if optimized_method:
                top_method = optimized_method

        # Apply budget constraint
        if budget_constraint:
            top_method = self._apply_budget_constraint(
                top_method, ranked_methods, budget_constraint
            )

        # Build selection result
        return self._build_selection(top_method, ranked_methods, task.type)

    def _rule_based_selection(
        self,
        task_type: TaskType,
        complexity: ComplexityLevel,
        risk_level: RiskLevel,
    ) -> list[ReasoningMethod]:
        """Rule-based method selection."""
        key = (task_type, complexity, risk_level)

        # Direct match
        if key in METHOD_SELECTION_RULES:
            return METHOD_SELECTION_RULES[key]

        # Fallback: try with lower complexity
        if complexity != ComplexityLevel.LOW:
            lower_key = (
                task_type,
                ComplexityLevel(max(0, list(ComplexityLevel).index(complexity) - 1)),
                risk_level,
            )
            if lower_key in METHOD_SELECTION_RULES:
                return METHOD_SELECTION_RULES[lower_key]

        # Fallback: try with lower risk
        if risk_level != RiskLevel.LOW:
            lower_key = (
                task_type,
                complexity,
                RiskLevel(max(0, list(RiskLevel).index(risk_level) - 1)),
            )
            if lower_key in METHOD_SELECTION_RULES:
                return METHOD_SELECTION_RULES[lower_key]

        # Default fallback
        return [ReasoningMethod.MULTI_PERSPECTIVE, ReasoningMethod.ITERATIVE]

    def _keyword_based_selection(self, prompt: str) -> dict[ReasoningMethod, int]:
        """Score methods based on keyword matching."""
        prompt_lower = prompt.lower()
        scores: dict[ReasoningMethod, int] = {}

        for method, keywords in self._method_keywords.items():
            score = sum(1 for keyword in keywords if keyword in prompt_lower)
            if score > 0:
                scores[method] = score

        return scores

    def _combine_rankings(
        self,
        rule_based: list[ReasoningMethod],
        keyword_scores: dict[ReasoningMethod, int],
    ) -> list[ReasoningMethod]:
        """Combine rule-based and keyword-based rankings."""
        # Assign scores
        method_scores: dict[ReasoningMethod, float] = {}

        # Rule-based scores (higher weight for top positions)
        for i, method in enumerate(rule_based):
            weight = len(rule_based) - i
            method_scores[method] = method_scores.get(method, 0) + weight * 2

        # Keyword scores
        for method, score in keyword_scores.items():
            method_scores[method] = method_scores.get(method, 0) + score

        # Sort by score
        ranked = sorted(method_scores.items(), key=lambda x: x[1], reverse=True)
        return [method for method, _ in ranked]

    def _llm_optimize_selection(
        self,
        task: Task,
        current_best: ReasoningMethod,
        top_candidates: list[ReasoningMethod],
        complexity: ComplexityLevel,
        risk_level: RiskLevel,
    ) -> ReasoningMethod | None:
        """Use LLM to optimize method selection."""
        if not self.client or not top_candidates:
            return None

        models = self._get_available_models()
        if not models:
            return None

        system_prompt = (
            "You are an expert at selecting optimal reasoning methods for AI tasks.\n"
            "Analyze the task and recommend the best method from the candidates.\n"
            "Consider: complexity, risk, cost, and expected quality.\n\n"
            "Available methods:\n"
            "- multi_perspective: 4 parallel perspectives (constructive, destructive, systemic, minimalist)\n"
            "- iterative: evolutionary refinement with early exit\n"
            "- debate: two-sided argument with judge\n"
            "- research: evidence-based with web search\n"
            "- jury: 4 generators + 3 critics + verifier (highest quality)\n"
            "- scientific: hypothesis → test → evaluate\n"
            "- socratic: iterative questioning for clarification\n"
            "- pre_mortem: failure analysis → hardened design\n"
            "- bayesian: probabilistic reasoning under uncertainty\n"
            "- dialectical: thesis → antithesis → synthesis\n"
            "- analogical: cross-domain analogy transfer\n"
            "- delphi: expert consensus with multiple rounds\n\n"
            "Return JSON: {'recommended_method': '', 'rationale': '', 'confidence': 0-1}"
        )

        user_prompt = f"""
Task: {task.prompt}
Task Type: {task.type.value}
Complexity: {complexity.value}
Risk Level: {risk_level.value}
Top Candidates: {[m.value for m in top_candidates]}
Current Best: {current_best.value}

Recommend the optimal method.
"""

        try:
            import asyncio

            async def call_llm():
                response, _ = await self.client.call(
                    model=models[0],
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_tokens=500,
                    temperature=0.2,
                )
                return response

            response = asyncio.run(call_llm())
            data = json.loads(response.text) if hasattr(response, "text") else {}

            recommended = data.get("recommended_method")
            if recommended:
                try:
                    return ReasoningMethod(recommended)
                except ValueError:
                    pass
        except Exception as e:
            logger.warning(f"LLM optimization failed: {e}")

        return None

    def _apply_budget_constraint(
        self,
        selected: ReasoningMethod,
        ranked_methods: list[ReasoningMethod],
        budget: float,
    ) -> ReasoningMethod:
        """Select best method within budget constraint."""
        for method in ranked_methods:
            cost_mult = METHOD_COST_MULTIPLIERS.get(method, 1.0)
            if cost_mult <= budget:
                return method

        # Fallback to cheapest option
        return ReasoningMethod.MULTI_PERSPECTIVE

    def _build_selection(
        self,
        method: ReasoningMethod,
        ranked_methods: list[ReasoningMethod],
        task_type: TaskType,
    ) -> MethodSelection:
        """Build MethodSelection result."""
        cost_mult = METHOD_COST_MULTIPLIERS.get(method, 1.0)
        time_mult = METHOD_TIME_MULTIPLIERS.get(method, 1.0)

        # Estimate confidence based on ranking position
        confidence = 0.9 if ranked_methods and ranked_methods[0] == method else 0.7

        # Build rationale
        rationale = self._build_rationale(method, task_type)

        return MethodSelection(
            method=method,
            confidence=confidence,
            rationale=rationale,
            alternative_methods=ranked_methods[1:3] if len(ranked_methods) > 1 else [],
            estimated_cost_multiplier=cost_mult,
            estimated_time_multiplier=time_mult,
        )

    def _build_rationale(self, method: ReasoningMethod, task_type: TaskType) -> str:
        """Build human-readable rationale for method selection."""
        rationales = {
            ReasoningMethod.MULTI_PERSPECTIVE: "Multi-perspective analysis provides balanced coverage of solution space",
            ReasoningMethod.ITERATIVE: "Iterative refinement optimizes solution quality with controlled iterations",
            ReasoningMethod.DEBATE: "Debate format exposes trade-offs and strengthens decision quality",
            ReasoningMethod.RESEARCH: "Evidence-based approach ensures factual accuracy",
            ReasoningMethod.JURY: "Jury system provides maximum quality through multi-agent evaluation",
            ReasoningMethod.SCIENTIFIC: "Scientific method ensures rigorous hypothesis testing",
            ReasoningMethod.SOCRATIC: "Socratic questioning clarifies ambiguous requirements",
            ReasoningMethod.PRE_MORTEM: "Pre-mortem analysis proactively identifies and mitigates risks",
            ReasoningMethod.BAYESIAN: "Bayesian reasoning quantifies uncertainty for informed decisions",
            ReasoningMethod.DIALECTICAL: "Dialectical synthesis transcends contradictions to higher-level solutions",
            ReasoningMethod.ANALOGICAL: "Analogical transfer enables cross-domain innovation",
            ReasoningMethod.DELPHI: "Delphi method aggregates expert consensus for reliable predictions",
        }

        base_rationale = rationales.get(method, "Selected based on task characteristics")
        return f"{base_rationale} for {task_type.value} task"

    def _get_available_models(self) -> list:
        """Get available models for LLM optimization."""
        from .models import ROUTING_TABLE, Model

        # Try to get a reasoning-capable model
        routing = ROUTING_TABLE.get(TaskType.REASONING, [])
        return routing[:3] if routing else [Model.GPT_4O_MINI]


# ─────────────────────────────────────────────
# Convenience Functions
# ─────────────────────────────────────────────


def select_method_for_task(
    task: Task,
    complexity: str = "medium",
    risk: str = "medium",
    use_llm: bool = True,
    client: UnifiedClient | None = None,
) -> MethodSelection:
    """
    Convenience function for method selection.

    Args:
        task: Task to select method for
        complexity: "low", "medium", "high", or "critical"
        risk: "low", "medium", "high", or "critical"
        use_llm: Whether to use LLM optimization
        client: Optional API client for LLM optimization

    Returns:
        MethodSelection with recommended method
    """
    selector = MethodSelector(client=client)

    complexity_map = {
        "low": ComplexityLevel.LOW,
        "medium": ComplexityLevel.MEDIUM,
        "high": ComplexityLevel.HIGH,
        "critical": ComplexityLevel.CRITICAL,
    }

    risk_map = {
        "low": RiskLevel.LOW,
        "medium": RiskLevel.MEDIUM,
        "high": RiskLevel.HIGH,
        "critical": RiskLevel.CRITICAL,
    }

    return selector.select_method(
        task=task,
        complexity=complexity_map.get(complexity, ComplexityLevel.MEDIUM),
        risk_level=risk_map.get(risk, RiskLevel.MEDIUM),
        use_llm_optimization=use_llm,
    )


# ─────────────────────────────────────────────
# Exports
# ─────────────────────────────────────────────

__all__ = [
    "MethodSelector",
    "MethodSelection",
    "ComplexityLevel",
    "RiskLevel",
    "select_method_for_task",
    "METHOD_COST_MULTIPLIERS",
    "METHOD_TIME_MULTIPLIERS",
]
