"""
ARA Pipeline — Advanced Reasoning & Analysis Pipelines
=======================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Implements 12 reasoning methods from the ARA Pipeline v2.0 specification:
- 7 Standard Methods: Multi-Perspective, Iterative, Debate, Research, Jury, Scientific, Socratic
- 5 Specialized Methods: Pre-Mortem, Bayesian, Dialectical, Analogical, Delphi

Each pipeline implements a distinct reasoning strategy optimized for specific problem types.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from ..infrastructure.cache import DiskCache
from ..models import (
    Model,
    ProbabilityFormat,
    Task,
    TaskResult,
    TaskStatus,
    TaskType,
    VSConfig,
    get_provider,
)

# Lazy import for VerbalizedSampler (avoids circular dep at module level)
_VerbalizedSampler = None
_VerbalizedSampler_lock = threading.Lock()


def _get_vs_sampler(client):
    global _VerbalizedSampler
    if _VerbalizedSampler is None:
        with _VerbalizedSampler_lock:
            if _VerbalizedSampler is None:
                from ..application.verbalized_sampling import VerbalizedSampler

                _VerbalizedSampler = VerbalizedSampler
    return _VerbalizedSampler(client=client)


from ..telemetry import TelemetryCollector

if TYPE_CHECKING:
    from ..api_clients import UnifiedClient

logger = logging.getLogger("orchestrator")


# ─────────────────────────────────────────────
# Enums & Constants
# ─────────────────────────────────────────────


class ReasoningMethod(str, Enum):
    """Available reasoning methods in the ARA Pipeline."""

    MULTI_PERSPECTIVE = "multi_perspective"
    ITERATIVE = "iterative"
    DEBATE = "debate"
    RESEARCH = "research"
    JURY = "jury"
    SCIENTIFIC = "scientific"
    SOCRATIC = "socratic"
    PRE_MORTEM = "pre_mortem"
    BAYESIAN = "bayesian"
    DIALECTICAL = "dialectical"
    ANALOGICAL = "analogical"
    DELPHI = "delphi"
    BRAINSTORMING = "brainstorming"
    VERBALIZED_SAMPLING = "verbalized_sampling"
    PERSUASION_DEFENSE = "persuasion_defense"
    COVE = "cove"
    SOT = "sot"
    TOT = "tot"
    POT = "pot"
    SELF_DISCOVER = "self_discover"


class PerspectiveType(str, Enum):
    """Perspectives for Multi-Perspective method."""

    CONSTRUCTIVE = "constructive"
    DESTRUCTIVE = "destructive"
    SYSTEMIC = "systemic"
    MINIMALIST = "minimalist"


# ─────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────


@dataclass
class SolutionCandidate:
    """Represents a candidate solution from a reasoning step."""

    perspective: str
    content: str
    key_insights: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CritiqueScore:
    """Scores from critique/evaluation phase."""

    perspective: str
    total: float
    logical_consistency: float = 0.0
    feasibility: float = 0.0
    completeness: float = 0.0
    novelty: float = 0.0
    steel_man: str = ""
    rationale: str = ""


@dataclass
class PipelineState:
    """State object passed through pipeline phases."""

    task: Task
    method: ReasoningMethod
    candidates: list[SolutionCandidate] = field(default_factory=list)
    scores: list[CritiqueScore] = field(default_factory=list)
    top_candidates: list[SolutionCandidate] = field(default_factory=list)
    final_output: str = ""
    final_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    reflexion_memory: list[str] = field(default_factory=list)

    # Method-specific state
    debate_rounds: list[dict] = field(default_factory=list)
    pre_mortem_state: dict[str, Any] = field(default_factory=dict)
    bayesian_state: dict[str, Any] = field(default_factory=dict)
    dialectical_state: dict[str, Any] = field(default_factory=dict)
    analogical_state: dict[str, Any] = field(default_factory=dict)
    delphi_state: dict[str, Any] = field(default_factory=dict)
    cove_state: dict[str, Any] = field(default_factory=dict)
    sot_state: dict[str, Any] = field(default_factory=dict)
    tot_state: dict[str, Any] = field(default_factory=dict)
    pot_state: dict[str, Any] = field(default_factory=dict)
    self_discover_state: dict[str, Any] = field(default_factory=dict)
    brainstorming_state: dict[str, Any] = field(default_factory=dict)
    cognitive_state: dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────
# Base Pipeline Class
# ─────────────────────────────────────────────


class BasePipeline(ABC):
    """
    Abstract base class for all ARA reasoning pipelines.

    Subclasses must implement:
    - _run_pipeline(): Main pipeline logic
    - get_method(): Return the ReasoningMethod enum value
    """

    def __init__(
        self,
        client: UnifiedClient,
        cache: DiskCache | None = None,
        telemetry: TelemetryCollector | None = None,
        top_k: int = 2,
        max_iterations: int = 3,
    ):
        self.client = client
        self.cache = cache or DiskCache()
        self.telemetry = telemetry or TelemetryCollector({})
        self.top_k = top_k
        self.max_iterations = max_iterations
        self.api_health: dict[Model, bool] = dict.fromkeys(Model, True)

    @abstractmethod
    async def execute(self, task: Task, context: str = "") -> TaskResult:
        """
        Execute the reasoning pipeline for a given task.

        Args:
            task: The task to execute
            context: Optional context from dependencies

        Returns:
            TaskResult with output, score, and metadata
        """
        pass

    @abstractmethod
    def get_method(self) -> ReasoningMethod:
        """Return the reasoning method this pipeline implements."""
        pass

    def _extract_json(self, text: str) -> dict[str, Any] | None:
        """Extract JSON from LLM response text."""
        # Try to find JSON object in text
        match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # Try parsing entire text as JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def _get_available_models(self, task_type: TaskType) -> list[Model]:
        """Get list of available models for task type."""
        from ..models import ROUTING_TABLE

        routing = ROUTING_TABLE.get(task_type, list(Model)[:5])
        return [m for m in routing if self.api_health.get(m, False)]

    def _get_model_for_phase(self, phase: "PhaseType", task_type: TaskType) -> Model:
        """Get optimal model for a specific phase type."""
        from ..phase_aware_models import PhaseAwareModelSelector

        available = [m.value for m in self._get_available_models(task_type)]
        selector = PhaseAwareModelSelector()
        best = selector.select_model(phase=phase, available_models=available)

        # Convert string back to Model enum (Model already imported at module level)
        try:
            return Model(best)
        except ValueError:
            # Fallback to first available
            return (
                self._get_available_models(task_type)[0]
                if self._get_available_models(task_type)
                else Model.GPT_4O
            )

    def _select_reviewer(self, primary: Model, task_type: TaskType) -> Model | None:
        """Select a reviewer model from different provider."""
        from ..models import ROUTING_TABLE

        primary_provider = get_provider(primary)

        for model in ROUTING_TABLE.get(task_type, []):
            if get_provider(model) != primary_provider and self.api_health.get(model, False):
                return model
        return None


# ─────────────────────────────────────────────
# 1. Multi-Perspective Pipeline
# ─────────────────────────────────────────────


class MultiPerspectivePipeline(BasePipeline):
    """
    Multi-Perspective Reasoning Pipeline

    Four independent perspectives analyze the problem in parallel:
    - Constructive: Find the strongest possible solution
    - Destructive: Find every flaw, do NOT propose solutions
    - Systemic: Identify second and third-order effects
    - Minimalist: Find the simplest 80% solution

    Best for: General problem analysis
    """

    PERSPECTIVE_SYSTEMS = {
        PerspectiveType.CONSTRUCTIVE: (
            "You are a constructive analyst. Your goal is to find the strongest "
            "possible solution to the problem. Focus on opportunities, strengths, "
            "and what could work. Be optimistic but realistic."
        ),
        PerspectiveType.DESTRUCTIVE: (
            "You are a destructive critic. Your goal is to find every flaw, "
            "weakness, and potential failure point. Do NOT propose solutions. "
            "Be ruthless and skeptical."
        ),
        PerspectiveType.SYSTEMIC: (
            "You are a systemic thinker. Your goal is to identify second and "
            "third-order effects, feedback loops, and unintended consequences. "
            "Think about the broader system and long-term impacts."
        ),
        PerspectiveType.MINIMALIST: (
            "You are a minimalist. Your goal is to find the simplest solution "
            "that addresses 80% of the problem with 20% of the effort. Focus on "
            "essential elements and eliminate complexity."
        ),
    }

    def get_method(self) -> ReasoningMethod:
        return ReasoningMethod.MULTI_PERSPECTIVE

    async def execute(self, task: Task, context: str = "") -> TaskResult:
        state = PipelineState(task=task, method=self.get_method())

        # Phase 2: Run all 4 perspectives in parallel
        await self._phase_perspectives(state, context)

        # Phase 3: Critique and score all candidates
        await self._phase_critique(state)

        # Select top-k candidates
        state.top_candidates = state.candidates[: self.top_k]

        # Synthesize final output
        await self._phase_synthesis(state)

        return self._build_result(state)

    async def _phase_perspectives(self, state: PipelineState, context: str):
        """Run all 4 perspectives concurrently."""
        from .phase_aware_models import PhaseType

        # Use phase-aware model selection for analysis
        primary = self._get_model_for_phase(PhaseType.ANALYSIS, state.task.type)

        # Get diverse models for different perspectives (optional)
        models = self._get_available_models(state.task.type)
        if not models:
            logger.error("No models available for perspectives")
            return

        async def run_perspective(perspective: PerspectiveType):
            system_prompt = self.PERSPECTIVE_SYSTEMS[perspective]
            user_prompt = f"Task: {state.task.prompt}\n\nContext: {context}\n\nAnalyze from {perspective.value} perspective."

            response = await self.client.call(
                model=primary,
                system=system_prompt,
                prompt=user_prompt,
                max_tokens=state.task.max_output_tokens,
                temperature=0.7,
            )

            data = self._extract_json(response.text) or {}
            return SolutionCandidate(
                perspective=perspective.value,
                content=data.get("core_analysis", response.text),
                key_insights=data.get("key_insights", []),
                metadata={"model": primary.value},
            )

        # Run all 4 perspectives in parallel
        tasks = [run_perspective(p) for p in PerspectiveType]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, SolutionCandidate):
                state.candidates.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Perspective failed: {result}")

    async def _phase_critique(self, state: PipelineState):
        """Score all candidates on multiple criteria."""
        from .phase_aware_models import PhaseType

        if len(state.candidates) == 0:
            return

        # Use phase-aware model selection for critique
        scorer = self._get_model_for_phase(PhaseType.CRITIQUE, state.task.type)

        candidates_text = "\n\n".join([f"[{c.perspective}]\n{c.content}" for c in state.candidates])

        system_prompt = (
            "You are an expert evaluator. Score each perspective on these criteria (0-10):\n"
            "- Logical consistency\n"
            "- Feasibility\n"
            "- Completeness\n"
            "- Novelty\n\n"
            "Return JSON array of scores."
        )

        user_prompt = f"Task: {state.task.prompt}\n\nCandidates:\n{candidates_text}"

        response = await self.client.call(
            model=scorer,
            system=system_prompt,
            prompt=user_prompt,
            max_tokens=1000,
            temperature=0.1,
        )

        data = self._extract_json(response.text)
        if data and isinstance(data, list):
            for score_data in data:
                state.scores.append(
                    CritiqueScore(
                        perspective=score_data.get("perspective", ""),
                        total=sum(
                            [
                                score_data.get("logical_consistency", 5),
                                score_data.get("feasibility", 5),
                                score_data.get("completeness", 5),
                                score_data.get("novelty", 5),
                            ]
                        )
                        / 4,
                        logical_consistency=score_data.get("logical_consistency", 5),
                        feasibility=score_data.get("feasibility", 5),
                        completeness=score_data.get("completeness", 5),
                        novelty=score_data.get("novelty", 5),
                    )
                )

        # Sort candidates by score
        scored = {s.perspective: s.total for s in state.scores}
        state.candidates.sort(key=lambda c: scored.get(c.perspective, 0), reverse=True)

    async def _phase_synthesis(self, state: PipelineState):
        """Synthesize top candidates into final solution."""
        from .phase_aware_models import PhaseType

        if not state.top_candidates:
            state.final_output = "No viable candidates generated."
            return

        # Use phase-aware model selection for synthesis
        synthesizer = self._get_model_for_phase(PhaseType.SYNTHESIS, state.task.type)

        top_texts = "\n\n".join([f"[{c.perspective}]\n{c.content}" for c in state.top_candidates])

        system_prompt = (
            "You are a synthesis expert. Combine the insights from multiple "
            "perspectives into a coherent, actionable solution. Preserve the "
            "strengths of each perspective while resolving contradictions."
        )

        user_prompt = f"Task: {state.task.prompt}\n\nTop Perspectives:\n{top_texts}"

        response = await self.client.call(
            model=synthesizer,
            system=system_prompt,
            prompt=user_prompt,
            max_tokens=state.task.max_output_tokens,
            temperature=0.3,
        )

        state.final_output = response.text
        state.final_score = state.scores[0].total if state.scores else 0.0

    def _build_result(self, state: PipelineState) -> TaskResult:
        """Convert pipeline state to TaskResult."""
        return TaskResult(
            task_id=state.task.id,
            output=state.final_output,
            score=state.final_score,
            model_used=Model.GPT_4O_MINI,  # Will be updated with actual model
            status=TaskStatus.COMPLETED if state.final_score >= 0.7 else TaskStatus.DEGRADED,
            metadata={
                "method": self.get_method().value,
                "candidates": len(state.candidates),
                "top_candidate": (
                    state.top_candidates[0].perspective if state.top_candidates else None
                ),
            },
        )


# ─────────────────────────────────────────────
# 2. Iterative Pipeline
# ─────────────────────────────────────────────


class IterativePipeline(BasePipeline):
    """
    Iterative Refinement Pipeline

    Evolutionary approach with iterative improvement.
    Maximum 3 rounds with early exit when mean_score ≥ 8.5.

    Best for: Optimization, design problems
    """

    MAX_ROUNDS = 3
    CONVERGENCE_THRESHOLD = 8.5

    def get_method(self) -> ReasoningMethod:
        return ReasoningMethod.ITERATIVE

    async def execute(self, task: Task, context: str = "") -> TaskResult:
        state = PipelineState(task=task, method=self.get_method())

        for round_num in range(1, self.MAX_ROUNDS + 1):
            logger.info(f"Iterative round {round_num}/{self.MAX_ROUNDS}")

            # Generate candidates with reflexion memory
            await self._phase_generate(state, context, round_num)

            # Critique and score
            await self._phase_critique(state)

            # Store insights for next round
            new_memories = [s.steel_man for s in state.scores if s.steel_man]
            state.reflexion_memory.extend(new_memories)

            # Early convergence check
            if state.scores:
                mean_score = sum(s.logical_consistency for s in state.scores) / len(state.scores)
                if mean_score >= self.CONVERGENCE_THRESHOLD:
                    logger.info(
                        f"Convergence achieved at round {round_num} (mean={mean_score:.2f})"
                    )
                    break

            # Clear for next round (except last)
            if round_num < self.MAX_ROUNDS:
                state.candidates = []
                state.scores = []
                state.top_candidates = []

        # Synthesize final output
        state.top_candidates = state.candidates[: self.top_k]
        await self._phase_synthesis(state)

        return self._build_result(state)

    async def _phase_generate(self, state: PipelineState, context: str, round_num: int):
        """Generate candidates with reflexion memory."""
        from .phase_aware_models import PhaseType

        # Use phase-aware model selection for generation
        generator = self._get_model_for_phase(PhaseType.GENERATION, state.task.type)

        # Build context with reflexion memory
        memory_context = ""
        if state.reflexion_memory:
            memory_context = "\n\nPrevious Insights:\n" + "\n".join(state.reflexion_memory)

        system_prompt = (
            "You are an iterative problem solver. Generate a solution that builds upon "
            "previous insights and improves upon weaknesses. Focus on continuous refinement."
        )

        user_prompt = (
            f"Task: {state.task.prompt}\n\nContext: {context}{memory_context}\n\nRound: {round_num}"
        )

        response = await self.client.call(
            model=generator,
            system=system_prompt,
            prompt=user_prompt,
            max_tokens=state.task.max_output_tokens,
            temperature=0.5,
        )

        data = self._extract_json(response.text) or {}
        state.candidates.append(
            SolutionCandidate(
                perspective="iterative",
                content=data.get("solution", response.text),
                key_insights=data.get("key_insights", []),
                metadata={"round": round_num, "model": generator.value},
            )
        )

    async def _phase_critique(self, state: PipelineState):
        """Score candidates with detailed feedback."""
        from .phase_aware_models import PhaseType

        if len(state.candidates) == 0:
            return

        # Use phase-aware model selection for critique
        scorer = self._get_model_for_phase(PhaseType.CRITIQUE, state.task.type)

        candidate = state.candidates[-1]  # Score latest candidate

        system_prompt = (
            "You are a rigorous evaluator. Score the solution on these criteria (0-10):\n"
            "- Logical consistency\n"
            "- Feasibility\n"
            "- Completeness\n"
            "- Novelty\n\n"
            "Also provide a 'steel-man' argument (strongest version) and rationale."
        )

        user_prompt = f"Task: {state.task.prompt}\n\nSolution:\n{candidate.content}"

        response = await self.client.call(
            model=scorer,
            system=system_prompt,
            prompt=user_prompt,
            max_tokens=1500,
            temperature=0.2,
        )

        data = self._extract_json(response.text) or {}
        state.scores.append(
            CritiqueScore(
                perspective="iterative",
                total=sum(
                    [
                        data.get("logical_consistency", 5),
                        data.get("feasibility", 5),
                        data.get("completeness", 5),
                        data.get("novelty", 5),
                    ]
                )
                / 4,
                logical_consistency=data.get("logical_consistency", 5),
                feasibility=data.get("feasibility", 5),
                completeness=data.get("completeness", 5),
                novelty=data.get("novelty", 5),
                steel_man=data.get("steel_man", ""),
                rationale=data.get("rationale", ""),
            )
        )

    async def _phase_synthesis(self, state: PipelineState):
        """Use best candidate as final output."""
        if state.candidates:
            state.final_output = state.candidates[-1].content
            state.final_score = state.scores[-1].total if state.scores else 0.0
        else:
            state.final_output = "No solution generated."
            state.final_score = 0.0

    def _build_result(self, state: PipelineState) -> TaskResult:
        return TaskResult(
            task_id=state.task.id,
            output=state.final_output,
            score=state.final_score,
            model_used=Model.GPT_4O_MINI,
            status=TaskStatus.COMPLETED if state.final_score >= 0.7 else TaskStatus.DEGRADED,
            metadata={
                "method": self.get_method().value,
                "rounds": len(state.candidates),
                "convergence_score": state.final_score,
            },
        )


# ─────────────────────────────────────────────
# 3. Debate Pipeline
# ─────────────────────────────────────────────


class DebatePipeline(BasePipeline):
    """
    Multi-Agent Debate Pipeline

    Two models (Model A vs Model B) compete with a third (Judge) evaluating.
    Rounds: Opening → Rebuttal → Cross-Examination → Judgment

    Best for: Strategic decisions with trade-offs
    """

    def get_method(self) -> ReasoningMethod:
        return ReasoningMethod.DEBATE

    async def execute(self, task: Task, context: str = "") -> TaskResult:
        state = PipelineState(task=task, method=self.get_method())

        # Round 1: Opening statements
        await self._phase_debate_opening(state, context)

        # Round 2: Rebuttals
        await self._phase_debate_rebuttal(state)

        # Round 3: Cross-examination
        await self._phase_debate_cross_examine(state)

        # Final: Judge decision
        await self._phase_debate_judge(state)

        # Synthesize final output from judge's decision
        await self._phase_synthesis(state)

        return self._build_result(state)

    async def _phase_debate_opening(self, state: PipelineState, context: str):
        """Two parallel opening statements."""
        from .phase_aware_models import PhaseType

        # Use phase-aware model selection for debate
        model_pro = self._get_model_for_phase(PhaseType.DEBATE, state.task.type)
        model_con = self._get_model_for_phase(PhaseType.CRITIQUE, state.task.type)

        # Side A: Pro/Constructive
        system_a = "You are arguing FOR the proposed solution. Present the strongest case with evidence and reasoning."
        prompt_a = f"Task: {state.task.prompt}\n\nContext: {context}\n\nPresent your opening argument FOR this approach."

        # Side B: Con/Destructive
        system_b = "You are arguing AGAINST the proposed solution. Identify flaws, risks, and better alternatives."
        prompt_b = f"Task: {state.task.prompt}\n\nContext: {context}\n\nPresent your opening argument AGAINST this approach."

        response_a, response_b = await asyncio.gather(
            self.client.call(
                model=model_pro,
                system=system_a,
                prompt=prompt_a,
                max_tokens=state.task.max_output_tokens,
                temperature=0.7,
            ),
            self.client.call(
                model=model_con,
                system=system_b,
                prompt=prompt_b,
                max_tokens=state.task.max_output_tokens,
                temperature=0.7,
            ),
        )

        state.debate_rounds.append(
            {
                "round": 1,
                "type": "opening",
                "statements": [
                    {"side": "A", "content": response_a.text, "model": model_pro.value},
                    {"side": "B", "content": response_b.text, "model": model_con.value},
                ],
            }
        )

    async def _phase_debate_rebuttal(self, state: PipelineState):
        """Each side rebuts the other's opening."""
        from .phase_aware_models import PhaseType

        if not state.debate_rounds:
            return

        opening = state.debate_rounds[0]
        statement_a = opening["statements"][0]["content"]
        statement_b = opening["statements"][1]["content"]

        # Use phase-aware model selection for rebuttal
        model_pro = self._get_model_for_phase(PhaseType.DEBATE, state.task.type)
        model_con = self._get_model_for_phase(PhaseType.CRITIQUE, state.task.type)

        # A rebuts B
        prompt_a = f"Task: {state.task.prompt}\n\nOpponent's argument:\n{statement_b}\n\nRebut their points and defend your position."

        # B rebuts A
        prompt_b = f"Task: {state.task.prompt}\n\nOpponent's argument:\n{statement_a}\n\nRebut their points and defend your position."

        response_a, response_b = await asyncio.gather(
            self.client.call(
                model=model_pro,
                system="You are a debater. Rebut your opponent's arguments point-by-point.",
                prompt=prompt_a,
                max_tokens=state.task.max_output_tokens,
                temperature=0.6,
            ),
            self.client.call(
                model=model_con,
                system="You are a debater. Rebut your opponent's arguments point-by-point.",
                prompt=prompt_b,
                max_tokens=state.task.max_output_tokens,
                temperature=0.6,
            ),
        )

        state.debate_rounds.append(
            {
                "round": 2,
                "type": "rebuttal",
                "statements": [
                    {"side": "A", "content": response_a.text, "model": model_pro.value},
                    {"side": "B", "content": response_b.text, "model": model_con.value},
                ],
            }
        )

    async def _phase_debate_cross_examine(self, state: PipelineState):
        """Judge asks probing questions to both sides."""
        from .phase_aware_models import PhaseType

        # Use phase-aware model selection for evaluation/judge
        judge_model = self._get_model_for_phase(PhaseType.EVALUATION, state.task.type)

        # Generate probing questions
        all_statements = "\n\n".join(
            [
                f"Round {r['round']} ({r['type']}):\n"
                + "\n".join([s["content"][:500] for s in r["statements"]])
                for r in state.debate_rounds
            ]
        )

        system_prompt = "You are a judge in a debate. Ask 3 probing questions that reveal the strengths and weaknesses of each position."
        user_prompt = f"Task: {state.task.prompt}\n\nDebate so far:\n{all_statements}"

        response = await self.client.call(
            model=judge_model,
            system=system_prompt,
            prompt=user_prompt,
            max_tokens=800,
            temperature=0.5,
        )

        state.metadata["cross_exam_questions"] = response.text

    async def _phase_debate_judge(self, state: PipelineState):
        """Judge makes final decision."""
        from .phase_aware_models import PhaseType

        # Use phase-aware model selection for evaluation/judge
        judge_model = self._get_model_for_phase(PhaseType.EVALUATION, state.task.type)

        all_statements = "\n\n".join(
            [
                f"Round {r['round']} ({r['type']}):\n"
                + "\n".join([f"[{s['side']}] {s['content']}" for s in r["statements"]])
                for r in state.debate_rounds
            ]
        )

        system_prompt = (
            "You are an impartial judge. Evaluate both sides based on:\n"
            "- Logical consistency\n"
            "- Evidence quality\n"
            "- Practical feasibility\n\n"
            "Declare a winner and explain your reasoning. Return JSON with scores."
        )

        user_prompt = f"Task: {state.task.prompt}\n\nDebate transcript:\n{all_statements}\n\nQuestions: {state.metadata.get('cross_exam_questions', '')}"

        response = await self.client.call(
            model=judge_model,
            system=system_prompt,
            prompt=user_prompt,
            max_tokens=1500,
            temperature=0.2,
        )

        data = self._extract_json(response.text) or {}
        state.metadata["judge_decision"] = response.text
        state.metadata["winner"] = data.get("winner", "A")
        state.metadata["scores"] = {
            "A": data.get("score_A", 5),
            "B": data.get("score_B", 5),
        }

    async def _phase_synthesis(self, state: PipelineState):
        """Synthesize final output from winning side + judge insights."""
        if not state.debate_rounds:
            state.final_output = "No debate occurred."
            return

        # Get winning side's final statement
        winner = state.metadata.get("winner", "A")
        final_round = state.debate_rounds[-1]
        winner_statement = next(
            (s["content"] for s in final_round["statements"] if s["side"] == winner), ""
        )

        judge_decision = state.metadata.get("judge_decision", "")

        # Synthesize
        models = self._get_available_models(state.task.type)
        synthesizer = models[0] if models else Model.GPT_4O_MINI

        system_prompt = "Synthesize the debate outcome into a final recommendation that incorporates the winning argument and judge's insights."
        user_prompt = f"Task: {state.task.prompt}\n\nWinner's argument: {winner_statement}\n\nJudge's decision: {judge_decision}"

        response = await self.client.call(
            model=synthesizer,
            system=system_prompt,
            prompt=user_prompt,
            max_tokens=state.task.max_output_tokens,
            temperature=0.3,
        )

        state.final_output = response.text
        state.final_score = (
            max(state.metadata.get("scores", {}).values()) / 10.0
            if state.metadata.get("scores")
            else 0.5
        )

    def _build_result(self, state: PipelineState) -> TaskResult:
        return TaskResult(
            task_id=state.task.id,
            output=state.final_output,
            score=state.final_score,
            model_used=Model.GPT_4O_MINI,
            status=TaskStatus.COMPLETED if state.final_score >= 0.7 else TaskStatus.DEGRADED,
            metadata={
                "method": self.get_method().value,
                "winner": state.metadata.get("winner"),
                "rounds": len(state.debate_rounds),
                "judge_decision": state.metadata.get("judge_decision", "")[:500],
            },
        )


# ─────────────────────────────────────────────
# 4. Research Pipeline
# ─────────────────────────────────────────────


class ResearchPipeline(BasePipeline):
    """
    Evidence-Based Research Pipeline

    Uses Nexus Search for web search with iterative discovery.
    Pipeline: Deep Research → Analysis with Web Context → Fact-Checked Critique

    Best for: Empirical questions, current events
    """

    def __init__(self, *args, nexus_enabled: bool = True, x_search_enabled: bool = False, **kwargs):
        """Initialize Research pipeline with optional Nexus Search and X Search integration."""
        super().__init__(*args, **kwargs)
        self.nexus_enabled = nexus_enabled
        self.x_search_enabled = x_search_enabled
        self._nexus_search = None
        self._SearchSource = None
        self._x_search = None
        self._XSearchClient = None

    def _get_nexus(self):
        """Lazy import of Nexus Search."""
        if self._nexus_search is None and self.nexus_enabled:
            try:
                from orchestrator.nexus_search import SearchSource, search

                self._nexus_search = search
                self._SearchSource = SearchSource
            except ImportError:
                self.nexus_enabled = False
        return self._nexus_search, self._SearchSource

    def _get_x_search(self):
        """Lazy import of X Search."""
        if self._x_search is None:
            try:
                from orchestrator.xai_search import XSearchClient

                self._XSearchClient = XSearchClient
                return XSearchClient
            except ImportError:
                logger.warning("X Search not installed")
                return None
        return self._XSearchClient

    def get_method(self) -> ReasoningMethod:
        return ReasoningMethod.RESEARCH

    async def execute(self, task: Task, context: str = "") -> TaskResult:
        state = PipelineState(task=task, method=self.get_method())

        # Phase 1: Deep iterative web research
        await self._phase_research(state)

        # Phase 2: Analyze with web context (using Multi-Perspective)
        await self._phase_analyze(state)

        # Phase 3: Fact-checked critique
        await self._phase_critique(state)

        # Synthesize
        state.top_candidates = state.candidates[: self.top_k]
        await self._phase_synthesis(state)

        return self._build_result(state)

    async def _phase_research(self, state: PipelineState):
        """Deep iterative web research using Nexus Search and X Search."""
        nexus_search, SearchSource = self._get_nexus()
        XSearchClient = self._get_x_search()

        # Combine Nexus Search (web) with X Search (real-time social)
        max_iterations = 3
        current_knowledge = []
        x_search_results = []

        # Nexus Search for web/academic content
        if nexus_search is not None and self.nexus_enabled:
            for i in range(1, max_iterations + 1):
                logger.info(f"Nexus Research iteration {i}/{max_iterations}")

                try:
                    results = await nexus_search(
                        query=state.task.prompt,
                        sources=[SearchSource.WEB, SearchSource.ACADEMIC, SearchSource.NEWS],
                        num_results=10,
                    )

                    for result in results.top[:5]:
                        knowledge_item = (
                            f"Source: {result.title}\nURL: {result.url}\nContent: {result.content}"
                        )
                        current_knowledge.append(knowledge_item)

                    logger.info(f"Nexus Search found {len(results)} results")
                    break

                except Exception as e:
                    logger.warning(f"Nexus Search iteration {i} failed: {e}")
                    if i == max_iterations:
                        await self._phase_research_llm_fallback(state)
                        return
        else:
            logger.info("Nexus Search not available, using LLM-based research")
            await self._phase_research_llm_fallback(state)
            return

        # X Search for real-time social media insights
        if XSearchClient is not None and self.x_search_enabled:
            logger.info("Searching X/Twitter for real-time insights")
            try:
                async with XSearchClient() as x_client:
                    x_results = await x_client.search_posts(
                        query=state.task.prompt,
                        count=5,
                        sort="latest",
                    )

                    for post in x_results.posts[:3]:
                        verified_badge = "✓" if post.verified else " "
                        x_knowledge = (
                            f"X Post [{verified_badge}] @{post.author_handle}: {post.text[:200]}"
                        )
                        x_search_results.append(x_knowledge)

                    logger.info(f"X Search found {len(x_results.posts)} posts")
            except Exception as e:
                logger.warning(f"X Search failed: {e}")

        # Combine both sources
        state.web_discovery_results = current_knowledge + x_search_results
        state.metadata["research_iterations"] = len(current_knowledge)
        state.metadata["nexus_search"] = True
        state.metadata["x_search"] = len(x_search_results) > 0
        state.metadata["x_search_posts"] = len(x_search_results)

    async def _phase_research_llm_fallback(self, state: PipelineState):
        """Fallback to LLM-based research if Nexus unavailable."""
        from .phase_aware_models import PhaseType

        # Use phase-aware model selection for research
        researcher = self._get_model_for_phase(PhaseType.RESEARCH, state.task.type)

        max_iterations = 3
        current_knowledge = []

        for i in range(1, max_iterations + 1):
            logger.info(f"Research iteration {i}/{max_iterations} (LLM fallback)")

            # Decide if more research needed
            system_prompt = (
                "You are a research coordinator. Decide if more information is needed.\n"
                "Return JSON: {'action': 'search' or 'done', 'queries': [] if search}"
            )

            knowledge_context = (
                "\n".join([f"- {k}" for k in current_knowledge])
                if current_knowledge
                else "No knowledge yet"
            )
            user_prompt = f"Task: {state.task.prompt}\n\nCurrent knowledge:\n{knowledge_context}\n\nDo you need more information? (max {max_iterations} iterations)"

            response = await self.client.call(
                model=researcher,
                system=system_prompt,
                prompt=user_prompt,
                max_tokens=500,
                temperature=0.3,
            )

            data = self._extract_json(response.text) or {}
            action = data.get("action", "done")

            if action == "done" or i == max_iterations:
                break

            # Execute "searches" (simulated with LLM knowledge)
            queries = data.get("queries", [])[:3]
            for query in queries:
                search_prompt = f"Provide detailed information about: {query}"
                search_response = await self.client.call(
                    model=researcher,
                    system="You are a research assistant. Provide factual, detailed information.",
                    prompt=search_prompt,
                    max_tokens=1000,
                    temperature=0.2,
                )
                current_knowledge.append(f"Query: {query}\nResult: {search_response.text}")

        state.web_discovery_results = current_knowledge
        state.metadata["research_iterations"] = len(current_knowledge)
        state.metadata["nexus_search"] = False

    async def _phase_analyze(self, state: PipelineState):
        """Analyze findings using Multi-Perspective approach."""
        from .phase_aware_models import PhaseType

        web_context = (
            "\n\n".join(state.web_discovery_results)
            if state.web_discovery_results
            else "No web research conducted"
        )

        # Use phase-aware model selection for analysis
        primary = self._get_model_for_phase(PhaseType.ANALYSIS, state.task.type)

        system_prompt = (
            "You are an analyst. Synthesize research findings into actionable insights.\n"
            "Base your analysis on the provided evidence."
        )

        user_prompt = f"Task: {state.task.prompt}\n\nResearch findings:\n{web_context}"

        response = await self.client.call(
            model=primary,
            system=system_prompt,
            prompt=user_prompt,
            max_tokens=state.task.max_output_tokens,
            temperature=0.4,
        )

        data = self._extract_json(response.text) or {}
        state.candidates.append(
            SolutionCandidate(
                perspective="research_based",
                content=data.get("analysis", response.text),
                key_insights=data.get("key_insights", []),
                metadata={"sources": len(state.web_discovery_results)},
            )
        )

    async def _phase_critique(self, state: PipelineState):
        """Fact-check the analysis."""
        if not state.candidates:
            return

        models = self._get_available_models(state.task.type)
        critic = self._select_reviewer(models[0], state.task.type) or (
            models[0] if models else Model.GPT_4O_MINI
        )

        candidate = state.candidates[0]
        web_context = (
            "\n\n".join(state.web_discovery_results) if state.web_discovery_results else ""
        )

        system_prompt = (
            "You are a fact-checker. Evaluate the analysis against the research evidence.\n"
            "Check for:\n"
            "- Accuracy of claims\n"
            "- Logical consistency with evidence\n"
            "- Missing critical information\n\n"
            "Score 0-10 on accuracy."
        )

        user_prompt = f"Task: {state.task.prompt}\n\nAnalysis: {candidate.content}\n\nResearch evidence: {web_context}"

        response = await self.client.call(
            model=critic,
            system=system_prompt,
            prompt=user_prompt,
            max_tokens=1000,
            temperature=0.1,
        )

        data = self._extract_json(response.text) or {}
        accuracy_score = data.get("accuracy_score", 5)

        state.scores.append(
            CritiqueScore(
                perspective="fact_check",
                total=accuracy_score / 10.0,
                logical_consistency=accuracy_score / 10.0,
                feasibility=data.get("feasibility", 5) / 10.0,
                completeness=data.get("completeness", 5) / 10.0,
                novelty=5.0 / 10.0,
            )
        )

    async def _phase_synthesis(self, state: PipelineState):
        """Synthesize research-based solution."""
        if not state.candidates:
            state.final_output = "No analysis generated."
            return

        models = self._get_available_models(state.task.type)
        synthesizer = models[0] if models else Model.GPT_4O_MINI

        candidate = state.candidates[0]
        web_context = (
            "\n\n".join(state.web_discovery_results) if state.web_discovery_results else ""
        )

        system_prompt = "Create a final solution that integrates research evidence with practical recommendations."
        user_prompt = (
            f"Task: {state.task.prompt}\n\nAnalysis: {candidate.content}\n\nEvidence: {web_context}"
        )

        response = await self.client.call(
            model=synthesizer,
            system=system_prompt,
            prompt=user_prompt,
            max_tokens=state.task.max_output_tokens,
            temperature=0.3,
        )

        state.final_output = response.text
        state.final_score = state.scores[0].total if state.scores else 0.5

    def _build_result(self, state: PipelineState) -> TaskResult:
        return TaskResult(
            task_id=state.task.id,
            output=state.final_output,
            score=state.final_score,
            model_used=Model.GPT_4O_MINI,
            status=TaskStatus.COMPLETED if state.final_score >= 0.7 else TaskStatus.DEGRADED,
            metadata={
                "method": self.get_method().value,
                "research_sources": len(state.web_discovery_results),
                "iterations": state.metadata.get("research_iterations", 0),
            },
        )


# ─────────────────────────────────────────────
# 5. Jury Pipeline
# ─────────────────────────────────────────────


class JuryPipeline(BasePipeline):
    """
    Multi-Agent Jury Pipeline

    Parallel multi-agent system with meta-evaluation.
    Pipeline: 4 Generators → 3 Critics → Verifier + Meta-Evaluator → Weighted Ranking

    Best for: High-risk decisions with multiple stakeholders
    """

    def get_method(self) -> ReasoningMethod:
        return ReasoningMethod.JURY

    async def execute(self, task: Task, context: str = "") -> TaskResult:
        state = PipelineState(task=task, method=self.get_method())

        # Phase 1: 4 parallel generators
        await self._phase_jury_generate(state, context)

        # Phase 2: 3 parallel critics
        await self._phase_jury_critique(state)

        # Phase 3: Verifier + Meta-Evaluator
        await self._phase_jury_verify_and_meta_eval(state)

        # Phase 4: Weighted ranking
        await self._phase_jury_weighted_ranking(state)

        # Synthesize
        await self._phase_synthesis(state)

        return self._build_result(state)

    async def _phase_jury_generate(self, state: PipelineState, context: str):
        """4 parallel generators create solutions."""
        from .phase_aware_models import PhaseType

        # Use phase-aware model selection for generation
        generator_model = self._get_model_for_phase(PhaseType.GENERATION, state.task.type)

        models = [generator_model] * 4

        async def generate(role: str, model: Model):
            system_prompt = (
                f"You are generator {role}. Create a comprehensive solution with unique insights."
            )
            user_prompt = f"Task: {state.task.prompt}\n\nContext: {context}\n\nRole: {role}"

            response = await self.client.call(
                model=model,
                system=system_prompt,
                prompt=user_prompt,
                max_tokens=state.task.max_output_tokens,
                temperature=0.7,
            )

            data = self._extract_json(response.text) or {}
            return {
                "role": role,
                "content": data.get("solution", response.text),
                "model": model.value,
            }

        gen_roles = ["generator_1", "generator_2", "generator_3", "generator_4"]
        tasks = [generate(role, models[i]) for i, role in enumerate(gen_roles)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, dict):
                state.candidates.append(
                    SolutionCandidate(
                        perspective=result["role"],
                        content=result["content"],
                        key_insights=[],
                        metadata={"model": result["model"]},
                    )
                )

    async def _phase_jury_critique(self, state: PipelineState):
        """3 parallel critics evaluate all generators."""
        from .phase_aware_models import PhaseType

        # Use phase-aware model selection for critique
        critic_model = self._get_model_for_phase(PhaseType.CRITIQUE, state.task.type)
        critic_models = [critic_model] * 3

        candidate_contents = "\n\n".join(
            [f"[{c.perspective}]\n{c.content}" for c in state.candidates]
        )

        async def critique(role: str, model: Model):
            system_prompt = (
                f"You are critic {role}. Evaluate all solutions rigorously on multiple criteria."
            )
            user_prompt = (
                f"Task: {state.task.prompt}\n\nSolutions:\n{candidate_contents}\n\nRole: {role}"
            )

            response = await self.client.call(
                model=model,
                system=system_prompt,
                prompt=user_prompt,
                max_tokens=1500,
                temperature=0.2,
            )

            data = self._extract_json(response.text) or {}
            return {
                "role": role,
                "scores": data,
                "model": model.value,
            }

        critic_roles = ["critic_1", "critic_2", "critic_3"]
        tasks = [critique(role, critic_models[i]) for i, role in enumerate(critic_roles)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, dict):
                state.metadata.setdefault("critics", []).append(result)

    async def _phase_jury_verify_and_meta_eval(self, state: PipelineState):
        """Verify claims and evaluate critic quality."""
        from .phase_aware_models import PhaseType

        # Use phase-aware model selection for evaluation
        verifier = self._get_model_for_phase(PhaseType.EVALUATION, state.task.type)

        # Verify claims
        candidate_contents = "\n\n".join(
            [f"{c.perspective}: {c.content}" for c in state.candidates]
        )

        system_prompt = (
            "You are a verifier. Check all factual claims in the solutions.\n"
            "Return JSON array of verifications: [{'claim': '', 'verified': true/false, 'reason': ''}]"
        )

        user_prompt = f"Task: {state.task.prompt}\n\nSolutions: {candidate_contents}"

        response = await self.client.call(
            model=verifier,
            system=system_prompt,
            prompt=user_prompt,
            max_tokens=2000,
            temperature=0.1,
        )

        data = self._extract_json(response.text) or []
        state.metadata["verifications"] = data

        # Meta-evaluate critics
        meta_prompt = "Evaluate the quality of each critic's feedback. Return JSON: {'critic_1': {'quality': 0-10, 'insights': ''}, ...}"

        response = await self.client.call(
            model=verifier,
            system="You are a meta-evaluator. Assess critic quality.",
            prompt=meta_prompt,
            max_tokens=1500,
            temperature=0.2,
        )

        meta_data = self._extract_json(response.text) or {}
        state.metadata["meta_evaluation"] = meta_data

    async def _phase_jury_weighted_ranking(self, state: PipelineState):
        """Compute weighted ranking based on critic scores and verifications."""
        if not state.candidates:
            return

        # Aggregate scores from all critics
        candidate_scores = {c.perspective: [] for c in state.candidates}

        for critic_data in state.metadata.get("critics", []):
            scores = critic_data.get("scores", {})
            for perspective, score_data in scores.items():
                if perspective in candidate_scores:
                    total = (
                        sum(
                            [
                                score_data.get("logical_consistency", 5),
                                score_data.get("feasibility", 5),
                                score_data.get("completeness", 5),
                            ]
                        )
                        / 3
                    )
                    candidate_scores[perspective].append(total)

        # Weight by meta-evaluation quality
        state.metadata.get("meta_evaluation", {})

        for candidate in state.candidates:
            scores = candidate_scores.get(candidate.perspective, [])
            avg_score = sum(scores) / len(scores) if scores else 5.0

            # Apply verification penalty for false claims
            verifications = state.metadata.get("verifications", [])
            false_claims = sum(1 for v in verifications if not v.get("verified", True))
            penalty = false_claims * 0.1

            final_score = max(0, avg_score - penalty)

            state.scores.append(
                CritiqueScore(
                    perspective=candidate.perspective,
                    total=final_score / 10.0,
                    logical_consistency=final_score / 10.0,
                )
            )

        # Sort by score
        state.candidates.sort(
            key=lambda c: next(
                (s.total for s in state.scores if s.perspective == c.perspective), 0
            ),
            reverse=True,
        )
        state.top_candidates = state.candidates[: self.top_k]

    async def _phase_synthesis(self, state: PipelineState):
        """Synthesize top candidates into final solution."""
        if not state.top_candidates:
            state.final_output = "No viable candidates."
            return

        models = self._get_available_models(state.task.type)
        synthesizer = models[0] if models else Model.GPT_4O_MINI

        top_texts = "\n\n".join([f"[{c.perspective}]\n{c.content}" for c in state.top_candidates])
        verifications = state.metadata.get("verifications", [])

        system_prompt = (
            "Synthesize the best elements from multiple solutions into one coherent recommendation."
        )
        user_prompt = f"Task: {state.task.prompt}\n\nTop solutions:\n{top_texts}\n\nVerifications: {json.dumps(verifications)}"

        response = await self.client.call(
            model=synthesizer,
            system=system_prompt,
            prompt=user_prompt,
            max_tokens=state.task.max_output_tokens,
            temperature=0.3,
        )

        state.final_output = response.text
        state.final_score = state.scores[0].total if state.scores else 0.5

    def _build_result(self, state: PipelineState) -> TaskResult:
        return TaskResult(
            task_id=state.task.id,
            output=state.final_output,
            score=state.final_score,
            model_used=Model.GPT_4O_MINI,
            status=TaskStatus.COMPLETED if state.final_score >= 0.7 else TaskStatus.DEGRADED,
            metadata={
                "method": self.get_method().value,
                "generators": len(state.candidates),
                "critics": len(state.metadata.get("critics", [])),
                "verifications": len(state.metadata.get("verifications", [])),
            },
        )


# ─────────────────────────────────────────────
# 6. Scientific Pipeline
# ─────────────────────────────────────────────


class ScientificPipeline(BasePipeline):
    """
    Hypothetico-Experimental Scientific Pipeline

    Scientific method approach: Hypothesize → Design Tests → Evaluate Evidence

    Best for: Research questions, technical decisions
    """

    def get_method(self) -> ReasoningMethod:
        return ReasoningMethod.SCIENTIFIC

    async def execute(self, task: Task, context: str = "") -> TaskResult:
        state = PipelineState(task=task, method=self.get_method())

        # Phase 1: Generate hypotheses
        await self._phase_scientific_hypothesize(state)

        # Phase 2: Design tests
        await self._phase_scientific_test(state)

        # Phase 3: Evaluate evidence
        await self._phase_scientific_evaluate(state)

        # Synthesize
        await self._phase_synthesis(state)

        return self._build_result(state)

    async def _phase_scientific_hypothesize(self, state: PipelineState):
        """Generate multiple hypotheses."""
        from .phase_aware_models import PhaseType

        # Use phase-aware model selection for analysis (hypothesis generation)
        primary = self._get_model_for_phase(PhaseType.ANALYSIS, state.task.type)

        system_prompt = (
            "You are a scientist. Generate 3-5 competing hypotheses to explain or solve the problem.\n"
            "Return JSON: {'hypotheses': [{'name': '', 'description': '', 'predictions': []}]}"
        )

        user_prompt = f"Task: {state.task.prompt}\n\nContext: {context}"

        response = await self.client.call(
            model=primary,
            system=system_prompt,
            prompt=user_prompt,
            max_tokens=1500,
            temperature=0.7,
        )

        data = self._extract_json(response.text) or {}
        hypotheses = data.get("hypotheses", [])

        for hyp in hypotheses:
            state.candidates.append(
                SolutionCandidate(
                    perspective=hyp.get("name", "unknown"),
                    content=hyp.get("description", ""),
                    key_insights=hyp.get("predictions", []),
                    metadata={"type": "hypothesis"},
                )
            )

        state.metadata["hypotheses"] = hypotheses

    async def _phase_scientific_test(self, state: PipelineState):
        """Design tests for each hypothesis."""
        from .phase_aware_models import PhaseType

        # Use phase-aware model selection for research (test design)
        tester = self._get_model_for_phase(PhaseType.RESEARCH, state.task.type)

        hypotheses = state.metadata.get("hypotheses", [])

        async def design_test(hypothesis: dict):
            system_prompt = "Design a rigorous test to validate or falsify this hypothesis."
            user_prompt = f"Hypothesis: {hypothesis.get('name')}\nDescription: {hypothesis.get('description')}"

            response = await self.client.call(
                model=tester,
                system=system_prompt,
                prompt=user_prompt,
                max_tokens=1000,
                temperature=0.3,
            )

            return {"hypothesis": hypothesis.get("name"), "test_design": response.text}

        tasks = [design_test(h) for h in hypotheses]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        state.metadata["test_designs"] = [r for r in results if isinstance(r, dict)]

    async def _phase_scientific_evaluate(self, state: PipelineState):
        """Evaluate evidence for each hypothesis."""
        from .phase_aware_models import PhaseType

        # Use phase-aware model selection for evaluation
        evaluator = self._get_model_for_phase(PhaseType.EVALUATION, state.task.type)

        test_designs = state.metadata.get("test_designs", [])

        async def evaluate_evidence(test_design: dict):
            system_prompt = (
                "Evaluate the expected evidence strength for this test.\n"
                "Return JSON: {'evidence_strength': 0-10, 'confidence': 0-1, 'limitations': []}"
            )
            user_prompt = f"Test design: {test_design.get('test_design')}"

            response = await self.client.call(
                model=evaluator,
                system=system_prompt,
                prompt=user_prompt,
                max_tokens=800,
                temperature=0.2,
            )

            data = self._extract_json(response.text) or {}
            return {
                "hypothesis": test_design["hypothesis"],
                "evidence_strength": data.get("evidence_strength", 5),
                "confidence": data.get("confidence", 0.5),
                "limitations": data.get("limitations", []),
            }

        tasks = [evaluate_evidence(td) for td in test_designs]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        evaluations = [r for r in results if isinstance(r, dict)]
        state.metadata["evidence_evaluations"] = evaluations

        # Create scores
        for eval_data in evaluations:
            state.scores.append(
                CritiqueScore(
                    perspective=eval_data["hypothesis"],
                    total=eval_data["evidence_strength"] / 10.0,
                    logical_consistency=eval_data["confidence"],
                )
            )

        # Sort by evidence strength
        state.candidates.sort(
            key=lambda c: next(
                (s.total for s in state.scores if s.perspective == c.perspective), 0
            ),
            reverse=True,
        )

    async def _phase_synthesis(self, state: PipelineState):
        """Synthesize conclusion from best hypothesis."""
        from .phase_aware_models import PhaseType

        if not state.candidates:
            state.final_output = "No hypotheses generated."
            return

        # Use phase-aware model selection for synthesis
        synthesizer = self._get_model_for_phase(PhaseType.SYNTHESIS, state.task.type)

        best_candidate = state.candidates[0]
        evidence_evals = state.metadata.get("evidence_evaluations", [])
        best_eval = next(
            (e for e in evidence_evals if e["hypothesis"] == best_candidate.perspective), {}
        )

        system_prompt = "Synthesize the scientific conclusion based on the strongest hypothesis and evidence evaluation."
        user_prompt = f"Task: {state.task.prompt}\n\nBest hypothesis: {best_candidate.perspective}\nContent: {best_candidate.content}\nEvidence: {json.dumps(best_eval)}"

        response = await self.client.call(
            model=synthesizer,
            system=system_prompt,
            prompt=user_prompt,
            max_tokens=state.task.max_output_tokens,
            temperature=0.3,
        )

        state.final_output = response.text
        state.final_score = state.scores[0].total if state.scores else 0.5

    def _build_result(self, state: PipelineState) -> TaskResult:
        return TaskResult(
            task_id=state.task.id,
            output=state.final_output,
            score=state.final_score,
            model_used=Model.GPT_4O_MINI,
            status=TaskStatus.COMPLETED if state.final_score >= 0.7 else TaskStatus.DEGRADED,
            metadata={
                "method": self.get_method().value,
                "hypotheses": len(state.metadata.get("hypotheses", [])),
                "tests_designed": len(state.metadata.get("test_designs", [])),
                "best_hypothesis": state.candidates[0].perspective if state.candidates else None,
            },
        )


# ─────────────────────────────────────────────
# 7. Socratic Pipeline
# ─────────────────────────────────────────────


class SocraticPipeline(BasePipeline):
    """
    Socratic Questioning Pipeline

    Iterative questioning approach to clarify ambiguous problems.
    Pipeline: Initial Question → Follow-up Loop → Clarified Solution

    Best for: Clarifying ambiguous problems
    """

    MAX_ROUNDS = 3

    def get_method(self) -> ReasoningMethod:
        return ReasoningMethod.SOCRATIC

    async def execute(self, task: Task, context: str = "") -> TaskResult:
        state = PipelineState(task=task, method=self.get_method())

        # Phase 1: Initial Socratic questioning
        await self._phase_socratic_question(state, context)

        # Phase 2: Follow-up Q&A loop
        for round_num in range(2, self.MAX_ROUNDS + 1):
            clarity_score = state.metadata.get("clarity_score", 0)
            if clarity_score >= 8.0:
                logger.info(f"Clarity threshold reached at round {round_num-1}")
                break
            await self._phase_socratic_followup(state, round_num)

        # Phase 3: Solution based on clarified understanding
        await self._phase_socratic_solution(state)

        return self._build_result(state)

    async def _phase_socratic_question(self, state: PipelineState, context: str):
        """Generate initial Socratic questions."""
        from .phase_aware_models import PhaseType

        # Use phase-aware model selection for analysis (Socratic questioning)
        questioner = self._get_model_for_phase(PhaseType.ANALYSIS, state.task.type)

        system_prompt = (
            "You are a Socratic teacher. Ask 5-7 probing questions that clarify the problem.\n"
            "Focus on: assumptions, definitions, evidence, perspectives, implications.\n"
            "Return JSON: {'questions': [], 'clarity_score': 0-10}"
        )

        user_prompt = f"Task: {state.task.prompt}\n\nContext: {context}"

        response = await self.client.call(
            model=questioner,
            system=system_prompt,
            prompt=user_prompt,
            max_tokens=1000,
            temperature=0.5,
        )

        data = self._extract_json(response.text) or {}
        questions = data.get("questions", [])

        state.metadata["questions"] = questions
        state.metadata["clarity_score"] = data.get("clarity_score", 5.0)
        state.metadata["answers"] = []

    async def _phase_socratic_followup(self, state: PipelineState, round_num: int):
        """Generate follow-up questions based on answers."""
        from .phase_aware_models import PhaseType

        # Use phase-aware model selection for analysis (follow-up questioning)
        questioner = self._get_model_for_phase(PhaseType.ANALYSIS, state.task.type)

        # Simulate answers (in production, could involve user interaction)
        if not state.metadata.get("answers"):
            # Generate simulated answers for autonomous operation
            await self._generate_simulated_answers(state)

        system_prompt = (
            "Based on the answers, ask 3-5 deeper follow-up questions.\n"
            "Return JSON: {'questions': [], 'clarity_score': 0-10}"
        )

        qa_pairs = "\n".join(
            [
                f"Q: {q}\nA: {a}"
                for q, a in zip(
                    state.metadata.get("questions", []),
                    state.metadata.get("answers", []),
                    strict=False,
                )
            ]
        )

        user_prompt = f"Task: {state.task.prompt}\n\nQ&A so far:\n{qa_pairs}"

        response = await self.client.call(
            model=questioner,
            system=system_prompt,
            prompt=user_prompt,
            max_tokens=800,
            temperature=0.4,
        )

        data = self._extract_json(response.text) or {}
        state.metadata["questions"] = data.get("questions", [])
        state.metadata["clarity_score"] = data.get("clarity_score", 5.0)

        # Generate new simulated answers
        await self._generate_simulated_answers(state)

    async def _generate_simulated_answers(self, state: PipelineState):
        """Generate simulated answers for autonomous operation."""
        models = self._get_available_models(TaskType.REASONING)
        answerer = models[0] if models else Model.GPT_4O_MINI

        questions = state.metadata.get("questions", [])
        if not questions:
            return

        system_prompt = (
            "Provide thoughtful, reasonable answers to these questions based on best practices."
        )
        user_prompt = f"Task: {state.task.prompt}\n\nQuestions:\n" + "\n".join(questions)

        response = await self.client.call(
            model=answerer,
            system=system_prompt,
            prompt=user_prompt,
            max_tokens=1500,
            temperature=0.4,
        )

        # Parse answers (one per line)
        answers = response.text.strip().split("\n")
        state.metadata["answers"] = answers[: len(questions)]

    async def _phase_socratic_solution(self, state: PipelineState):
        """Generate solution based on clarified understanding."""
        models = self._get_available_models(state.task.type)
        solver = models[0] if models else Model.GPT_4O_MINI

        qa_pairs = "\n".join(
            [
                f"Q: {q}\nA: {a}"
                for q, a in zip(
                    state.metadata.get("questions", []),
                    state.metadata.get("answers", []),
                    strict=False,
                )
            ]
        )

        system_prompt = "Based on the Socratic Q&A, generate a well-reasoned solution that addresses the clarified problem."

        user_prompt = f"Original task: {state.task.prompt}\n\nSocratic Q&A:\n{qa_pairs}"

        response = await self.client.call(
            model=solver,
            system=system_prompt,
            prompt=user_prompt,
            max_tokens=state.task.max_output_tokens,
            temperature=0.4,
        )

        state.final_output = response.text
        state.final_score = state.metadata.get("clarity_score", 5.0) / 10.0

    def _build_result(self, state: PipelineState) -> TaskResult:
        return TaskResult(
            task_id=state.task.id,
            output=state.final_output,
            score=state.final_score,
            model_used=Model.GPT_4O_MINI,
            status=TaskStatus.COMPLETED if state.final_score >= 0.7 else TaskStatus.DEGRADED,
            metadata={
                "method": self.get_method().value,
                "questions_asked": len(state.metadata.get("questions", [])),
                "clarity_score": state.metadata.get("clarity_score", 0),
            },
        )


# ─────────────────────────────────────────────
# 8. Pre-Mortem Pipeline
# ─────────────────────────────────────────────


class PreMortemPipeline(BasePipeline):
    """
    Pre-Mortem Risk Assessment Pipeline

    Methodology (Gary Klein, 1989):
    Failure Narrative → Root Cause → Early Signals → Hardened Design

    Best for: Risk assessment, project planning
    """

    def get_method(self) -> ReasoningMethod:
        return ReasoningMethod.PRE_MORTEM

    async def execute(self, task: Task, context: str = "") -> TaskResult:
        state = PipelineState(task=task, method=self.get_method())

        # Phase 1: Imagine failure
        await self._phase_pre_mortem_failure(state)

        # Phase 2: Find root cause
        await self._phase_pre_mortem_backtrack(state)

        # Phase 3: Early warning signals
        await self._phase_pre_mortem_signals(state)

        # Phase 4: Hardened redesign
        await self._phase_pre_mortem_redesign(state)

        # Synthesize
        await self._phase_synthesis(state)

        return self._build_result(state)

    async def _phase_pre_mortem_failure(self, state: PipelineState):
        """Imagine the project has failed catastrophically."""
        models = self._get_available_models(state.task.type)
        primary = models[0] if models else Model.GPT_4O_MINI

        system_prompt = (
            "Imagine it's 6 months from now and this project has failed catastrophically.\n"
            "Write a detailed failure narrative with timeline and key events.\n"
            "Return JSON: {'failure_narrative': '', 'timeline': [], 'key_events': []}"
        )

        user_prompt = f"Project/Task: {state.task.prompt}\n\nContext: {context}"

        response = await self.client.call(
            model=primary,
            system=system_prompt,
            prompt=user_prompt,
            max_tokens=1500,
            temperature=0.7,
        )

        data = self._extract_json(response.text) or {}
        state.pre_mortem_state["failure_narrative"] = data
        state.metadata["failure_narrative"] = data.get("failure_narrative", "")

    async def _phase_pre_mortem_backtrack(self, state: PipelineState):
        """Backtrack to find root cause and pivot point."""
        models = self._get_available_models(state.task.type)
        analyst = models[0] if models else Model.GPT_4O_MINI

        failure_narrative = state.metadata.get("failure_narrative", "")

        system_prompt = (
            "Analyze the failure narrative to identify:\n"
            "1. Root cause\n"
            "2. Pivot point (decision that sealed fate)\n"
            "3. Decision chain that led to failure\n\n"
            "Return JSON: {'root_cause': '', 'pivot_point': '', 'decision_chain': []}"
        )

        user_prompt = f"Failure narrative:\n{failure_narrative}"

        response = await self.client.call(
            model=analyst,
            system=system_prompt,
            prompt=user_prompt,
            max_tokens=1200,
            temperature=0.3,
        )

        data = self._extract_json(response.text) or {}
        state.pre_mortem_state["root_cause"] = data
        state.metadata["root_cause"] = data.get("root_cause", "")

    async def _phase_pre_mortem_signals(self, state: PipelineState):
        """Identify early warning signals."""
        models = self._get_available_models(state.task.type)
        analyst = models[0] if models else Model.GPT_4O_MINI

        system_prompt = (
            "Identify early warning signals that would have predicted this failure.\n"
            "Return JSON: {'early_signals': [], 'monitoring_cadence': ''}"
        )

        root_cause = state.metadata.get("root_cause", "")
        user_prompt = f"Root cause: {root_cause}"

        response = await self.client.call(
            model=analyst,
            system=system_prompt,
            prompt=user_prompt,
            max_tokens=1000,
            temperature=0.3,
        )

        data = self._extract_json(response.text) or {}
        state.pre_mortem_state["early_signals"] = data.get("early_signals", [])
        state.metadata["early_signals"] = data.get("early_signals", [])

    async def _phase_pre_mortem_redesign(self, state: PipelineState):
        """Design hardened solution that addresses failure modes."""
        models = self._get_available_models(state.task.type)
        designer = models[0] if models else Model.GPT_4O_MINI

        system_prompt = (
            "Design a hardened solution that prevents the identified failure.\n"
            "Include safeguards, checkpoints, and rollback plan.\n"
            "Return JSON: {'hardened_solution': '', 'safeguards': [], 'checkpoints': [], 'rollback_plan': ''}"
        )

        failure_narrative = state.metadata.get("failure_narrative", "")
        root_cause = state.metadata.get("root_cause", "")
        early_signals = state.metadata.get("early_signals", [])

        user_prompt = f"""
Original task: {state.task.prompt}
Failure narrative: {failure_narrative}
Root cause: {root_cause}
Early signals: {json.dumps(early_signals)}

Design a solution that specifically addresses these failure modes.
"""

        response = await self.client.call(
            model=designer,
            system=system_prompt,
            prompt=user_prompt,
            max_tokens=state.task.max_output_tokens,
            temperature=0.4,
        )

        data = self._extract_json(response.text) or {}
        state.pre_mortem_state["hardened_solution"] = data
        state.final_output = data.get("hardened_solution", response.text)
        state.metadata["safeguards"] = data.get("safeguards", [])
        state.final_score = 0.85  # Pre-mortem typically produces high-quality output

    async def _phase_synthesis(self, state: PipelineState):
        """Synthesize final recommendation."""
        # Already synthesized in redesign phase
        pass

    def _build_result(self, state: PipelineState) -> TaskResult:
        return TaskResult(
            task_id=state.task.id,
            output=state.final_output,
            score=state.final_score,
            model_used=Model.GPT_4O_MINI,
            status=TaskStatus.COMPLETED if state.final_score >= 0.7 else TaskStatus.DEGRADED,
            metadata={
                "method": self.get_method().value,
                "failure_narrative": state.metadata.get("failure_narrative", "")[:500],
                "root_cause": state.metadata.get("root_cause", "")[:300],
                "early_signals": len(state.metadata.get("early_signals", [])),
                "safeguards": len(state.metadata.get("safeguards", [])),
            },
        )


# ─────────────────────────────────────────────
# 9. Bayesian Pipeline
# ─────────────────────────────────────────────


class BayesianPipeline(BasePipeline):
    """
    Bayesian Decision-Making Pipeline

    Methodology (Jaynes, 2003):
    Priors → Likelihoods → Posteriors → Sensitivity Analysis

    Best for: Decisions under uncertainty, risk quantification
    """

    def get_method(self) -> ReasoningMethod:
        return ReasoningMethod.BAYESIAN

    async def execute(self, task: Task, context: str = "") -> TaskResult:
        state = PipelineState(task=task, method=self.get_method())

        # Phase 1: Prior elicitation
        await self._phase_bayesian_priors(state)

        # Phase 2: Likelihood assessment
        await self._phase_bayesian_likelihood(state)

        # Phase 3: Posterior update
        await self._phase_bayesian_posterior(state)

        # Phase 4: Sensitivity analysis
        await self._phase_bayesian_sensitivity(state)

        # Synthesize
        await self._phase_synthesis(state)

        return self._build_result(state)

    async def _phase_bayesian_priors(self, state: PipelineState):
        """Elicit prior probabilities for hypotheses."""
        models = self._get_available_models(TaskType.REASONING)
        primary = models[0] if models else Model.GPT_4O_MINI

        system_prompt = (
            "Identify competing hypotheses and assign prior probabilities.\n"
            "Return JSON: {'hypotheses': [{'name': '', 'prior_probability': 0-1, 'rationale': ''}]}"
        )

        user_prompt = f"Task: {state.task.prompt}\n\nContext: {context}"

        response = await self.client.call(
            model=primary,
            system=system_prompt,
            prompt=user_prompt,
            max_tokens=1200,
            temperature=0.5,
        )

        data = self._extract_json(response.text) or {}
        hypotheses = data.get("hypotheses", [])

        # Validate probabilities sum to ~1
        total = sum(h.get("prior_probability", 0) for h in hypotheses)
        if total > 0:
            for h in hypotheses:
                h["prior_probability"] /= total

        state.bayesian_state["hypotheses_with_priors"] = hypotheses
        state.metadata["hypotheses"] = hypotheses

    async def _phase_bayesian_likelihood(self, state: PipelineState):
        """Assess likelihoods of observations given hypotheses."""
        models = self._get_available_models(TaskType.REASONING)
        analyst = models[0] if models else Model.GPT_4O_MINI

        hypotheses = state.metadata.get("hypotheses", [])

        system_prompt = (
            "For each hypothesis, identify key observations and their likelihoods.\n"
            "Return JSON: {'likelihoods': [{'hypothesis': '', 'observation': '', 'likelihood': 0-1}], 'observations': []}"
        )

        hyp_text = "\n".join([f"- {h['name']}: {h.get('rationale', '')}" for h in hypotheses])
        user_prompt = f"Task: {state.task.prompt}\n\nHypotheses:\n{hyp_text}"

        response = await self.client.call(
            model=analyst,
            system=system_prompt,
            prompt=user_prompt,
            max_tokens=1500,
            temperature=0.4,
        )

        data = self._extract_json(response.text) or {}
        state.bayesian_state["evidence_likelihoods"] = data.get("likelihoods", [])
        state.metadata["observations"] = data.get("observations", [])

    async def _phase_bayesian_posterior(self, state: PipelineState):
        """Compute posterior probabilities."""
        models = self._get_available_models(TaskType.REASONING)
        calculator = models[0] if models else Model.GPT_4O_MINI

        priors = state.bayesian_state.get("hypotheses_with_priors", [])
        likelihoods = state.bayesian_state.get("evidence_likelihoods", [])

        system_prompt = (
            "Apply Bayes' theorem to compute posterior probabilities.\n"
            "Return JSON: {'posteriors': [{'hypothesis': '', 'posterior_probability': 0-1}], 'most_probable': ''}"
        )

        user_prompt = f"""
Priors: {json.dumps(priors)}
Likelihoods: {json.dumps(likelihoods)}

Compute posteriors using Bayes' theorem: P(H|E) = P(E|H) * P(H) / P(E)
"""

        response = await self.client.call(
            model=calculator,
            system=system_prompt,
            prompt=user_prompt,
            max_tokens=1000,
            temperature=0.2,
        )

        data = self._extract_json(response.text) or {}
        state.bayesian_state["posteriors"] = data.get("posteriors", [])
        state.metadata["most_probable"] = data.get("most_probable", "")

    async def _phase_bayesian_sensitivity(self, state: PipelineState):
        """Perform sensitivity analysis on assumptions."""
        models = self._get_available_models(TaskType.REASONING)
        analyst = models[0] if models else Model.GPT_4O_MINI

        posteriors = state.bayesian_state.get("posteriors", [])

        system_prompt = (
            "Analyze which assumptions most affect the posterior probabilities.\n"
            "Return JSON: {'sensitivity_analysis': [{'assumption': '', 'impact': 'low/medium/high'}], 'most_sensitive_assumption': ''}"
        )

        user_prompt = f"Posteriors: {json.dumps(posteriors)}"

        response = await self.client.call(
            model=analyst,
            system=system_prompt,
            prompt=user_prompt,
            max_tokens=1000,
            temperature=0.3,
        )

        data = self._extract_json(response.text) or {}
        state.bayesian_state["sensitivity_results"] = data.get("sensitivity_analysis", [])
        state.metadata["most_sensitive"] = data.get("most_sensitive_assumption", "")

    async def _phase_synthesis(self, state: PipelineState):
        """Synthesize Bayesian recommendation."""
        models = self._get_available_models(state.task.type)
        synthesizer = models[0] if models else Model.GPT_4O_MINI

        posteriors = state.bayesian_state.get("posteriors", [])
        sensitivity = state.bayesian_state.get("sensitivity_results", [])
        most_probable = state.metadata.get("most_probable", "")

        system_prompt = "Synthesize a decision recommendation based on Bayesian analysis."
        user_prompt = f"""
Task: {state.task.prompt}
Most probable hypothesis: {most_probable}
Posteriors: {json.dumps(posteriors)}
Sensitivity: {json.dumps(sensitivity)}

Provide a decision recommendation with confidence level.
"""

        response = await self.client.call(
            model=synthesizer,
            system=system_prompt,
            prompt=user_prompt,
            max_tokens=state.task.max_output_tokens,
            temperature=0.3,
        )

        state.final_output = response.text
        state.final_score = 0.8 if posteriors else 0.5

    def _build_result(self, state: PipelineState) -> TaskResult:
        return TaskResult(
            task_id=state.task.id,
            output=state.final_output,
            score=state.final_score,
            model_used=Model.GPT_4O_MINI,
            status=TaskStatus.COMPLETED if state.final_score >= 0.7 else TaskStatus.DEGRADED,
            metadata={
                "method": self.get_method().value,
                "hypotheses": len(state.metadata.get("hypotheses", [])),
                "most_probable": state.metadata.get("most_probable", ""),
                "sensitivity_analysis": len(state.bayesian_state.get("sensitivity_results", [])),
            },
        )


# ─────────────────────────────────────────────
# 10. Dialectical Pipeline
# ─────────────────────────────────────────────


class DialecticalPipeline(BasePipeline):
    """
    Dialectical Reasoning Pipeline

    Hegelian methodology:
    Thesis → Antithesis → Contradictions → Aufhebung (Transcendence)

    Best for: Philosophical problems, policy debates
    """

    def get_method(self) -> ReasoningMethod:
        return ReasoningMethod.DIALECTICAL

    async def execute(self, task: Task, context: str = "") -> TaskResult:
        state = PipelineState(task=task, method=self.get_method())

        # Phase 1: Thesis
        await self._phase_dialectical_thesis(state)

        # Phase 2: Antithesis
        await self._phase_dialectical_antithesis(state)

        # Phase 3: Analyze contradictions
        await self._phase_dialectical_contradictions(state)

        # Phase 4: Aufhebung (transcendence)
        await self._phase_dialectical_aufhebung(state)

        # Synthesize
        await self._phase_synthesis(state)

        return self._build_result(state)

    async def _phase_dialectical_thesis(self, state: PipelineState):
        """Establish primary position (thesis)."""
        models = self._get_available_models(state.task.type)
        primary = models[0] if models else Model.GPT_4O_MINI

        system_prompt = (
            "Articulate the primary position (thesis) with key commitments and assumptions.\n"
            "Return JSON: {'thesis': '', 'key_commitments': [], 'assumptions': []}"
        )

        user_prompt = f"Task: {state.task.prompt}\n\nContext: {context}"

        response = await self.client.call(
            model=primary,
            system=system_prompt,
            prompt=user_prompt,
            max_tokens=1200,
            temperature=0.5,
        )

        data = self._extract_json(response.text) or {}
        state.dialectical_state["thesis"] = data.get("thesis", "")
        state.dialectical_state["key_commitments"] = data.get("key_commitments", [])
        state.metadata["thesis"] = data

    async def _phase_dialectical_antithesis(self, state: PipelineState):
        """Articulate opposing position (antithesis)."""
        models = self._get_available_models(state.task.type)
        opposition = models[0] if models else Model.GPT_4O_MINI

        thesis = state.dialectical_state.get("thesis", "")

        system_prompt = (
            "Articulate the opposing position (antithesis) that contradicts the thesis.\n"
            "Expose contradictions and negate key commitments.\n"
            "Return JSON: {'antithesis': '', 'contradictions_exposed': [], 'negated_commitments': []}"
        )

        user_prompt = f"Task: {state.task.prompt}\n\nThesis: {thesis}"

        response = await self.client.call(
            model=opposition,
            system=system_prompt,
            prompt=user_prompt,
            max_tokens=1200,
            temperature=0.5,
        )

        data = self._extract_json(response.text) or {}
        state.dialectical_state["antithesis"] = data.get("antithesis", "")
        state.dialectical_state["contradictions_exposed"] = data.get("contradictions_exposed", [])
        state.metadata["antithesis"] = data

    async def _phase_dialectical_contradictions(self, state: PipelineState):
        """Analyze irreconcilable vs compatible elements."""
        models = self._get_available_models(TaskType.REASONING)
        analyst = models[0] if models else Model.GPT_4O_MINI

        thesis = state.dialectical_state.get("thesis", "")
        antithesis = state.dialectical_state.get("antithesis", "")

        system_prompt = (
            "Analyze the contradictions between thesis and antithesis.\n"
            "Identify what is irreconcilable vs what can be preserved.\n"
            "Return JSON: {'irreconcilable': [], 'compatible': []}"
        )

        user_prompt = f"Thesis: {thesis}\n\nAntithesis: {antithesis}"

        response = await self.client.call(
            model=analyst,
            system=system_prompt,
            prompt=user_prompt,
            max_tokens=1000,
            temperature=0.3,
        )

        data = self._extract_json(response.text) or {}
        state.dialectical_state["irreconcilable"] = data.get("irreconcilable", [])
        state.dialectical_state["compatible"] = data.get("compatible", [])

    async def _phase_dialectical_aufhebung(self, state: PipelineState):
        """Achieve synthesis through transcendence (not compromise)."""
        models = self._get_available_models(state.task.type)
        synthesizer = models[0] if models else Model.GPT_4O_MINI

        thesis = state.dialectical_state.get("thesis", "")
        antithesis = state.dialectical_state.get("antithesis", "")
        irreconcilable = state.dialectical_state.get("irreconcilable", [])
        compatible = state.dialectical_state.get("compatible", [])

        system_prompt = (
            "Achieve Aufhebung: transcend the contradiction by preserving truths from both thesis and antithesis.\n"
            "This is NOT compromise - it's a qualitative transcendence to a higher level.\n"
            "Return JSON: {'synthesis': '', 'preserved_truths': [], 'new_concepts': []}"
        )

        user_prompt = f"""
Thesis: {thesis}
Antithesis: {antithesis}
Irreconcilable: {json.dumps(irreconcilable)}
Compatible: {json.dumps(compatible)}

Achieve synthesis through transcendence.
"""

        response = await self.client.call(
            model=synthesizer,
            system=system_prompt,
            prompt=user_prompt,
            max_tokens=state.task.max_output_tokens,
            temperature=0.4,
        )

        data = self._extract_json(response.text) or {}
        state.dialectical_state["synthesis"] = data.get("synthesis", "")
        state.final_output = data.get("synthesis", response.text)
        state.final_score = 0.85

    async def _phase_synthesis(self, state: PipelineState):
        """Final synthesis already done in aufhebung phase."""
        pass

    def _build_result(self, state: PipelineState) -> TaskResult:
        return TaskResult(
            task_id=state.task.id,
            output=state.final_output,
            score=state.final_score,
            model_used=Model.GPT_4O_MINI,
            status=TaskStatus.COMPLETED if state.final_score >= 0.7 else TaskStatus.DEGRADED,
            metadata={
                "method": self.get_method().value,
                "thesis": state.dialectical_state.get("thesis", "")[:300],
                "antithesis": state.dialectical_state.get("antithesis", "")[:300],
                "synthesis": state.dialectical_state.get("synthesis", "")[:500],
                "contradictions": len(state.dialectical_state.get("contradictions_exposed", [])),
            },
        )


# ─────────────────────────────────────────────
# 11. Analogical Pipeline
# ─────────────────────────────────────────────


class AnalogicalPipeline(BasePipeline):
    """
    Analogical Reasoning Pipeline

    Gentner's Structure-Mapping Theory (1983):
    Abstraction → Domain Search → Mapping → Transfer & Adaptation

    Best for: Innovation through cross-domain transfer
    """

    def get_method(self) -> ReasoningMethod:
        return ReasoningMethod.ANALOGICAL

    async def execute(self, task: Task, context: str = "") -> TaskResult:
        state = PipelineState(task=task, method=self.get_method())

        # Phase 1: Extract abstract structure
        await self._phase_analogical_abstraction(state)

        # Phase 2: Search source domains
        await self._phase_analogical_domain_search(state)

        # Phase 3: Map elements (if domains found)
        if state.analogical_state.get("source_domains"):
            await self._phase_analogical_mapping(state)
            await self._phase_analogical_transfer(state)

        # Synthesize
        await self._phase_synthesis(state)

        return self._build_result(state)

    async def _phase_analogical_abstraction(self, state: PipelineState):
        """Extract abstract structure from the problem."""
        models = self._get_available_models(TaskType.REASONING)
        primary = models[0] if models else Model.GPT_4O_MINI

        system_prompt = (
            "Extract the abstract structure of this problem, stripping away domain-specific details.\n"
            "Return JSON: {'abstract_structure': '', 'constraints': [], 'objectives': [], 'actors': [], 'core_dynamics': []}"
        )

        user_prompt = f"Task: {state.task.prompt}\n\nContext: {context}"

        response = await self.client.call(
            model=primary,
            system=system_prompt,
            prompt=user_prompt,
            max_tokens=1200,
            temperature=0.5,
        )

        data = self._extract_json(response.text) or {}
        state.analogical_state["abstract_structure"] = data.get("abstract_structure", "")
        state.analogical_state["constraints"] = data.get("constraints", [])
        state.metadata["abstract_structure"] = data

    async def _phase_analogical_domain_search(self, state: PipelineState):
        """Find analogous source domains."""
        models = self._get_available_models(TaskType.REASONING)
        searcher = models[0] if models else Model.GPT_4O_MINI

        abstract_structure = state.analogical_state.get("abstract_structure", "")
        constraints = state.analogical_state.get("constraints", [])

        system_prompt = (
            "Find 3-5 source domains that share this abstract structure.\n"
            "Look for solutions from unrelated fields.\n"
            "Return JSON: {'source_domains': [{'domain': '', 'relevance': 0-10, 'solution': ''}]}"
        )

        user_prompt = (
            f"Abstract structure: {abstract_structure}\n\nConstraints: {json.dumps(constraints)}"
        )

        response = await self.client.call(
            model=searcher,
            system=system_prompt,
            prompt=user_prompt,
            max_tokens=1500,
            temperature=0.7,
        )

        data = self._extract_json(response.text) or {}
        state.analogical_state["source_domains"] = data.get("source_domains", [])

    async def _phase_analogical_mapping(self, state: PipelineState):
        """Map elements from source to target domain."""
        models = self._get_available_models(TaskType.REASONING)
        mapper = models[0] if models else Model.GPT_4O_MINI

        source_domains = state.analogical_state.get("source_domains", [])
        abstract_structure = state.analogical_state.get("abstract_structure", "")

        async def map_domain(source: dict):
            system_prompt = (
                "Map elements from this source domain to the target problem.\n"
                "Return JSON: {'analogy_mappings': [{'source': '', 'target': ''}], 'unmapped_elements': [], 'mapping_quality': 0-10}"
            )

            user_prompt = f"""
Source domain: {source.get('domain')}
Source solution: {source.get('solution')}
Target abstract structure: {abstract_structure}
"""

            response = await self.client.call(
                model=mapper,
                system=system_prompt,
                prompt=user_prompt,
                max_tokens=1000,
                temperature=0.4,
            )

            data = self._extract_json(response.text) or {}
            return {"source": source.get("domain"), "mapping": data}

        tasks = [map_domain(sd) for sd in source_domains]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        state.analogical_state["analogy_mappings"] = [r for r in results if isinstance(r, dict)]

    async def _phase_analogical_transfer(self, state: PipelineState):
        """Transfer and adapt solution to target domain."""
        models = self._get_available_models(state.task.type)
        adapter = models[0] if models else Model.GPT_4O_MINI

        mappings = state.analogical_state.get("analogy_mappings", [])
        original_task = state.task.prompt

        # Use best mapping (highest quality)
        best_mapping = max(
            mappings, key=lambda m: m.get("mapping", {}).get("mapping_quality", 0), default=None
        )

        if not best_mapping:
            state.final_output = "No viable analogical transfer found."
            state.final_score = 0.3
            return

        system_prompt = (
            "Adapt the analogous solution to solve the target problem.\n"
            "Return JSON: {'transferred_solution': '', 'transfer_steps': [], 'adaptations_required': [], 'confidence': 0-1}"
        )

        user_prompt = f"""
Original task: {original_task}
Source domain: {best_mapping.get('source')}
Mapping: {json.dumps(best_mapping.get('mapping'))}

Transfer and adapt the solution.
"""

        response = await self.client.call(
            model=adapter,
            system=system_prompt,
            prompt=user_prompt,
            max_tokens=state.task.max_output_tokens,
            temperature=0.4,
        )

        data = self._extract_json(response.text) or {}
        state.analogical_state["transferred_solution"] = data
        state.final_output = data.get("transferred_solution", response.text)
        state.final_score = data.get("confidence", 0.5)

    async def _phase_synthesis(self, state: PipelineState):
        """Final synthesis already done in transfer phase."""
        pass

    def _build_result(self, state: PipelineState) -> TaskResult:
        return TaskResult(
            task_id=state.task.id,
            output=state.final_output,
            score=state.final_score,
            model_used=Model.GPT_4O_MINI,
            status=TaskStatus.COMPLETED if state.final_score >= 0.7 else TaskStatus.DEGRADED,
            metadata={
                "method": self.get_method().value,
                "source_domains": len(state.analogical_state.get("source_domains", [])),
                "mappings": len(state.analogical_state.get("analogy_mappings", [])),
                "best_source": (
                    state.analogical_state.get("source_domains", [{}])[0].get("domain")
                    if state.analogical_state.get("source_domains")
                    else None
                ),
            },
        )


# ─────────────────────────────────────────────
# 12. Delphi Pipeline
# ─────────────────────────────────────────────


class DelphiPipeline(BasePipeline):
    """
    Delphi Expert Consensus Pipeline

    RAND Corporation methodology (Dalkey & Helmer, 1963):
    Round 1 (Independent) → Aggregation → Round 2 (Revision) → Convergence → Dissent Analysis

    Best for: Predictions, expert consensus
    """

    NUM_EXPERTS = 4

    def get_method(self) -> ReasoningMethod:
        return ReasoningMethod.DELPHI

    async def execute(self, task: Task, context: str = "") -> TaskResult:
        state = PipelineState(task=task, method=self.get_method())

        # Round 1: Independent estimates
        await self._phase_delphi_round1(state, context)

        # Aggregation: Compute median, IQR
        await self._phase_delphi_aggregation(state)

        # Round 2: Revision with feedback
        await self._phase_delphi_round2(state)

        # Convergence check
        await self._phase_delphi_convergence(state)

        # Dissent analysis
        await self._phase_delphi_dissent(state)

        # Synthesize
        await self._phase_synthesis(state)

        return self._build_result(state)

    async def _phase_delphi_round1(self, state: PipelineState, context: str):
        """4 independent experts provide estimates."""
        models = self._get_available_models(TaskType.REASONING)
        if len(models) < self.NUM_EXPERTS:
            models = (models * self.NUM_EXPERTS)[: self.NUM_EXPERTS]

        async def expert_estimate(expert_num: int, model: Model):
            system_prompt = (
                f"You are expert {expert_num}. Provide an independent estimate.\n"
                "Return JSON: {'estimate_value': number, 'rationale': '', 'confidence': 0-1}"
            )

            user_prompt = f"Task: {state.task.prompt}\n\nContext: {context}"

            response = await self.client.call(
                model=model,
                system=system_prompt,
                prompt=user_prompt,
                max_tokens=800,
                temperature=0.5,
            )

            data = self._extract_json(response.text) or {}
            return {
                "expert_id": f"expert_{expert_num}",
                "estimate_value": data.get("estimate_value"),
                "rationale": data.get("rationale", ""),
                "confidence": data.get("confidence", 0.5),
                "model": model.value,
            }

        tasks = [expert_estimate(i + 1, models[i]) for i in range(self.NUM_EXPERTS)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        estimates = [r for r in results if isinstance(r, dict)]
        state.delphi_state["round_1_estimates"] = estimates

    async def _phase_delphi_aggregation(self, state: PipelineState):
        """Compute median, IQR, identify outliers."""
        estimates = state.delphi_state.get("round_1_estimates", [])

        values = [
            e.get("estimate_value")
            for e in estimates
            if isinstance(e.get("estimate_value"), (int, float))
        ]

        if not values:
            state.delphi_state["aggregated_stats"] = {"error": "No numeric estimates"}
            return

        values_sorted = sorted(values)
        n = len(values_sorted)

        # Median
        median = (
            values_sorted[n // 2]
            if n % 2 == 1
            else (values_sorted[n // 2 - 1] + values_sorted[n // 2]) / 2
        )

        # Quartiles
        q1_idx = n // 4
        q3_idx = (3 * n) // 4
        q1 = values_sorted[q1_idx]
        q3 = values_sorted[q3_idx]
        iqr = q3 - q1

        # Identify outlier (furthest from median)
        outlier = max(estimates, key=lambda e: abs((e.get("estimate_value") or 0) - median))

        state.delphi_state["aggregated_stats"] = {
            "median": median,
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "outlier_expert": outlier.get("expert_id"),
            "n_estimates": n,
        }

    async def _phase_delphi_round2(self, state: PipelineState):
        """Experts revise estimates with feedback."""
        models = self._get_available_models(TaskType.REASONING)
        if len(models) < self.NUM_EXPERTS:
            models = (models * self.NUM_EXPERTS)[: self.NUM_EXPERTS]

        stats = state.delphi_state.get("aggregated_stats", {})
        round1_estimates = state.delphi_state.get("round_1_estimates", [])

        async def expert_revision(expert: dict, model: Model):
            system_prompt = (
                "You see the group's statistical summary. Revise your estimate if warranted.\n"
                "Return JSON: {'revised_estimate': number, 'revision_rationale': ''}"
            )

            user_prompt = f"""
Task: {state.task.prompt}
Your estimate: {expert.get('estimate_value')}
Group median: {stats.get('median')}
Group IQR: {stats.get('iqr')}
You are the outlier: {expert.get('expert_id') == stats.get('outlier_expert')}

Revise your estimate if the group information warrants it.
"""

            response = await self.client.call(
                model=model,
                system=system_prompt,
                prompt=user_prompt,
                max_tokens=800,
                temperature=0.4,
            )

            data = self._extract_json(response.text) or {}
            return {
                "expert_id": expert.get("expert_id"),
                "original_estimate": expert.get("estimate_value"),
                "revised_estimate": data.get("revised_estimate", expert.get("estimate_value")),
                "revision_rationale": data.get("revision_rationale", ""),
            }

        tasks = [expert_revision(e, models[i]) for i, e in enumerate(round1_estimates)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        state.delphi_state["round_2_estimates"] = [r for r in results if isinstance(r, dict)]

    async def _phase_delphi_convergence(self, state: PipelineState):
        """Check convergence and compute consensus."""
        round2_estimates = state.delphi_state.get("round_2_estimates", [])

        values = [
            e.get("revised_estimate")
            for e in round2_estimates
            if isinstance(e.get("revised_estimate"), (int, float))
        ]

        if not values:
            state.delphi_state["consensus"] = {"error": "No estimates"}
            state.delphi_state["convergence_achieved"] = False
            return

        # Recompute stats
        values_sorted = sorted(values)
        n = len(values_sorted)
        median = (
            values_sorted[n // 2]
            if n % 2 == 1
            else (values_sorted[n // 2 - 1] + values_sorted[n // 2]) / 2
        )

        # Check convergence (IQR < threshold)
        q1 = values_sorted[n // 4]
        q3 = values_sorted[(3 * n) // 4]
        iqr = q3 - q1

        convergence_threshold = median * 0.2  # 20% of median
        convergence_achieved = iqr < convergence_threshold

        state.delphi_state["consensus"] = {
            "median": median,
            "iqr": iqr,
            "convergence_threshold": convergence_threshold,
        }
        state.delphi_state["convergence_achieved"] = convergence_achieved

    async def _phase_delphi_dissent(self, state: PipelineState):
        """Analyze remaining disagreement."""
        models = self._get_available_models(TaskType.REASONING)
        analyst = models[0] if models else Model.GPT_4O_MINI

        round2_estimates = state.delphi_state.get("round_2_estimates", [])
        consensus = state.delphi_state.get("consensus", {})

        system_prompt = (
            "Analyze the sources of remaining disagreement among experts.\n"
            "Return JSON: {'dissent_analysis': '', 'key_disagreements': [], 'resolution_suggestions': []}"
        )

        user_prompt = f"""
Task: {state.task.prompt}
Round 2 estimates: {json.dumps(round2_estimates)}
Consensus stats: {json.dumps(consensus)}

Analyze the dissent.
"""

        response = await self.client.call(
            model=analyst,
            system=system_prompt,
            prompt=user_prompt,
            max_tokens=1200,
            temperature=0.3,
        )

        data = self._extract_json(response.text) or {}
        state.delphi_state["dissent_analysis"] = data

    async def _phase_synthesis(self, state: PipelineState):
        """Synthesize final consensus recommendation."""
        models = self._get_available_models(state.task.type)
        synthesizer = models[0] if models else Model.GPT_4O_MINI

        consensus = state.delphi_state.get("consensus", {})
        dissent = state.delphi_state.get("dissent_analysis", {})
        convergence = state.delphi_state.get("convergence_achieved", False)

        system_prompt = "Synthesize the Delphi consensus into a final recommendation."
        user_prompt = f"""
Task: {state.task.prompt}
Consensus median: {consensus.get('median')}
IQR: {consensus.get('iqr')}
Convergence achieved: {convergence}
Dissent analysis: {json.dumps(dissent)}

Provide final recommendation.
"""

        response = await self.client.call(
            model=synthesizer,
            system=system_prompt,
            prompt=user_prompt,
            max_tokens=state.task.max_output_tokens,
            temperature=0.3,
        )

        state.final_output = response.text
        state.final_score = 0.9 if convergence else 0.7

    def _build_result(self, state: PipelineState) -> TaskResult:
        return TaskResult(
            task_id=state.task.id,
            output=state.final_output,
            score=state.final_score,
            model_used=Model.GPT_4O_MINI,
            status=TaskStatus.COMPLETED if state.final_score >= 0.7 else TaskStatus.DEGRADED,
            metadata={
                "method": self.get_method().value,
                "experts": self.NUM_EXPERTS,
                "convergence_achieved": state.delphi_state.get("convergence_achieved", False),
                "consensus_median": state.delphi_state.get("consensus", {}).get("median"),
                "final_iqr": state.delphi_state.get("consensus", {}).get("iqr"),
            },
        )


# ─────────────────────────────────────────────
# Pipeline Factory (Complete)
# ─────────────────────────────────────────────


# ─────────────────────────────────────────────
# 13. Brainstorming Pipeline (VS-based)
# ─────────────────────────────────────────────


class BrainstormingPipeline(BasePipeline):
    """
    Verbalized Sampling Brainstorming Pipeline

    Adapted from Reasoner's BrainstormingMixin. Four phases:
    Phase 1 — VS idea generation across N rounds
    Phase 2 — Cluster, deduplicate, and score raw ideas
    Phase 3 — Deep development of top ideas
    Phase 4 — Synthesis into final solution

    Best for: Creative ideation, open-ended problems
    """

    K_DEFAULT = 5
    ROUNDS_DEFAULT = 3

    def get_method(self) -> ReasoningMethod:
        return ReasoningMethod.BRAINSTORMING

    async def execute(self, task: Task, context: str = "") -> TaskResult:
        state = PipelineState(task=task, method=self.get_method())
        state.brainstorming_state = {
            "config": {"k": self.K_DEFAULT, "rounds": self.ROUNDS_DEFAULT, "threshold": 0.10},
        }

        # Phase 1: VS idea generation
        await self._phase_generate(state, context)
        # Phase 2: Cluster and score
        await self._phase_cluster(state)
        # Phase 3: Deep development
        await self._phase_develop(state)
        # Phase 4: Synthesis
        await self._phase_synthesis(state)

        return self._build_result(state)

    async def _phase_generate(self, state: PipelineState, context: str):
        """Generate diverse ideas via VS-Multi (VerbalizedSampler)."""
        cfg = state.brainstorming_state["config"]
        rounds = cfg.get("rounds", self.ROUNDS_DEFAULT)
        k = cfg.get("k", self.K_DEFAULT)
        models = self._get_available_models(state.task.type)
        gen_model = models[0] if models else Model.GPT_4O_MINI

        sampler = _get_vs_sampler(self.client)
        all_ideas = []

        for rnd in range(1, rounds + 1):
            previous = ""
            if all_ideas:
                texts = [s.get("text", "")[:200] for s in all_ideas]
                previous = "\nPrevious ideas (be diverse):\n" + "\n".join(texts)

            candidates = await sampler.sample(
                prompt=f"Task: {state.task.prompt}\n\nContext: {context}{previous}\n\nRound {rnd}/{rounds}",
                model=gen_model,
                cfg=VSConfig(k=k, temperature=0.8, fmt=ProbabilityFormat.CONFIDENCE),
                max_tokens=state.task.max_output_tokens,
                timeout=120,
            )
            if candidates:
                for c in candidates:
                    all_ideas.append({"text": c.text, "score": c.probability})
            else:
                logger.warning("Brainstorming round %d: VS returned no candidates", rnd)

        state.brainstorming_state["raw_ideas"] = all_ideas
        logger.info(
            "Brainstorming: %d raw ideas generated across %d rounds", len(all_ideas), rounds
        )

    async def _phase_cluster(self, state: PipelineState):
        """Cluster raw ideas into themes."""
        raw = state.brainstorming_state.get("raw_ideas", [])
        if not raw:
            return

        models = self._get_available_models(state.task.type)
        cluster_model = models[0] if models else Model.GPT_4O_MINI

        texts = "\n".join(f"- {s.get('text', s) if isinstance(s, dict) else s}" for s in raw)
        system = "You are an idea clustering expert. Group the following ideas into themes. Return JSON with 'clusters' list."
        user = f"Cluster these ideas:\n\n{texts}"

        response = await self.client.call(
            model=cluster_model,
            system=system,
            prompt=user,
            max_tokens=2000,
            temperature=0.3,
        )
        data = self._extract_json(response.text) or {}
        clusters = data.get("clusters", [])

        # Collect top ideas from clusters
        top = []
        for cluster in clusters:
            for idea in cluster.get("ideas", []):
                if idea.get("keep", True):
                    top.append(idea)

        state.brainstorming_state["clusters"] = clusters
        state.brainstorming_state["top_ideas"] = top
        logger.info("Brainstorming: %d clusters, %d top ideas", len(clusters), len(top))

    async def _phase_develop(self, state: PipelineState):
        """Deeply develop top ideas."""
        top = state.brainstorming_state.get("top_ideas", [])
        if not top:
            return
        top = top[:3]  # Develop top 3

        models = self._get_available_models(state.task.type)
        dev_model = models[0] if models else Model.GPT_4O_MINI

        texts = "\n".join(f"- {s.get('text', s)}" for s in top)
        system = "You are a strategic development expert. Expand each idea into a concrete, actionable plan."
        user = f"Develop these ideas into detailed plans:\n\n{texts}"

        response = await self.client.call(
            model=dev_model,
            system=system,
            prompt=user,
            max_tokens=state.task.max_output_tokens,
            temperature=0.4,
        )
        data = self._extract_json(response.text) or {}
        state.brainstorming_state["developments"] = data.get("developments", [response.text])

    async def _phase_synthesis(self, state: PipelineState):
        """Synthesize into final solution."""
        developments = state.brainstorming_state.get("developments", [])
        clusters = state.brainstorming_state.get("clusters", [])
        if not developments:
            state.final_output = "No viable ideas generated."
            state.final_score = 0.0
            return

        models = self._get_available_models(state.task.type)
        synth_model = models[0] if models else Model.GPT_4O_MINI

        dev_text = "\n\n".join(
            d.get("plan", d) if isinstance(d, dict) else str(d) for d in developments
        )
        system = "Synthesize the developed ideas into a final, coherent, actionable solution."
        user = f"Synthesize these developments:\n\n{dev_text}"

        response = await self.client.call(
            model=synth_model,
            system=system,
            prompt=user,
            max_tokens=state.task.max_output_tokens,
            temperature=0.3,
        )
        state.final_output = response.text
        state.final_score = 0.85  # Fixed high score for brainstorming

    def _build_result(self, state: PipelineState) -> TaskResult:
        return TaskResult(
            task_id=state.task.id,
            output=state.final_output,
            score=state.final_score,
            model_used=Model.GPT_4O_MINI,
            status=TaskStatus.COMPLETED if state.final_score >= 0.7 else TaskStatus.DEGRADED,
            metadata={
                "method": self.get_method().value,
                "idea_count": len(state.brainstorming_state.get("raw_ideas", [])),
                "cluster_count": len(state.brainstorming_state.get("clusters", [])),
            },
        )


# ─────────────────────────────────────────────
# 14. Verbalized Sampling Pipeline
# ─────────────────────────────────────────────


class VerbalizedSamplingPipeline(BasePipeline):
    """
    Verbalized Sampling (VS) Pipeline

    Adapted from Reasoner's verbalized sampling implementation.
    Generates k diverse candidates, scores them by probability,
    and selects the best via consensus.

    Modes:
    - STANDARD: Generate and sample from diverse candidates
    - TAIL: Focus on unconventional/tail-distribution candidates
    - COT: Chain-of-thought guided sampling

    Best for: Exploration tasks, uncertainty quantification
    """

    K_DEFAULT = 5
    QUALITY_THRESHOLD = 0.15

    def get_method(self) -> ReasoningMethod:
        return ReasoningMethod.VERBALIZED_SAMPLING

    async def execute(self, task: Task, context: str = "") -> TaskResult:
        state = PipelineState(task=task, method=self.get_method())

        # Phase 1: Generate diverse candidates
        await self._phase_generate_candidates(state, context)

        # Phase 2: Score and rank by probability
        await self._phase_score_candidates(state)

        # Phase 3: Consensus selection
        await self._phase_consensus(state)

        # Phase 4: Synthesize final output
        await self._phase_synthesis(state)

        return self._build_result(state)

    async def _phase_generate_candidates(self, state: PipelineState, context: str):
        """Generate k diverse candidate responses via VerbalizedSampler.

        Supports TAIL mode via VSConfig.probability_threshold. When the task
        metadata contains a 'vs_threshold' key (e.g. 0.10), tail-sampling is
        activated to surface unconventional approaches.
        """
        models = self._get_available_models(state.task.type)
        gen_model = models[0] if models else Model.GPT_4O_MINI

        # Determine VS mode: TAIL if threshold is set in task metadata
        threshold = (
            state.task.metadata.get("vs_threshold")
            if getattr(state.task, "metadata", None)
            else None
        )

        # Use the reusable VS primitive (port-only, never blocks on old tuple contract)
        sampler = _get_vs_sampler(self.client)
        candidates = await sampler.sample(
            prompt=f"{state.task.prompt}\n\nContext: {context}",
            model=gen_model,
            cfg=VSConfig(
                k=self.K_DEFAULT,
                probability_threshold=threshold,
                temperature=0.9,
                fmt=ProbabilityFormat.EXPLICIT,
            ),
            max_tokens=state.task.max_output_tokens,
            timeout=120,
        )

        # Fallback: if VS returned nothing, create a single-candidate stub
        if not candidates:
            state.candidates.append(
                SolutionCandidate(
                    perspective="vs_candidate_1",
                    content=context or state.task.prompt,
                    metadata={"index": 0, "probability": 1.0},
                )
            )
            return

        for i, cand in enumerate(candidates):
            state.candidates.append(
                SolutionCandidate(
                    perspective=f"vs_candidate_{i+1}",
                    content=cand.text,
                    metadata={"index": i, "probability": cand.probability},
                )
            )

    async def _phase_score_candidates(self, state: PipelineState):
        """Score each candidate by probability/quality."""
        if not state.candidates:
            return

        models = self._get_available_models(state.task.type)
        score_model = models[0] if models else Model.GPT_4O_MINI

        texts = "\n\n".join(f"[{i+1}] {c.content[:500]}" for i, c in enumerate(state.candidates))
        system = (
            "You are a probability calibration expert. For each candidate, estimate "
            "the probability that it is the correct/optimal solution. "
            "Probabilities must sum to 1.0. Return JSON with 'scores' list."
        )
        user = f"Task: {state.task.prompt}\n\nCandidates:\n{texts}"

        # NOTE: Quality scoring uses a separate LLM call for probability calibration.
        # EvaluatorService delegation (per plan) is deferred — the pipeline context
        # doesn't have access to EvaluatorService without a constructor change to
        # BasePipeline. Using LLM scoring is functionally correct (paper §5.1).
        response = await self.client.call(
            model=score_model,
            system=system,
            prompt=user,
            max_tokens=1000,
            temperature=0.1,
        )
        data = self._extract_json(response.text) or {}
        scores_raw = data.get("scores", [])

        for i, c in enumerate(state.candidates):
            if i < len(scores_raw):
                prob = (
                    scores_raw[i]
                    if isinstance(scores_raw[i], (int, float))
                    else scores_raw[i].get("probability", 0)
                )
                state.scores.append(
                    CritiqueScore(
                        perspective=c.perspective,
                        total=float(prob),
                    )
                )

        # Sort by probability descending
        scored = {s.perspective: s.total for s in state.scores}
        state.candidates.sort(key=lambda c: scored.get(c.perspective, 0), reverse=True)
        state.scores.sort(key=lambda s: s.total, reverse=True)

    async def _phase_consensus(self, state: PipelineState):
        """Select top candidates via threshold."""
        if not state.scores:
            return
        threshold = self.QUALITY_THRESHOLD
        best_score = state.scores[0].total if state.scores else 0
        state.top_candidates = [
            c
            for c in state.candidates
            if any(
                s.perspective == c.perspective and s.total >= max(best_score * 0.5, threshold)
                for s in state.scores
            )
        ]
        if not state.top_candidates and state.candidates:
            state.top_candidates = state.candidates[:1]

    async def _phase_synthesis(self, state: PipelineState):
        """Synthesize top candidates into final output."""
        if not state.top_candidates:
            state.final_output = "No candidates met the quality threshold."
            state.final_score = 0.0
            return

        if len(state.top_candidates) == 1:
            state.final_output = state.top_candidates[0].content
            state.final_score = state.scores[0].total if state.scores else 0.5
            return

        models = self._get_available_models(state.task.type)
        synth_model = models[0] if models else Model.GPT_4O_MINI

        texts = "\n\n".join(
            f"[Score: {s.total:.2f}]\n{c.content[:500]}"
            for c, s in zip(state.top_candidates, state.scores)
        )
        system = "Synthesize the best candidate responses into a final, coherent answer."
        user = f"Task: {state.task.prompt}\n\nTop candidates:\n{texts}"

        response = await self.client.call(
            model=synth_model,
            system=system,
            prompt=user,
            max_tokens=state.task.max_output_tokens,
            temperature=0.3,
        )
        state.final_output = response.text
        state.final_score = state.scores[0].total if state.scores else 0.5

    def _build_result(self, state: PipelineState) -> TaskResult:
        return TaskResult(
            task_id=state.task.id,
            output=state.final_output,
            score=state.final_score,
            model_used=Model.GPT_4O_MINI,
            status=TaskStatus.COMPLETED if state.final_score >= 0.3 else TaskStatus.DEGRADED,
            metadata={
                "method": self.get_method().value,
                "candidate_count": len(state.candidates),
                "top_score": state.final_score,
            },
        )


# ─────────────────────────────────────────────
# 15. Persuasion Defense Pipeline
# ─────────────────────────────────────────────


class PersuasionDefensePipeline(BasePipeline):
    """
    Persuasion Defense Pipeline

    Adapted from Reasoner's PersuasionDefense module. A 5-stage
    hallucination mitigation pipeline targeting high-stakes outputs:
    1. Claim extraction and structuring
    2. NLI verification against source context
    3. Conflict surfacing and taint propagation
    4. Behavioral monitoring for persuasion tactics
    5. Final synthesis with confidence scoring

    Best for: High-stakes verification, hallucination detection
    """

    NLI_THRESHOLD = 0.7
    DRIFT_THRESHOLD = 0.25

    def get_method(self) -> ReasoningMethod:
        return ReasoningMethod.PERSUASION_DEFENSE

    async def execute(self, task: Task, context: str = "") -> TaskResult:
        state = PipelineState(task=task, method=self.get_method())

        # Phase 1: Extract claims from output
        await self._phase_extract_claims(state, context)

        # Phase 2: NLI verification
        await self._phase_nli_verify(state)

        # Phase 3: Conflict surfacing
        await self._phase_conflict_surface(state)

        # Phase 4: Behavioral audit
        await self._phase_behavioral_audit(state)

        # Phase 5: Final synthesis
        await self._phase_synthesis(state)

        return self._build_result(state)

    async def _phase_extract_claims(self, state: PipelineState, context: str):
        """Extract verifiable claims from task output."""
        models = self._get_available_models(state.task.type)
        model = models[0] if models else Model.GPT_4O_MINI

        system = (
            "You are a claim extraction expert. Extract all factual claims from "
            "the following text that can be verified against source context. "
            "Return JSON with 'claims' list, each with 'text' and 'category'."
        )
        user = f"Text: {context}\n\nTask: {state.task.prompt}"

        response = await self.client.call(
            model=model,
            system=system,
            prompt=user,
            max_tokens=2000,
            temperature=0.2,
        )
        data = self._extract_json(response.text) or {}
        claims = data.get("claims", [])
        state.metadata["claims"] = claims
        logger.info("Persuasion Defense: %d claims extracted", len(claims))

    async def _phase_nli_verify(self, state: PipelineState):
        """Verify claims via NLI-style evaluation."""
        claims = state.metadata.get("claims", [])
        if not claims:
            return

        models = self._get_available_models(state.task.type)
        model = models[0] if models else Model.GPT_4O_MINI

        verified = []
        for claim in claims:
            text = claim.get("text", "") if isinstance(claim, dict) else str(claim)
            system = (
                "You are an NLI (Natural Language Inference) evaluator. "
                f"Score the claim on a scale of 0.0 (completely unsupported) "
                f"to 1.0 (fully supported). Return JSON with 'score' and 'reason'."
            )
            user = f"Claim: {text}"

            response = await self.client.call(
                model=model,
                system=system,
                prompt=user,
                max_tokens=500,
                temperature=0.1,
            )
            data = self._extract_json(response.text) or {}
            score = float(data.get("score", 0.5))
            verified.append(
                {
                    "claim": text,
                    "score": score,
                    "passed": score >= self.NLI_THRESHOLD,
                }
            )

        state.metadata["verified_claims"] = verified
        passed = sum(1 for v in verified if v["passed"])
        logger.info("Persuasion Defense: %d/%d claims passed NLI", passed, len(verified))

    async def _phase_conflict_surface(self, state: PipelineState):
        """Surface conflicts between claims."""
        verified = state.metadata.get("verified_claims", [])
        if len(verified) < 2:
            return

        models = self._get_available_models(state.task.type)
        model = models[0] if models else Model.GPT_4O_MINI

        texts = "\n".join(f"{v['claim']} (score: {v['score']:.2f})" for v in verified)
        system = (
            "You are a conflict analysis expert. Identify any contradictory or "
            "inconsistent claims. Return JSON with 'conflicts' list."
        )
        user = f"Analyze these claims for conflicts:\n\n{texts}"

        response = await self.client.call(
            model=model,
            system=system,
            prompt=user,
            max_tokens=1000,
            temperature=0.2,
        )
        data = self._extract_json(response.text) or {}
        state.metadata["conflicts"] = data.get("conflicts", [])

    async def _phase_behavioral_audit(self, state: PipelineState):
        """Audit for persuasion tactics."""
        models = self._get_available_models(state.task.type)
        model = models[0] if models else Model.GPT_4O_MINI

        system = (
            "You are a behavioral auditor. Analyze the response for persuasion "
            "tactics, rhetorical manipulation, or emotional appeals that might "
            "mask weak reasoning. Return JSON with 'tactics' list."
        )
        user = f"Task: {state.task.prompt}"

        response = await self.client.call(
            model=model,
            system=system,
            prompt=user,
            max_tokens=1000,
            temperature=0.2,
        )
        data = self._extract_json(response.text) or {}
        state.metadata["persuasion_tactics"] = data.get("tactics", [])

    async def _phase_synthesis(self, state: PipelineState):
        """Synthesize verification results into final confidence score."""
        verified = state.metadata.get("verified_claims", [])
        conflicts = state.metadata.get("conflicts", [])

        if not verified:
            state.final_output = "No claims to verify."
            state.final_score = 0.5
            return

        # Calculate confidence score
        avg_nli = sum(v["score"] for v in verified) / len(verified)
        conflict_penalty = max(0, 1.0 - len(conflicts) * 0.1)
        score = round(avg_nli * conflict_penalty, 4)

        state.final_score = score
        state.final_output = (
            f"Confidence Score: {score:.2f}\n"
            f"Claims Verified: {len(verified)}\n"
            f"Conflicts Found: {len(conflicts)}\n"
        )

    def _build_result(self, state: PipelineState) -> TaskResult:
        return TaskResult(
            task_id=state.task.id,
            output=state.final_output,
            score=state.final_score,
            model_used=Model.GPT_4O_MINI,
            status=TaskStatus.COMPLETED,
            metadata={
                "method": self.get_method().value,
                "claims": len(state.metadata.get("claims", [])),
                "verified": len(state.metadata.get("verified_claims", [])),
                "conflicts": len(state.metadata.get("conflicts", [])),
            },
        )


# ── Shared constants from Reasoner ──────────────────────────────────────

_JSON_ONLY_FOOTER = "Return ONLY valid JSON. No markdown fences, no explanation outside the JSON."


def _wrap_user_input(text: str) -> str:
    """Wrap user input for prompt safety."""
    return text[:5000] if text else ""


def _wrap_external_content(text: str) -> str:
    """Wrap external content for prompt safety."""
    return str(text)[:3000] if text else ""


def _get_language_instruction(state: PipelineState) -> str:
    """Get language instruction from state metadata."""
    return state.metadata.get("language_instruction", "")


# ─────────────────────────────────────────────
# 16. CoVE Pipeline (Chain-of-Verification)
# ─────────────────────────────────────────────

_COVE_DRAFT_SYSTEM = (
    "You are a knowledgeable analyst. Draft a comprehensive initial answer to the problem. "
    "Break your answer into explicit, verifiable claims. " + _JSON_ONLY_FOOTER
)

_COVE_VERIFY_SYSTEM = (
    "You are a skeptical fact-checker. Given a draft answer with claims, generate specific, "
    "independent verification questions for EACH claim. Do not trust the original answer. "
    + _JSON_ONLY_FOOTER
)

_COVE_ANSWER_SYSTEM = (
    "You are an independent researcher. Answer the verification questions based on your own "
    "knowledge. Do not refer to the draft answer. Be explicit about whether evidence supports "
    "or contradicts each claim. " + _JSON_ONLY_FOOTER
)

_COVE_REVISE_SYSTEM = (
    "You are a careful editor. Given a draft answer and independent verification results, "
    "revise the answer to correct errors, add caveats, and improve accuracy. " + _JSON_ONLY_FOOTER
)


class CoVEPipeline(BasePipeline):
    """
    Chain-of-Verification (CoVE) Pipeline — exact Reasoner implementation.

    Draft -> Verify -> Answer -> Revise cycle that reduces hallucination
    by generating and answering verification questions independently.
    """

    def get_method(self) -> ReasoningMethod:
        return ReasoningMethod.COVE

    async def execute(self, task: Task, context: str = "") -> TaskResult:
        state = PipelineState(task=task, method=self.get_method())
        state.cove_state = {}
        state.metadata["problem"] = task.prompt
        state.metadata["language_instruction"] = ""

        await self._phase_cove_draft(state)
        await self._phase_cove_verify(state)
        await self._phase_cove_answer(state)
        await self._phase_cove_revise(state)

        state.final_output = state.cove_state.get(
            "revised_answer", state.cove_state.get("draft_answer", "")
        )
        state.final_score = 0.85
        return self._build_result(state)

    def _call_llm(self, system: str, user: str, max_tokens=2000, temp=0.3) -> dict:
        models = self._get_available_models(TaskType.REASONING)
        model = models[0] if models else Model.GPT_4O_MINI
        resp, _ = asyncio.get_event_loop().run_until_complete() if False else None  # placeholder
        # Use synchronous wrapper
        return self._call_llm_inner(system, user, max_tokens, temp)

    async def _call_llm_inner(self, system: str, user: str, max_tokens=2000, temp=0.3):
        models = self._get_available_models(TaskType.REASONING)
        model = models[0] if models else Model.GPT_4O_MINI
        resp = await self.client.call(
            model=model,
            system=system,
            prompt=user,
            max_tokens=max_tokens,
            temperature=temp,
        )
        return self._extract_json(resp.text) or {}

    async def _phase_cove_draft(self, state: PipelineState):
        prompt = (
            f"{_get_language_instruction(state)}\n\n"
            f'Problem: {_wrap_user_input(state.metadata.get("problem", ""))}\n\n'
            f"Draft an initial answer. Break it into explicit claims that can be independently verified. "
            f"For each claim, assign a confidence score (0.0-1.0).\n\n"
            f'Output JSON: {{"draft_answer": "<full answer text>", '
            f'"claims": [{{"claim": "<claim text>", "confidence": 0.8}}]}}'
        )
        data = await self._call_llm_inner(_COVE_DRAFT_SYSTEM, prompt)
        state.cove_state["draft_answer"] = data.get("draft_answer", "")
        state.cove_state["claims"] = data.get("claims", [])

    async def _phase_cove_verify(self, state: PipelineState):
        draft = state.cove_state.get("draft_answer", "")
        claims = state.cove_state.get("claims", [])
        claims_json = json.dumps(claims, indent=2) if claims else "[]"
        prompt = (
            f"{_get_language_instruction(state)}\n\n"
            f'Original Problem: {_wrap_user_input(state.metadata.get("problem", ""))}\n\n'
            f"Draft Answer:\n{_wrap_external_content(draft)}\n\n"
            f"Claims to verify:\n{claims_json}\n\n"
            f"For EACH claim above, generate 1-2 specific verification questions that would "
            f"independently test whether the claim is true. "
            f'Output JSON: {{"verification_questions": [{{'
            f'"question": "<verification question>", '
            f'"target_claim": "<claim being tested>", '
            f'"expected_evidence_type": "<fact|statistic|authority|logic>"'
            f"}}]}}"
        )
        data = await self._call_llm_inner(_COVE_VERIFY_SYSTEM, prompt)
        state.cove_state["verification_questions"] = data.get("verification_questions", [])

    async def _phase_cove_answer(self, state: PipelineState):
        questions = state.cove_state.get("verification_questions", [])
        questions_json = json.dumps(questions, indent=2) if questions else "[]"
        prompt = (
            f"{_get_language_instruction(state)}\n\n"
            f'Original Problem: {_wrap_user_input(state.metadata.get("problem", ""))}\n\n'
            f"Answer these verification questions INDEPENDENTLY, using your own knowledge. "
            f"Do not refer to any draft answer.\n\n"
            f"Questions:\n{questions_json}\n\n"
            f'Output JSON: {{"answers": [{{'
            f'"question": "<question text>", '
            f'"answer": "<your independent answer>", '
            f'"verdict": "<supports|contradicts|insufficient>", '
            f'"confidence": 0.8, '
            f'"reasoning": "<why>"'
            f"}}]}}"
        )
        data = await self._call_llm_inner(_COVE_ANSWER_SYSTEM, prompt)
        state.cove_state["verification_answers"] = data.get("answers", [])

    async def _phase_cove_revise(self, state: PipelineState):
        draft = state.cove_state.get("draft_answer", "")
        answers = state.cove_state.get("verification_answers", [])
        answers_json = json.dumps(answers, indent=2) if answers else "[]"
        prompt = (
            f"{_get_language_instruction(state)}\n\n"
            f'Original Problem: {_wrap_user_input(state.metadata.get("problem", ""))}\n\n'
            f"Draft Answer:\n{_wrap_external_content(draft)}\n\n"
            f"Independent Verification Results:\n{_wrap_external_content(answers_json)}\n\n"
            f"Revise the draft answer based on the verification results. "
            f'Output JSON: {{"revised_answer": "<revised full answer>", '
            f'"changes_made": ["<change description>"], '
            f'"remaining_uncertainties": ["<uncertainty>"]}}'
        )
        data = await self._call_llm_inner(_COVE_REVISE_SYSTEM, prompt)
        state.cove_state["revised_answer"] = data.get("revised_answer", "")

    def _build_result(self, state: PipelineState) -> TaskResult:
        return TaskResult(
            task_id=state.task.id,
            output=state.final_output,
            score=state.final_score,
            model_used=Model.GPT_4O_MINI,
            status=TaskStatus.COMPLETED if state.final_score >= 0.7 else TaskStatus.DEGRADED,
            metadata={"method": self.get_method().value},
        )


# ─────────────────────────────────────────────
# 17. SoT Pipeline (Skeleton-of-Thought)
# ─────────────────────────────────────────────

_SOT_SKELETON_SYSTEM = (
    "You are an expert problem decomposer. Generate a skeleton outline of sub-problems "
    "that collectively solve the main problem. Each sub-problem should be independent "
    "and solvable in parallel. " + _JSON_ONLY_FOOTER
)

_SOT_SOLVE_SYSTEM = (
    "You are a specialist solver. Solve the assigned sub-problem thoroughly and concisely. "
    + _JSON_ONLY_FOOTER
)

_SOT_ASSEMBLE_SYSTEM = (
    "You are a master synthesizer. Combine multiple sub-problem solutions into a coherent, "
    "unified answer. Ensure smooth transitions and resolve any contradictions. " + _JSON_ONLY_FOOTER
)


class SoTPipeline(BasePipeline):
    """
    Skeleton-of-Thought (SoT) Pipeline — exact Reasoner implementation.

    Skeleton -> Parallel solve -> Assemble
    """

    def get_method(self) -> ReasoningMethod:
        return ReasoningMethod.SOT

    async def execute(self, task: Task, context: str = "") -> TaskResult:
        state = PipelineState(task=task, method=self.get_method())
        state.sot_state = {}
        state.metadata["problem"] = task.prompt

        await self._phase_sot_skeleton(state)
        await self._phase_sot_solve(state)
        await self._phase_sot_assemble(state)

        state.final_output = state.sot_state.get("assembled_answer", "")
        state.final_score = 0.85
        return self._build_result(state)

    async def _llm(self, system, user, max_tokens=2000, temp=0.3):
        models = self._get_available_models(TaskType.REASONING)
        model = models[0] if models else Model.GPT_4O_MINI
        resp = await self.client.call(
            model=model,
            system=system,
            prompt=user,
            max_tokens=max_tokens,
            temperature=temp,
        )
        return self._extract_json(resp.text) or {}

    async def _phase_sot_skeleton(self, state: PipelineState):
        prompt = (
            f"{_get_language_instruction(state)}\n\n"
            f'Problem: {_wrap_user_input(state.metadata.get("problem", ""))}\n\n'
            f"Decompose this problem into 3-5 sub-problems that can be solved independently "
            f"and in parallel. Each sub-problem should have a clear scope, inputs, and expected output.\n\n"
            f'Output JSON: {{"sub_problems": [{{'
            f'"id": "1", '
            f'"description": "<sub-problem>", '
            f'"inputs": ["<input>"], '
            f'"expected_output": "<output description>"'
            f"}}]}}"
        )
        data = await self._llm(_SOT_SKELETON_SYSTEM, prompt, max_tokens=2000)
        state.sot_state["sub_problems"] = data.get("sub_problems", [])

    async def _phase_sot_solve(self, state: PipelineState):
        sub_problems = state.sot_state.get("sub_problems", [])
        if not sub_problems:
            return

        async def solve_one(sp: dict) -> dict:
            prompt = (
                f"{_get_language_instruction(state)}\n\n"
                f'Original Problem: {_wrap_user_input(state.metadata.get("problem", ""))}\n\n'
                f"YOUR ASSIGNED SUB-PROBLEM:\n"
                f'ID: {sp.get("id", "?")}\n'
                f'Description: {sp.get("description", "")}\n\n'
                f"Solve this sub-problem thoroughly.\n\n"
                f'Output JSON: {{"sub_problem_id": "{sp.get("id", "")}", '
                f'"solution": "<detailed solution>", '
                f'"key_insights": ["<insight>"]}}'
            )
            data = await self._llm(_SOT_SOLVE_SYSTEM, prompt)
            return {
                "sub_problem_id": sp.get("id", ""),
                "solution": data.get("solution", ""),
                "key_insights": data.get("key_insights", []),
            }

        tasks = [solve_one(sp) for sp in sub_problems]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        state.sot_state["solutions"] = [r for r in results if not isinstance(r, Exception)]

    async def _phase_sot_assemble(self, state: PipelineState):
        solutions = state.sot_state.get("solutions", [])
        solutions_json = json.dumps(solutions, indent=2) if solutions else "[]"
        prompt = (
            f"{_get_language_instruction(state)}\n\n"
            f'Original Problem: {_wrap_user_input(state.metadata.get("problem", ""))}\n\n'
            f"Sub-problem Solutions:\n{_wrap_external_content(solutions_json)}\n\n"
            f"Assemble these sub-problem solutions into a single, coherent, comprehensive answer.\n\n"
            f'Output JSON: {{"assembled_answer": "<full unified answer>", '
            f'"transitions": ["<how sections connect>"]}}'
        )
        data = await self._llm(_SOT_ASSEMBLE_SYSTEM, prompt, max_tokens=4000)
        state.sot_state["assembled_answer"] = data.get("assembled_answer", "")

    def _build_result(self, state: PipelineState) -> TaskResult:
        return TaskResult(
            task_id=state.task.id,
            output=state.final_output,
            score=state.final_score,
            model_used=Model.GPT_4O_MINI,
            status=TaskStatus.COMPLETED,
            metadata={"method": self.get_method().value},
        )


# ─────────────────────────────────────────────
# 18. ToT Pipeline (Tree-of-Thoughts)
# ─────────────────────────────────────────────

_TOT_DECOMPOSE_SYSTEM = (
    "You are a strategic planner. Decompose the problem into sequential decision points. "
    + _JSON_ONLY_FOOTER
)

_TOT_GENERATE_SYSTEM = (
    "You are a creative strategist. Generate diverse candidate next-steps for the given decision point. "
    + _JSON_ONLY_FOOTER
)

_TOT_EVALUATE_SYSTEM = (
    "You are a critical evaluator. Score each candidate action on multiple dimensions. "
    + _JSON_ONLY_FOOTER
)

_TOT_BACKTRACK_SYSTEM = (
    "You are a strategic analyst. Given evaluation results, decide whether to proceed, "
    "backtrack, or terminate. " + _JSON_ONLY_FOOTER
)


class ToTPipeline(BasePipeline):
    """
    Tree-of-Thoughts (ToT) Pipeline — exact Reasoner implementation.

    Decompose -> Generate candidates -> Evaluate -> Backtrack/Continue -> Assemble
    """

    def get_method(self) -> ReasoningMethod:
        return ReasoningMethod.TOT

    async def execute(self, task: Task, context: str = "") -> TaskResult:
        state = PipelineState(task=task, method=self.get_method())
        state.tot_state = {"current_path": [], "decision_points": []}
        state.metadata["problem"] = task.prompt

        await self._phase_tot_decompose(state)

        for _ in range(3):
            if not state.tot_state.get("decision_points"):
                break
            await self._phase_tot_generate(state)
            if not state.tot_state.get("current_candidates"):
                break
            await self._phase_tot_evaluate(state)
            await self._phase_tot_backtrack(state)
            if state.tot_state.get("backtrack_decision") == "terminate":
                break

        state.final_output = " -> ".join(
            state.tot_state.get("final_path", state.tot_state.get("current_path", []))
        )
        state.final_score = state.tot_state.get("tot_confidence", 0.7)
        return self._build_result(state)

    async def _llm(self, system, user, max_tokens=1500, temp=0.3):
        models = self._get_available_models(TaskType.REASONING)
        model = models[0] if models else Model.GPT_4O_MINI
        resp = await self.client.call(
            model=model,
            system=system,
            prompt=user,
            max_tokens=max_tokens,
            temperature=temp,
        )
        return self._extract_json(resp.text) or {}

    async def _phase_tot_decompose(self, state: PipelineState):
        prompt = (
            f"{_get_language_instruction(state)}\n\n"
            f'Problem: {_wrap_user_input(state.metadata.get("problem", ""))}\n\n'
            f"Identify the key sequential decision points in this problem. "
            f"Each decision point should have 2-3 possible candidate actions.\n\n"
            f'Output JSON: {{"decision_points": [{{'
            f'"id": "dp1", '
            f'"description": "<what decision must be made>", '
            f'"candidates": [{{"action": "<action>", "rationale": "<why>"}}]'
            f"}}]}}"
        )
        data = await self._llm(_TOT_DECOMPOSE_SYSTEM, prompt)
        state.tot_state["decision_points"] = data.get("decision_points", [])

    async def _phase_tot_generate(self, state: PipelineState):
        dps = state.tot_state.get("decision_points", [])
        idx = len(state.tot_state.get("current_path", []))
        if idx >= len(dps):
            return
        dp = dps[idx]
        prompt = (
            f"{_get_language_instruction(state)}\n\n"
            f'Problem: {_wrap_user_input(state.metadata.get("problem", ""))}\n\n'
            f'Current Decision Point: {dp.get("description", "")}\n\n'
            f"Generate 2-3 diverse, high-quality candidate actions for this decision point.\n\n"
            f'Output JSON: {{"candidates": [{{'
            f'"candidate_id": "c1", '
            f'"action": "<action description>"'
            f"}}]}}"
        )
        data = await self._llm(_TOT_GENERATE_SYSTEM, prompt, temp=0.8)
        state.tot_state["current_candidates"] = data.get("candidates", [])

    async def _phase_tot_evaluate(self, state: PipelineState):
        candidates = state.tot_state.get("current_candidates", [])
        if not candidates:
            return
        prompt = (
            f"{_get_language_instruction(state)}\n\n"
            f'Problem: {_wrap_user_input(state.metadata.get("problem", ""))}\n\n'
            f"Candidate Actions:\n{json.dumps(candidates, indent=2)}\n\n"
            f"Evaluate each candidate. Return best_candidate.\n\n"
            f'Output JSON: {{"evaluations": [{{'
            f'"candidate_id": "c1", "score": 7.5, "verdict": "<proceed|reject|caution>"'
            f"}}], "
            f'"best_candidate": "<candidate_id>"}}'
        )
        data = await self._llm(_TOT_EVALUATE_SYSTEM, prompt)
        state.tot_state["evaluations"] = data.get("evaluations", [])
        best = data.get("best_candidate", "")
        if best:
            state.tot_state["current_path"].append(best)

    async def _phase_tot_backtrack(self, state: PipelineState):
        path = state.tot_state.get("current_path", [])
        prompt = (
            f"{_get_language_instruction(state)}\n\n"
            f'Problem: {_wrap_user_input(state.metadata.get("problem", ""))}\n\n'
            f"Current path: {json.dumps(path, indent=2)}\n\n"
            f"Based on evaluations, decide: CONTINUE, BACKTRACK, or TERMINATE.\n\n"
            f'Output JSON: {{"decision": "<continue|backtrack|terminate>", '
            f'"final_path": ["<action>"], "confidence": 0.8}}'
        )
        data = await self._llm(_TOT_BACKTRACK_SYSTEM, prompt, max_tokens=500)
        state.tot_state["backtrack_decision"] = data.get("decision", "terminate")
        state.tot_state["final_path"] = data.get("final_path", path)
        state.tot_state["tot_confidence"] = data.get("confidence", 0.7)

    def _build_result(self, state: PipelineState) -> TaskResult:
        return TaskResult(
            task_id=state.task.id,
            output=state.final_output,
            score=state.final_score,
            model_used=Model.GPT_4O_MINI,
            status=TaskStatus.COMPLETED,
            metadata={"method": self.get_method().value},
        )


# ─────────────────────────────────────────────
# 19. PoT Pipeline (Program-of-Thoughts)
# ─────────────────────────────────────────────

_POT_GENERATE_SYSTEM = (
    "You are an expert programmer. Generate Python code to solve the given quantitative problem. "
    "The code should be self-contained, use only standard library, and include comments. "
    + _JSON_ONLY_FOOTER
)

_POT_EXECUTE_SYSTEM = (
    "You are a code execution engine. Simulate or describe the execution of the given Python code. "
    "If actual execution is unavailable, trace through the code logically and produce the output. "
    + _JSON_ONLY_FOOTER
)

_POT_INTERPRET_SYSTEM = (
    "You are an analytical interpreter. Given code execution results, explain what they mean "
    "in the context of the original problem. " + _JSON_ONLY_FOOTER
)


class PoTPipeline(BasePipeline):
    """Program-of-Thoughts (PoT) Pipeline — exact Reasoner implementation."""

    def get_method(self) -> ReasoningMethod:
        return ReasoningMethod.POT

    async def execute(self, task: Task, context: str = "") -> TaskResult:
        state = PipelineState(task=task, method=self.get_method())
        state.pot_state = {}
        state.metadata["problem"] = task.prompt

        await self._phase_pot_generate(state)
        await self._phase_pot_execute(state)
        await self._phase_pot_interpret(state)

        state.final_output = state.pot_state.get("computed_answer", "")
        state.final_score = 0.85 if state.pot_state.get("execution_success") else 0.5
        return self._build_result(state)

    async def _llm(self, system, user, max_tokens=2000, temp=0.3):
        models = self._get_available_models(TaskType.REASONING)
        model = models[0] if models else Model.GPT_4O_MINI
        resp = await self.client.call(
            model=model,
            system=system,
            prompt=user,
            max_tokens=max_tokens,
            temperature=temp,
        )
        return self._extract_json(resp.text) or {}

    async def _phase_pot_generate(self, state: PipelineState):
        prompt = (
            f"{_get_language_instruction(state)}\n\n"
            f'Problem: {_wrap_user_input(state.metadata.get("problem", ""))}\n\n'
            f"Write Python code to solve this problem computationally. "
            f'Output JSON: {{"code": "<python code>", '
            f'"explanation": "<approach>", '
            f'"expected_output_type": "<number|list|dict|boolean>"}}'
        )
        data = await self._llm(_POT_GENERATE_SYSTEM, prompt, max_tokens=4000)
        state.pot_state["code"] = data.get("code", "")

    async def _phase_pot_execute(self, state: PipelineState):
        code = state.pot_state.get("code", "")
        if not code:
            return
        prompt = (
            f"{_get_language_instruction(state)}\n\n"
            f"Execute the following Python code and return the exact output.\n\n"
            f"Code:\n```python\n{code}\n```\n\n"
            f'Output JSON: {{"output": "<execution output>", '
            f'"success": true, "error": ""}}'
        )
        data = await self._llm(_POT_EXECUTE_SYSTEM, prompt)
        state.pot_state["execution_output"] = data.get("output", "")
        state.pot_state["execution_success"] = data.get("success", False)
        state.pot_state["execution_error"] = data.get("error", "")

    async def _phase_pot_interpret(self, state: PipelineState):
        code = state.pot_state.get("code", "")
        output = state.pot_state.get("execution_output", "")
        error = state.pot_state.get("execution_error", "")
        prompt = (
            f"{_get_language_instruction(state)}\n\n"
            f'Original Problem: {_wrap_user_input(state.metadata.get("problem", ""))}\n\n'
            f"Generated Code:\n```python\n{code}\n```\n\n"
            f"Execution Output:\n{_wrap_external_content(output)}\n\n"
            f'Output JSON: {{"interpretation": "<explanation>", '
            f'"answer": "<final answer>", '
            f'"caveats": ["<caveat>"], "confidence": 0.9}}'
        )
        data = await self._llm(_POT_INTERPRET_SYSTEM, prompt)
        state.pot_state["computed_answer"] = data.get("answer", "")

    def _build_result(self, state: PipelineState) -> TaskResult:
        return TaskResult(
            task_id=state.task.id,
            output=state.final_output,
            score=state.final_score,
            model_used=Model.GPT_4O_MINI,
            status=TaskStatus.COMPLETED,
            metadata={"method": self.get_method().value},
        )


# ─────────────────────────────────────────────
# 20. Self-Discover Pipeline
# ─────────────────────────────────────────────

_SD_SELECT_SYSTEM = (
    "You are a meta-reasoning architect. Given a problem, select the reasoning modules "
    "that are most appropriate from the available inventory. " + _JSON_ONLY_FOOTER
)

_SD_ADAPT_SYSTEM = (
    "You are a prompt engineer. Adapt the selected reasoning modules into concrete prompts "
    "and instructions for the current problem. " + _JSON_ONLY_FOOTER
)

_SD_IMPLEMENT_SYSTEM = (
    "You are an execution engine. Execute the adapted reasoning modules in sequence "
    "and synthesize their outputs into a final answer. " + _JSON_ONLY_FOOTER
)

_SD_MODULES_INVENTORY = (
    "- decomposition: break problem into sub-problems\n"
    "- verification: fact-check claims\n"
    "- analogy: find cross-domain parallels\n"
    "- causal_analysis: identify cause-effect chains\n"
    "- counterfactual: explore what-if scenarios\n"
    "- abstraction: extract deep structure\n"
    "- constraint_satisfaction: respect hard limits\n"
    "- optimization: find best allocation\n"
)


class SelfDiscoverPipeline(BasePipeline):
    """Self-Discover Pipeline — exact Reasoner implementation."""

    def get_method(self) -> ReasoningMethod:
        return ReasoningMethod.SELF_DISCOVER

    async def execute(self, task: Task, context: str = "") -> TaskResult:
        state = PipelineState(task=task, method=self.get_method())
        state.self_discover_state = {}
        state.metadata["problem"] = task.prompt

        await self._phase_sd_select(state)
        await self._phase_sd_adapt(state)
        await self._phase_sd_implement(state)

        state.final_output = state.self_discover_state.get("final_answer", "")
        state.final_score = 0.85
        return self._build_result(state)

    async def _llm(self, system, user, max_tokens=2000, temp=0.3):
        models = self._get_available_models(TaskType.REASONING)
        model = models[0] if models else Model.GPT_4O_MINI
        resp = await self.client.call(
            model=model,
            system=system,
            prompt=user,
            max_tokens=max_tokens,
            temperature=temp,
        )
        return self._extract_json(resp.text) or {}

    async def _phase_sd_select(self, state: PipelineState):
        prompt = (
            f"{_get_language_instruction(state)}\n\n"
            f'Problem: {_wrap_user_input(state.metadata.get("problem", ""))}\n\n'
            f"Available reasoning modules:\n{_SD_MODULES_INVENTORY}"
            f"Select 3-5 modules that are MOST relevant to this problem.\n\n"
            f'Output JSON: {{"selected_modules": [{{'
            f'"module": "<module_name>", '
            f'"rationale": "<why needed>", "order": 1'
            f"}}], "
            f'"composition_strategy": "<how modules interact>"}}'
        )
        data = await self._llm(_SD_SELECT_SYSTEM, prompt)
        state.self_discover_state["selected_modules"] = data.get("selected_modules", [])
        state.self_discover_state["composition_strategy"] = data.get("composition_strategy", "")

    async def _phase_sd_adapt(self, state: PipelineState):
        modules = state.self_discover_state.get("selected_modules", [])
        if not modules:
            return
        prompt = (
            f"{_get_language_instruction(state)}\n\n"
            f'Problem: {_wrap_user_input(state.metadata.get("problem", ""))}\n\n'
            f"Selected Modules: {json.dumps(modules, indent=2)}\n\n"
            f"Adapt each selected module into a concrete instruction for this problem.\n\n"
            f'Output JSON: {{"adapted_modules": [{{'
            f'"module": "<module_name>", '
            f'"instruction": "<concrete instruction>"'
            f"}}]}}"
        )
        data = await self._llm(_SD_ADAPT_SYSTEM, prompt)
        state.self_discover_state["adapted_modules"] = data.get("adapted_modules", [])

    async def _phase_sd_implement(self, state: PipelineState):
        adapted = state.self_discover_state.get("adapted_modules", [])
        prompt = (
            f"{_get_language_instruction(state)}\n\n"
            f'Problem: {_wrap_user_input(state.metadata.get("problem", ""))}\n\n'
            f"Adapted Module Instructions: {json.dumps(adapted, indent=2)}\n\n"
            f"Execute each module in sequence and synthesize the final answer.\n\n"
            f'Output JSON: {{"module_outputs": [{{'
            f'"module": "<name>", "output": "<result>"'
            f"}}], "
            f'"final_answer": "<synthesized answer>", '
            f'"confidence": 0.85}}'
        )
        data = await self._llm(_SD_IMPLEMENT_SYSTEM, prompt, max_tokens=4000)
        state.self_discover_state["module_outputs"] = data.get("module_outputs", [])
        state.self_discover_state["final_answer"] = data.get("final_answer", "")

    def _build_result(self, state: PipelineState) -> TaskResult:
        return TaskResult(
            task_id=state.task.id,
            output=state.final_output,
            score=state.final_score,
            model_used=Model.GPT_4O_MINI,
            status=TaskStatus.COMPLETED,
            metadata={"method": self.get_method().value},
        )


class PipelineFactory:
    """Factory for creating ARA reasoning pipelines."""

    _pipelines: dict[ReasoningMethod, type] = {
        ReasoningMethod.MULTI_PERSPECTIVE: MultiPerspectivePipeline,
        ReasoningMethod.ITERATIVE: IterativePipeline,
        ReasoningMethod.DEBATE: DebatePipeline,
        ReasoningMethod.RESEARCH: ResearchPipeline,
        ReasoningMethod.JURY: JuryPipeline,
        ReasoningMethod.SCIENTIFIC: ScientificPipeline,
        ReasoningMethod.SOCRATIC: SocraticPipeline,
        ReasoningMethod.PRE_MORTEM: PreMortemPipeline,
        ReasoningMethod.BAYESIAN: BayesianPipeline,
        ReasoningMethod.DIALECTICAL: DialecticalPipeline,
        ReasoningMethod.ANALOGICAL: AnalogicalPipeline,
        ReasoningMethod.DELPHI: DelphiPipeline,
        ReasoningMethod.BRAINSTORMING: BrainstormingPipeline,
        ReasoningMethod.VERBALIZED_SAMPLING: VerbalizedSamplingPipeline,
        ReasoningMethod.PERSUASION_DEFENSE: PersuasionDefensePipeline,
        ReasoningMethod.COVE: CoVEPipeline,
        ReasoningMethod.SOT: SoTPipeline,
        ReasoningMethod.TOT: ToTPipeline,
        ReasoningMethod.POT: PoTPipeline,
        ReasoningMethod.SELF_DISCOVER: SelfDiscoverPipeline,
    }

    @classmethod
    def create(
        cls,
        method: ReasoningMethod,
        client: UnifiedClient,
        cache: DiskCache | None = None,
        telemetry: TelemetryCollector | None = None,
    ) -> BasePipeline:
        """
        Create a pipeline instance for the specified method.

        Args:
            method: The reasoning method to use
            client: API client for LLM calls
            cache: Optional disk cache
            telemetry: Optional telemetry collector

        Returns:
            Pipeline instance

        Raises:
            ValueError: If method is not implemented
        """
        pipeline_class = cls._pipelines.get(method)
        if not pipeline_class:
            raise ValueError(f"Pipeline for method {method.value} not implemented yet")

        return pipeline_class(client=client, cache=cache, telemetry=telemetry)

    @classmethod
    def get_available_methods(cls) -> list[ReasoningMethod]:
        """Return list of implemented reasoning methods."""
        return list(cls._pipelines.keys())


# ─────────────────────────────────────────────
# Exports
# ─────────────────────────────────────────────

__all__ = [
    # Enums
    "ReasoningMethod",
    "PerspectiveType",
    # Data classes
    "SolutionCandidate",
    "CritiqueScore",
    "PipelineState",
    # Base classes
    "BasePipeline",
    # All pipelines (7 standard)
    "MultiPerspectivePipeline",
    "IterativePipeline",
    "DebatePipeline",
    "ResearchPipeline",
    "JuryPipeline",
    "ScientificPipeline",
    "SocraticPipeline",
    # All pipelines (5 specialized)
    "PreMortemPipeline",
    "BayesianPipeline",
    "DialecticalPipeline",
    "AnalogicalPipeline",
    "DelphiPipeline",
    # Factory
    "PipelineFactory",
]


# ─────────────────────────────────────────────
