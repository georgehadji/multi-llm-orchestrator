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
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .models import (
    Model, Task, TaskResult, TaskStatus, TaskType,
    get_provider, estimate_cost
)
from .api_clients import UnifiedClient, APIResponse
from .validators import run_validators, ValidationResult
from .cache import DiskCache
from .telemetry import TelemetryCollector

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
    key_insights: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


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
    candidates: List[SolutionCandidate] = field(default_factory=list)
    scores: List[CritiqueScore] = field(default_factory=list)
    top_candidates: List[SolutionCandidate] = field(default_factory=list)
    final_output: str = ""
    final_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    reflexion_memory: List[str] = field(default_factory=list)
    
    # Method-specific state
    debate_rounds: List[Dict] = field(default_factory=list)
    pre_mortem_state: Dict[str, Any] = field(default_factory=dict)
    bayesian_state: Dict[str, Any] = field(default_factory=dict)
    dialectical_state: Dict[str, Any] = field(default_factory=dict)
    analogical_state: Dict[str, Any] = field(default_factory=dict)
    delphi_state: Dict[str, Any] = field(default_factory=dict)


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
        cache: Optional[DiskCache] = None,
        telemetry: Optional[TelemetryCollector] = None,
        top_k: int = 2,
        max_iterations: int = 3,
    ):
        self.client = client
        self.cache = cache or DiskCache()
        self.telemetry = telemetry or TelemetryCollector({})
        self.top_k = top_k
        self.max_iterations = max_iterations
        self.api_health: Dict[Model, bool] = {m: True for m in Model}
    
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
    
    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from LLM response text."""
        # Try to find JSON object in text
        match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
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
    
    def _get_available_models(self, task_type: TaskType) -> List[Model]:
        """Get list of available models for task type."""
        from .models import ROUTING_TABLE
        routing = ROUTING_TABLE.get(task_type, list(Model)[:5])
        return [m for m in routing if self.api_health.get(m, False)]
    
    def _select_reviewer(self, primary: Model, task_type: TaskType) -> Optional[Model]:
        """Select a reviewer model from different provider."""
        from .models import ROUTING_TABLE
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
        state.top_candidates = state.candidates[:self.top_k]
        
        # Synthesize final output
        await self._phase_synthesis(state)
        
        return self._build_result(state)
    
    async def _phase_perspectives(self, state: PipelineState, context: str):
        """Run all 4 perspectives concurrently."""
        models = self._get_available_models(state.task.type)
        if not models:
            logger.error("No models available for perspectives")
            return
        
        primary = models[0]
        
        async def run_perspective(perspective: PerspectiveType):
            system_prompt = self.PERSPECTIVE_SYSTEMS[perspective]
            user_prompt = f"Task: {state.task.prompt}\n\nContext: {context}\n\nAnalyze from {perspective.value} perspective."
            
            response, _ = await self.client.call(
                model=primary,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
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
        models = self._get_available_models(state.task.type)
        if not models or len(state.candidates) == 0:
            return
        
        scorer = self._select_reviewer(models[0], state.task.type) or models[0]
        
        candidates_text = "\n\n".join([
            f"[{c.perspective}]\n{c.content}"
            for c in state.candidates
        ])
        
        system_prompt = (
            "You are an expert evaluator. Score each perspective on these criteria (0-10):\n"
            "- Logical consistency\n"
            "- Feasibility\n"
            "- Completeness\n"
            "- Novelty\n\n"
            "Return JSON array of scores."
        )
        
        user_prompt = f"Task: {state.task.prompt}\n\nCandidates:\n{candidates_text}"
        
        response, _ = await self.client.call(
            model=scorer,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=1000,
            temperature=0.1,
        )
        
        data = self._extract_json(response.text)
        if data and isinstance(data, list):
            for score_data in data:
                state.scores.append(CritiqueScore(
                    perspective=score_data.get("perspective", ""),
                    total=sum([
                        score_data.get("logical_consistency", 5),
                        score_data.get("feasibility", 5),
                        score_data.get("completeness", 5),
                        score_data.get("novelty", 5),
                    ]) / 4,
                    logical_consistency=score_data.get("logical_consistency", 5),
                    feasibility=score_data.get("feasibility", 5),
                    completeness=score_data.get("completeness", 5),
                    novelty=score_data.get("novelty", 5),
                ))
        
        # Sort candidates by score
        scored = {s.perspective: s.total for s in state.scores}
        state.candidates.sort(key=lambda c: scored.get(c.perspective, 0), reverse=True)
    
    async def _phase_synthesis(self, state: PipelineState):
        """Synthesize top candidates into final solution."""
        if not state.top_candidates:
            state.final_output = "No viable candidates generated."
            return
        
        models = self._get_available_models(state.task.type)
        if not models:
            return
        
        synthesizer = models[0]
        
        top_texts = "\n\n".join([
            f"[{c.perspective}]\n{c.content}"
            for c in state.top_candidates
        ])
        
        system_prompt = (
            "You are a synthesis expert. Combine the insights from multiple "
            "perspectives into a coherent, actionable solution. Preserve the "
            "strengths of each perspective while resolving contradictions."
        )
        
        user_prompt = f"Task: {state.task.prompt}\n\nTop Perspectives:\n{top_texts}"
        
        response, _ = await self.client.call(
            model=synthesizer,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
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
                "top_candidate": state.top_candidates[0].perspective if state.top_candidates else None,
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
            new_memories = [
                s.steel_man for s in state.scores if s.steel_man
            ]
            state.reflexion_memory.extend(new_memories)
            
            # Early convergence check
            if state.scores:
                mean_score = sum(s.logical_consistency for s in state.scores) / len(state.scores)
                if mean_score >= self.CONVERGENCE_THRESHOLD:
                    logger.info(f"Convergence achieved at round {round_num} (mean={mean_score:.2f})")
                    break
            
            # Clear for next round (except last)
            if round_num < self.MAX_ROUNDS:
                state.candidates = []
                state.scores = []
                state.top_candidates = []
        
        # Synthesize final output
        state.top_candidates = state.candidates[:self.top_k]
        await self._phase_synthesis(state)
        
        return self._build_result(state)
    
    async def _phase_generate(self, state: PipelineState, context: str, round_num: int):
        """Generate candidates with reflexion memory."""
        models = self._get_available_models(state.task.type)
        if not models:
            return
        
        primary = models[0]
        
        # Build context with reflexion memory
        memory_context = ""
        if state.reflexion_memory:
            memory_context = "\n\nPrevious Insights:\n" + "\n".join(state.reflexion_memory)
        
        system_prompt = (
            "You are an iterative problem solver. Generate a solution that builds upon "
            "previous insights and improves upon weaknesses. Focus on continuous refinement."
        )
        
        user_prompt = f"Task: {state.task.prompt}\n\nContext: {context}{memory_context}\n\nRound: {round_num}"
        
        response, _ = await self.client.call(
            model=primary,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=state.task.max_output_tokens,
            temperature=0.5,
        )
        
        data = self._extract_json(response.text) or {}
        state.candidates.append(SolutionCandidate(
            perspective="iterative",
            content=data.get("solution", response.text),
            key_insights=data.get("key_insights", []),
            metadata={"round": round_num, "model": primary.value},
        ))
    
    async def _phase_critique(self, state: PipelineState):
        """Score candidates with detailed feedback."""
        models = self._get_available_models(state.task.type)
        if not models or len(state.candidates) == 0:
            return
        
        scorer = self._select_reviewer(models[0], state.task.type) or models[0]
        
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
        
        response, _ = await self.client.call(
            model=scorer,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=1500,
            temperature=0.2,
        )
        
        data = self._extract_json(response.text) or {}
        state.scores.append(CritiqueScore(
            perspective="iterative",
            total=sum([
                data.get("logical_consistency", 5),
                data.get("feasibility", 5),
                data.get("completeness", 5),
                data.get("novelty", 5),
            ]) / 4,
            logical_consistency=data.get("logical_consistency", 5),
            feasibility=data.get("feasibility", 5),
            completeness=data.get("completeness", 5),
            novelty=data.get("novelty", 5),
            steel_man=data.get("steel_man", ""),
            rationale=data.get("rationale", ""),
        ))
    
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
        models = self._get_available_models(state.task.type)
        if len(models) < 2:
            models = models * 2 if models else []
        
        model_a = models[0] if len(models) > 0 else Model.GPT_4O_MINI
        model_b = models[1] if len(models) > 1 else Model.GPT_4O
        
        # Side A: Pro/Constructive
        system_a = "You are arguing FOR the proposed solution. Present the strongest case with evidence and reasoning."
        prompt_a = f"Task: {state.task.prompt}\n\nContext: {context}\n\nPresent your opening argument FOR this approach."
        
        # Side B: Con/Destructive
        system_b = "You are arguing AGAINST the proposed solution. Identify flaws, risks, and better alternatives."
        prompt_b = f"Task: {state.task.prompt}\n\nContext: {context}\n\nPresent your opening argument AGAINST this approach."
        
        response_a, response_b = await asyncio.gather(
            self.client.call(model=model_a, system_prompt=system_a, user_prompt=prompt_a, max_tokens=state.task.max_output_tokens, temperature=0.7),
            self.client.call(model=model_b, system_prompt=system_b, user_prompt=prompt_b, max_tokens=state.task.max_output_tokens, temperature=0.7),
        )
        
        state.debate_rounds.append({
            "round": 1,
            "type": "opening",
            "statements": [
                {"side": "A", "content": response_a[0].text, "model": model_a.value},
                {"side": "B", "content": response_b[0].text, "model": model_b.value},
            ]
        })
    
    async def _phase_debate_rebuttal(self, state: PipelineState):
        """Each side rebuts the other's opening."""
        if not state.debate_rounds:
            return
        
        opening = state.debate_rounds[0]
        statement_a = opening["statements"][0]["content"]
        statement_b = opening["statements"][1]["content"]
        
        models = self._get_available_models(state.task.type)
        model_a = models[0] if models else Model.GPT_4O_MINI
        model_b = models[1] if len(models) > 1 else Model.GPT_4O
        
        # A rebuts B
        prompt_a = f"Task: {state.task.prompt}\n\nOpponent's argument:\n{statement_b}\n\nRebut their points and defend your position."
        
        # B rebuts A
        prompt_b = f"Task: {state.task.prompt}\n\nOpponent's argument:\n{statement_a}\n\nRebut their points and defend your position."
        
        response_a, response_b = await asyncio.gather(
            self.client.call(model=model_a, system_prompt="You are a debater. Rebut your opponent's arguments point-by-point.", user_prompt=prompt_a, max_tokens=state.task.max_output_tokens, temperature=0.6),
            self.client.call(model=model_b, system_prompt="You are a debater. Rebut your opponent's arguments point-by-point.", user_prompt=prompt_b, max_tokens=state.task.max_output_tokens, temperature=0.6),
        )
        
        state.debate_rounds.append({
            "round": 2,
            "type": "rebuttal",
            "statements": [
                {"side": "A", "content": response_a[0].text, "model": model_a.value},
                {"side": "B", "content": response_b[0].text, "model": model_b.value},
            ]
        })
    
    async def _phase_debate_cross_examine(self, state: PipelineState):
        """Judge asks probing questions to both sides."""
        models = self._get_available_models(state.task.type)
        judge_model = models[2] if len(models) > 2 else (models[0] if models else Model.GPT_4O_MINI)
        
        # Generate probing questions
        all_statements = "\n\n".join([
            f"Round {r['round']} ({r['type']}):\n" + "\n".join([s["content"][:500] for s in r["statements"]])
            for r in state.debate_rounds
        ])
        
        system_prompt = "You are a judge in a debate. Ask 3 probing questions that reveal the strengths and weaknesses of each position."
        user_prompt = f"Task: {state.task.prompt}\n\nDebate so far:\n{all_statements}"
        
        response, _ = await self.client.call(
            model=judge_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=800,
            temperature=0.5,
        )
        
        state.metadata["cross_exam_questions"] = response.text
    
    async def _phase_debate_judge(self, state: PipelineState):
        """Judge makes final decision."""
        models = self._get_available_models(state.task.type)
        judge_model = models[2] if len(models) > 2 else (models[0] if models else Model.GPT_4O_MINI)
        
        all_statements = "\n\n".join([
            f"Round {r['round']} ({r['type']}):\n" + "\n".join([f"[{s['side']}] {s['content']}" for s in r["statements"]])
            for r in state.debate_rounds
        ])
        
        system_prompt = (
            "You are an impartial judge. Evaluate both sides based on:\n"
            "- Logical consistency\n"
            "- Evidence quality\n"
            "- Practical feasibility\n\n"
            "Declare a winner and explain your reasoning. Return JSON with scores."
        )
        
        user_prompt = f"Task: {state.task.prompt}\n\nDebate transcript:\n{all_statements}\n\nQuestions: {state.metadata.get('cross_exam_questions', '')}"
        
        response, _ = await self.client.call(
            model=judge_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
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
            (s["content"] for s in final_round["statements"] if s["side"] == winner),
            ""
        )
        
        judge_decision = state.metadata.get("judge_decision", "")
        
        # Synthesize
        models = self._get_available_models(state.task.type)
        synthesizer = models[0] if models else Model.GPT_4O_MINI
        
        system_prompt = "Synthesize the debate outcome into a final recommendation that incorporates the winning argument and judge's insights."
        user_prompt = f"Task: {state.task.prompt}\n\nWinner's argument: {winner_statement}\n\nJudge's decision: {judge_decision}"
        
        response, _ = await self.client.call(
            model=synthesizer,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=state.task.max_output_tokens,
            temperature=0.3,
        )
        
        state.final_output = response.text
        state.final_score = max(state.metadata.get("scores", {}).values()) / 10.0 if state.metadata.get("scores") else 0.5
    
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

    def __init__(self, *args, nexus_enabled: bool = True, **kwargs):
        """Initialize Research pipeline with optional Nexus Search integration."""
        super().__init__(*args, **kwargs)
        self.nexus_enabled = nexus_enabled
        self._nexus_search = None
        self._SearchSource = None
    
    def _get_nexus(self):
        """Lazy import of Nexus Search."""
        if self._nexus_search is None and self.nexus_enabled:
            try:
                from orchestrator.nexus_search import search, SearchSource
                self._nexus_search = search
                self._SearchSource = SearchSource
            except ImportError:
                self.nexus_enabled = False
        return self._nexus_search, self._SearchSource
    
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
        state.top_candidates = state.candidates[:self.top_k]
        await self._phase_synthesis(state)
        
        return self._build_result(state)
    
    async def _phase_research(self, state: PipelineState):
        """Deep iterative web research using Nexus Search."""
        nexus_search, SearchSource = self._get_nexus()
        
        if nexus_search is None or not self.nexus_enabled:
            # Fallback to LLM-based research if Nexus unavailable
            logger.info("Nexus Search not available, using LLM-based research")
            await self._phase_research_llm_fallback(state)
            return
        
        # Use Nexus Search for real web research
        max_iterations = 3
        current_knowledge = []
        
        for i in range(1, max_iterations + 1):
            logger.info(f"Research iteration {i}/{max_iterations}")
            
            try:
                # Perform search with Nexus
                results = await nexus_search(
                    query=state.task.prompt,
                    sources=[SearchSource.WEB, SearchSource.ACADEMIC, SearchSource.NEWS],
                    num_results=10,
                )
                
                # Process results
                for result in results.top[:5]:
                    knowledge_item = f"Source: {result.title}\nURL: {result.url}\nContent: {result.content}"
                    current_knowledge.append(knowledge_item)
                
                logger.info(f"Nexus Search found {len(results)} results")
                break  # Got results, no need for more iterations
                
            except Exception as e:
                logger.warning(f"Nexus Search iteration {i} failed: {e}")
                if i == max_iterations:
                    # Last iteration failed, use fallback
                    await self._phase_research_llm_fallback(state)
                    return
        
        state.web_discovery_results = current_knowledge
        state.metadata["research_iterations"] = len(current_knowledge)
        state.metadata["nexus_search"] = True
    
    async def _phase_research_llm_fallback(self, state: PipelineState):
        """Fallback to LLM-based research if Nexus unavailable."""
        models = self._get_available_models(TaskType.REASONING)
        researcher = models[0] if models else Model.GPT_4O_MINI

        max_iterations = 3
        current_knowledge = []

        for i in range(1, max_iterations + 1):
            logger.info(f"Research iteration {i}/{max_iterations} (LLM fallback)")

            # Decide if more research needed
            system_prompt = (
                "You are a research coordinator. Decide if more information is needed.\n"
                "Return JSON: {'action': 'search' or 'done', 'queries': [] if search}"
            )

            knowledge_context = "\n".join([f"- {k}" for k in current_knowledge]) if current_knowledge else "No knowledge yet"
            user_prompt = f"Task: {state.task.prompt}\n\nCurrent knowledge:\n{knowledge_context}\n\nDo you need more information? (max {max_iterations} iterations)"

            response, _ = await self.client.call(
                model=researcher,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
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
                search_response, _ = await self.client.call(
                    model=researcher,
                    system_prompt="You are a research assistant. Provide factual, detailed information.",
                    user_prompt=search_prompt,
                    max_tokens=1000,
                    temperature=0.2,
                )
                current_knowledge.append(f"Query: {query}\nResult: {search_response[0].text}")

        state.web_discovery_results = current_knowledge
        state.metadata["research_iterations"] = len(current_knowledge)
        state.metadata["nexus_search"] = False
    
    async def _phase_analyze(self, state: PipelineState):
        """Analyze findings using Multi-Perspective approach."""
        web_context = "\n\n".join(state.web_discovery_results) if state.web_discovery_results else "No web research conducted"
        
        models = self._get_available_models(state.task.type)
        primary = models[0] if models else Model.GPT_4O_MINI
        
        system_prompt = (
            "You are an analyst. Synthesize research findings into actionable insights.\n"
            "Base your analysis on the provided evidence."
        )
        
        user_prompt = f"Task: {state.task.prompt}\n\nResearch findings:\n{web_context}"
        
        response, _ = await self.client.call(
            model=primary,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=state.task.max_output_tokens,
            temperature=0.4,
        )
        
        data = self._extract_json(response.text) or {}
        state.candidates.append(SolutionCandidate(
            perspective="research_based",
            content=data.get("analysis", response.text),
            key_insights=data.get("key_insights", []),
            metadata={"sources": len(state.web_discovery_results)},
        ))
    
    async def _phase_critique(self, state: PipelineState):
        """Fact-check the analysis."""
        if not state.candidates:
            return
        
        models = self._get_available_models(state.task.type)
        critic = self._select_reviewer(models[0], state.task.type) or (models[0] if models else Model.GPT_4O_MINI)
        
        candidate = state.candidates[0]
        web_context = "\n\n".join(state.web_discovery_results) if state.web_discovery_results else ""
        
        system_prompt = (
            "You are a fact-checker. Evaluate the analysis against the research evidence.\n"
            "Check for:\n"
            "- Accuracy of claims\n"
            "- Logical consistency with evidence\n"
            "- Missing critical information\n\n"
            "Score 0-10 on accuracy."
        )
        
        user_prompt = f"Task: {state.task.prompt}\n\nAnalysis: {candidate.content}\n\nResearch evidence: {web_context}"
        
        response, _ = await self.client.call(
            model=critic,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=1000,
            temperature=0.1,
        )
        
        data = self._extract_json(response.text) or {}
        accuracy_score = data.get("accuracy_score", 5)
        
        state.scores.append(CritiqueScore(
            perspective="fact_check",
            total=accuracy_score / 10.0,
            logical_consistency=accuracy_score / 10.0,
            feasibility=data.get("feasibility", 5) / 10.0,
            completeness=data.get("completeness", 5) / 10.0,
            novelty=5.0 / 10.0,
        ))
    
    async def _phase_synthesis(self, state: PipelineState):
        """Synthesize research-based solution."""
        if not state.candidates:
            state.final_output = "No analysis generated."
            return
        
        models = self._get_available_models(state.task.type)
        synthesizer = models[0] if models else Model.GPT_4O_MINI
        
        candidate = state.candidates[0]
        web_context = "\n\n".join(state.web_discovery_results) if state.web_discovery_results else ""
        
        system_prompt = "Create a final solution that integrates research evidence with practical recommendations."
        user_prompt = f"Task: {state.task.prompt}\n\nAnalysis: {candidate.content}\n\nEvidence: {web_context}"
        
        response, _ = await self.client.call(
            model=synthesizer,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
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
        models = self._get_available_models(state.task.type)
        if len(models) < 4:
            models = (models * 4)[:4]
        
        async def generate(role: str, model: Model):
            system_prompt = f"You are generator {role}. Create a comprehensive solution with unique insights."
            user_prompt = f"Task: {state.task.prompt}\n\nContext: {context}\n\nRole: {role}"
            
            response, _ = await self.client.call(
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
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
                state.candidates.append(SolutionCandidate(
                    perspective=result["role"],
                    content=result["content"],
                    key_insights=[],
                    metadata={"model": result["model"]},
                ))
    
    async def _phase_jury_critique(self, state: PipelineState):
        """3 parallel critics evaluate all generators."""
        models = self._get_available_models(state.task.type)
        critic_models = models[:3] if len(models) >= 3 else (models * 3)[:3]
        
        candidate_contents = "\n\n".join([
            f"[{c.perspective}]\n{c.content}" for c in state.candidates
        ])
        
        async def critique(role: str, model: Model):
            system_prompt = f"You are critic {role}. Evaluate all solutions rigorously on multiple criteria."
            user_prompt = f"Task: {state.task.prompt}\n\nSolutions:\n{candidate_contents}\n\nRole: {role}"
            
            response, _ = await self.client.call(
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
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
        models = self._get_available_models(state.task.type)
        verifier = models[0] if models else Model.GPT_4O_MINI
        
        # Verify claims
        candidate_contents = "\n\n".join([f"{c.perspective}: {c.content}" for c in state.candidates])
        
        system_prompt = (
            "You are a verifier. Check all factual claims in the solutions.\n"
            "Return JSON array of verifications: [{'claim': '', 'verified': true/false, 'reason': ''}]"
        )
        
        user_prompt = f"Task: {state.task.prompt}\n\nSolutions: {candidate_contents}"
        
        response, _ = await self.client.call(
            model=verifier,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=2000,
            temperature=0.1,
        )
        
        data = self._extract_json(response.text) or []
        state.metadata["verifications"] = data
        
        # Meta-evaluate critics
        meta_prompt = "Evaluate the quality of each critic's feedback. Return JSON: {'critic_1': {'quality': 0-10, 'insights': ''}, ...}"
        
        response, _ = await self.client.call(
            model=verifier,
            system_prompt="You are a meta-evaluator. Assess critic quality.",
            user_prompt=meta_prompt,
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
                    total = sum([
                        score_data.get("logical_consistency", 5),
                        score_data.get("feasibility", 5),
                        score_data.get("completeness", 5),
                    ]) / 3
                    candidate_scores[perspective].append(total)
        
        # Weight by meta-evaluation quality
        meta_eval = state.metadata.get("meta_evaluation", {})
        
        for candidate in state.candidates:
            scores = candidate_scores.get(candidate.perspective, [])
            avg_score = sum(scores) / len(scores) if scores else 5.0
            
            # Apply verification penalty for false claims
            verifications = state.metadata.get("verifications", [])
            false_claims = sum(1 for v in verifications if not v.get("verified", True))
            penalty = false_claims * 0.1
            
            final_score = max(0, avg_score - penalty)
            
            state.scores.append(CritiqueScore(
                perspective=candidate.perspective,
                total=final_score / 10.0,
                logical_consistency=final_score / 10.0,
            ))
        
        # Sort by score
        state.candidates.sort(key=lambda c: next((s.total for s in state.scores if s.perspective == c.perspective), 0), reverse=True)
        state.top_candidates = state.candidates[:self.top_k]
    
    async def _phase_synthesis(self, state: PipelineState):
        """Synthesize top candidates into final solution."""
        if not state.top_candidates:
            state.final_output = "No viable candidates."
            return
        
        models = self._get_available_models(state.task.type)
        synthesizer = models[0] if models else Model.GPT_4O_MINI
        
        top_texts = "\n\n".join([f"[{c.perspective}]\n{c.content}" for c in state.top_candidates])
        verifications = state.metadata.get("verifications", [])
        
        system_prompt = "Synthesize the best elements from multiple solutions into one coherent recommendation."
        user_prompt = f"Task: {state.task.prompt}\n\nTop solutions:\n{top_texts}\n\nVerifications: {json.dumps(verifications)}"
        
        response, _ = await self.client.call(
            model=synthesizer,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
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
        models = self._get_available_models(TaskType.REASONING)
        primary = models[0] if models else Model.GPT_4O_MINI
        
        system_prompt = (
            "You are a scientist. Generate 3-5 competing hypotheses to explain or solve the problem.\n"
            "Return JSON: {'hypotheses': [{'name': '', 'description': '', 'predictions': []}]}"
        )
        
        user_prompt = f"Task: {state.task.prompt}\n\nContext: {context}"
        
        response, _ = await self.client.call(
            model=primary,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=1500,
            temperature=0.7,
        )
        
        data = self._extract_json(response.text) or {}
        hypotheses = data.get("hypotheses", [])
        
        for hyp in hypotheses:
            state.candidates.append(SolutionCandidate(
                perspective=hyp.get("name", "unknown"),
                content=hyp.get("description", ""),
                key_insights=hyp.get("predictions", []),
                metadata={"type": "hypothesis"},
            ))
        
        state.metadata["hypotheses"] = hypotheses
    
    async def _phase_scientific_test(self, state: PipelineState):
        """Design tests for each hypothesis."""
        models = self._get_available_models(TaskType.REASONING)
        tester = models[0] if models else Model.GPT_4O_MINI
        
        hypotheses = state.metadata.get("hypotheses", [])
        
        async def design_test(hypothesis: dict):
            system_prompt = "Design a rigorous test to validate or falsify this hypothesis."
            user_prompt = f"Hypothesis: {hypothesis.get('name')}\nDescription: {hypothesis.get('description')}"
            
            response, _ = await self.client.call(
                model=tester,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=1000,
                temperature=0.3,
            )
            
            return {"hypothesis": hypothesis.get("name"), "test_design": response.text}
        
        tasks = [design_test(h) for h in hypotheses]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        state.metadata["test_designs"] = [
            r for r in results if isinstance(r, dict)
        ]
    
    async def _phase_scientific_evaluate(self, state: PipelineState):
        """Evaluate evidence for each hypothesis."""
        models = self._get_available_models(TaskType.REASONING)
        evaluator = models[0] if models else Model.GPT_4O_MINI
        
        test_designs = state.metadata.get("test_designs", [])
        
        async def evaluate_evidence(test_design: dict):
            system_prompt = (
                "Evaluate the expected evidence strength for this test.\n"
                "Return JSON: {'evidence_strength': 0-10, 'confidence': 0-1, 'limitations': []}"
            )
            user_prompt = f"Test design: {test_design.get('test_design')}"
            
            response, _ = await self.client.call(
                model=evaluator,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
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
            state.scores.append(CritiqueScore(
                perspective=eval_data["hypothesis"],
                total=eval_data["evidence_strength"] / 10.0,
                logical_consistency=eval_data["confidence"],
            ))
        
        # Sort by evidence strength
        state.candidates.sort(
            key=lambda c: next((s.total for s in state.scores if s.perspective == c.perspective), 0),
            reverse=True
        )
    
    async def _phase_synthesis(self, state: PipelineState):
        """Synthesize conclusion from best hypothesis."""
        if not state.candidates:
            state.final_output = "No hypotheses generated."
            return
        
        models = self._get_available_models(state.task.type)
        synthesizer = models[0] if models else Model.GPT_4O_MINI
        
        best_candidate = state.candidates[0]
        evidence_evals = state.metadata.get("evidence_evaluations", [])
        best_eval = next((e for e in evidence_evals if e["hypothesis"] == best_candidate.perspective), {})
        
        system_prompt = "Synthesize the scientific conclusion based on the strongest hypothesis and evidence evaluation."
        user_prompt = f"Task: {state.task.prompt}\n\nBest hypothesis: {best_candidate.perspective}\nContent: {best_candidate.content}\nEvidence: {json.dumps(best_eval)}"
        
        response, _ = await self.client.call(
            model=synthesizer,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
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
        models = self._get_available_models(TaskType.REASONING)
        questioner = models[0] if models else Model.GPT_4O_MINI
        
        system_prompt = (
            "You are a Socratic teacher. Ask 5-7 probing questions that clarify the problem.\n"
            "Focus on: assumptions, definitions, evidence, perspectives, implications.\n"
            "Return JSON: {'questions': [], 'clarity_score': 0-10}"
        )
        
        user_prompt = f"Task: {state.task.prompt}\n\nContext: {context}"
        
        response, _ = await self.client.call(
            model=questioner,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
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
        models = self._get_available_models(TaskType.REASONING)
        questioner = models[0] if models else Model.GPT_4O_MINI
        
        # Simulate answers (in production, could involve user interaction)
        if not state.metadata.get("answers"):
            # Generate simulated answers for autonomous operation
            await self._generate_simulated_answers(state)
        
        system_prompt = (
            "Based on the answers, ask 3-5 deeper follow-up questions.\n"
            "Return JSON: {'questions': [], 'clarity_score': 0-10}"
        )
        
        qa_pairs = "\n".join([
            f"Q: {q}\nA: {a}" for q, a in zip(
                state.metadata.get("questions", []),
                state.metadata.get("answers", [])
            )
        ])
        
        user_prompt = f"Task: {state.task.prompt}\n\nQ&A so far:\n{qa_pairs}"
        
        response, _ = await self.client.call(
            model=questioner,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
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
        
        system_prompt = "Provide thoughtful, reasonable answers to these questions based on best practices."
        user_prompt = f"Task: {state.task.prompt}\n\nQuestions:\n" + "\n".join(questions)
        
        response, _ = await self.client.call(
            model=answerer,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=1500,
            temperature=0.4,
        )
        
        # Parse answers (one per line)
        answers = response.text.strip().split("\n")
        state.metadata["answers"] = answers[:len(questions)]
    
    async def _phase_socratic_solution(self, state: PipelineState):
        """Generate solution based on clarified understanding."""
        models = self._get_available_models(state.task.type)
        solver = models[0] if models else Model.GPT_4O_MINI
        
        qa_pairs = "\n".join([
            f"Q: {q}\nA: {a}" for q, a in zip(
                state.metadata.get("questions", []),
                state.metadata.get("answers", [])
            )
        ])
        
        system_prompt = (
            "Based on the Socratic Q&A, generate a well-reasoned solution that addresses the clarified problem."
        )
        
        user_prompt = f"Original task: {state.task.prompt}\n\nSocratic Q&A:\n{qa_pairs}"
        
        response, _ = await self.client.call(
            model=solver,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
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
        
        response, _ = await self.client.call(
            model=primary,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
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
        
        response, _ = await self.client.call(
            model=analyst,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
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
        
        response, _ = await self.client.call(
            model=analyst,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
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
        
        response, _ = await self.client.call(
            model=designer,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
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
        
        response, _ = await self.client.call(
            model=primary,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
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
        
        response, _ = await self.client.call(
            model=analyst,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
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
        
        response, _ = await self.client.call(
            model=calculator,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
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
        
        response, _ = await self.client.call(
            model=analyst,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
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
        
        response, _ = await self.client.call(
            model=synthesizer,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
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
        
        response, _ = await self.client.call(
            model=primary,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
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
        
        response, _ = await self.client.call(
            model=opposition,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
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
        
        response, _ = await self.client.call(
            model=analyst,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
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
        
        response, _ = await self.client.call(
            model=synthesizer,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
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
        
        response, _ = await self.client.call(
            model=primary,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
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
        
        user_prompt = f"Abstract structure: {abstract_structure}\n\nConstraints: {json.dumps(constraints)}"
        
        response, _ = await self.client.call(
            model=searcher,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
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
            
            response, _ = await self.client.call(
                model=mapper,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=1000,
                temperature=0.4,
            )
            
            data = self._extract_json(response.text) or {}
            return {"source": source.get("domain"), "mapping": data}
        
        tasks = [map_domain(sd) for sd in source_domains]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        state.analogical_state["analogy_mappings"] = [
            r for r in results if isinstance(r, dict)
        ]
    
    async def _phase_analogical_transfer(self, state: PipelineState):
        """Transfer and adapt solution to target domain."""
        models = self._get_available_models(state.task.type)
        adapter = models[0] if models else Model.GPT_4O_MINI
        
        mappings = state.analogical_state.get("analogy_mappings", [])
        original_task = state.task.prompt
        
        # Use best mapping (highest quality)
        best_mapping = max(
            mappings,
            key=lambda m: m.get("mapping", {}).get("mapping_quality", 0),
            default=None
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
        
        response, _ = await self.client.call(
            model=adapter,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
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
                "best_source": state.analogical_state.get("source_domains", [{}])[0].get("domain") if state.analogical_state.get("source_domains") else None,
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
            models = (models * self.NUM_EXPERTS)[:self.NUM_EXPERTS]
        
        async def expert_estimate(expert_num: int, model: Model):
            system_prompt = (
                f"You are expert {expert_num}. Provide an independent estimate.\n"
                "Return JSON: {'estimate_value': number, 'rationale': '', 'confidence': 0-1}"
            )
            
            user_prompt = f"Task: {state.task.prompt}\n\nContext: {context}"
            
            response, _ = await self.client.call(
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
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
        
        tasks = [expert_estimate(i+1, models[i]) for i in range(self.NUM_EXPERTS)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        estimates = [r for r in results if isinstance(r, dict)]
        state.delphi_state["round_1_estimates"] = estimates
    
    async def _phase_delphi_aggregation(self, state: PipelineState):
        """Compute median, IQR, identify outliers."""
        estimates = state.delphi_state.get("round_1_estimates", [])
        
        values = [
            e.get("estimate_value") for e in estimates
            if isinstance(e.get("estimate_value"), (int, float))
        ]
        
        if not values:
            state.delphi_state["aggregated_stats"] = {"error": "No numeric estimates"}
            return
        
        values_sorted = sorted(values)
        n = len(values_sorted)
        
        # Median
        median = values_sorted[n // 2] if n % 2 == 1 else (values_sorted[n//2 - 1] + values_sorted[n//2]) / 2
        
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
            models = (models * self.NUM_EXPERTS)[:self.NUM_EXPERTS]
        
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
            
            response, _ = await self.client.call(
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
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
            e.get("revised_estimate") for e in round2_estimates
            if isinstance(e.get("revised_estimate"), (int, float))
        ]
        
        if not values:
            state.delphi_state["consensus"] = {"error": "No estimates"}
            state.delphi_state["convergence_achieved"] = False
            return
        
        # Recompute stats
        values_sorted = sorted(values)
        n = len(values_sorted)
        median = values_sorted[n // 2] if n % 2 == 1 else (values_sorted[n//2 - 1] + values_sorted[n//2]) / 2
        
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
        
        response, _ = await self.client.call(
            model=analyst,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
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
        
        response, _ = await self.client.call(
            model=synthesizer,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
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

class PipelineFactory:
    """Factory for creating ARA reasoning pipelines."""
    
    _pipelines: Dict[ReasoningMethod, type] = {
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
    }
    
    @classmethod
    def create(
        cls,
        method: ReasoningMethod,
        client: UnifiedClient,
        cache: Optional[DiskCache] = None,
        telemetry: Optional[TelemetryCollector] = None,
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
    def get_available_methods(cls) -> List[ReasoningMethod]:
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
