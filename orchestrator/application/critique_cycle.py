"""
Critique Cycle — Generate → Critique → Revise Pipeline
=======================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Implements the iterative generate-critique-revise loop with plateau detection,
deterministic validation, and multi-perspective review.

Part of Application Layer (Phase 4) — Canonical location.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..model_registry import ModelRegistry
from ..models import AttemptRecord, TaskType
from ..prompt_builder import CritiquePrompt, DeltaPrompt

if TYPE_CHECKING:
    from ..api_clients import APIResponse, UnifiedClient
    from ..models import Model, Task

logger = logging.getLogger(__name__)


@dataclass
class CritiqueState:
    """State tracked during critique cycle."""

    best_output: str = ""
    best_score: float = 0.0
    best_critique: str = ""
    total_cost: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    degraded_count: int = 0
    scores_history: list[float] = field(default_factory=list)
    attempt_history: list[AttemptRecord] = field(default_factory=list)
    failed_validators: list[str] = field(default_factory=list)
    model_escalated: bool = False


class CritiqueCycle:
    """
    Implements the generate → critique → revise → evaluate pipeline.

    Responsibilities:
    1. Generate output with LLM
    2. Critique with cross-model review
    3. Revise based on feedback
    4. Evaluate quality and detect plateaus
    5. Enforce iteration limits
    """

    PLATEAU_THRESHOLD = 0.05
    DEFAULT_MAX_ITERATIONS = 5
    EXCELLENCE_THRESHOLD = 0.95
    ACCEPTABLE_THRESHOLD = 0.75

    def __init__(
        self,
        client: UnifiedClient,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        enable_streaming: bool = False,
    ):
        self.client = client
        self.max_iterations = max_iterations
        self.enable_streaming = enable_streaming
        self._partial_output_buffer = ""

    async def run_cycle(
        self,
        task: Task,
        primary_model: Model,
        reviewer_model: Model | None,
        full_prompt: str,
    ) -> CritiqueState:
        """Run full generate-critique-revise-evaluate cycle."""
        state = CritiqueState()

        for iteration in range(task.max_iterations):
            logger.info(f"  {task.id}: Iteration {iteration + 1}/{task.max_iterations}")

            generate_response = await self._generate(
                model=primary_model,
                prompt=full_prompt,
                task_type=task.type,
                max_tokens=task.max_output_tokens,
            )

            if generate_response is None:
                logger.error(f"  {task.id}: Generation failed")
                break

            output = generate_response.text
            state.total_cost += generate_response.cost_usd
            state.total_input_tokens += generate_response.input_tokens
            state.total_output_tokens += generate_response.output_tokens

            if task.type == TaskType.CODE_GEN:
                output = self._clean_code_output(output)

            critique = ""
            score = 0.0

            if reviewer_model:
                critique_response = await self._critique(
                    model=reviewer_model,
                    original_prompt=full_prompt,
                    generated_output=output,
                    task_type=task.type,
                )

                if critique_response:
                    critique = critique_response.text
                    score = self._extract_score(critique)
                    state.total_cost += critique_response.cost_usd
                    state.total_input_tokens += critique_response.input_tokens
                    state.total_output_tokens += critique_response.output_tokens

            state.scores_history.append(score)

            if score > state.best_score:
                logger.info(f"  {task.id}: Score improved {state.best_score:.2f} -> {score:.2f}")
                state.best_output = output
                state.best_score = score
                state.best_critique = critique
            else:
                logger.info(f"  {task.id}: No improvement ({score:.2f} <= {state.best_score:.2f})")

            if len(state.scores_history) >= 2:
                recent_improvement = max(state.scores_history[-2:]) - min(state.scores_history[-2:])
                if recent_improvement < self.PLATEAU_THRESHOLD:
                    logger.info(
                        f"  {task.id}: Plateau detected "
                        f"(improvement {recent_improvement:.3f} < threshold)"
                    )
                    break

            if score >= self.EXCELLENCE_THRESHOLD:
                logger.info(f"  {task.id}: Excellence threshold reached ({score:.2f})")
                break

            if iteration < task.max_iterations - 1 and score < self.EXCELLENCE_THRESHOLD:
                attempt = AttemptRecord(
                    attempt_num=iteration + 1,
                    model_used=primary_model,
                    output_snippet=output[:500],
                    failure_reason=critique or "Score below excellence threshold",
                    validators_failed=[],
                )
                revise_prompt = DeltaPrompt.build(full_prompt, attempt)
                full_prompt = revise_prompt

        return state

    async def _generate(
        self,
        model: Model,
        prompt: str,
        task_type: TaskType,
        max_tokens: int,
    ) -> APIResponse | None:
        timeout, effective_max_tokens = self._get_model_params(model, task_type, max_tokens)
        try:
            response = await self.client.call_with_retry(
                model=model,
                prompt=prompt,
                max_tokens=effective_max_tokens,
                timeout=timeout,
            )
            return response
        except Exception as e:
            logger.error(f"Generation failed for {model.value}: {e}")
            return None

    async def _critique(
        self,
        model: Model,
        original_prompt: str,
        generated_output: str,
        task_type: TaskType,
    ) -> APIResponse | None:
        critique_prompt = CritiquePrompt.build_score(
            original_prompt, generated_output, task_type.value
        )
        try:
            response = await self.client.call_with_retry(
                model=model,
                prompt=critique_prompt,
                max_tokens=1000,
                timeout=60,
            )
            return response
        except Exception as e:
            logger.error(f"Critique failed for {model.value}: {e}")
            return None

    def _extract_score(self, critique_text: str) -> float:
        try:
            json_match = re.search(r'\{[^}]*"score"[^}]*\}', critique_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                score = float(data.get("score", 0.5))
                return max(0.0, min(1.0, score))
        except (json.JSONDecodeError, ValueError):
            pass

        score_match = re.search(r"score[:\s]+([0-9.]+)", critique_text, re.IGNORECASE)
        if score_match:
            try:
                score = float(score_match.group(1))
                return max(0.0, min(1.0, score))
            except ValueError:
                pass

        logger.warning("Could not extract score from critique, using default 0.5")
        return 0.5

    def _clean_code_output(self, text: str) -> str:
        text = re.sub(r"^```\w*\n?", "", text, flags=re.MULTILINE)
        text = re.sub(r"\n?```\s*$", "", text, flags=re.MULTILINE)

        placeholder_patterns = [
            r"//\s*[Aa]dd\s+(?:content|code|your|more|placeholder).*?\n",
            r"//\s*[Rr]eplace\s+this.*?(?:\n|$)",
            r"#\s*[Aa]dd\s+(?:content|code|your|more).*?(?:\n|$)",
        ]

        for pattern in placeholder_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)

        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _get_model_params(
        self,
        model: Model,
        task_type: TaskType,
        max_tokens: int,
    ) -> tuple[int, int]:
        from ..models import MODEL_MAX_TOKENS

        is_reasoning_model = ModelRegistry.is_reasoning_model(model.value)

        if is_reasoning_model:
            timeout = 240
            effective_max_tokens = (
                min(max_tokens * 2, 16384)
                if task_type in (TaskType.CODE_GEN, TaskType.CODE_REVIEW)
                else max_tokens
            )
        elif model.value == "deepseek/deepseek-v4-flash":
            timeout = 180
            effective_max_tokens = max_tokens
        elif task_type in (TaskType.CODE_GEN, TaskType.CODE_REVIEW):
            timeout = 120
            effective_max_tokens = max_tokens
        else:
            timeout = 60
            effective_max_tokens = max_tokens

        model_limit = MODEL_MAX_TOKENS.get(model)
        if model_limit:
            effective_max_tokens = min(effective_max_tokens, model_limit)

        return timeout, effective_max_tokens

    def _validate_syntax(self, output: str) -> bool:
        try:
            ast.parse(output)
            return True
        except SyntaxError:
            return False

    def _extract_function_name(self, code: str) -> str | None:
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if not node.name.startswith("__"):
                        return node.name
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    return node.name
        except SyntaxError:
            match = re.search(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", code)
            if match:
                return match.group(1)
            class_match = re.search(r"class\s+([a-zA-Z_][a-zA-Z0-9_]*)", code)
            if class_match:
                return class_match.group(1)
        return None
