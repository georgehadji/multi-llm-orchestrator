"""
Critique Cycle — Generate → Critique → Revise Pipeline
=======================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Implements the iterative generate-critique-revise loop with plateau detection,
deterministic validation, and multi-perspective review.

Part of Engine Decomposition (Phase 1) - Extracted from engine.py
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

    Cycle Phases:
    1. GENERATE: Initial output from primary model
    2. CRITIQUE: Cross-model review for quality assessment
    3. REVISE: Incorporate feedback and regenerate
    4. EVALUATE: Score quality and check for plateau
    """

    # Plateau detection: score improvement threshold
    PLATEAU_THRESHOLD = 0.05  # 5% improvement needed to continue

    # Maximum iterations before forced stop
    DEFAULT_MAX_ITERATIONS = 5

    # Score thresholds
    EXCELLENCE_THRESHOLD = 0.95  # Early exit if exceeded
    ACCEPTABLE_THRESHOLD = 0.75  # Minimum acceptable score

    def __init__(
        self,
        client: UnifiedClient,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        enable_streaming: bool = False,
    ):
        """
        Initialize critique cycle.

        Args:
            client: Unified API client for LLM calls
            max_iterations: Maximum generate-critique cycles
            enable_streaming: Whether to use streaming responses
        """
        self.client = client
        self.max_iterations = max_iterations
        self.enable_streaming = enable_streaming

        # Streaming syntax validator state
        self._partial_output_buffer = ""

    async def run_cycle(
        self,
        task: Task,
        primary_model: Model,
        reviewer_model: Model | None,
        full_prompt: str,
    ) -> CritiqueState:
        """
        Run full generate-critique-revise-evaluate cycle.

        Args:
            task: Task to execute
            primary_model: Model for generation
            reviewer_model: Model for critique (cross-provider)
            full_prompt: Complete prompt with context

        Returns:
            CritiqueState with results
        """
        state = CritiqueState()

        for iteration in range(task.max_iterations):
            logger.info(f"  {task.id}: Iteration {iteration + 1}/{task.max_iterations}")

            # GENERATE phase
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

            # Post-process code output
            if task.type == TaskType.CODE_GEN:
                output = self._clean_code_output(output)

            # CRITIQUE phase
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

            # EVALUATE phase
            state.scores_history.append(score)

            # Check for improvement
            if score > state.best_score:
                logger.info(f"  {task.id}: Score improved {state.best_score:.2f} → {score:.2f}")
                state.best_output = output
                state.best_score = score
                state.best_critique = critique
            else:
                logger.info(f"  {task.id}: No improvement ({score:.2f} ≤ {state.best_score:.2f})")

            # Check for plateau (no improvement for 2 iterations)
            if len(state.scores_history) >= 2:
                recent_improvement = max(state.scores_history[-2:]) - min(state.scores_history[-2:])
                if recent_improvement < self.PLATEAU_THRESHOLD:
                    logger.info(
                        f"  {task.id}: Plateau detected (improvement {recent_improvement:.3f} < threshold)"
                    )
                    break

            # Early exit on excellence
            if score >= self.EXCELLENCE_THRESHOLD:
                logger.info(f"  {task.id}: Excellence threshold reached ({score:.2f})")
                break

            # REVISE phase (if not last iteration and score not excellent)
            if iteration < task.max_iterations - 1 and score < self.EXCELLENCE_THRESHOLD:
                # Build delta prompt with feedback
                attempt = AttemptRecord(
                    attempt_num=iteration + 1,
                    model_used=primary_model,
                    output_snippet=output[:500],
                    failure_reason=critique or "Score below excellence threshold",
                    validators_failed=[],
                )

                revise_prompt = DeltaPrompt.build(full_prompt, attempt)
                full_prompt = revise_prompt  # Use revised prompt for next iteration

        return state

    async def _generate(
        self,
        model: Model,
        prompt: str,
        task_type: TaskType,
        max_tokens: int,
    ) -> APIResponse | None:
        """
        Generate output with LLM.

        Args:
            model: Model to use
            prompt: Full prompt
            task_type: Type of task
            max_tokens: Maximum output tokens

        Returns:
            API response or None on failure
        """
        # Adjust parameters for reasoning models
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
        """
        Critique generated output with cross-model review.

        Args:
            model: Reviewer model
            original_prompt: Original task prompt
            generated_output: Output to critique
            task_type: Type of task

        Returns:
            API response with critique and score
        """
        critique_prompt = CritiquePrompt.build_score(
            original_prompt, generated_output, task_type.value
        )

        try:
            response = await self.client.call_with_retry(
                model=model,
                prompt=critique_prompt,
                max_tokens=1000,  # Critique doesn't need many tokens
                timeout=60,
            )
            return response
        except Exception as e:
            logger.error(f"Critique failed for {model.value}: {e}")
            return None

    def _extract_score(self, critique_text: str) -> float:
        """
        Extract score from critique response.

        Args:
            critique_text: Critique response text

        Returns:
            Score between 0.0 and 1.0
        """
        # Try to parse JSON
        try:
            # Look for JSON object in text
            json_match = re.search(r'\{[^}]*"score"[^}]*\}', critique_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                score = float(data.get("score", 0.5))
                return max(0.0, min(1.0, score))
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: look for score pattern
        score_match = re.search(r"score[:\s]+([0-9.]+)", critique_text, re.IGNORECASE)
        if score_match:
            try:
                score = float(score_match.group(1))
                return max(0.0, min(1.0, score))
            except ValueError:
                pass

        # Default: moderate score
        logger.warning("Could not extract score from critique, using default 0.5")
        return 0.5

    def _clean_code_output(self, text: str) -> str:
        """
        Post-process code output to remove LLM artifacts.

        Args:
            text: Raw LLM output

        Returns:
            Cleaned code
        """
        # Remove markdown code fences
        text = re.sub(r"^```\w*\n?", "", text, flags=re.MULTILINE)
        text = re.sub(r"\n?```\s*$", "", text, flags=re.MULTILINE)

        # Remove placeholder comments
        placeholder_patterns = [
            r"//\s*[Aa]dd\s+(?:content|code|your|more|placeholder).*?\n",
            r"//\s*[Rr]eplace\s+this.*?(?:\n|$)",
            r"#\s*[Aa]dd\s+(?:content|code|your|more).*?(?:\n|$)",
        ]

        for pattern in placeholder_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)

        # Clean up multiple blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def _get_model_params(
        self,
        model: Model,
        task_type: TaskType,
        max_tokens: int,
    ) -> tuple[int, int]:
        """
        Get model-specific timeout and token limits.

        Args:
            model: Model to get params for
            task_type: Type of task
            max_tokens: Requested max tokens

        Returns:
            Tuple of (timeout_seconds, effective_max_tokens)
        """
        from .models import MODEL_MAX_TOKENS, get_provider

        provider = get_provider(model)

        # Reasoning models need more time and tokens
        is_reasoning_model = ModelRegistry.is_reasoning_model(model.value)

        if is_reasoning_model:
            timeout = 240
            effective_max_tokens = (
                min(max_tokens * 2, 16384)
                if task_type in (TaskType.CODE_GEN, TaskType.CODE_REVIEW)
                else max_tokens
            )
        elif model.value == "deepseek/deepseek-chat":
            timeout = 180
            effective_max_tokens = max_tokens
        elif task_type in (TaskType.CODE_GEN, TaskType.CODE_REVIEW):
            timeout = 120
            effective_max_tokens = max_tokens
        else:
            timeout = 60
            effective_max_tokens = max_tokens

        # Apply model-specific limits
        model_limit = MODEL_MAX_TOKENS.get(model)
        if model_limit:
            effective_max_tokens = min(effective_max_tokens, model_limit)

        return timeout, effective_max_tokens

    def _validate_syntax(self, output: str) -> bool:
        """
        Validate Python syntax of generated code.

        Args:
            output: Code to validate

        Returns:
            True if syntax is valid
        """
        try:
            ast.parse(output)
            return True
        except SyntaxError:
            return False

    def _extract_function_name(self, code: str) -> str | None:
        """
        Extract main function name from generated code.

        Args:
            code: Python source code

        Returns:
            Function name or None
        """
        try:
            tree = ast.parse(code)

            # Look for first function definition
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if not node.name.startswith("__"):
                        return node.name

            # Fallback to class name
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    return node.name

        except SyntaxError:
            # Fallback to regex
            match = re.search(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", code)
            if match:
                return match.group(1)

            class_match = re.search(r"class\s+([a-zA-Z_][a-zA-Z0-9_]*)", code)
            if class_match:
                return class_match.group(1)

        return None
